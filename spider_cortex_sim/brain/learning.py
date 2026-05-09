from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..direct_policy_affordances import (
    AFFORDANCE_GEOMETRY_TARGET_NAMES,
    AFFORDANCE_SHELTER_COLUMN_NAMES,
    AFFORDANCE_SHELTER_POSITION_NAMES,
    AFFORDANCE_SHELTER_ROLE_NAMES,
)
from ..direct_policy_options import OPTION_NAMES, OPTION_TO_INDEX
from ..distillation.dataset import (
    DistillationConfig,
    DistillationDataset,
    DistillationSample,
    DistillationLossConfig,
)
from ..interfaces import ACTION_CONTEXT_INTERFACE, ACTION_TO_INDEX
from ..modules import ModuleResult
from ..phase import PHASE_LABELS, PHASE_TO_INDEX
from ..nn import one_hot, softmax
from ..nn_utils import cross_entropy_loss, kl_divergence

from .types import BrainStep


class BrainLearningMixin:
    CONTINUATION_MARGIN = 1.0
    POST_REST_SEQUENCE_REPLAY_PASSES = 6
    POST_REST_SEQUENCE_REPLAY_LR_SCALE = 1.0

    @staticmethod
    def _post_rest_release_sequence_active(decision: BrainStep) -> bool:
        return bool(
            decision.teacher_action_target_stage in {
                "handoff_release",
                "handoff_hold",
                "handoff_continue",
                "handoff_forage_window",
                "handoff_post_rest_forage",
            }
            or decision.teacher_option_target_stage in {
                "option_reactivate",
                "option_post_rest_inside",
                "option_post_rest_forage",
                "option_forage_window",
            }
        )

    @staticmethod
    def _continuation_auxiliary_weights(decision: BrainStep) -> Dict[str, float]:
        weights = {
            "teacher_action": 1.0,
            "teacher_option": 1.0,
            "phase": 0.2,
            "affordance_blocked": 0.1,
            "affordance_role": 0.1,
            "geometry": 0.1,
            "shelter_column": 0.1,
            "shelter_position": 0.1,
            "transition_prediction": 0.1,
            "transition_rollout_prediction": 0.1,
        }
        scenario_name = str(decision.scenario_name or "")
        if scenario_name in {
            "continuous_survival_post_rest_inside_v1",
            "continuous_survival_post_rest_entrance_v1",
        }:
            weights["teacher_action"] = 3.0
            weights["teacher_option"] = 3.0
            weights["phase"] = 0.6
            weights["affordance_blocked"] = 0.2
            weights["affordance_role"] = 0.2
            weights["shelter_position"] = 0.2
            weights["transition_prediction"] = 0.2
            weights["transition_rollout_prediction"] = 0.2
        elif scenario_name == "continuous_survival_return_after_late_forage_v1":
            weights["phase"] = 0.6
            weights["affordance_blocked"] = 0.2
            weights["affordance_role"] = 0.2
            weights["geometry"] = 0.2
            weights["shelter_column"] = 0.2
            weights["shelter_position"] = 0.3
            weights["transition_prediction"] = 0.2
            weights["transition_rollout_prediction"] = 0.2
        elif scenario_name == "continuous_survival_re_rest_after_return_v1":
            weights["phase"] = 0.6
            weights["affordance_blocked"] = 0.2
            weights["affordance_role"] = 0.2
            weights["geometry"] = 0.2
            weights["shelter_position"] = 0.2
            weights["transition_prediction"] = 0.2
            weights["transition_rollout_prediction"] = 0.2
        if decision.teacher_action_target_stage in {
            "handoff_release",
            "handoff_hold",
            "handoff_continue",
            "handoff_forage_window",
            "handoff_post_rest_forage",
        }:
            weights["teacher_action"] = max(weights["teacher_action"], 3.0)
        if decision.teacher_option_target_stage in {
            "option_reactivate",
            "option_post_rest_inside",
            "option_post_rest_forage",
            "option_forage_window",
        }:
            weights["teacher_option"] = max(weights["teacher_option"], 3.0)
        if decision.phase_target in {
            "POST_REST_REACTIVATE",
            "RECOVERED_IN_SHELTER",
            "RETURN_AFTER_LATE_FORAGE",
        }:
            weights["phase"] = max(weights["phase"], 0.6)
        return weights

    @staticmethod
    def _continuation_replay_focus_active(decision: BrainStep) -> bool:
        continuation_weights = BrainLearningMixin._continuation_auxiliary_weights(
            decision
        )
        if 0 <= int(decision.teacher_action_target_idx):
            return True
        if 0 <= int(decision.teacher_option_target_idx):
            return True
        if (
            decision.phase_target_idx >= 0
            and float(continuation_weights["phase"]) > 0.2
        ):
            return True
        return False

    @staticmethod
    def _weighted_distribution(
        size: int,
        weights: Dict[int, float],
        *,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        probs = np.full(max(1, int(size)), float(epsilon), dtype=float)
        for idx, weight in weights.items():
            index = int(idx)
            if 0 <= index < probs.size:
                probs[index] += max(0.0, float(weight))
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            return np.full(probs.shape, 1.0 / max(1, probs.size), dtype=float)
        return probs / total

    def _post_rest_release_sequence_distillation_targets(
        self,
        decision: BrainStep,
    ) -> Dict[str, np.ndarray]:
        action_stage = str(decision.teacher_action_target_stage or "")
        option_stage = str(decision.teacher_option_target_stage or "")

        action_weights: Dict[int, float] = {}
        if action_stage == "handoff_release":
            action_weights = {
                int(ACTION_TO_INDEX["MOVE_UP"]): 0.84,
                int(ACTION_TO_INDEX["STAY"]): 0.08,
                int(ACTION_TO_INDEX["MOVE_RIGHT"]): 0.06,
                int(ACTION_TO_INDEX["MOVE_DOWN"]): 0.015,
                int(ACTION_TO_INDEX["MOVE_LEFT"]): 0.005,
            }
        elif action_stage == "handoff_hold":
            action_weights = {
                int(ACTION_TO_INDEX["STAY"]): 0.82,
                int(ACTION_TO_INDEX["MOVE_UP"]): 0.08,
                int(ACTION_TO_INDEX["MOVE_DOWN"]): 0.06,
                int(ACTION_TO_INDEX["MOVE_RIGHT"]): 0.03,
                int(ACTION_TO_INDEX["MOVE_LEFT"]): 0.01,
            }
        elif action_stage == "handoff_continue":
            action_weights = {
                int(ACTION_TO_INDEX["MOVE_DOWN"]): 0.84,
                int(ACTION_TO_INDEX["STAY"]): 0.08,
                int(ACTION_TO_INDEX["MOVE_RIGHT"]): 0.04,
                int(ACTION_TO_INDEX["MOVE_UP"]): 0.03,
                int(ACTION_TO_INDEX["MOVE_LEFT"]): 0.01,
            }
        elif action_stage in {"handoff_forage_window", "handoff_post_rest_forage"}:
            action_weights = {
                int(ACTION_TO_INDEX["MOVE_RIGHT"]): 0.86,
                int(ACTION_TO_INDEX["STAY"]): 0.07,
                int(ACTION_TO_INDEX["MOVE_DOWN"]): 0.03,
                int(ACTION_TO_INDEX["MOVE_UP"]): 0.03,
                int(ACTION_TO_INDEX["MOVE_LEFT"]): 0.01,
            }

        option_weights: Dict[int, float] = {}
        if option_stage in {"option_reactivate", "option_post_rest_inside"}:
            option_weights = {
                int(OPTION_TO_INDEX["POST_REST_REACTIVATE"]): 0.82,
                int(OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]): 0.08,
                int(OPTION_TO_INDEX["FORAGE"]): 0.05,
                int(OPTION_TO_INDEX["RETURN_TO_SHELTER"]): 0.03,
                int(OPTION_TO_INDEX["REST"]): 0.01,
                int(OPTION_TO_INDEX["ESCAPE"]): 0.01,
            }
        elif option_stage in {"option_post_rest_forage", "option_forage_window"}:
            option_weights = {
                int(OPTION_TO_INDEX["FORAGE"]): 0.82,
                int(OPTION_TO_INDEX["POST_REST_REACTIVATE"]): 0.1,
                int(OPTION_TO_INDEX["RETURN_TO_SHELTER"]): 0.04,
                int(OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]): 0.02,
                int(OPTION_TO_INDEX["REST"]): 0.01,
                int(OPTION_TO_INDEX["ESCAPE"]): 0.01,
            }

        phase_weights: Dict[int, float] = {}
        if action_stage in {"handoff_release", "handoff_hold", "handoff_continue"}:
            phase_weights = {
                int(PHASE_TO_INDEX["POST_REST_REACTIVATE"]): 0.78,
                int(PHASE_TO_INDEX["RECOVERED_IN_SHELTER"]): 0.16,
                int(PHASE_TO_INDEX["RETURNING_TO_SHELTER"]): 0.03,
                int(PHASE_TO_INDEX["INITIAL_FORAGE"]): 0.02,
                int(PHASE_TO_INDEX["RESTING"]): 0.01,
            }
        elif action_stage in {"handoff_forage_window", "handoff_post_rest_forage"}:
            phase_weights = {
                int(PHASE_TO_INDEX["POST_REST_REACTIVATE"]): 0.56,
                int(PHASE_TO_INDEX["LATE_FORAGE"]): 0.24,
                int(PHASE_TO_INDEX["INITIAL_FORAGE"]): 0.1,
                int(PHASE_TO_INDEX["RECOVERED_IN_SHELTER"]): 0.05,
                int(PHASE_TO_INDEX["RETURN_AFTER_LATE_FORAGE"]): 0.03,
                int(PHASE_TO_INDEX["RETURNING_TO_SHELTER"]): 0.02,
            }

        return {
            "action": self._weighted_distribution(self.action_dim, action_weights),
            "option": self._weighted_distribution(len(OPTION_NAMES), option_weights),
            "phase": self._weighted_distribution(len(PHASE_LABELS), phase_weights),
        }

    @staticmethod
    def _copy_continuation_targets(
        source: BrainStep,
        target: BrainStep,
    ) -> None:
        target.scenario_name = source.scenario_name
        target.phase_target = source.phase_target
        target.phase_target_idx = int(source.phase_target_idx)
        target.affordance_blocked_targets = np.asarray(
            source.affordance_blocked_targets,
            dtype=float,
        ).copy()
        target.affordance_role_targets = np.asarray(
            source.affordance_role_targets,
            dtype=int,
        ).copy()
        target.geometry_targets = np.asarray(
            source.geometry_targets,
            dtype=float,
        ).copy()
        target.shelter_column_targets = np.asarray(
            source.shelter_column_targets,
            dtype=int,
        ).copy()
        target.shelter_position_targets = np.asarray(
            source.shelter_position_targets,
            dtype=int,
        ).copy()
        target.transition_prediction_targets = np.asarray(
            source.transition_prediction_targets,
            dtype=float,
        ).copy()
        target.transition_rollout_prediction_targets = np.asarray(
            source.transition_rollout_prediction_targets,
            dtype=float,
        ).copy()
        target.teacher_action_target_idx = int(source.teacher_action_target_idx)
        target.teacher_action_target_name = source.teacher_action_target_name
        target.teacher_action_target_stage = source.teacher_action_target_stage
        target.teacher_option_target_idx = int(source.teacher_option_target_idx)
        target.teacher_option_target_name = source.teacher_option_target_name
        target.teacher_option_target_stage = source.teacher_option_target_stage

    @staticmethod
    def _continuation_margin_loss_and_grad(
        logits: np.ndarray,
        *,
        target_idx: int,
        competitor_idx: int,
        margin: float,
        scale: float,
    ) -> tuple[float, np.ndarray]:
        logits = np.asarray(logits, dtype=float)
        grad = np.zeros_like(logits, dtype=float)
        if (
            logits.size <= 0
            or target_idx < 0
            or competitor_idx < 0
            or target_idx >= logits.size
            or competitor_idx >= logits.size
            or target_idx == competitor_idx
            or scale <= 0.0
        ):
            return 0.0, grad
        gap = float(logits[target_idx] - logits[competitor_idx])
        shortfall = float(margin - gap)
        if shortfall <= 0.0:
            return 0.0, grad
        grad[target_idx] -= float(scale)
        grad[competitor_idx] += float(scale)
        return float(scale * shortfall), grad

    def _true_monolithic_continuation_replay_step(
        self,
        source_decision: BrainStep,
        *,
        lr_scale: float,
    ) -> Dict[str, float]:
        observation = source_decision.observation
        if not isinstance(observation, dict) or not observation:
            return {
                "active": False,
                "loss": 0.0,
                "grad_norm": 0.0,
            }
        replay_decision = self.act(
            observation,
            bus=None,
            sample=False,
            policy_mode="normal",
            training=True,
        )
        self._copy_continuation_targets(source_decision, replay_decision)
        continuation_weights = self._continuation_auxiliary_weights(replay_decision)
        if bool(
            getattr(
                self.config,
                "direct_policy_post_rest_release_sequence_replay_boost",
                False,
            )
            and self._post_rest_release_sequence_active(replay_decision)
        ):
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
            and self._post_rest_release_sequence_active(replay_decision)
        )
        post_rest_sequence_distill_loss = 0.0
        post_rest_sequence_distill_action_weight = 2.0
        post_rest_sequence_distill_option_weight = 1.5
        post_rest_sequence_distill_phase_weight = 1.0
        sequence_distill_targets = (
            self._post_rest_release_sequence_distillation_targets(replay_decision)
            if post_rest_sequence_distill_active
            else {}
        )
        handoff_teacher_grad_logits = np.zeros(self.action_dim, dtype=float)
        handoff_option_teacher_grad_logits = np.zeros_like(
            replay_decision.option_logits,
            dtype=float,
        )
        phase_grad_logits = np.zeros_like(replay_decision.phase_logits, dtype=float)
        affordance_blocked_grad_logits = np.zeros(self.action_dim, dtype=float)
        affordance_role_grad_logits = np.zeros(0, dtype=float)
        geometry_grad_logits = np.zeros(0, dtype=float)
        shelter_column_grad_logits = np.zeros(0, dtype=float)
        shelter_position_grad_logits = np.zeros(0, dtype=float)
        total_loss = 0.0
        continuation_margin_weight = float(
            getattr(self.config, "direct_policy_continuation_margin_weight", 0.0)
        )
        stay_action_idx = int(ACTION_TO_INDEX["STAY"])
        return_option_idx = int(OPTION_NAMES.index("RETURN_TO_SHELTER"))
        initial_forage_phase_idx = int(PHASE_LABELS.index("INITIAL_FORAGE"))

        if (
            self.config.direct_policy_handoff_teacher
            and 0 <= int(replay_decision.teacher_action_target_idx) < self.action_dim
            and replay_decision.total_logits.size == self.action_dim
        ):
            teacher_target = one_hot(
                int(replay_decision.teacher_action_target_idx),
                self.action_dim,
            )
            handoff_teacher_weight = float(continuation_weights["teacher_action"])
            total_loss += handoff_teacher_weight * cross_entropy_loss(
                replay_decision.total_logits,
                teacher_target,
            )
            handoff_teacher_grad_logits = handoff_teacher_weight * (
                softmax(replay_decision.total_logits) - teacher_target
            )
            if continuation_margin_weight > 0.0:
                margin_loss, margin_grad = self._continuation_margin_loss_and_grad(
                    replay_decision.total_logits,
                    target_idx=int(replay_decision.teacher_action_target_idx),
                    competitor_idx=stay_action_idx,
                    margin=self.CONTINUATION_MARGIN,
                    scale=continuation_margin_weight * handoff_teacher_weight,
                )
                total_loss += float(margin_loss)
                handoff_teacher_grad_logits += margin_grad
        if (
            post_rest_sequence_distill_active
            and replay_decision.total_logits.size == self.action_dim
        ):
            action_distill_target = np.asarray(
                sequence_distill_targets.get("action", ()),
                dtype=float,
            )
            if action_distill_target.size == self.action_dim:
                post_rest_sequence_distill_loss += (
                    post_rest_sequence_distill_action_weight
                    * cross_entropy_loss(
                        replay_decision.total_logits,
                        action_distill_target,
                    )
                )
                handoff_teacher_grad_logits += (
                    post_rest_sequence_distill_action_weight
                    * (
                        softmax(replay_decision.total_logits)
                        - action_distill_target
                    )
                )
        if (
            self.config.direct_policy_handoff_option_teacher
            and 0 <= int(replay_decision.teacher_option_target_idx) < len(OPTION_NAMES)
            and replay_decision.option_logits.size == len(OPTION_NAMES)
        ):
            option_teacher_target = one_hot(
                int(replay_decision.teacher_option_target_idx),
                len(OPTION_NAMES),
            )
            handoff_option_teacher_weight = float(
                continuation_weights["teacher_option"]
            )
            total_loss += handoff_option_teacher_weight * cross_entropy_loss(
                replay_decision.option_logits,
                option_teacher_target,
            )
            handoff_option_teacher_grad_logits = handoff_option_teacher_weight * (
                softmax(replay_decision.option_logits) - option_teacher_target
            )
            if continuation_margin_weight > 0.0:
                margin_loss, margin_grad = self._continuation_margin_loss_and_grad(
                    replay_decision.option_logits,
                    target_idx=int(replay_decision.teacher_option_target_idx),
                    competitor_idx=return_option_idx,
                    margin=self.CONTINUATION_MARGIN,
                    scale=continuation_margin_weight
                    * handoff_option_teacher_weight,
                )
                total_loss += float(margin_loss)
                handoff_option_teacher_grad_logits += margin_grad
        if (
            post_rest_sequence_distill_active
            and replay_decision.option_logits.size == len(OPTION_NAMES)
        ):
            option_distill_target = np.asarray(
                sequence_distill_targets.get("option", ()),
                dtype=float,
            )
            if option_distill_target.size == len(OPTION_NAMES):
                post_rest_sequence_distill_loss += (
                    post_rest_sequence_distill_option_weight
                    * cross_entropy_loss(
                        replay_decision.option_logits,
                        option_distill_target,
                    )
                )
                handoff_option_teacher_grad_logits += (
                    post_rest_sequence_distill_option_weight
                    * (
                        softmax(replay_decision.option_logits)
                        - option_distill_target
                    )
                )
        if (
            self.config.direct_policy_phase_head
            and replay_decision.phase_target_idx >= 0
            and replay_decision.phase_logits.size == len(PHASE_LABELS)
        ):
            phase_target = one_hot(
                replay_decision.phase_target_idx,
                len(PHASE_LABELS),
            )
            phase_weight = float(continuation_weights["phase"])
            total_loss += phase_weight * cross_entropy_loss(
                replay_decision.phase_logits,
                phase_target,
            )
            phase_grad_logits = phase_weight * (
                softmax(replay_decision.phase_logits) - phase_target
            )
            if continuation_margin_weight > 0.0:
                margin_loss, margin_grad = self._continuation_margin_loss_and_grad(
                    replay_decision.phase_logits,
                    target_idx=int(replay_decision.phase_target_idx),
                    competitor_idx=initial_forage_phase_idx,
                    margin=self.CONTINUATION_MARGIN,
                    scale=continuation_margin_weight * phase_weight,
                )
                total_loss += float(margin_loss)
                phase_grad_logits += margin_grad
        if (
            post_rest_sequence_distill_active
            and replay_decision.phase_logits.size == len(PHASE_LABELS)
        ):
            phase_distill_target = np.asarray(
                sequence_distill_targets.get("phase", ()),
                dtype=float,
            )
            if phase_distill_target.size == len(PHASE_LABELS):
                post_rest_sequence_distill_loss += (
                    post_rest_sequence_distill_phase_weight
                    * cross_entropy_loss(
                        replay_decision.phase_logits,
                        phase_distill_target,
                    )
                )
                phase_grad_logits += (
                    post_rest_sequence_distill_phase_weight
                    * (
                        softmax(replay_decision.phase_logits)
                        - phase_distill_target
                    )
                )
        total_loss += float(post_rest_sequence_distill_loss)
        if (
            self.config.direct_policy_affordance_head
            and replay_decision.affordance_blocked_logits.size == self.action_dim
            and replay_decision.affordance_blocked_targets.size == self.action_dim
        ):
            affordance_blocked_target = np.clip(
                np.asarray(replay_decision.affordance_blocked_targets, dtype=float),
                0.0,
                1.0,
            )
            blocked_probs = 1.0 / (
                1.0
                + np.exp(
                    -np.asarray(
                        replay_decision.affordance_blocked_logits,
                        dtype=float,
                    )
                )
            )
            blocked_weight = float(continuation_weights["affordance_blocked"])
            total_loss += float(
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
        affordance_role_dim = len(AFFORDANCE_SHELTER_ROLE_NAMES)
        expected_affordance_role_size = self.action_dim * affordance_role_dim
        if (
            self.config.direct_policy_affordance_head
            and replay_decision.affordance_role_logits.size
            == expected_affordance_role_size
            and replay_decision.affordance_role_targets.size == self.action_dim
        ):
            role_weight = float(continuation_weights["affordance_role"])
            affordance_role_grad_matrix = np.zeros(
                (self.action_dim, affordance_role_dim),
                dtype=float,
            )
            role_logits_matrix = np.asarray(
                replay_decision.affordance_role_logits,
                dtype=float,
            ).reshape(self.action_dim, affordance_role_dim)
            role_targets = np.asarray(
                replay_decision.affordance_role_targets,
                dtype=int,
            )
            role_loss = 0.0
            for action_idx, role_target_idx in enumerate(role_targets.tolist()):
                role_target = one_hot(role_target_idx, affordance_role_dim)
                role_loss += cross_entropy_loss(
                    role_logits_matrix[action_idx],
                    role_target,
                )
                affordance_role_grad_matrix[action_idx] = (
                    softmax(role_logits_matrix[action_idx]) - role_target
                )
            total_loss += float(role_weight * role_loss / max(1, self.action_dim))
            affordance_role_grad_logits = (
                role_weight
                * affordance_role_grad_matrix.reshape(-1)
                / max(1, self.action_dim)
            )
        geometry_dim = len(AFFORDANCE_GEOMETRY_TARGET_NAMES)
        expected_geometry_size = self.action_dim * geometry_dim
        if (
            self.config.direct_policy_geometry_head
            and replay_decision.geometry_logits.size == expected_geometry_size
            and replay_decision.geometry_targets.size == expected_geometry_size
        ):
            geometry_targets = np.clip(
                np.asarray(replay_decision.geometry_targets, dtype=float),
                0.0,
                1.0,
            )
            geometry_probs = 1.0 / (
                1.0 + np.exp(-np.asarray(replay_decision.geometry_logits, dtype=float))
            )
            geometry_weight = float(continuation_weights["geometry"])
            total_loss += float(
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
        shelter_column_dim = len(AFFORDANCE_SHELTER_COLUMN_NAMES)
        expected_shelter_column_size = self.action_dim * shelter_column_dim
        if (
            self.config.direct_policy_shelter_column_head
            and replay_decision.shelter_column_logits.size
            == expected_shelter_column_size
            and replay_decision.shelter_column_targets.size == self.action_dim
        ):
            shelter_column_weight = float(continuation_weights["shelter_column"])
            shelter_column_grad_matrix = np.zeros(
                (self.action_dim, shelter_column_dim),
                dtype=float,
            )
            shelter_column_logits_matrix = np.asarray(
                replay_decision.shelter_column_logits,
                dtype=float,
            ).reshape(self.action_dim, shelter_column_dim)
            shelter_column_targets = np.asarray(
                replay_decision.shelter_column_targets,
                dtype=int,
            )
            shelter_column_loss = 0.0
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
            total_loss += float(
                shelter_column_weight
                * shelter_column_loss
                / max(1, self.action_dim)
            )
            shelter_column_grad_logits = (
                shelter_column_weight
                * shelter_column_grad_matrix.reshape(-1)
                / max(1, self.action_dim)
            )
        shelter_position_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        expected_shelter_position_size = self.action_dim * shelter_position_dim
        if (
            self.config.direct_policy_shelter_position_head
            and replay_decision.shelter_position_logits.size
            == expected_shelter_position_size
            and replay_decision.shelter_position_targets.size == self.action_dim
        ):
            shelter_position_weight = float(
                continuation_weights["shelter_position"]
            )
            shelter_position_grad_matrix = np.zeros(
                (self.action_dim, shelter_position_dim),
                dtype=float,
            )
            shelter_position_logits_matrix = np.asarray(
                replay_decision.shelter_position_logits,
                dtype=float,
            ).reshape(self.action_dim, shelter_position_dim)
            shelter_position_targets = np.asarray(
                replay_decision.shelter_position_targets,
                dtype=int,
            )
            shelter_position_loss = 0.0
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
            total_loss += float(
                shelter_position_weight
                * shelter_position_loss
                / max(1, self.action_dim)
            )
            shelter_position_grad_logits = (
                shelter_position_weight
                * shelter_position_grad_matrix.reshape(-1)
                / max(1, self.action_dim)
            )
        transition_prediction_grad_logits = np.zeros(0, dtype=float)
        transition_rollout_prediction_grad_logits = np.zeros(0, dtype=float)
        if (
            getattr(self.config, "direct_policy_transition_prediction_head", False)
            and replay_decision.transition_prediction_logits.size
            == replay_decision.transition_prediction_targets.size
            and replay_decision.transition_prediction_logits.size > 0
        ):
            transition_prediction_targets = np.clip(
                np.asarray(
                    replay_decision.transition_prediction_targets,
                    dtype=float,
                ),
                -1.0,
                1.0,
            )
            transition_prediction_values = np.clip(
                np.asarray(
                    replay_decision.transition_prediction_logits,
                    dtype=float,
                ),
                -1.0,
                1.0,
            )
            transition_prediction_weight = float(
                continuation_weights["transition_prediction"]
            )
            total_loss += float(
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
        if (
            getattr(
                self.config,
                "direct_policy_transition_rollout_prediction_head",
                False,
            )
            and replay_decision.transition_rollout_prediction_logits.size
            == replay_decision.transition_rollout_prediction_targets.size
            and replay_decision.transition_rollout_prediction_logits.size > 0
        ):
            transition_rollout_prediction_targets = np.clip(
                np.asarray(
                    replay_decision.transition_rollout_prediction_targets,
                    dtype=float,
                ),
                -1.0,
                1.0,
            )
            transition_rollout_prediction_values = np.clip(
                np.asarray(
                    replay_decision.transition_rollout_prediction_logits,
                    dtype=float,
                ),
                -1.0,
                1.0,
            )
            transition_rollout_prediction_weight = float(
                continuation_weights["transition_rollout_prediction"]
            )
            total_loss += float(
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
        grad_norm = float(
            np.linalg.norm(
                np.concatenate(
                    [
                        np.asarray(handoff_teacher_grad_logits, dtype=float),
                        np.asarray(handoff_option_teacher_grad_logits, dtype=float),
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
            )
        )
        if grad_norm <= 0.0:
            return {
                "active": False,
                "loss": float(total_loss),
                "grad_norm": 0.0,
            }
        if self.TRUE_MONOLITHIC_POLICY_NAME not in self._frozen_modules:
            backward_kwargs = {
                "grad_policy_logits": handoff_teacher_grad_logits,
                "grad_value": 0.0,
                "lr": self.module_lr * float(lr_scale),
            }
            if (
                self.config.direct_policy_handoff_option_teacher
                and replay_decision.option_logits.size == len(OPTION_NAMES)
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
        return {
            "active": True,
            "loss": float(total_loss),
            "grad_norm": grad_norm,
        }

    @staticmethod
    def _normalized_teacher_probs(
        teacher_probs: np.ndarray,
        *,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        teacher_probs = np.asarray(teacher_probs, dtype=float)
        if teacher_probs.shape != shape:
            raise ValueError(
                "teacher_probs shape must match the student distribution shape."
            )
        teacher_probs = np.nan_to_num(
            teacher_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        teacher_probs = np.clip(teacher_probs, 0.0, None)
        total = float(np.sum(teacher_probs))
        if total <= 0.0 or not np.isfinite(total):
            return np.full(shape, 1.0 / max(1, int(np.prod(shape))), dtype=float)
        return teacher_probs / total

    def distill_step(
        self,
        observation: Dict[str, object],
        teacher_policy: np.ndarray,
        teacher_valence_logits: np.ndarray | None = None,
        *,
        teacher_policy_logits: np.ndarray | None = None,
        teacher_action_center_policy: np.ndarray | None = None,
        teacher_action_center_logits: np.ndarray | None = None,
        teacher_action_intent_idx: int | None = None,
        teacher_module_policies: Dict[str, np.ndarray] | None = None,
        temperature: float = 1.0,
        module_lr: float | None = None,
        motor_lr: float | None = None,
        arbitration_lr: float | None = None,
        match_local_proposals: bool = False,
        loss_weights: DistillationLossConfig | None = None,
    ) -> Dict[str, float]:
        resolved_module_lr = self.module_lr if module_lr is None else float(module_lr)
        resolved_motor_lr = self.motor_lr if motor_lr is None else float(motor_lr)
        resolved_arbitration_lr = (
            self.arbitration_lr if arbitration_lr is None else float(arbitration_lr)
        )
        resolved_loss_weights = (
            DistillationLossConfig()
            if loss_weights is None
            else loss_weights
        )
        decision = self.act(
            observation,
            bus=None,
            sample=False,
            policy_mode="normal",
            training=True,
        )

        final_policy_loss = 0.0
        final_policy_grad = np.zeros(self.action_dim, dtype=float)
        if np.asarray(teacher_policy, dtype=float).size == self.action_dim:
            teacher_policy_target = (
                softmax(teacher_policy_logits, temperature=temperature)
                if teacher_policy_logits is not None
                else self._normalized_teacher_probs(
                    teacher_policy,
                    shape=decision.policy.shape,
                )
            )
            final_policy_loss = (
                resolved_loss_weights.final_policy_weight
                * cross_entropy_loss(
                    decision.total_logits,
                    teacher_policy_target,
                    temperature=temperature,
                )
            )
            final_policy_grad = (
                resolved_loss_weights.final_policy_weight
                * (
                    softmax(decision.total_logits, temperature=temperature)
                    - teacher_policy_target
                )
            )

        action_center_loss = 0.0
        action_center_direct_grad = np.zeros(self.action_dim, dtype=float)
        if (
            teacher_action_center_policy is not None
            and np.asarray(teacher_action_center_policy, dtype=float).size == self.action_dim
        ):
            teacher_action_center_target = (
                softmax(teacher_action_center_logits, temperature=temperature)
                if teacher_action_center_logits is not None
                else self._normalized_teacher_probs(
                    teacher_action_center_policy,
                    shape=decision.action_center_policy.shape,
                )
            )
            action_center_loss = (
                resolved_loss_weights.action_center_weight
                * cross_entropy_loss(
                    decision.action_center_logits,
                    teacher_action_center_target,
                    temperature=temperature,
                )
            )
            action_center_direct_grad = (
                resolved_loss_weights.action_center_weight
                * (
                    softmax(
                        decision.action_center_logits,
                        temperature=temperature,
                    )
                    - teacher_action_center_target
                )
            )

        proposal_stage_grad = final_policy_grad + action_center_direct_grad
        proposal_grad_width = self.action_dim * len(decision.module_results)
        module_gradient_norms: Dict[str, float] = {}
        proposal_alignment_losses: Dict[str, float] = {}

        if self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
                )
            if self.TRUE_MONOLITHIC_POLICY_NAME not in self._frozen_modules:
                self.true_monolithic_policy.backward(
                    grad_policy_logits=final_policy_grad,
                    grad_value=0.0,
                    lr=resolved_module_lr,
                )
            module_gradient_norms[self.TRUE_MONOLITHIC_POLICY_NAME] = float(
                0.0
                if self.TRUE_MONOLITHIC_POLICY_NAME in self._frozen_modules
                else np.linalg.norm(final_policy_grad)
            )
        else:
            if self.motor_cortex is None or self.action_center is None:
                raise RuntimeError(
                    "Distillation requires action_center and motor_cortex for the configured architecture."
                )
            self.motor_cortex.backward(final_policy_grad, lr=resolved_motor_lr)
            action_center_input_grads = self.action_center.backward(
                grad_policy_logits=proposal_stage_grad,
                grad_value=0.0,
                lr=resolved_motor_lr,
            )
            proposal_input_grads = np.asarray(
                action_center_input_grads[:proposal_grad_width],
                dtype=float,
            )
            per_result_input_grads = proposal_input_grads.reshape(
                len(decision.module_results),
                self.action_dim,
            )
            if self.config.is_modular:
                if self.module_bank is None:
                    raise RuntimeError(
                        "Module bank unavailable for modular architecture."
                    )
                module_total_grads: Dict[str, np.ndarray] = {}
                trainable_module_names: List[str] = []
                for result, extra_grad in zip(
                    decision.module_results,
                    per_result_input_grads,
                    strict=True,
                ):
                    if not result.active:
                        continue
                    if result.name in self._frozen_modules:
                        module_gradient_norms[result.name] = 0.0
                        continue
                    total_grad = (
                        result.gate_weight
                        * (
                            np.asarray(proposal_stage_grad, dtype=float)
                            + np.asarray(extra_grad, dtype=float)
                        )
                    )
                    if (
                        match_local_proposals
                        and teacher_module_policies is not None
                        and result.name in teacher_module_policies
                    ):
                        teacher_module_policy = np.asarray(
                            teacher_module_policies[result.name],
                            dtype=float,
                        )
                        teacher_module_policy = self._normalized_teacher_probs(
                            teacher_module_policy,
                            shape=result.probs.shape,
                        )
                        local_grad = (
                            resolved_loss_weights.proposal_weight
                            * (
                                softmax(result.logits, temperature=temperature)
                                - teacher_module_policy
                            )
                        )
                        total_grad += local_grad
                        proposal_alignment_losses[result.name] = cross_entropy_loss(
                            result.logits,
                            teacher_module_policy,
                            temperature=temperature,
                        )
                    module_total_grads[result.name] = module_total_grads.get(
                        result.name,
                        np.zeros(self.action_dim, dtype=float),
                    ) + total_grad
                    module_gradient_norms[result.name] = float(
                        np.linalg.norm(total_grad)
                    )
                    trainable_module_names.append(result.name)
                if trainable_module_names:
                    original_active_names = list(self.module_bank._active_names)
                    try:
                        self.module_bank._active_names = trainable_module_names
                        self.module_bank.backward(
                            np.zeros(self.action_dim, dtype=float),
                            lr=resolved_module_lr,
                            aux_grads=module_total_grads,
                        )
                    finally:
                        self.module_bank._active_names = original_active_names
            else:
                if self.monolithic_policy is None:
                    raise RuntimeError(
                        "Monolithic policy unavailable for the configured architecture."
                    )
                grad_for_monolithic = proposal_stage_grad + proposal_input_grads
                if self.MONOLITHIC_POLICY_NAME not in self._frozen_modules:
                    self.monolithic_policy.backward(
                        grad_for_monolithic,
                        lr=resolved_module_lr,
                    )
                module_gradient_norms[self.MONOLITHIC_POLICY_NAME] = float(
                    0.0
                    if self.MONOLITHIC_POLICY_NAME in self._frozen_modules
                    else np.linalg.norm(grad_for_monolithic)
                )

        valence_loss = 0.0
        valence_grad_norm = 0.0
        student_valence_probs = np.zeros(0, dtype=float)
        if (
            teacher_valence_logits is not None
            and np.asarray(teacher_valence_logits, dtype=float).size > 0
            and decision.arbitration_decision is not None
            and self.arbitration_network is not None
            and self.arbitration_network.cache is not None
        ):
            teacher_valence_logits = np.asarray(teacher_valence_logits, dtype=float)
            student_valence_probs = np.array(
                [
                    float(
                        decision.arbitration_decision.valence_scores.get(name, 0.0)
                    )
                    for name in self.VALENCE_ORDER
                ],
                dtype=float,
            )
            teacher_valence_probs = softmax(
                teacher_valence_logits,
                temperature=temperature,
            )
            if student_valence_probs.shape == teacher_valence_probs.shape:
                grad_valence = (
                    resolved_loss_weights.valence_weight
                    * (
                        student_valence_probs
                        - teacher_valence_probs
                    )
                )
                valence_loss = (
                    resolved_loss_weights.valence_weight
                    * cross_entropy_loss(
                        np.array(
                            [
                                float(
                                    decision.arbitration_decision.valence_logits.get(
                                        name,
                                        0.0,
                                    )
                                )
                                for name in self.VALENCE_ORDER
                            ],
                            dtype=float,
                        ),
                        teacher_valence_probs,
                        temperature=temperature,
                    )
                )
                self.arbitration_network.backward(
                    grad_valence_logits=grad_valence,
                    grad_gate_adjustments=np.zeros(
                        self.arbitration_network.gate_dim,
                        dtype=float,
                    ),
                    grad_value=0.0,
                    lr=resolved_arbitration_lr,
                )
                valence_grad_norm = float(np.linalg.norm(grad_valence))

        total_loss = (
            final_policy_loss
            + action_center_loss
            + valence_loss
            + sum(proposal_alignment_losses.values()) * resolved_loss_weights.proposal_weight
        )
        return {
            "total_loss": float(total_loss),
            "final_policy_loss": float(final_policy_loss),
            "action_center_loss": float(action_center_loss),
            "valence_loss": float(valence_loss),
            "local_proposal_loss": float(
                sum(proposal_alignment_losses.values())
                * resolved_loss_weights.proposal_weight
            ),
            "policy_loss": float(final_policy_loss + action_center_loss),
            "policy_kl": float(
                0.0
                if np.asarray(teacher_policy, dtype=float).size != self.action_dim
                else kl_divergence(
                    decision.total_logits,
                    (
                        softmax(teacher_policy_logits, temperature=temperature)
                        if teacher_policy_logits is not None
                        else self._normalized_teacher_probs(
                            teacher_policy,
                            shape=decision.policy.shape,
                        )
                    ),
                    temperature=temperature,
                )
            ),
            "final_policy_kl_grad_norm": float(np.linalg.norm(final_policy_grad)),
            "action_center_grad_norm": float(np.linalg.norm(action_center_direct_grad)),
            "valence_grad_norm": float(valence_grad_norm),
            "module_gradient_norms": {
                name: float(value)
                for name, value in sorted(module_gradient_norms.items())
            },
            "teacher_action_intent_idx": (
                -1
                if teacher_action_intent_idx is None
                else int(teacher_action_intent_idx)
            ),
            "student_action_intent_idx": int(decision.action_intent_idx),
        }

    def _distillation_step(
        self,
        sample: DistillationSample,
        *,
        distillation_config: DistillationConfig,
    ) -> Dict[str, float]:
        return self.distill_step(
            sample.observation,
            sample.teacher_policy,
            sample.teacher_valence_logits,
            teacher_policy_logits=sample.teacher_total_logits,
            teacher_action_center_policy=sample.teacher_action_center_policy,
            teacher_action_center_logits=sample.teacher_action_center_logits,
            teacher_action_intent_idx=sample.teacher_action_intent_idx,
            teacher_module_policies=sample.teacher_module_policies,
            temperature=distillation_config.temperature,
            module_lr=distillation_config.module_lr,
            motor_lr=distillation_config.motor_lr,
            arbitration_lr=distillation_config.arbitration_lr,
            match_local_proposals=distillation_config.match_local_proposals,
            loss_weights=distillation_config.loss,
        )

    def distill(
        self,
        dataset: DistillationDataset,
        *,
        distillation_config: DistillationConfig,
    ) -> Dict[str, object]:
        if len(dataset) <= 0:
            raise ValueError("Distillation requires at least one collected sample.")
        epoch_summaries: List[Dict[str, object]] = []
        rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        for epoch in range(int(distillation_config.distillation_epochs)):
            total_loss = 0.0
            total_final_policy_loss = 0.0
            total_action_center_loss = 0.0
            total_valence_loss = 0.0
            total_local_proposal_loss = 0.0
            total_policy_kl = 0.0
            last_episode: int | None = None
            self.reset_hidden_states()
            for sample in dataset.iter_samples(
                shuffle=bool(distillation_config.shuffle),
                rng=rng,
            ):
                if (
                    distillation_config.shuffle
                    or last_episode is None
                    or int(sample.episode) != last_episode
                ):
                    self.reset_hidden_states()
                step_summary = self._distillation_step(
                    sample,
                    distillation_config=distillation_config,
                )
                total_loss += float(step_summary["total_loss"])
                total_final_policy_loss += float(
                    step_summary["final_policy_loss"]
                )
                total_action_center_loss += float(
                    step_summary["action_center_loss"]
                )
                total_valence_loss += float(step_summary["valence_loss"])
                total_local_proposal_loss += float(
                    step_summary["local_proposal_loss"]
                )
                total_policy_kl += float(step_summary["policy_kl"])
                last_episode = int(sample.episode)
            sample_count = max(1, len(dataset))
            epoch_summaries.append(
                {
                    "epoch": epoch + 1,
                    "samples": len(dataset),
                    "mean_total_loss": float(total_loss / sample_count),
                    "mean_final_policy_loss": float(
                        total_final_policy_loss / sample_count
                    ),
                    "mean_action_center_loss": float(
                        total_action_center_loss / sample_count
                    ),
                    "mean_valence_loss": float(total_valence_loss / sample_count),
                    "mean_local_proposal_loss": float(
                        total_local_proposal_loss / sample_count
                    ),
                    "mean_policy_kl": float(total_policy_kl / sample_count),
                }
            )
        return {
            "epochs": int(distillation_config.distillation_epochs),
            "samples": len(dataset),
            "shuffle": bool(distillation_config.shuffle),
            "loss": distillation_config.loss.to_summary(),
            "history": epoch_summaries,
            "final_epoch": dict(epoch_summaries[-1]),
        }

    def _compute_counterfactual_credit(
        self,
        module_results: List[ModuleResult],
        observation: Dict[str, np.ndarray],
        action_idx: int,
    ) -> Dict[str, float]:
        """
        Compute per-module credit weights by measuring how masking each module's proposal changes the probability of the selected action.
        
        For each module, the method evaluates a counterfactual policy obtained by zeroing that module's proposal logits before running the action-center forward pass (the action-center cache is not stored). The raw importance for a module is the signed change in the chosen-action probability (actual minus counterfactual). Returned weights are the normalized absolute importances so they sum to 1.0. If all importances are numerically zero, the method falls back to a uniform distribution over active modules (or over all modules if none are active).
        
        Parameters:
            module_results: Ordered list of ModuleResult objects whose proposal logits are concatenated to form the action-center input; the order determines each module's slice in the flat proposal vector.
            observation: Observation mapping used to build the action-context vector passed to the action-center.
            action_idx: Index of the action whose probability change is used to compute importance.
        
        Returns:
            Mapping from module name to a nonnegative weight that sums to 1.0. Inactive modules receive weight 0.0 except in the uniform-fallback case, where active modules share the mass equally (inactive modules remain 0.0).
        """
        if not module_results:
            return {}
        action_idx = int(action_idx)
        action_context_mapping = self._bound_action_context(observation)
        action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
        logits_by_module = [
            np.asarray(
                result.gated_logits if result.gated_logits is not None else result.logits,
                dtype=float,
            )
            for result in module_results
        ]
        logits_flat = np.concatenate(logits_by_module, axis=0)
        actual_proposal_sum = np.sum(np.stack(logits_by_module, axis=0), axis=0)
        actual_correction_logits, _ = self.action_center.forward(
            np.concatenate([logits_flat, action_context], axis=0),
            store_cache=False,
        )
        actual_policy = softmax(actual_proposal_sum + actual_correction_logits)

        raw_importance: Dict[str, float] = {}
        for module_index, result in enumerate(module_results):
            if not result.active:
                raw_importance[result.name] = 0.0
                continue
            counterfactual_logits_flat = logits_flat.copy()
            start = module_index * self.action_dim
            stop = start + self.action_dim
            counterfactual_logits_flat[start:stop] = 0.0
            counterfactual_correction_logits, _ = self.action_center.forward(
                np.concatenate([counterfactual_logits_flat, action_context], axis=0),
                store_cache=False,
            )
            counterfactual_policy = softmax(
                actual_proposal_sum
                - logits_by_module[module_index]
                + counterfactual_correction_logits
            )
            raw_importance[result.name] = float(
                actual_policy[action_idx] - counterfactual_policy[action_idx]
            )

        total = sum(abs(v) for v in raw_importance.values())
        if total > 1e-8:
            return {name: abs(value) / total for name, value in raw_importance.items()}
        active_results = [r for r in module_results if r.active]
        pool = active_results or module_results
        uniform = 1.0 / len(pool)
        pool_names = {r.name for r in pool}
        return {r.name: (uniform if r.name in pool_names else 0.0) for r in module_results}

    def _compute_route_mask_credit(
        self,
        module_results: List[ModuleResult],
        *,
        threshold: float,
        dominant_module: str = "",
    ) -> tuple[list[str], Dict[str, float]]:
        """
        Selects proposer modules that meet a route-mask causal threshold and computes normalized route-credit weights.
        
        If no modules are provided, returns ([], {}). If no modules are active, returns an empty selection and a weight map with 0.0 for each module. When one or more active modules meet or exceed `threshold`, those modules are returned as the selected route owners and their nonnegative weights are normalized to sum to 1. If no active module meets `threshold`, `dominant_module` is used when present among active modules; otherwise the single best active module is selected. If the selected modules' raw scores sum to a value too small for stable normalization, weights are distributed uniformly across the selected modules.
        
        Parameters:
            module_results (List[ModuleResult]): List of proposer module results to evaluate; each result must expose `name`, `active`, `gate_weight`, and `contribution_share`.
            threshold (float): Minimum raw route score (gate_weight * contribution_share) required to include a module as a route owner.
            dominant_module (str): Optional module name to prefer as the sole route owner when no module meets `threshold`.
        
        Returns:
            tuple[list[str], Dict[str, float]]: A pair consisting of the list of selected module names and a mapping from every module name to its route-credit weight (selected modules have positive weights that sum to 1, others map to 0.0).
        """
        if not module_results:
            return [], {}
        active_results = [result for result in module_results if result.active]
        if not active_results:
            return [], {result.name: 0.0 for result in module_results}
        raw_scores = {
            result.name: (
                max(0.0, float(result.gate_weight))
                * max(0.0, float(result.contribution_share))
                if result.active
                else 0.0
            )
            for result in module_results
        }
        active_names = [
            result.name
            for result in module_results
            if result.active and raw_scores.get(result.name, 0.0) >= threshold
        ]
        if not active_names:
            active_result_names = {result.name for result in active_results}
            if dominant_module in active_result_names:
                active_names = [dominant_module]
            else:
                best_result = max(
                    active_results,
                    key=lambda result: (
                        raw_scores.get(result.name, 0.0),
                        float(result.contribution_share),
                        -module_results.index(result),
                    ),
                )
                active_names = [best_result.name]
        active_name_set = set(active_names)
        active_total = sum(raw_scores.get(name, 0.0) for name in active_names)
        if active_total > 1e-8:
            route_credit_weights = {
                result.name: (
                    float(raw_scores.get(result.name, 0.0) / active_total)
                    if result.name in active_name_set
                    else 0.0
                )
                for result in module_results
            }
        else:
            uniform = 1.0 / float(len(active_names)) if active_names else 0.0
            route_credit_weights = {
                result.name: (uniform if result.name in active_name_set else 0.0)
                for result in module_results
            }
        return active_names, route_credit_weights

    def _auxiliary_module_gradients(self, module_results: List[ModuleResult]) -> Dict[str, np.ndarray]:
        """
        Compute auxiliary gradient targets for proposal modules based on reflex decisions.
        
        If auxiliary targets are disabled via the brain configuration, returns an empty dict.
        For each module result that is active, has a non-None reflex, and whose reflex specifies
        a positive auxiliary_weight, produces a gradient array equal to
        `auxiliary_weight * (probs - reflex.target_probs)`.
        
        Parameters:
            module_results (List[ModuleResult]): List of per-module proposal outputs; each item must
                expose `name`, `active`, `probs`, and `reflex.target_probs`/`reflex.auxiliary_weight`.
        
        Returns:
            aux_grads (Dict[str, np.ndarray]): Mapping from module name to the auxiliary gradient array
            for modules that contribute auxiliary targets. Modules that do not meet the conditions are
            omitted.
        """
        if not self.config.enable_auxiliary_targets:
            return {}
        aux_grads: Dict[str, np.ndarray] = {}
        for result in module_results:
            if not result.active or result.reflex is None:
                continue
            weight = result.reflex.auxiliary_weight
            if weight <= 0.0:
                continue
            aux_grads[result.name] = weight * (result.probs - result.reflex.target_probs)
        return aux_grads

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
