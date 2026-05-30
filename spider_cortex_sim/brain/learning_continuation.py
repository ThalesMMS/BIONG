from __future__ import annotations

from .learning_shared import *


class _BrainLearningContinuationMixin:
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
        continuation_weights = _BrainLearningContinuationMixin._continuation_auxiliary_weights(
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
