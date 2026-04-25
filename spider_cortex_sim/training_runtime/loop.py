from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

from .common import (
    AnnealingSchedule,
    CheckpointSelectionConfig,
    DistillationConfig,
    TrainingRegimeSpec,
    checkpoint_candidate_sort_key,
    resolve_checkpoint_selection_config,
    resolve_training_regime,
    validate_curriculum_profile,
)


class SimulationTrainingLoopMixin:
    def train(
        self,
        episodes: int,
        *,
        evaluation_episodes: int = 3,
        render_last_evaluation: bool = False,
        capture_evaluation_trace: bool = True,
        debug_trace: bool = False,
        checkpoint_selection: str = "none",
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_scenario_names: Sequence[str] | None = None,
        selection_scenario_episodes: int = 1,
        checkpoint_selection_config: CheckpointSelectionConfig | None = None,
        reflex_anneal_final_scale: float | None = None,
        curriculum_profile: str = "none",
        training_regime: str | TrainingRegimeSpec | None = None,
        teacher_checkpoint: str | Path | None = None,
        distillation_config: DistillationConfig | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Train the agent for a specified number of episodes, optionally perform checkpoint-based selection, and run a post-training evaluation.
        
        Performs training according to the resolved budget and curriculum, applies an optional reflex-annealing regime or named training regime (which may include late finetuning), optionally captures checkpoint candidates and selects/persists the best checkpoint, and produces a consolidated summary plus an optional evaluation trace.
        
        Parameters:
            episodes: Number of training episodes to execute.
            evaluation_episodes: Number of evaluation episodes to run after training.
            render_last_evaluation: Render the environment on the final evaluation episode when True.
            capture_evaluation_trace: Capture and return a per-tick trace for the final evaluation episode when True.
            debug_trace: Include expanded diagnostic fields in captured traces when True.
            checkpoint_selection: "none" to skip checkpoint candidate selection, "best" to capture and choose the best checkpoint.
            checkpoint_interval: Episode interval for capturing checkpoint candidates; when None the resolved budget default is used.
            checkpoint_dir: Destination directory to persist selected "best" and "last" checkpoints; persistence is skipped when None.
            checkpoint_scenario_names: Scenario names used to evaluate checkpoint candidates; defaults are used when None.
            selection_scenario_episodes: Episodes per scenario used when evaluating checkpoint candidates.
            checkpoint_selection_config: Normalized checkpoint selection settings including metric and penalty behavior.
            reflex_anneal_final_scale: If provided, target reflex scale for annealing during training; when None no custom anneal is applied. Mutually exclusive with providing a TrainingRegimeSpec via training_regime.
            curriculum_profile: Curriculum profile name to use during training (e.g., "none", "ecological_v1", "ecological_v2").
            training_regime: Optional named training regime or explicit TrainingRegimeSpec that defines annealing schedule and optional late finetuning.
            teacher_checkpoint: Teacher checkpoint directory required by the distillation training regime.
            distillation_config: Optional distillation dataset/loss overrides; the training regime remains the source of distillation epochs, temperature, and learning rate.
        
        Returns:
            summary: Consolidated training and evaluation summary suitable for serialization.
            evaluation_trace: Captured trace for the final evaluation run when capture_evaluation_trace is True; otherwise an empty list.
        
        Raises:
            ValueError: If checkpoint_selection is not one of "none" or "best", if curriculum_profile is not supported, or if both training_regime and reflex_anneal_final_scale are provided.
        """
        if checkpoint_selection not in {"none", "best"}:
            raise ValueError(
                "Invalid checkpoint_selection. Use 'none' or 'best'."
            )
        checkpoint_selection_config = resolve_checkpoint_selection_config(
            checkpoint_selection_config
        )
        self._latest_distillation_summary = None
        if training_regime is not None and reflex_anneal_final_scale is not None:
            raise ValueError(
                "training_regime and reflex_anneal_final_scale are mutually exclusive."
            )
        training_regime_spec: TrainingRegimeSpec | None = None
        if isinstance(training_regime, TrainingRegimeSpec):
            training_regime_spec = training_regime
        elif training_regime is not None:
            training_regime_spec = resolve_training_regime(str(training_regime))
        if (
            training_regime_spec is not None
            and bool(training_regime_spec.distillation_enabled)
            and teacher_checkpoint is None
            and distillation_config is None
        ):
            raise ValueError(
                "The distillation training regime requires teacher_checkpoint."
            )
        if teacher_checkpoint is None and distillation_config is not None:
            teacher_checkpoint = distillation_config.teacher_checkpoint
        if (
            training_regime_spec is not None
            and bool(training_regime_spec.distillation_enabled)
            and teacher_checkpoint is None
        ):
            raise ValueError(
                "The distillation training regime requires teacher_checkpoint."
            )
        curriculum_profile = validate_curriculum_profile(curriculum_profile)
        if checkpoint_selection == "best":
            checkpoint_candidate_sort_key(
                {},
                selection_config=checkpoint_selection_config,
            )
        self._set_runtime_budget(
            episodes=episodes,
            evaluation_episodes=evaluation_episodes,
            checkpoint_interval=checkpoint_interval,
        )
        self._latest_checkpointing_summary = None
        start_reflex_scale = float(self.brain.config.reflex_scale)
        if training_regime_spec is not None:
            reflex_annealing_schedule = training_regime_spec.annealing_schedule
            reflex_anneal_warmup_fraction = float(
                training_regime_spec.anneal_warmup_fraction
            )
            final_training_reflex_scale = (
                start_reflex_scale
                if reflex_annealing_schedule is AnnealingSchedule.NONE
                else float(training_regime_spec.anneal_target_scale)
            )
            training_regime_name = training_regime_spec.name
        else:
            reflex_annealing_schedule = (
                AnnealingSchedule.LINEAR
                if reflex_anneal_final_scale is not None
                else AnnealingSchedule.NONE
            )
            reflex_anneal_warmup_fraction = 0.0
            final_training_reflex_scale = (
                start_reflex_scale
                if reflex_anneal_final_scale is None
                else max(0.0, float(reflex_anneal_final_scale))
            )
            training_regime_name = (
                "custom_reflex_anneal"
                if reflex_anneal_final_scale is not None
                else "baseline"
            )
        final_runtime_reflex_scale = final_training_reflex_scale
        if (
            training_regime_spec is not None
            and int(training_regime_spec.finetuning_episodes) > 0
        ):
            final_runtime_reflex_scale = max(
                0.0,
                float(training_regime_spec.finetuning_reflex_scale),
            )
        primary_evaluation_reflex_scale = (
            0.0 if self.brain.config.is_modular else None
        )
        self._latest_reflex_schedule_summary = {
            "enabled": (
                episodes > 0
                and reflex_annealing_schedule is not AnnealingSchedule.NONE
                and abs(final_training_reflex_scale - start_reflex_scale) > 1e-8
            ),
            "start_scale": start_reflex_scale,
            "final_scale": final_training_reflex_scale,
            "episodes": int(episodes),
            "mode": reflex_annealing_schedule.value,
            "schedule": reflex_annealing_schedule.value,
            "warmup_fraction": reflex_anneal_warmup_fraction,
            "finetuning_episodes": (
                int(training_regime_spec.finetuning_episodes)
                if training_regime_spec is not None
                else 0
            ),
        }
        self._latest_evaluation_without_reflex_support = None
        self.checkpoint_source = "final"
        if (
            training_regime_spec is not None
            and bool(training_regime_spec.distillation_enabled)
        ):
            return self._train_with_distillation(
                collection_episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                render_last_evaluation=render_last_evaluation,
                capture_evaluation_trace=capture_evaluation_trace,
                debug_trace=debug_trace,
                checkpoint_selection=checkpoint_selection,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
                checkpoint_scenario_names=checkpoint_scenario_names,
                selection_scenario_episodes=selection_scenario_episodes,
                checkpoint_selection_config=checkpoint_selection_config,
                curriculum_profile=curriculum_profile,
                training_regime_spec=training_regime_spec,
                teacher_checkpoint=str(teacher_checkpoint),
                distillation_config=distillation_config,
            )
        if checkpoint_selection == "best":
            effective_checkpoint_interval = (
                int(checkpoint_interval)
                if checkpoint_interval is not None
                else int(
                    self.budget_summary.get("resolved", {}).get("checkpoint_interval", 1)
                )
            )
            training_history, evaluation_history, evaluation_trace = (
                self._train_with_best_checkpoint_selection(
                    episodes=episodes,
                    evaluation_episodes=evaluation_episodes,
                    render_last_evaluation=render_last_evaluation,
                    capture_evaluation_trace=capture_evaluation_trace,
                    debug_trace=debug_trace,
                    checkpoint_interval=effective_checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_scenario_names=checkpoint_scenario_names,
                    selection_scenario_episodes=selection_scenario_episodes,
                    checkpoint_selection_config=checkpoint_selection_config,
                    reflex_anneal_final_scale=final_training_reflex_scale,
                    reflex_annealing_schedule=reflex_annealing_schedule,
                    reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
                    evaluation_reflex_scale=primary_evaluation_reflex_scale,
                    curriculum_profile=curriculum_profile,
                    training_regime_spec=training_regime_spec,
                )
            )
        else:
            training_history, evaluation_history, evaluation_trace = self._train_histories(
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                render_last_evaluation=render_last_evaluation,
                capture_evaluation_trace=capture_evaluation_trace,
                debug_trace=debug_trace,
                reflex_anneal_final_scale=final_training_reflex_scale,
                reflex_annealing_schedule=reflex_annealing_schedule,
                reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
                evaluation_reflex_scale=primary_evaluation_reflex_scale,
                curriculum_profile=curriculum_profile,
                training_regime_spec=training_regime_spec,
            )
        if training_regime_spec is None:
            self._set_training_regime_metadata(
                curriculum_profile=curriculum_profile,
                episodes=int(episodes),
                curriculum_summary=self._latest_curriculum_summary,
                training_regime_name=training_regime_name,
                annealing_schedule=reflex_annealing_schedule,
                anneal_target_scale=final_training_reflex_scale,
                anneal_warmup_fraction=reflex_anneal_warmup_fraction,
            )
        summary = self._build_summary(training_history, evaluation_history)
        if evaluation_episodes > 0 and self.brain.config.is_modular:
            primary_evaluation_summary = self._label_evaluation_summary(
                self._aggregate_group(evaluation_history),
                eval_reflex_scale=0.0,
                competence_type="self_sufficient",
            )
            reflex_diagnostic_eval_scale = self._effective_reflex_scale(
                final_runtime_reflex_scale
            )
            _, evaluation_with_reflex_support, _ = self._train_histories(
                episodes=0,
                evaluation_episodes=evaluation_episodes,
                render_last_evaluation=False,
                capture_evaluation_trace=False,
                debug_trace=False,
                episode_start=episodes,
                reflex_anneal_final_scale=final_runtime_reflex_scale,
                reflex_annealing_schedule=reflex_annealing_schedule,
                reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
                evaluation_reflex_scale=reflex_diagnostic_eval_scale,
                curriculum_profile=curriculum_profile,
                preserve_training_metadata=True,
                training_regime_spec=training_regime_spec,
            )
            reflex_diagnostic_summary = self._aggregate_group(
                evaluation_with_reflex_support
            )
            reflex_diagnostic_summary = self._label_evaluation_summary(
                reflex_diagnostic_summary,
                eval_reflex_scale=reflex_diagnostic_eval_scale,
                competence_type="scaffolded",
            )
            summary["evaluation"] = self._evaluation_summary_block(
                primary=primary_evaluation_summary,
                self_sufficient=primary_evaluation_summary,
                scaffolded=reflex_diagnostic_summary,
            )
            summary["evaluation_with_reflex_support"] = {
                "eval_reflex_scale": reflex_diagnostic_eval_scale,
                "summary": reflex_diagnostic_summary,
                "delta_vs_primary_eval": {
                    "mean_reward_delta": round(
                        float(
                            reflex_diagnostic_summary["mean_reward"]
                            - primary_evaluation_summary["mean_reward"]
                        ),
                        6,
                    ),
                    "scenario_reflex_usage_delta": round(
                        float(
                            reflex_diagnostic_summary["mean_reflex_usage_rate"]
                            - primary_evaluation_summary["mean_reflex_usage_rate"]
                        ),
                        6,
                    ),
                },
            }
            self._latest_evaluation_without_reflex_support = {
                "eval_reflex_scale": 0.0,
                "primary": True,
                "summary": deepcopy(primary_evaluation_summary),
                "delta_vs_reflex_support_eval": {
                    "mean_reward_delta": round(
                        float(
                            primary_evaluation_summary["mean_reward"]
                            - reflex_diagnostic_summary["mean_reward"]
                        ),
                        6,
                    ),
                    "scenario_reflex_usage_delta": round(
                        float(
                            primary_evaluation_summary["mean_reflex_usage_rate"]
                            - reflex_diagnostic_summary["mean_reflex_usage_rate"]
                        ),
                        6,
                    ),
                },
            }
            summary["evaluation_without_reflex_support"] = deepcopy(
                self._latest_evaluation_without_reflex_support
            )
        return summary, evaluation_trace


__all__ = ["SimulationTrainingLoopMixin"]
