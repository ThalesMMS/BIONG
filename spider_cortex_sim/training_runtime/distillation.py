from __future__ import annotations

from .common import (
    CheckpointSelectionConfig,
    Dict,
    DistillationConfig,
    List,
    Path,
    Sequence,
    TrainingRegimeSpec,
    deepcopy,
    load_teacher_checkpoint,
)


class SimulationDistillationMixin:
    @staticmethod
    def _normalize_distillation_answer(answer: object) -> str:
        answer_text = str(answer or "inconclusive")
        return {
            "yes": "supported",
            "yes_after_rl_finetuning": "supported_after_rl_finetuning",
            "partially": "partially_supported",
            "no": "not_supported_yet",
        }.get(answer_text, answer_text if answer_text else "inconclusive")

    @staticmethod
    def _distillation_expressivity_assessment(
        *,
        teacher_summary: Dict[str, object],
        distilled_summary: Dict[str, object],
        finetuned_summary: Dict[str, object] | None,
        baseline_summary: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        """
        Classify whether a distilled student recovers or supports the teacher's performance using scenario success rates.
        
        When a baseline_summary is provided, the assessment is derived from the baseline comparison report; otherwise the assessment is computed by comparing the numeric `scenario_success_rate` values found in the supplied summaries and selecting one of: "inconclusive", "supported", "supported_after_rl_finetuning", or "not_supported_yet".
        
        Parameters:
            teacher_summary (Dict[str, object]): Evaluation summary for the teacher; expected to contain a numeric `scenario_success_rate`.
            distilled_summary (Dict[str, object]): Evaluation summary for the distilled student; expected to contain a numeric `scenario_success_rate`.
            finetuned_summary (Dict[str, object] | None): Evaluation summary for the student after optional RL fine-tuning; may be None. If present, expected to contain a numeric `scenario_success_rate`.
            baseline_summary (Dict[str, object] | None): Optional modular-from-scratch baseline evaluation summary; when provided the final `answer` and `rationale` are derived from the baseline comparison.
        
        Returns:
            Dict[str, object]: A dictionary containing:
                - "answer": one of "inconclusive", "supported", "supported_after_rl_finetuning", or "not_supported_yet".
                - "rationale": a short explanatory string for the chosen answer.
                - "teacher_scenario_success_rate": float success rate extracted from teacher_summary (defaults to 0.0 if missing).
                - "distilled_scenario_success_rate": float success rate extracted from distilled_summary (defaults to 0.0 if missing).
                - "finetuned_scenario_success_rate": float success rate extracted from finetuned_summary (0.0 if finetuned_summary is None or missing).
        """
        if baseline_summary is not None:
            from ..learning_comparison import build_distillation_comparison_report

            report = build_distillation_comparison_report(
                teacher=teacher_summary,
                modular_rl_from_scratch=baseline_summary,
                modular_distilled=distilled_summary,
                modular_distilled_plus_rl_finetuning=finetuned_summary,
            )
            return {
                "answer": SimulationDistillationMixin._normalize_distillation_answer(
                    report.get("answer", "inconclusive")
                ),
                "rationale": str(report.get("rationale", "")),
                "teacher_scenario_success_rate": float(
                    teacher_summary.get("scenario_success_rate", 0.0)
                ),
                "distilled_scenario_success_rate": float(
                    distilled_summary.get("scenario_success_rate", 0.0)
                ),
                "finetuned_scenario_success_rate": float(
                    0.0
                    if finetuned_summary is None
                    else finetuned_summary.get("scenario_success_rate", 0.0)
                ),
            }
        teacher_success = float(teacher_summary.get("scenario_success_rate", 0.0))
        distilled_success = float(
            distilled_summary.get("scenario_success_rate", 0.0)
        )
        finetuned_success = float(
            0.0 if finetuned_summary is None else finetuned_summary.get("scenario_success_rate", 0.0)
        )
        if teacher_success <= 0.0:
            answer = "inconclusive"
            rationale = (
                "Teacher performance did not establish an effective reference policy."
            )
        elif distilled_success > 0.0:
            answer = "supported"
            rationale = (
                "The student achieved non-zero post-distillation success against the same evaluation suite."
            )
        elif finetuned_summary is not None and finetuned_success > 0.0:
            answer = "supported_after_rl_finetuning"
            rationale = (
                "The student only recovered measurable success after RL fine-tuning."
            )
        else:
            answer = "not_supported_yet"
            rationale = (
                "The student failed to reproduce measurable success after distillation or fine-tuning."
            )
        return {
            "answer": answer,
            "rationale": rationale,
            "teacher_scenario_success_rate": teacher_success,
            "distilled_scenario_success_rate": distilled_success,
            "finetuned_scenario_success_rate": finetuned_success,
        }

    def _train_with_distillation(
        self,
        *,
        collection_episodes: int,
        evaluation_episodes: int,
        render_last_evaluation: bool,
        capture_evaluation_trace: bool,
        debug_trace: bool,
        checkpoint_selection: str,
        checkpoint_interval: int | None,
        checkpoint_dir: str | Path | None,
        checkpoint_scenario_names: Sequence[str] | None,
        selection_scenario_episodes: int,
        checkpoint_selection_config: CheckpointSelectionConfig,
        curriculum_profile: str,
        training_regime_spec: TrainingRegimeSpec,
        teacher_checkpoint: str | Path,
        distillation_config: DistillationConfig | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Run a full distillation workflow: collect teacher rollouts, distill a student, evaluate teacher and student, optionally fine-tune the student using a checkpoint selection strategy, and assemble a comprehensive summary.
        
        Parameters:
            collection_episodes (int): Number of episodes used to collect the teacher rollout dataset.
            evaluation_episodes (int): Number of episodes to use for evaluation runs.
            render_last_evaluation (bool): If True, render the final evaluation.
            capture_evaluation_trace (bool): If True, capture and return the evaluation trace.
            debug_trace (bool): If True, enable additional debug tracing during training.
            checkpoint_selection (str): Checkpoint selection strategy, e.g., "best" or other strategy that uses the history-based flow.
            checkpoint_interval (int | None): Interval (in episodes) between checkpoints when applicable; if None, uses resolved budget defaults.
            checkpoint_dir (str | Path | None): Directory to store or read checkpoints from when selection requires it.
            checkpoint_scenario_names (Sequence[str] | None): Optional list of scenario names used for checkpoint selection evaluations.
            selection_scenario_episodes (int): Number of episodes per scenario when running checkpoint selection evaluations.
            checkpoint_selection_config (CheckpointSelectionConfig): Configuration object for checkpoint selection behavior.
            curriculum_profile (str): Curriculum profile identifier to apply when running subsequent training/fine-tuning.
            training_regime_spec (TrainingRegimeSpec): Specification of training hyperparameters including distillation and fine-tuning regimes.
            teacher_checkpoint (str | Path): Path or identifier of the teacher checkpoint to load for rollouts and distillation.
            distillation_config (DistillationConfig | None): Optional distillation dataset and matching/loss configuration; when omitted, defaults are derived from training_regime_spec.
        
        Returns:
            tuple[Dict[str, object], List[Dict[str, object]]]: 
                - summary: A comprehensive training summary that includes evaluation histories and a `distillation` section describing dataset stats, distillation hyperparameters, loss curve, and a post-distillation comparison/assessment.
                - evaluation_trace: Captured evaluation trace entries when `capture_evaluation_trace` is True (may be an empty list otherwise).
        """
        teacher_brain, teacher_metadata = load_teacher_checkpoint(
            teacher_checkpoint
        )
        teacher_architecture = str(
            teacher_metadata.get("ablation_config", {}).get("architecture", "")
        )
        if (
            teacher_architecture in {"monolithic", "true_monolithic"}
            and not self.brain.config.is_modular
        ):
            raise ValueError(
                "A monolithic teacher requires a modular student architecture for distillation."
            )
        if distillation_config is None:
            resolved_distillation_config = DistillationConfig(
                teacher_checkpoint=teacher_checkpoint,
                distillation_epochs=int(training_regime_spec.distillation_epochs),
                temperature=float(training_regime_spec.distillation_temperature),
                module_lr=float(training_regime_spec.distillation_lr),
                motor_lr=float(training_regime_spec.distillation_lr),
                arbitration_lr=float(training_regime_spec.distillation_lr),
            )
        else:
            resolved_distillation_config = DistillationConfig(
                teacher_checkpoint=teacher_checkpoint,
                dataset_episodes=distillation_config.dataset_episodes,
                distillation_epochs=int(training_regime_spec.distillation_epochs),
                temperature=float(training_regime_spec.distillation_temperature),
                module_lr=float(training_regime_spec.distillation_lr),
                motor_lr=float(training_regime_spec.distillation_lr),
                arbitration_lr=float(training_regime_spec.distillation_lr),
                shuffle=distillation_config.shuffle,
                match_local_proposals=distillation_config.match_local_proposals,
                loss=distillation_config.loss,
            )
        dataset_episodes = (
            int(collection_episodes)
            if resolved_distillation_config.dataset_episodes is None
            else int(resolved_distillation_config.dataset_episodes)
        )
        dataset = self.collect_teacher_rollout(
            teacher_brain,
            episodes=dataset_episodes,
            teacher_metadata=teacher_metadata,
        )
        distillation_run_summary = self._execute_distillation_phase(
            dataset=dataset,
            distillation_config=resolved_distillation_config,
        )

        teacher_eval_scale = 0.0 if teacher_brain.config.is_modular else None
        student_eval_scale = 0.0 if self.brain.config.is_modular else None
        with self._temporary_brain(teacher_brain):
            teacher_evaluation = self._evaluate_active_brain_summary(
                evaluation_episodes=evaluation_episodes,
                evaluation_reflex_scale=teacher_eval_scale,
                episode_start=max(0, int(dataset_episodes)),
            )
        distilled_evaluation = self._evaluate_active_brain_summary(
            evaluation_episodes=evaluation_episodes,
            evaluation_reflex_scale=student_eval_scale,
            episode_start=max(0, int(dataset_episodes)),
        )

        self._latest_checkpointing_summary = None
        self.checkpoint_source = "final"
        if checkpoint_selection == "best":
            training_history, evaluation_history, evaluation_trace = (
                self._train_with_best_checkpoint_selection(
                    episodes=0,
                    evaluation_episodes=evaluation_episodes,
                    render_last_evaluation=render_last_evaluation,
                    capture_evaluation_trace=capture_evaluation_trace,
                    debug_trace=debug_trace,
                    checkpoint_interval=(
                        int(checkpoint_interval)
                        if checkpoint_interval is not None
                        else int(
                            self.budget_summary.get("resolved", {}).get(
                                "checkpoint_interval",
                                1,
                            )
                        )
                    ),
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_scenario_names=checkpoint_scenario_names,
                    selection_scenario_episodes=selection_scenario_episodes,
                    checkpoint_selection_config=checkpoint_selection_config,
                    reflex_anneal_final_scale=float(
                        training_regime_spec.anneal_target_scale
                    ),
                    reflex_annealing_schedule=training_regime_spec.annealing_schedule,
                    reflex_anneal_warmup_fraction=float(
                        training_regime_spec.anneal_warmup_fraction
                    ),
                    evaluation_reflex_scale=student_eval_scale,
                    curriculum_profile=curriculum_profile,
                    training_regime_spec=training_regime_spec,
                )
            )
        else:
            training_history, evaluation_history, evaluation_trace = self._train_histories(
                episodes=0,
                evaluation_episodes=evaluation_episodes,
                render_last_evaluation=render_last_evaluation,
                capture_evaluation_trace=capture_evaluation_trace,
                debug_trace=debug_trace,
                reflex_anneal_final_scale=float(training_regime_spec.anneal_target_scale),
                reflex_annealing_schedule=training_regime_spec.annealing_schedule,
                reflex_anneal_warmup_fraction=float(
                    training_regime_spec.anneal_warmup_fraction
                ),
                evaluation_reflex_scale=student_eval_scale,
                curriculum_profile=curriculum_profile,
                training_regime_spec=training_regime_spec,
            )

        self._set_training_regime_metadata(
            curriculum_profile=curriculum_profile,
            episodes=int(collection_episodes),
            curriculum_summary=self._latest_curriculum_summary,
            training_regime_spec=training_regime_spec,
        )
        self._latest_training_regime_summary.setdefault("resolved_budget", {})[
            "distillation_collection_episodes"
        ] = int(dataset_episodes)
        self._latest_training_regime_summary["resolved_budget"][
            "total_training_episodes"
        ] = int(dataset_episodes) + int(training_regime_spec.finetuning_episodes)

        summary = self._build_summary(training_history, evaluation_history)
        final_student_summary = (
            dict(summary.get("evaluation", {}).get("primary", {}))
            if isinstance(summary.get("evaluation"), dict)
            else {}
        )
        post_distillation_comparison = {
            "teacher": teacher_evaluation,
            # This path does not run a separate from-scratch RL baseline.
            "modular_rl_from_scratch": None,
            "modular_distilled": distilled_evaluation,
            "modular_distilled_plus_rl_finetuning": (
                final_student_summary if final_student_summary else None
            ),
        }
        post_distillation_comparison["assessment"] = (
            self._distillation_expressivity_assessment(
                teacher_summary=post_distillation_comparison["teacher"],
                distilled_summary=post_distillation_comparison[
                    "modular_distilled"
                ],
                finetuned_summary=post_distillation_comparison[
                    "modular_distilled_plus_rl_finetuning"
                ],
                baseline_summary=post_distillation_comparison[
                    "modular_rl_from_scratch"
                ],
            )
        )
        dataset_summary = dataset.to_summary()
        self._latest_distillation_summary = {
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_architecture": teacher_architecture,
            "dataset_episodes": int(dataset_summary.get("episodes", 0)),
            "dataset_sample_count": int(dataset_summary.get("samples", 0)),
            "distillation_epochs": int(training_regime_spec.distillation_epochs),
            "distillation_temperature": float(
                training_regime_spec.distillation_temperature
            ),
            "distillation_lr": float(training_regime_spec.distillation_lr),
            "final_loss": float(
                distillation_run_summary.get("final_epoch", {}).get(
                    "mean_total_loss",
                    0.0,
                )
            ),
            "loss_curve": [
                float(epoch_summary.get("mean_total_loss", 0.0))
                for epoch_summary in distillation_run_summary.get("history", [])
                if isinstance(epoch_summary, dict)
            ],
            "post_distillation_comparison": post_distillation_comparison,
        }
        summary["distillation"] = deepcopy(self._latest_distillation_summary)
        return summary, evaluation_trace

__all__ = ["SimulationDistillationMixin"]
