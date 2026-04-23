"""Simulation runtime facade.

Multi-seed comparison workflows and comparison statistics live in
`spider_cortex_sim.comparison`; claim-test evaluation lives in
`spider_cortex_sim.claim_evaluation`.

``default_behavior_evaluation`` and ``ensure_behavior_evaluation`` are the
public homes for the default behavior-evaluation helpers previously kept in
the CLI entrypoint.
"""

from __future__ import annotations

import inspect
import math
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence

from .ablations import (
    BrainAblationConfig,
    PROPOSAL_SOURCE_NAMES,
    default_brain_config,
    resolve_ablation_configs,
    resolve_ablation_scenario_group,
)
from .agent import BrainStep, SpiderBrain
from .bus import MessageBus
from .interfaces import MODULE_INTERFACES
from .maps import MAP_TEMPLATE_NAMES
from .metrics import (
    BehavioralEpisodeScore,
    EpisodeMetricAccumulator,
    EpisodeStats,
    PREDATOR_TYPE_NAMES,
    aggregate_behavior_scores,
    aggregate_episode_stats,
    competence_label_from_eval_reflex_scale,
    flatten_behavior_rows,
    normalize_competence_label,
    summarize_behavior_suite,
)
from .checkpointing import (
    CHECKPOINT_METRIC_ORDER,
    CHECKPOINT_PENALTY_MODE_NAMES,
    CheckpointPenaltyMode,
    CheckpointSelectionConfig,
    checkpoint_candidate_composite_score,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    file_sha256,
    jsonify_observation,
    mean_reward_from_behavior_payload,
    persist_checkpoint_pair,
    resolve_checkpoint_selection_config,
    resolve_checkpoint_load_dir,
)
from .curriculum import (
    CURRICULUM_COLUMNS,
    CURRICULUM_FOCUS_SCENARIOS,
    CURRICULUM_PROFILE_NAMES,
    SUBSKILL_CHECK_MAPPINGS,
    CurriculumPhaseDefinition,
    PromotionCheckCriteria,
    empty_curriculum_summary,
    evaluate_promotion_check_specs,
    promotion_check_spec_records,
    regime_row_metadata_from_summary,
    resolve_curriculum_phase_budgets,
    resolve_curriculum_profile,
    validate_curriculum_profile,
)
from .distillation import DistillationConfig
from .distillation.teacher import load_teacher_checkpoint
from .export import (
    compact_aggregate,
    compact_behavior_payload,
    jsonify,
    save_behavior_csv,
    save_summary,
    save_trace,
)
from .noise import (
    NoiseConfig,
    resolve_noise_profile,
)
from .operational_profiles import OperationalProfile, runtime_operational_profile
from .predator import PREDATOR_STATES
from .scenarios import (
    CAPABILITY_PROBE_SCENARIOS as SCENARIO_CAPABILITY_PROBE_SCENARIOS,
    BenchmarkTier,
    ProbeType,
    SCENARIO_NAMES,
    get_scenario,
)
from .training_regimes import (
    AnnealingSchedule,
    TrainingRegimeSpec,
    resolve_training_regime,
)
from .world import ACTIONS, REWARD_COMPONENT_NAMES, SpiderWorld

from .simulation_helpers import EXPERIMENT_OF_RECORD_REGIME, CAPABILITY_PROBE_SCENARIOS, default_behavior_evaluation, ensure_behavior_evaluation, is_capability_probe, _probe_metadata_for_scenario


class SimulationTrainingMixin:
    def _set_runtime_budget(
        self,
        *,
        episodes: int,
        evaluation_episodes: int,
        scenario_episodes: int | None = None,
        behavior_seeds: Sequence[int] | None = None,
        ablation_seeds: Sequence[int] | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        """
        Update the simulation's runtime budget by resolving and storing episode counts, step limits, seed lists, and checkpoint interval into the instance budget summary.
        
        Parameters:
            episodes (int): Number of training episodes to resolve into the budget.
            evaluation_episodes (int): Number of evaluation episodes to resolve into the budget.
            scenario_episodes (int | None): Number of episodes to run per scenario; if omitted, preserves existing value or defaults to 1.
            behavior_seeds (Sequence[int] | None): Sequence of integer seeds to use for behavior evaluations; if provided, replaces the budget's behavior seed list.
            ablation_seeds (Sequence[int] | None): Sequence of integer seeds to use for ablation evaluations; if provided, replaces the budget's ablation seed list.
            checkpoint_interval (int | None): Interval (in completed episodes) at which to capture checkpoint candidates; if provided, updates the resolved checkpoint interval.
        """
        budget = deepcopy(self.budget_summary)
        resolved = budget.setdefault("resolved", {})
        resolved["episodes"] = int(episodes)
        resolved["eval_episodes"] = int(evaluation_episodes)
        resolved["max_steps"] = int(self.max_steps)
        if scenario_episodes is not None:
            resolved["scenario_episodes"] = int(scenario_episodes)
        elif "scenario_episodes" not in resolved:
            resolved["scenario_episodes"] = 1
        if behavior_seeds is not None:
            resolved["behavior_seeds"] = [int(seed) for seed in behavior_seeds]
        if ablation_seeds is not None:
            resolved["ablation_seeds"] = [int(seed) for seed in ablation_seeds]
        if checkpoint_interval is not None:
            resolved["checkpoint_interval"] = int(checkpoint_interval)
        self.budget_summary = budget
        self.budget_profile_name = str(budget.get("profile", self.budget_profile_name))
        self.benchmark_strength = str(
            budget.get("benchmark_strength", self.benchmark_strength)
        )

    def set_runtime_budget(
        self,
        *,
        episodes: int,
        evaluation_episodes: int,
        scenario_episodes: int | None = None,
        behavior_seeds: Sequence[int] | None = None,
        ablation_seeds: Sequence[int] | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        """Package-internal API for entrypoints to update runtime budget state."""
        self._set_runtime_budget(
            episodes=episodes,
            evaluation_episodes=evaluation_episodes,
            scenario_episodes=scenario_episodes,
            behavior_seeds=behavior_seeds,
            ablation_seeds=ablation_seeds,
            checkpoint_interval=checkpoint_interval,
        )

    @contextmanager
    def _temporary_brain(self, brain: SpiderBrain) -> Iterator[None]:
        original_brain = self.brain
        self.brain = brain
        try:
            yield
        finally:
            self.brain = original_brain

    def _evaluate_active_brain_summary(
        self,
        *,
        evaluation_episodes: int,
        evaluation_reflex_scale: float | None,
        episode_start: int = 0,
    ) -> Dict[str, object]:
        _, evaluation_history, _ = self._train_histories(
            episodes=0,
            evaluation_episodes=evaluation_episodes,
            render_last_evaluation=False,
            capture_evaluation_trace=False,
            debug_trace=False,
            episode_start=episode_start,
            evaluation_reflex_scale=evaluation_reflex_scale,
            preserve_training_metadata=True,
            training_regime_spec=None,
        )
        effective_scale = (
            self._effective_reflex_scale(evaluation_reflex_scale)
            if evaluation_reflex_scale is not None
            else self._effective_reflex_scale(self.brain.current_reflex_scale)
        )
        competence_type = competence_label_from_eval_reflex_scale(effective_scale)
        return self._label_evaluation_summary(
            self._aggregate_group(evaluation_history),
            eval_reflex_scale=effective_scale,
            competence_type=competence_type,
        )

    @staticmethod
    def _distillation_expressivity_assessment(
        *,
        teacher_summary: Dict[str, object],
        distilled_summary: Dict[str, object],
        finetuned_summary: Dict[str, object] | None,
        baseline_summary: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        if baseline_summary is not None:
            from .comparison_learning import build_distillation_comparison_report

            report = build_distillation_comparison_report(
                teacher=teacher_summary,
                modular_rl_from_scratch=baseline_summary,
                modular_distilled=distilled_summary,
                modular_distilled_plus_rl_finetuning=finetuned_summary,
            )
            return {
                "answer": str(report.get("answer", "inconclusive")),
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
                episode_start=max(0, int(collection_episodes)),
            )
        distilled_evaluation = self._evaluate_active_brain_summary(
            evaluation_episodes=evaluation_episodes,
            evaluation_reflex_scale=student_eval_scale,
            episode_start=max(0, int(collection_episodes)),
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

    def _train_histories(
        self,
        *,
        episodes: int,
        evaluation_episodes: int,
        render_last_evaluation: bool,
        capture_evaluation_trace: bool,
        debug_trace: bool,
        episode_start: int = 0,
        reflex_anneal_final_scale: float | None = None,
        reflex_annealing_schedule: AnnealingSchedule | str = AnnealingSchedule.LINEAR,
        reflex_anneal_warmup_fraction: float = 0.0,
        evaluation_reflex_scale: float | None = None,
        curriculum_profile: str = "none",
        preserve_training_metadata: bool = False,
        training_regime_spec: TrainingRegimeSpec | None = None,
    ) -> tuple[List[EpisodeStats], List[EpisodeStats], List[Dict[str, object]]]:
        """
        Run a training schedule (including optional finetuning) and then run evaluation episodes, returning per-episode statistics and an optional evaluation trace.
        
        Training episodes are executed according to the configured schedule or the provided TrainingRegimeSpec; if `reflex_anneal_final_scale` is given the brain's runtime reflex scale is annealed per episode toward that value (clamped to >= 0.0). When `training_regime_spec` contains a finetuning phase, its finetuning episodes and reflex scale extend the total training episodes and determine the post-training reflex scale. If `evaluation_reflex_scale` is provided it temporarily overrides the brain's runtime reflex scale for all evaluation episodes and is restored afterward. When trace capture is enabled, only the last evaluation run that produced a non-empty trace is returned.
        
        Parameters:
            episodes (int): Number of training episodes to execute (does not include finetuning episodes from a TrainingRegimeSpec).
            evaluation_episodes (int): Number of evaluation episodes to run after training.
            render_last_evaluation (bool): If true, render the final evaluation episode.
            capture_evaluation_trace (bool): If true, capture a trace for the final evaluation episode (if any).
            debug_trace (bool): If true and trace capture is enabled, include extended debug fields in the captured trace.
            episode_start (int): Base index added to episode indices when running episodes.
            reflex_anneal_final_scale (float | None): Target reflex scale for annealing during training; if `None`, the brain's configured reflex scale is used.
            reflex_annealing_schedule (AnnealingSchedule | str): Annealing schedule to use during training.
            reflex_anneal_warmup_fraction (float): Fraction of training held at the initial reflex scale before annealing begins.
            evaluation_reflex_scale (float | None): If provided, temporarily override the brain's runtime reflex scale for the evaluation phase.
            curriculum_profile (str): Curriculum profile name used when executing curriculum-based training; `"none"` disables curriculum.
            preserve_training_metadata (bool): If true and `episodes == 0`, keep training metadata empty rather than forcing a no-op training_history; otherwise run the training schedule when episodes > 0.
            training_regime_spec (TrainingRegimeSpec | None): Optional regime spec that can add a finetuning phase and regime-specific annealing; when present its finetuning episodes and reflex scale are applied after the main training episodes.
        
        Returns:
            tuple:
                - training_history (List[EpisodeStats]): Per-episode statistics for all executed training episodes (includes finetuning episodes when a TrainingRegimeSpec is used).
                - evaluation_history (List[EpisodeStats]): Per-episode statistics for all evaluation episodes.
                - evaluation_trace (List[Dict[str, object]]): Trace captured from the last non-empty evaluation run when `capture_evaluation_trace` is true; otherwise an empty list.
        """
        training_history: List[EpisodeStats] = []
        if episodes > 0 or not preserve_training_metadata:
            training_history = self._execute_training_schedule_with_finetuning(
                episodes=episodes,
                episode_start=episode_start,
                reflex_anneal_final_scale=reflex_anneal_final_scale,
                reflex_annealing_schedule=reflex_annealing_schedule,
                reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
                curriculum_profile=curriculum_profile,
                training_regime_spec=training_regime_spec,
            )
        final_training_reflex_scale = (
            float(self.brain.config.reflex_scale)
            if reflex_anneal_final_scale is None
            else max(0.0, float(reflex_anneal_final_scale))
        )
        if (
            training_regime_spec is not None
            and int(training_regime_spec.finetuning_episodes) > 0
        ):
            final_training_reflex_scale = max(
                0.0,
                float(training_regime_spec.finetuning_reflex_scale),
            )
        total_training_episodes = max(0, int(episodes))
        if training_regime_spec is not None:
            total_training_episodes += max(0, int(training_regime_spec.finetuning_episodes))

        evaluation_history: List[EpisodeStats] = []
        evaluation_trace: List[Dict[str, object]] = []
        previous_reflex_scale = float(self.brain.current_reflex_scale)
        if evaluation_reflex_scale is not None:
            self.brain.set_runtime_reflex_scale(evaluation_reflex_scale)
        try:
            for i in range(evaluation_episodes):
                stats, trace = self.run_episode(
                    episode_start + total_training_episodes + i,
                    training=False,
                    sample=False,
                    render=render_last_evaluation and i == evaluation_episodes - 1,
                    capture_trace=capture_evaluation_trace and i == evaluation_episodes - 1,
                    debug_trace=debug_trace and capture_evaluation_trace and i == evaluation_episodes - 1,
                )
                evaluation_history.append(stats)
                if trace:
                    evaluation_trace = trace
        finally:
            self.brain.set_runtime_reflex_scale(previous_reflex_scale)
        return training_history, evaluation_history, evaluation_trace

    def _consume_episodes_without_learning(
        self,
        *,
        episodes: int,
        episode_start: int = 0,
        policy_mode: str = "normal",
    ) -> List[EpisodeStats]:
        """
        Execute a series of evaluation episodes without performing learning updates.
        
        Parameters:
            episodes (int): Number of episodes to run.
            episode_start (int): Base episode index used to derive deterministic per-episode seeds.
            policy_mode (str): Inference path used for action selection (e.g., "normal" or "reflex_only").
        
        Returns:
            List[EpisodeStats]: Per-episode statistics for the executed episodes.
        """
        history: List[EpisodeStats] = []
        for offset in range(max(0, int(episodes))):
            stats, _ = self.run_episode(
                episode_start + offset,
                training=False,
                sample=True,
                render=False,
                capture_trace=False,
                debug_trace=False,
                policy_mode=policy_mode,
            )
            history.append(stats)
        return history

    def _train_with_best_checkpoint_selection(
        self,
        *,
        episodes: int,
        evaluation_episodes: int,
        render_last_evaluation: bool,
        capture_evaluation_trace: bool,
        debug_trace: bool,
        checkpoint_interval: int,
        checkpoint_dir: str | Path | None,
        checkpoint_scenario_names: Sequence[str] | None,
        selection_scenario_episodes: int,
        checkpoint_selection_config: CheckpointSelectionConfig,
        reflex_anneal_final_scale: float | None = None,
        reflex_annealing_schedule: AnnealingSchedule | str = AnnealingSchedule.LINEAR,
        reflex_anneal_warmup_fraction: float = 0.0,
        evaluation_reflex_scale: float | None = None,
        curriculum_profile: str = "none",
        training_regime_spec: TrainingRegimeSpec | None = None,
    ) -> tuple[List[EpisodeStats], List[EpisodeStats], List[Dict[str, object]]]:
        """
        Train while capturing checkpoint candidates, select and load the best candidate by evaluating behavior suites, optionally persist best/last checkpoints, and run a final evaluation pass.
        
        Parameters:
            episodes (int): Number of training episodes to run (primary training phase).
            evaluation_episodes (int): Number of episodes to run for the final evaluation pass after loading the selected checkpoint.
            render_last_evaluation (bool): If true, render the last evaluation episodes.
            capture_evaluation_trace (bool): If true, capture trace data for the final evaluation episodes.
            debug_trace (bool): If true, include extended debug fields in captured traces.
            checkpoint_interval (int): Interval (in completed episodes) at which to capture checkpoint candidates.
            checkpoint_dir (str | Path | None): Optional directory in which to persist selected best/last checkpoints.
            checkpoint_scenario_names (Sequence[str] | None): Sequence of scenario names used to evaluate checkpoint candidates; defaults to all scenarios.
            selection_scenario_episodes (int): Episodes per scenario to use when evaluating each checkpoint candidate.
            checkpoint_selection_config (CheckpointSelectionConfig): Configuration that controls checkpoint ranking and summary metadata.
            reflex_anneal_final_scale (float | None): Final reflex scale applied during training annealing (overridden by training_regime_spec finetuning settings when present).
            reflex_annealing_schedule (AnnealingSchedule | str): Annealing schedule for reflex scale during training.
            reflex_anneal_warmup_fraction (float): Warmup fraction during reflex annealing (hold progress at zero for this fraction).
            evaluation_reflex_scale (float | None): Reflex scale to apply during the final evaluation pass (if None, current runtime scale is used).
            curriculum_profile (str): Curriculum profile name used during training and evaluation.
            training_regime_spec (TrainingRegimeSpec | None): Optional regime spec that may add a finetuning phase; finetuning episodes are captured and considered for checkpoint numbering.
        
        Returns:
            tuple:
                - training_history (List[EpisodeStats]): Episode statistics collected during the training phase used to generate checkpoint candidates.
                - evaluation_history (List[EpisodeStats]): Episode statistics produced by the final evaluation pass after loading the selected checkpoint.
                - evaluation_trace (List[Dict[str, object]]): Captured trace items from the final evaluation episodes (empty list if tracing was disabled).
        """
        scenario_names = list(checkpoint_scenario_names or SCENARIO_NAMES)
        interval = max(1, int(checkpoint_interval))
        selection_runs = max(1, int(selection_scenario_episodes))
        selection_config = checkpoint_selection_config
        training_history: List[EpisodeStats] = []
        self.checkpoint_source = "best"
        final_training_reflex_scale = (
            float(self.brain.config.reflex_scale)
            if reflex_anneal_final_scale is None
            else max(0.0, float(reflex_anneal_final_scale))
        )
        if (
            training_regime_spec is not None
            and int(training_regime_spec.finetuning_episodes) > 0
        ):
            final_training_reflex_scale = max(
                0.0,
                float(training_regime_spec.finetuning_reflex_scale),
            )
        total_training_episodes = max(0, int(episodes)) + (
            max(0, int(training_regime_spec.finetuning_episodes))
            if training_regime_spec is not None
            else 0
        )

        with tempfile.TemporaryDirectory(prefix="spider_budget_ckpt_") as tmpdir:
            checkpoint_root = Path(tmpdir)
            candidates: List[Dict[str, object]] = []
            if episodes <= 0:
                candidates.append(
                    self._capture_checkpoint_candidate(
                        root_dir=checkpoint_root,
                        episode=0,
                        scenario_names=scenario_names,
                        metric=selection_config.metric,
                        selection_scenario_episodes=selection_runs,
                        eval_reflex_scale=0.0,
                    )
                )
            def maybe_capture_candidate(completed_episode: int) -> None:
                """
                Append a checkpoint candidate when a completed episode meets the checkpoint capture condition.
                
                If `completed_episode` is divisible by the configured `interval` or equals `total_training_episodes`, this function calls `self._capture_checkpoint_candidate` with `eval_reflex_scale=0.0` and appends the returned candidate metadata to the surrounding `candidates` list.
                
                Parameters:
                    completed_episode (int): The index or number of the episode that just finished.
                """
                should_checkpoint = (
                    completed_episode % interval == 0
                    or completed_episode == total_training_episodes
                )
                if should_checkpoint:
                    candidates.append(
                        self._capture_checkpoint_candidate(
                            root_dir=checkpoint_root,
                            episode=completed_episode,
                            scenario_names=scenario_names,
                            metric=selection_config.metric,
                            selection_scenario_episodes=selection_runs,
                            eval_reflex_scale=0.0,
                        )
                    )

            training_history = self._execute_training_schedule_with_finetuning(
                episodes=episodes,
                episode_start=0,
                reflex_anneal_final_scale=final_training_reflex_scale,
                reflex_annealing_schedule=reflex_annealing_schedule,
                reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
                curriculum_profile=curriculum_profile,
                checkpoint_callback=maybe_capture_candidate,
                training_regime_spec=training_regime_spec,
            )

            if not candidates:
                candidates.append(
                    self._capture_checkpoint_candidate(
                        root_dir=checkpoint_root,
                        episode=total_training_episodes,
                        scenario_names=scenario_names,
                        metric=selection_config.metric,
                        selection_scenario_episodes=selection_runs,
                        eval_reflex_scale=0.0,
                    )
                )

            best_candidate = max(
                candidates,
                key=lambda item: checkpoint_candidate_sort_key(
                    item,
                    selection_config=selection_config,
                ),
            )
            last_candidate = max(candidates, key=lambda item: int(item.get("episode", 0)))
            self.brain.load(best_candidate["path"])
            persisted = persist_checkpoint_pair(
                checkpoint_dir=checkpoint_dir,
                best_candidate=best_candidate,
                last_candidate=last_candidate,
            )
            def checkpoint_summary(candidate: Dict[str, object]) -> Dict[str, object]:
                """
                Builds a compact summary dictionary for a checkpoint candidate suitable for CSV annotation and metadata.
                
                Parameters:
                    candidate (dict): Candidate metadata containing at minimum the keys
                        "name", "episode", "scenario_success_rate", "episode_success_rate", and "mean_reward".
                        May include "eval_reflex_scale" and an "evaluation_summary" dict with
                        "mean_final_reflex_override_rate" and "mean_reflex_dominance".
                
                Returns:
                    dict: A typed summary with these keys:
                        - name (str): Candidate name.
                        - episode (int): Episode index when the candidate was captured.
                        - scenario_success_rate (float): Aggregate scenario success rate.
                        - episode_success_rate (float): Aggregate episode success rate.
                        - mean_reward (float): Mean reward from the evaluation payload.
                        - eval_reflex_scale (float): Evaluation reflex scale (defaults to 0.0).
                        - mean_final_reflex_override_rate (float): Mean final reflex override rate (defaults to 0.0).
                        - mean_reflex_dominance (float): Mean reflex dominance (defaults to 0.0).
                        - composite_score (float, optional): Composite penalized score when present in the candidate summary.
                """
                evaluation_summary = candidate.get("evaluation_summary", {})
                if not isinstance(evaluation_summary, dict):
                    evaluation_summary = {}
                result = {
                    "name": str(candidate["name"]),
                    "episode": int(candidate["episode"]),
                    "scenario_success_rate": float(candidate["scenario_success_rate"]),
                    "episode_success_rate": float(candidate["episode_success_rate"]),
                    "mean_reward": float(candidate["mean_reward"]),
                    "eval_reflex_scale": float(candidate.get("eval_reflex_scale", 0.0)),
                    "mean_final_reflex_override_rate": float(
                        evaluation_summary.get("mean_final_reflex_override_rate", 0.0)
                    ),
                    "mean_reflex_dominance": float(
                        evaluation_summary.get("mean_reflex_dominance", 0.0)
                    ),
                }
                if selection_config.penalty_mode is CheckpointPenaltyMode.DIRECT:
                    result["composite_score"] = (
                        checkpoint_candidate_composite_score(
                            candidate,
                            selection_config,
                        )
                    )
                return result

            generated = [checkpoint_summary(candidate) for candidate in candidates]
            selected_summary = checkpoint_summary(best_candidate)
            if "best" in persisted:
                selected_summary["path"] = persisted["best"]
            self._latest_checkpointing_summary = {
                "enabled": True,
                "selection": "best",
                "metric": selection_config.metric,
                "penalty_mode": selection_config.penalty_mode.value,
                "penalty_config": selection_config.to_summary(),
                "checkpoint_interval": interval,
                "evaluation_source": "behavior_suite",
                "eval_reflex_scale": 0.0,
                "selection_scenario_episodes": selection_runs,
                "generated_checkpoints": generated,
                "selected_checkpoint": selected_summary,
                "last_checkpoint": {
                    "name": str(last_candidate["name"]),
                    "episode": int(last_candidate["episode"]),
                },
            }
            if persisted:
                self._latest_checkpointing_summary["persisted"] = persisted
        self.brain.set_runtime_reflex_scale(final_training_reflex_scale)

        _, evaluation_history, evaluation_trace = self._train_histories(
            episodes=0,
            evaluation_episodes=evaluation_episodes,
            render_last_evaluation=render_last_evaluation,
            capture_evaluation_trace=capture_evaluation_trace,
            debug_trace=debug_trace,
            episode_start=episodes,
            reflex_anneal_final_scale=final_training_reflex_scale,
            reflex_annealing_schedule=reflex_annealing_schedule,
            reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
            evaluation_reflex_scale=evaluation_reflex_scale,
            curriculum_profile=curriculum_profile,
            preserve_training_metadata=True,
            training_regime_spec=training_regime_spec,
        )
        return training_history, evaluation_history, evaluation_trace

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
