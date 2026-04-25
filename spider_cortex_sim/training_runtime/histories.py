from __future__ import annotations

from typing import Dict, List

from .common import AnnealingSchedule, EpisodeStats, TrainingRegimeSpec


class SimulationTrainingHistoriesMixin:
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
        if evaluation_reflex_scale is None:
            evaluation_reflex_scale = final_training_reflex_scale
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
        Executes a sequence of evaluation episodes without performing learning updates.
        
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

__all__ = ["SimulationTrainingHistoriesMixin"]
