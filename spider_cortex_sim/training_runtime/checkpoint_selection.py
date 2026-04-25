from __future__ import annotations

from .common import *


class SimulationCheckpointSelectionTrainingMixin:
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
                Append a checkpoint candidate to the surrounding `candidates` list when the given episode meets the capture condition.
                
                If `completed_episode` is divisible by `interval` or equals `total_training_episodes`, capture a checkpoint candidate for that episode (using the surrounding `scenario_names`, `selection_config.metric`, `selection_runs`, and an evaluation reflex scale of 0.0) and append its metadata to `candidates`.
                
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
            episode_start=total_training_episodes,
            reflex_anneal_final_scale=final_training_reflex_scale,
            reflex_annealing_schedule=reflex_annealing_schedule,
            reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
            evaluation_reflex_scale=evaluation_reflex_scale,
            curriculum_profile=curriculum_profile,
            preserve_training_metadata=True,
            training_regime_spec=None,
        )
        return training_history, evaluation_history, evaluation_trace

__all__ = [name for name in globals() if not name.startswith("__")]
