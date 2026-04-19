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


class SimulationTrainingScheduleMixin:
    def _set_training_regime_metadata(
        self,
        *,
        curriculum_profile: str,
        episodes: int,
        curriculum_summary: Dict[str, object] | None = None,
        training_regime_spec: TrainingRegimeSpec | None = None,
        training_regime_name: str | None = None,
        annealing_schedule: AnnealingSchedule | str | None = None,
        anneal_target_scale: float | None = None,
        anneal_warmup_fraction: float | None = None,
        finetuning_episodes: int = 0,
        finetuning_reflex_scale: float = 0.0,
    ) -> None:
        """
        Cache and normalize training-regime and curriculum metadata used for summaries and CSV export.
        
        Parameters:
            curriculum_profile (str): Curriculum profile name; validated and resolved into internal profile name.
            episodes (int): Total main training episodes (does not include finetuning episodes).
            curriculum_summary (dict | None): Optional precomputed curriculum summary to cache as-is; when None the cached value is cleared.
            training_regime_spec (TrainingRegimeSpec | None): Optional regime spec whose summary will be used verbatim when provided.
            training_regime_name (str | None): Name for an ad-hoc regime when `training_regime_spec` is not supplied.
            annealing_schedule (AnnealingSchedule | str | None): Reflex-annealing schedule to record when `training_regime_spec` is not supplied.
            anneal_target_scale (float | None): Target reflex scale for annealing; when None uses the brain's configured reflex scale.
            anneal_warmup_fraction (float | None): Warmup fraction for the annealing schedule; recorded as 0.0 when not supplied.
            finetuning_episodes (int): Additional fixed-phase finetuning episode count to include after main training.
            finetuning_reflex_scale (float): Reflex scale to apply during the finetuning phase.
        
        Side effects:
            Updates the instance cached fields `_latest_curriculum_summary` and `_latest_training_regime_summary`,
            resolving phase episode budgets when a curriculum profile other than "none" is used.
        """
        profile_name = validate_curriculum_profile(curriculum_profile)
        phase_budgets: list[int] = []
        if profile_name != "none":
            phase_budgets = resolve_curriculum_phase_budgets(episodes)
        if training_regime_spec is not None:
            regime_summary = training_regime_spec.to_summary()
        else:
            schedule = (
                AnnealingSchedule.NONE
                if annealing_schedule is None
                else AnnealingSchedule(annealing_schedule)
            )
            regime_summary = {
                "name": str(training_regime_name or "baseline"),
                "annealing_schedule": schedule.value,
                "anneal_target_scale": (
                    float(self.brain.config.reflex_scale)
                    if anneal_target_scale is None
                    else max(0.0, float(anneal_target_scale))
                ),
                "anneal_warmup_fraction": (
                    0.0
                    if anneal_warmup_fraction is None
                    else max(0.0, float(anneal_warmup_fraction))
                ),
                "finetuning_episodes": max(0, int(finetuning_episodes)),
                "finetuning_reflex_scale": max(0.0, float(finetuning_reflex_scale)),
                "loss_override_penalty_weight": 0.0,
                "loss_dominance_penalty_weight": 0.0,
            }
        main_training_episodes = max(0, int(episodes))
        resolved_finetuning_episodes = max(
            0,
            int(regime_summary.get("finetuning_episodes", finetuning_episodes)),
        )
        self._latest_curriculum_summary = (
            deepcopy(curriculum_summary) if curriculum_summary is not None else None
        )
        self._latest_training_regime_summary = {
            **regime_summary,
            "is_experiment_of_record": (
                str(regime_summary.get("name", "baseline"))
                == EXPERIMENT_OF_RECORD_REGIME
            ),
            "mode": "curriculum" if profile_name != "none" else "flat",
            "curriculum_profile": profile_name,
            "resolved_budget": {
                "total_training_episodes": int(
                    main_training_episodes + resolved_finetuning_episodes
                ),
                "main_training_episodes": int(main_training_episodes),
                "finetuning_episodes": int(resolved_finetuning_episodes),
                "phase_episode_budgets": phase_budgets,
            },
        }

    def _execute_training_schedule(
        self,
        *,
        episodes: int,
        episode_start: int = 0,
        reflex_anneal_final_scale: float | None = None,
        reflex_annealing_schedule: AnnealingSchedule | str = AnnealingSchedule.LINEAR,
        reflex_anneal_warmup_fraction: float = 0.0,
        curriculum_profile: str = "none",
        checkpoint_callback: Callable[[int], None] | None = None,
        training_regime_spec: TrainingRegimeSpec | None = None,
    ) -> List[EpisodeStats]:
        """
        Execute the configured training schedule, either flat or curriculum-based, and return per-episode statistics.
        
        Runs up to `episodes` training episodes. If `curriculum_profile` is "none" episodes are executed sequentially; otherwise phases from the selected curriculum are executed in order with per-phase budgets, micro-evaluations against promotion scenarios and optional promotion-check criteria, and possible episode carryover when phases are advanced early. Per-episode reflex annealing is applied according to `reflex_annealing_schedule`, `reflex_anneal_warmup_fraction`, and `reflex_anneal_final_scale`. If `checkpoint_callback` is provided it is invoked after each completed training episode with the 1-based completed-episode count.
        
        Parameters:
            episodes (int): Total number of training episodes to execute.
            episode_start (int): Base index added to per-episode indices passed to run_episode.
            reflex_anneal_final_scale (float | None): Final reflex scale to use at the end of training; when None the brain's configured reflex scale is used. Values are clamped to be >= 0.0.
            reflex_annealing_schedule (AnnealingSchedule | str): Annealing schedule name; one of "none", "linear", or "cosine".
            reflex_anneal_warmup_fraction (float): Fraction of main training episodes to hold the start scale before annealing (clamped to [0.0, 0.999999]).
            curriculum_profile (str): Curriculum identifier; "none" disables curriculum and runs flat training.
            checkpoint_callback (Callable[[int], None] | None): Optional callback invoked after each completed training episode with the completed episode count.
            training_regime_spec (TrainingRegimeSpec | None): Optional regime specification used to populate runtime regime metadata.
        
        Returns:
            List[EpisodeStats]: Per-episode statistics in execution order.
        """
        training_history: List[EpisodeStats] = []
        total_episodes = max(0, int(episodes))
        final_training_reflex_scale = (
            float(self.brain.config.reflex_scale)
            if reflex_anneal_final_scale is None
            else max(0.0, float(reflex_anneal_final_scale))
        )
        schedule = AnnealingSchedule(reflex_annealing_schedule)
        warmup_fraction = max(0.0, min(0.999999, float(reflex_anneal_warmup_fraction)))
        profile_name = validate_curriculum_profile(curriculum_profile)

        if profile_name == "none":
            self._set_training_regime_metadata(
                curriculum_profile="none",
                episodes=total_episodes,
                training_regime_spec=training_regime_spec,
                annealing_schedule=schedule,
                anneal_target_scale=final_training_reflex_scale,
                anneal_warmup_fraction=warmup_fraction,
            )
            for episode in range(total_episodes):
                self._set_training_episode_reflex_scale(
                    episode_index=episode,
                    total_episodes=total_episodes,
                    final_scale=final_training_reflex_scale,
                    schedule=schedule,
                    warmup_fraction=warmup_fraction,
                )
                stats, _ = self.run_episode(
                    episode_start + episode,
                    training=True,
                    sample=True,
                    render=False,
                    capture_trace=False,
                    debug_trace=False,
                )
                training_history.append(stats)
                if checkpoint_callback is not None:
                    checkpoint_callback(episode + 1)
            self.brain.set_runtime_reflex_scale(final_training_reflex_scale)
            return training_history

        phases = self._resolve_curriculum_profile(
            curriculum_profile=profile_name,
            total_episodes=total_episodes,
        )
        curriculum_summary = empty_curriculum_summary(
            profile_name,
            total_episodes,
        )
        completed_episodes = 0
        carryover_episodes = 0
        final_phase_index = len(phases) - 1
        for phase_index, phase in enumerate(phases):
            remaining_budget = max(0, total_episodes - completed_episodes)
            carryover_in = int(carryover_episodes)
            allocated_episodes = min(
                remaining_budget,
                int(phase.max_episodes) + carryover_in,
            )
            carryover_episodes = 0
            phase_record: Dict[str, object] = {
                "name": phase.name,
                "skill_name": phase.skill_name,
                "training_scenarios": list(phase.training_scenarios),
                "promotion_scenarios": list(phase.promotion_scenarios),
                "promotion_check_specs": promotion_check_spec_records(
                    phase.promotion_check_specs
                ),
                "success_threshold": float(phase.success_threshold),
                "max_episodes": int(phase.max_episodes),
                "min_episodes": int(phase.min_episodes),
                "allocated_episodes": int(allocated_episodes),
                "carryover_in": carryover_in,
                "episodes_executed": 0,
                "status": "pending",
                "promotion_checks": [],
                "final_metrics": {},
                "final_check_results": {},
                "promotion_reason": "",
            }
            if allocated_episodes <= 0 or completed_episodes >= total_episodes:
                phase_record["status"] = "skipped_no_budget"
                curriculum_summary["phases"].append(phase_record)
                continue

            promoted = False
            for local_episode in range(allocated_episodes):
                if completed_episodes >= total_episodes:
                    break
                scenario_name = phase.training_scenarios[
                    local_episode % len(phase.training_scenarios)
                ]
                self._set_training_episode_reflex_scale(
                    episode_index=completed_episodes,
                    total_episodes=total_episodes,
                    final_scale=final_training_reflex_scale,
                    schedule=schedule,
                    warmup_fraction=warmup_fraction,
                )
                stats, _ = self.run_episode(
                    episode_start + completed_episodes,
                    training=True,
                    sample=True,
                    render=False,
                    capture_trace=False,
                    debug_trace=False,
                    scenario_name=scenario_name,
                )
                training_history.append(stats)
                completed_episodes += 1
                phase_record["episodes_executed"] = int(local_episode + 1)
                if checkpoint_callback is not None:
                    checkpoint_callback(completed_episodes)
                if promoted or local_episode + 1 < phase.min_episodes:
                    continue

                microeval_scenarios = list(
                    dict.fromkeys(
                        [
                            *phase.promotion_scenarios,
                            *(
                                spec.scenario
                                for spec in phase.promotion_check_specs
                            ),
                        ]
                    )
                )
                microeval_payload, _, _ = self.evaluate_behavior_suite(
                    microeval_scenarios,
                    episodes_per_scenario=1,
                    capture_trace=False,
                    debug_trace=False,
                    summary_only=True,
                )
                microeval_summary = dict(microeval_payload.get("summary", {}))
                check = {
                    "after_episode": int(completed_episodes),
                    "phase_episode": int(local_episode + 1),
                    "scenario_success_rate": float(
                        microeval_summary.get("scenario_success_rate", 0.0)
                    ),
                    "episode_success_rate": float(
                        microeval_summary.get("episode_success_rate", 0.0)
                    ),
                    "mean_reward": mean_reward_from_behavior_payload(
                        microeval_payload
                    ),
                }
                if phase.promotion_check_specs:
                    check_results, promotion_passed, promotion_reason = (
                        evaluate_promotion_check_specs(
                            microeval_payload,
                            phase.promotion_check_specs,
                        )
                    )
                    check["check_results"] = check_results
                    check["promotion_criteria_passed"] = bool(promotion_passed)
                    check["promotion_reason"] = promotion_reason
                else:
                    promotion_passed = (
                        check["scenario_success_rate"] >= phase.success_threshold
                    )
                    check["check_results"] = {}
                    check["promotion_criteria_passed"] = bool(promotion_passed)
                    check["promotion_reason"] = "threshold_fallback"
                phase_record["promotion_checks"].append(check)
                phase_record["final_metrics"] = dict(check)
                phase_record["final_check_results"] = dict(check["check_results"])
                phase_record["promotion_reason"] = str(check["promotion_reason"])
                if promotion_passed:
                    phase_record["status"] = "promoted"
                    promoted = True
                    if phase_index < final_phase_index:
                        carryover_episodes = max(
                            0,
                            allocated_episodes - phase_record["episodes_executed"],
                        )
                        break

            if not promoted:
                phase_record["status"] = "max_budget_exhausted"
            curriculum_summary["phases"].append(phase_record)

        curriculum_summary["executed_training_episodes"] = int(completed_episodes)
        if curriculum_summary["phases"]:
            latest_executed_phase = None
            for phase in reversed(curriculum_summary["phases"]):
                if not isinstance(phase, dict):
                    continue
                if str(phase.get("status", "")) in {"skipped", "skipped_no_budget"}:
                    continue
                latest_executed_phase = phase
                break
            curriculum_summary["status"] = str(
                (
                    latest_executed_phase
                    if latest_executed_phase is not None
                    else curriculum_summary["phases"][-1]
                ).get("status", "completed")
            )
        else:
            curriculum_summary["status"] = "no_training_budget"
        self._set_training_regime_metadata(
            curriculum_profile=profile_name,
            episodes=total_episodes,
            curriculum_summary=curriculum_summary,
            training_regime_spec=training_regime_spec,
            annealing_schedule=schedule,
            anneal_target_scale=final_training_reflex_scale,
            anneal_warmup_fraction=warmup_fraction,
        )
        self.brain.set_runtime_reflex_scale(final_training_reflex_scale)
        return training_history

    def _execute_finetuning_phase(
        self,
        *,
        episodes: int,
        episode_start: int,
        completed_episode_start: int,
        reflex_scale: float,
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> List[EpisodeStats]:
        """
        Run a fixed-scale finetuning phase of training episodes.
        
        Each episode is executed in training mode with the brain's runtime reflex scale set to the provided `reflex_scale`. After each completed episode, `checkpoint_callback`, if given, is invoked with the 1-based completed-episode count computed as `completed_episode_start + i` where `i` is the completed-episode index (starting at 1). The brain's runtime reflex scale is set to `reflex_scale` before the phase and left at that value when the function returns.
        
        Parameters:
            episodes (int): Number of finetuning episodes to run. Non-positive values result in no episodes.
            episode_start (int): Episode index supplied to `run_episode` for the first finetuning episode; subsequent episodes increment this value.
            completed_episode_start (int): Base count used when invoking `checkpoint_callback`; the callback receives `completed_episode_start + completed_index`.
            reflex_scale (float): Fixed reflex scale to apply during the finetuning phase (clamped to >= 0.0).
            checkpoint_callback (Callable[[int], None] | None): Optional callback invoked after each completed episode with the completed-episode count.
        
        Returns:
            List[EpisodeStats]: A list of per-episode statistics for all finetuning episodes that were run.
        """
        history: List[EpisodeStats] = []
        total_episodes = max(0, int(episodes))
        fixed_scale = max(0.0, float(reflex_scale))
        for offset in range(total_episodes):
            self.brain.set_runtime_reflex_scale(fixed_scale)
            stats, _ = self.run_episode(
                episode_start + offset,
                training=True,
                sample=True,
                render=False,
                capture_trace=False,
                debug_trace=False,
            )
            history.append(stats)
            if checkpoint_callback is not None:
                checkpoint_callback(completed_episode_start + offset + 1)
        self.brain.set_runtime_reflex_scale(fixed_scale)
        return history

    def _execute_training_schedule_with_finetuning(
        self,
        *,
        episodes: int,
        episode_start: int = 0,
        reflex_anneal_final_scale: float | None = None,
        reflex_annealing_schedule: AnnealingSchedule | str = AnnealingSchedule.LINEAR,
        reflex_anneal_warmup_fraction: float = 0.0,
        curriculum_profile: str = "none",
        checkpoint_callback: Callable[[int], None] | None = None,
        training_regime_spec: TrainingRegimeSpec | None = None,
    ) -> List[EpisodeStats]:
        """
        Run the configured training schedule and, if the provided training regime specifies it, an additional fixed-reflex fine-tuning phase.
        
        Parameters:
            episodes (int): Number of episodes for the main training schedule.
            episode_start (int): Episode index offset to apply when numbering episodes.
            reflex_anneal_final_scale (float | None): Final reflex scale for the annealing schedule; if None, no annealing override is applied.
            reflex_annealing_schedule (AnnealingSchedule | str): Annealing curve to use for reflex scaling during the main training schedule.
            reflex_anneal_warmup_fraction (float): Fraction of the main training budget to hold at initial reflex scale before annealing progress begins.
            curriculum_profile (str): Curriculum profile name determining phased training behavior; use "none" for flat training.
            checkpoint_callback (Callable[[int], None] | None): Optional callback invoked after each completed training episode with the completed-episode count.
            training_regime_spec (TrainingRegimeSpec | None): Optional regime specification that may declare a late fine-tuning phase (`finetuning_episodes` and `finetuning_reflex_scale`).
        
        Returns:
            List[EpisodeStats]: Concatenated episode statistics from the main training schedule followed by any fine-tuning episodes.
        """
        main_episodes = max(0, int(episodes))
        training_history = self._execute_training_schedule(
            episodes=main_episodes,
            episode_start=episode_start,
            reflex_anneal_final_scale=reflex_anneal_final_scale,
            reflex_annealing_schedule=reflex_annealing_schedule,
            reflex_anneal_warmup_fraction=reflex_anneal_warmup_fraction,
            curriculum_profile=curriculum_profile,
            checkpoint_callback=checkpoint_callback,
            training_regime_spec=training_regime_spec,
        )
        finetuning_episodes = (
            max(0, int(training_regime_spec.finetuning_episodes))
            if training_regime_spec is not None
            else 0
        )
        if finetuning_episodes <= 0:
            return training_history
        fine_history = self._execute_finetuning_phase(
            episodes=finetuning_episodes,
            episode_start=episode_start + main_episodes,
            completed_episode_start=main_episodes,
            reflex_scale=training_regime_spec.finetuning_reflex_scale,
            checkpoint_callback=checkpoint_callback,
        )
        training_history.extend(fine_history)
        return training_history

    def _set_training_episode_reflex_scale(
        self,
        *,
        episode_index: int,
        total_episodes: int,
        final_scale: float,
        schedule: AnnealingSchedule | str = AnnealingSchedule.LINEAR,
        warmup_fraction: float = 0.0,
    ) -> float:
        """
        Set and return the runtime reflex scale for a training episode by interpolating from the configured start scale toward `final_scale` according to `schedule`, applying an optional warmup that holds the start scale for an initial fraction of episodes.
        
        Parameters:
            episode_index (int): Zero-based index of the current training episode.
            total_episodes (int): Number of episodes over which annealing is defined.
            final_scale (float): Target reflex scale at the end of annealing; values below 0 are treated as 0.0.
            schedule (AnnealingSchedule | str): One of "none", "linear", or "cosine" determining the interpolation shape.
            warmup_fraction (float): Fraction of `total_episodes` to hold at the start scale before annealing begins (clamped to [0, 0.999999]).
        
        Returns:
            float: The reflex scale applied for this episode.
        
        Side effects:
            Applies the computed scale to the brain by calling `brain.set_runtime_reflex_scale`.
        """
        start_scale = float(self.brain.config.reflex_scale)
        target_scale = max(0.0, float(final_scale))
        schedule_name = AnnealingSchedule(schedule)
        if schedule_name is AnnealingSchedule.NONE or total_episodes <= 1:
            scale = start_scale
        else:
            total = max(1, int(total_episodes))
            index = min(max(0, int(episode_index)), total - 1)
            warmup = max(0.0, min(0.999999, float(warmup_fraction)))
            warmup_episodes = min(total - 1, int(total * warmup))
            if index < warmup_episodes:
                progress = 0.0
            else:
                denominator = max(1, total - warmup_episodes - 1)
                progress = float(index - warmup_episodes) / float(denominator)
            if schedule_name is AnnealingSchedule.COSINE:
                scale = target_scale + (start_scale - target_scale) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            else:
                scale = start_scale + (target_scale - start_scale) * progress
        self.brain.set_runtime_reflex_scale(scale)
        return scale

    def _effective_reflex_scale(self, requested_scale: float | None = None) -> float:
        """
        Determine the effective global reflex scale used by the brain, optionally overriding the brain's current runtime scale.
        
        Parameters:
            requested_scale (float | None): If provided, use this value instead of the brain's current runtime reflex scale.
        
        Returns:
            float: Effective reflex scale (greater than or equal to 0.0). Returns 0.0 when reflexes are disabled or the brain architecture is not modular.
        """
        scale = self.brain.current_reflex_scale if requested_scale is None else requested_scale
        if not self.brain.config.enable_reflexes or not self.brain.config.is_modular:
            return 0.0
        return max(0.0, float(scale))
