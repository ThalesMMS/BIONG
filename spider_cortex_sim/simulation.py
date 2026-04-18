"""Simulation runtime facade.

Multi-seed comparison workflows and comparison statistics live in
`spider_cortex_sim.comparison`; claim-test evaluation lives in
`spider_cortex_sim.claim_evaluation`.
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


EXPERIMENT_OF_RECORD_REGIME = "late_finetuning"
CAPABILITY_PROBE_SCENARIOS: tuple[str, ...] = SCENARIO_CAPABILITY_PROBE_SCENARIOS


def is_capability_probe(scenario_name: str) -> bool:
    """Return whether a scenario is classified as a capability probe."""
    return (
        _probe_metadata_for_scenario(scenario_name)["probe_type"]
        == ProbeType.CAPABILITY_PROBE.value
    )


def _probe_metadata_for_scenario(scenario_name: str) -> Dict[str, object]:
    """Return probe metadata from the ScenarioSpec registry."""
    try:
        scenario = get_scenario(str(scenario_name))
    except ValueError:
        return {
            "probe_type": ProbeType.EMERGENCE_GATE.value,
            "target_skill": None,
            "geometry_assumptions": None,
            "benchmark_tier": BenchmarkTier.PRIMARY.value,
            "acceptable_partial_progress": None,
        }
    return {
        "probe_type": scenario.probe_type,
        "target_skill": scenario.target_skill,
        "geometry_assumptions": scenario.geometry_assumptions,
        "benchmark_tier": scenario.benchmark_tier,
        "acceptable_partial_progress": scenario.acceptable_partial_progress,
    }


class SpiderSimulation:
    _resolve_curriculum_phase_budgets = staticmethod(resolve_curriculum_phase_budgets)
    _resolve_curriculum_profile = staticmethod(resolve_curriculum_profile)
    _promotion_check_spec_records = staticmethod(promotion_check_spec_records)
    _evaluate_promotion_check_specs = staticmethod(evaluate_promotion_check_specs)
    _empty_curriculum_summary = staticmethod(empty_curriculum_summary)
    _regime_row_metadata_from_summary = staticmethod(regime_row_metadata_from_summary)
    _resolve_checkpoint_load_dir = staticmethod(resolve_checkpoint_load_dir)
    _checkpoint_run_fingerprint = staticmethod(checkpoint_run_fingerprint)
    _file_sha256 = staticmethod(file_sha256)
    _checkpoint_preload_fingerprint = staticmethod(checkpoint_preload_fingerprint)
    _jsonify_observation = staticmethod(jsonify_observation)
    _mean_reward_from_behavior_payload = staticmethod(mean_reward_from_behavior_payload)
    _checkpoint_candidate_sort_key = staticmethod(checkpoint_candidate_sort_key)
    _checkpoint_candidate_composite_score = staticmethod(
        checkpoint_candidate_composite_score
    )
    _persist_checkpoint_pair = staticmethod(persist_checkpoint_pair)
    _jsonify = staticmethod(jsonify)
    _compact_aggregate = staticmethod(compact_aggregate)
    _compact_behavior_payload = staticmethod(compact_behavior_payload)
    save_summary = staticmethod(save_summary)
    save_trace = staticmethod(save_trace)
    save_behavior_csv = staticmethod(save_behavior_csv)

    def __init__(
        self,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int = 120,
        seed: int = 0,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        brain_config: BrainAblationConfig | None = None,
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile_name: str = "custom",
        benchmark_strength: str = "custom",
        budget_summary: Dict[str, object] | None = None,
    ) -> None:
        """
        Create a SpiderSimulation and configure its world, brain, messaging bus, and runtime budget defaults.
        
        When `brain_config` is omitted, a default ablation-aware brain configuration is created using `module_dropout`. Provided `operational_profile` and `noise_profile` values are resolved (by name or passed-through object); invalid profile names raise ValueError during resolution. If `budget_summary` is omitted, a default resolved budget structure is populated and stored.
        
        Parameters:
            brain_config (BrainAblationConfig | None): Optional ablation-aware brain configuration; when None a default is created using `module_dropout`.
            module_dropout (float): Dropout probability used to construct the default `brain_config` when `brain_config` is not provided.
            operational_profile (str | OperationalProfile | None): Operational profile name or explicit OperationalProfile; None selects the default operational profile.
            noise_profile (str | NoiseConfig | None): Noise profile name or explicit NoiseConfig; None selects the default noise profile.
            map_template (str): Default map template name used for episodes when a scenario-specific template is not supplied.
            budget_profile_name (str): Logical budget profile identifier stored in the simulation summary and used when building a default `budget_summary`.
            benchmark_strength (str): Benchmark strength identifier stored in the simulation summary and used when building a default `budget_summary`.
            budget_summary (Dict[str, object] | None): Optional budget override structure; when omitted the simulation creates and stores a default resolved budget summary.
        """
        self.seed = seed
        self.brain_config = brain_config if brain_config is not None else default_brain_config(module_dropout=module_dropout)
        self.module_dropout = self.brain_config.module_dropout
        self.default_map_template = map_template
        self.operational_profile = runtime_operational_profile(operational_profile)
        self.noise_profile = resolve_noise_profile(noise_profile)
        self.world = SpiderWorld(
            width=width,
            height=height,
            food_count=food_count,
            day_length=day_length,
            night_length=night_length,
            seed=seed,
            reward_profile=reward_profile,
            map_template=map_template,
            operational_profile=self.operational_profile,
            noise_profile=self.noise_profile,
        )
        self.brain = SpiderBrain(
            seed=seed,
            gamma=gamma,
            module_lr=module_lr,
            motor_lr=motor_lr,
            module_dropout=self.brain_config.module_dropout,
            config=self.brain_config,
            operational_profile=self.operational_profile,
        )
        self.max_steps = max_steps
        self.bus = MessageBus()
        self.budget_profile_name = str(budget_profile_name)
        self.benchmark_strength = str(benchmark_strength)
        self.budget_summary = (
            deepcopy(budget_summary)
            if budget_summary is not None
            else {
                "profile": self.budget_profile_name,
                "benchmark_strength": self.benchmark_strength,
                "resolved": {
                    "episodes": 0,
                    "eval_episodes": 0,
                    "max_steps": self.max_steps,
                    "scenario_episodes": 1,
                    "comparison_seeds": [7, 17, 29],
                    "checkpoint_interval": 10,
                    "selection_scenario_episodes": 1,
                    "behavior_seeds": [7, 17, 29],
                    "ablation_seeds": [7, 17, 29],
                },
                "overrides": {},
            }
        )
        self.budget_profile_name = str(
            self.budget_summary.get("profile", self.budget_profile_name)
        )
        self.benchmark_strength = str(
            self.budget_summary.get(
                "benchmark_strength",
                self.benchmark_strength,
            )
        )
        self.checkpoint_source = "final"
        self._latest_checkpointing_summary: Dict[str, object] | None = None
        self._latest_reflex_schedule_summary: Dict[str, object] | None = None
        self._latest_evaluation_without_reflex_support: Dict[str, object] | None = None
        self._latest_curriculum_summary: Dict[str, object] | None = None
        self._latest_training_regime_summary: Dict[str, object] = {
            "name": "baseline",
            "mode": "flat",
            "curriculum_profile": "none",
            "annealing_schedule": AnnealingSchedule.NONE.value,
            "anneal_target_scale": float(self.brain.config.reflex_scale),
            "anneal_warmup_fraction": 0.0,
            "finetuning_episodes": 0,
            "finetuning_reflex_scale": 0.0,
            "loss_override_penalty_weight": 0.0,
            "loss_dominance_penalty_weight": 0.0,
            "is_experiment_of_record": False,
            "resolved_budget": {
                "total_training_episodes": int(
                    self.budget_summary.get("resolved", {}).get("episodes", 0)
                ),
                "main_training_episodes": int(
                    self.budget_summary.get("resolved", {}).get("episodes", 0)
                ),
                "finetuning_episodes": 0,
                "phase_episode_budgets": [],
            },
        }

    @staticmethod
    def _episode_stats_behavior_metrics(stats: EpisodeStats) -> Dict[str, object]:
        """
        Convert independence metrics from EpisodeStats into behavior-metric fields.

        These fields are merged into scenario scorecards so they flow naturally to
        behavior_evaluation summaries and to `behavior_csv` as `metric_*` columns.
        """
        metrics: Dict[str, object] = {
            "dominant_module": stats.dominant_module,
            "dominant_module_share": float(stats.dominant_module_share),
            "effective_module_count": float(stats.effective_module_count),
            "module_agreement_rate": float(stats.module_agreement_rate),
            "module_disagreement_rate": float(stats.module_disagreement_rate),
            "motor_slip_rate": float(stats.motor_slip_rate),
            "mean_orientation_alignment": float(stats.mean_orientation_alignment),
            "mean_terrain_difficulty": float(stats.mean_terrain_difficulty),
        }
        for module_name in PROPOSAL_SOURCE_NAMES:
            metrics[f"module_contribution_{module_name}"] = float(
                stats.module_contribution_share.get(module_name, 0.0)
            )
        for terrain, rate in sorted(stats.terrain_slip_rates.items()):
            metrics[f"terrain_slip_rate_{terrain}"] = float(rate)
        return metrics

    @staticmethod
    def _attach_motor_execution_info(decision: BrainStep, info: Dict[str, object]) -> None:
        """Attach authoritative post-world motor execution diagnostics to a BrainStep-like object."""
        slip_info = info.get("motor_slip", {})
        if not isinstance(slip_info, dict):
            slip_info = {}
        components = info.get("motor_execution_components", {})
        if not isinstance(components, dict):
            components = slip_info.get("components", {})
        if not isinstance(components, dict):
            components = {}

        slip_occurred = bool(
            slip_info.get("occurred", info.get("motor_noise_applied", False))
        )
        decision.execution_slip_occurred = slip_occurred
        decision.motor_slip_occurred = slip_occurred
        decision.motor_noise_applied = slip_occurred
        decision.slip_reason = str(
            info.get(
                "slip_reason",
                info.get("motor_slip_reason", slip_info.get("reason", "none")),
            )
        )
        decision.orientation_alignment = float(
            components.get(
                "orientation_alignment",
                getattr(decision, "orientation_alignment", 1.0),
            )
        )
        decision.terrain_difficulty = float(
            components.get(
                "terrain_difficulty",
                getattr(decision, "terrain_difficulty", 0.0),
            )
        )
        decision.execution_difficulty = float(
            info.get(
                "motor_execution_difficulty",
                slip_info.get(
                    "execution_difficulty",
                    getattr(decision, "execution_difficulty", 0.0),
                ),
            )
        )

    @staticmethod
    def _motor_stage_summary(history: List[EpisodeStats]) -> Dict[str, object]:
        """Aggregate motor execution observability metrics for summary output."""
        if not history:
            return {
                "episodes": 0,
                "mean_motor_slip_rate": 0.0,
                "mean_orientation_alignment": 0.0,
                "mean_terrain_difficulty": 0.0,
                "mean_terrain_slip_rates": {},
            }
        terrain_names = sorted(
            {
                terrain
                for stats in history
                for terrain in stats.terrain_slip_rates
            }
        )
        return {
            "episodes": len(history),
            "mean_motor_slip_rate": float(
                sum(stats.motor_slip_rate for stats in history) / len(history)
            ),
            "mean_orientation_alignment": float(
                sum(stats.mean_orientation_alignment for stats in history) / len(history)
            ),
            "mean_terrain_difficulty": float(
                sum(stats.mean_terrain_difficulty for stats in history) / len(history)
            ),
            "mean_terrain_slip_rates": {
                terrain: float(
                    sum(
                        stats.terrain_slip_rates.get(terrain, 0.0)
                        for stats in history
                    )
                    / len(history)
                )
                for terrain in terrain_names
            },
        }

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

    def run_episode(
        self,
        episode_index: int,
        *,
        training: bool,
        sample: bool,
        render: bool = False,
        capture_trace: bool = False,
        scenario_name: str | None = None,
        debug_trace: bool = False,
        policy_mode: str = "normal",
    ) -> tuple[EpisodeStats, List[Dict[str, object]]]:
        """
        Run a single simulation episode and return aggregated episode statistics and an optional per-tick trace.
        
        Executes environment-agent interaction for up to the episode's max steps, optionally performs learning updates, and can collect a per-tick trace with optional debug diagnostics. Validation: `policy_mode` must be either `"normal"` or `"reflex_only"`; `policy_mode="reflex_only"` requires a modular brain with reflexes enabled; `training=True` is only allowed with `policy_mode="normal"`.
        
        Parameters:
            episode_index (int): Index used to derive the episode RNG seed and to tag trace items.
            training (bool): Enable learning updates during the episode.
            sample (bool): Allow stochastic action sampling when True; otherwise use deterministic selection.
            render (bool): Render the world each tick when True.
            capture_trace (bool): Collect a list of per-tick dictionaries describing state, actions, rewards, and messages when True.
            scenario_name (str | None): Optional scenario identifier; when provided the scenario may override the map template and max steps.
            debug_trace (bool): When True and `capture_trace` is True, include expanded diagnostic fields in each trace item.
            policy_mode (str): Inference mode passed to the brain; valid values are `"normal"` or `"reflex_only"`.
        
        Returns:
            tuple[EpisodeStats, List[Dict[str, object]]]: EpisodeStats containing aggregated episode metrics, and a list of per-tick trace dictionaries (empty when `capture_trace` is False).
        """
        # Validate policy_mode and brain compatibility before any state mutations
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "Invalid policy_mode. Use 'normal' or 'reflex_only'."
            )
        if policy_mode == "reflex_only" and not self.brain.config.is_modular:
            raise ValueError(
                "policy_mode='reflex_only' requires the modular architecture."
            )
        if policy_mode == "reflex_only" and not self.brain.config.enable_reflexes:
            raise ValueError(
                "policy_mode='reflex_only' requires reflexes to be enabled."
            )
        if training and policy_mode != "normal":
            raise ValueError(
                "run_episode() only supports training=True with policy_mode='normal'."
            )
        episode_seed = self.seed + 997 * (episode_index + 1)
        scenario = get_scenario(scenario_name) if scenario_name is not None else None
        episode_map_template = scenario.map_template if scenario is not None else self.default_map_template
        if self.world.map_template_name != episode_map_template:
            self.world.configure_map_template(episode_map_template)
        observation = self.world.reset(seed=episode_seed)
        self.brain.reset_hidden_states()
        episode_max_steps = self.max_steps
        if scenario is not None:
            scenario.setup(self.world)
            observation = self.world.observe()
            episode_max_steps = scenario.max_steps

        trace: List[Dict[str, object]] = []
        total_reward = 0.0
        metrics = EpisodeMetricAccumulator(
            reward_component_names=REWARD_COMPONENT_NAMES,
            predator_states=PREDATOR_STATES,
        )

        for step in range(episode_max_steps):
            self.bus.set_tick(step)
            self.bus.publish(
                sender="environment",
                topic="observation",
                payload={
                    "state": self.world.state_dict(),
                    "meta": observation["meta"],
                },
            )

            act_kwargs: Dict[str, object] = {
                "sample": sample,
                "policy_mode": policy_mode,
            }
            try:
                act_signature = inspect.signature(self.brain.act)
                supports_training_kwarg = (
                    "training" in act_signature.parameters
                    or any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in act_signature.parameters.values()
                    )
                )
            except (TypeError, ValueError):
                supports_training_kwarg = True
            if supports_training_kwarg:
                act_kwargs["training"] = training

            decision = self.brain.act(
                observation,
                self.bus,
                **act_kwargs,
            )
            metrics.record_decision(decision)
            predator_state_before = self.world.lizard.mode
            next_observation, reward, done, info = self.world.step(decision.action_idx)
            self._attach_motor_execution_info(decision, info)
            learn_stats: Dict[str, object] = {}
            if training:
                learn_stats = self.brain.learn(decision, reward, next_observation, done)
                metrics.record_learning(learn_stats)
                self.bus.publish(
                    sender="learning",
                    topic="td_update",
                    payload=dict(learn_stats),
                )

            self.bus.publish(
                sender="environment",
                topic="transition",
                payload={
                    "selected_action": ACTIONS[decision.action_idx],
                    "executed_action": info.get("executed_action", ACTIONS[decision.action_idx]),
                    "motor_noise_applied": bool(info.get("motor_noise_applied", False)),
                    "reward": round(float(reward), 6),
                    "reward_components": info["reward_components"],
                    "done": bool(done),
                    "info": info,
                },
            )

            metrics.record_transition(
                step=step,
                observation_meta=observation["meta"],
                next_meta=next_observation["meta"],
                info=info,
                state=self.world.state,
                predator_state_before=predator_state_before,
                predator_state=self.world.lizard.mode,
            )

            if capture_trace:
                item: Dict[str, object] = {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "tick": step,
                    "training": training,
                    "policy_mode": policy_mode,
                    "scenario": scenario_name,
                    "action": info.get("action"),
                    "intended_action": info.get("intended_action"),
                    "executed_action": info.get("executed_action"),
                    "motor_noise_applied": bool(info.get("motor_noise_applied", False)),
                    "slip_reason": str(info.get("slip_reason", "none")),
                    "ate": bool(info.get("ate")),
                    "slept": bool(info.get("slept")),
                    "predator_contact": bool(info.get("predator_contact")),
                    "predator_escape": bool(info.get("predator_escape")),
                    "state": self.world.state_dict(),
                    "reward": float(reward),
                    "reward_components": info["reward_components"],
                    "predator_transition": info.get("predator_transition"),
                    "distance_deltas": info.get("distance_deltas", {}),
                    "event_log": info["event_log"],
                    "done": bool(done),
                    "messages": self.bus.serialize_current_tick(),
                }
                if debug_trace:
                    arbitration_payload = (
                        decision.arbitration_decision.to_payload()
                        if decision.arbitration_decision is not None
                        else {}
                    )
                    item["observation"] = jsonify_observation(observation)
                    item["next_observation"] = jsonify_observation(next_observation)
                    item["debug"] = {
                        "memory_vectors": next_observation["meta"].get("memory_vectors", {}),
                        "reflexes": {
                            result.name: {
                                "active": bool(result.active),
                                "best_action": ACTIONS[int(result.probs.argmax())],
                                "neural_logits": result.neural_logits.round(6).tolist() if result.neural_logits is not None else None,
                                "reflex_delta_logits": result.reflex_delta_logits.round(6).tolist() if result.reflex_delta_logits is not None else None,
                                "post_reflex_logits": result.post_reflex_logits.round(6).tolist() if result.post_reflex_logits is not None else None,
                                "reflex_applied": bool(result.reflex_applied),
                                "effective_reflex_scale": round(float(result.effective_reflex_scale), 6),
                                "module_reflex_override": bool(result.module_reflex_override),
                                "module_reflex_dominance": round(float(result.module_reflex_dominance), 6),
                                "contribution_share": round(float(result.contribution_share), 6),
                                "reflex": result.reflex.to_payload() if result.reflex is not None else None,
                            }
                            for result in decision.module_results
                        },
                        "action_center": {
                            "policy_mode": decision.policy_mode,
                            "input": decision.action_center_input.round(6).tolist(),
                            "logits": decision.action_center_logits.round(6).tolist(),
                            "policy": decision.action_center_policy.round(6).tolist(),
                            "selected_intent": ACTIONS[decision.action_intent_idx],
                            **dict(arbitration_payload),
                        },
                        "arbitration": dict(arbitration_payload),
                        "motor_cortex": {
                            "policy_mode": decision.policy_mode,
                            "input": decision.motor_input.round(6).tolist(),
                            "correction_logits": decision.motor_correction_logits.round(6).tolist(),
                            "motor_override": bool(decision.motor_override),
                            "selected_intent": ACTIONS[decision.action_intent_idx],
                            "arbitrated_intent": ACTIONS[decision.action_intent_idx],
                            "selected_action": ACTIONS[decision.motor_action_idx],
                            "motor_selected_action": ACTIONS[decision.motor_action_idx],
                            "sampled_action": ACTIONS[decision.action_idx],
                            "intended_action": info.get("intended_action", ACTIONS[decision.action_idx]),
                            "executed_action": info.get("executed_action", ACTIONS[decision.action_idx]),
                            "execution_slip_occurred": bool(decision.execution_slip_occurred),
                            "motor_noise_applied": bool(decision.motor_noise_applied),
                            "slip_reason": str(decision.slip_reason),
                            "orientation_alignment": round(float(decision.orientation_alignment), 6),
                            "terrain_difficulty": round(float(decision.terrain_difficulty), 6),
                            "execution_difficulty": round(float(decision.execution_difficulty), 6),
                        },
                        "final_reflex_override": bool(decision.final_reflex_override),
                        "total_logits_without_reflex": decision.total_logits_without_reflex.round(6).tolist(),
                        "total_logits": decision.total_logits.round(6).tolist(),
                        "vision": next_observation["meta"].get("vision", {}),
                        "predator": {
                            "mode": self.world.lizard.mode,
                            "mode_ticks": self.world.lizard.mode_ticks,
                            "patrol_target": self.world.lizard.patrol_target,
                            "last_known_spider": self.world.lizard.last_known_spider,
                            "investigate_ticks": self.world.lizard.investigate_ticks,
                            "investigate_target": self.world.lizard.investigate_target,
                            "recover_ticks": self.world.lizard.recover_ticks,
                            "wait_target": self.world.lizard.wait_target,
                            "ambush_ticks": self.world.lizard.ambush_ticks,
                            "chase_streak": self.world.lizard.chase_streak,
                            "failed_chases": self.world.lizard.failed_chases,
                        },
                    }
                trace.append(item)

            total_reward += reward
            observation = next_observation

            if render:
                print(self.world.render())
                print("-" * self.world.width)

            if done:
                break

        state = self.world.state
        stats = metrics.finalize(
            episode=episode_index,
            seed=episode_seed,
            training=training,
            scenario=scenario_name,
            total_reward=float(total_reward),
            state=state,
        )
        return stats, trace

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

    def _capture_checkpoint_candidate(
        self,
        *,
        root_dir: Path,
        episode: int,
        scenario_names: Sequence[str],
        metric: str,
        selection_scenario_episodes: int,
        eval_reflex_scale: float | None = 0.0,
    ) -> Dict[str, object]:
        """
        Create a checkpoint candidate by saving the current brain state and evaluating its behavior on the provided scenarios.
        
        Parameters:
            root_dir (Path): Directory where the checkpoint directory will be created.
            episode (int): Episode index used to name the checkpoint directory (e.g., episode_00042).
            scenario_names (Sequence[str]): Names of scenarios to evaluate for candidate scoring.
            metric (str): Primary metric label attached to the candidate metadata.
            selection_scenario_episodes (int): Number of evaluation episodes to run per scenario for scoring.
            eval_reflex_scale (float | None): Reflex scale override used during candidate evaluation; when None uses the simulation's effective reflex setting. Defaults to 0.0.
        
        Returns:
            candidate (Dict[str, object]): Metadata for the checkpoint candidate containing:
                - name (str): Checkpoint directory name.
                - episode (int): Episode index.
                - path (Path): Path to the saved checkpoint directory.
                - metric (str): Provided metric name.
                - eval_reflex_scale (float): Effective reflex scale used for evaluation.
                - evaluation_summary (Dict[str, object]): Behavior-suite summary produced during evaluation (includes `eval_reflex_scale`, `mean_final_reflex_override_rate`, `mean_reflex_dominance`, and `mean_reward`).
                - scenario_success_rate (float): Aggregated scenario success rate from the evaluation summary.
                - episode_success_rate (float): Aggregated episode success rate from the evaluation summary.
                - mean_reward (float): Mean reward derived from the evaluated behavior payload.
        """
        checkpoint_name = f"episode_{int(episode):05d}"
        checkpoint_path = root_dir / checkpoint_name
        self.brain.save(checkpoint_path)
        payload, _, _ = self.evaluate_behavior_suite(
            list(scenario_names),
            episodes_per_scenario=max(1, int(selection_scenario_episodes)),
            capture_trace=False,
            debug_trace=False,
            eval_reflex_scale=eval_reflex_scale,
        )
        summary = dict(payload.get("summary", {}))
        legacy_summaries = [
            data
            for data in payload.get("legacy_scenarios", {}).values()
            if isinstance(data, dict)
        ]
        mean_final_reflex_override_rate = (
            sum(
                float(data.get("mean_final_reflex_override_rate", 0.0))
                for data in legacy_summaries
            )
            / len(legacy_summaries)
            if legacy_summaries
            else 0.0
        )
        mean_reflex_dominance = (
            sum(
                float(data.get("mean_reflex_dominance", 0.0))
                for data in legacy_summaries
            )
            / len(legacy_summaries)
            if legacy_summaries
            else 0.0
        )
        summary["eval_reflex_scale"] = (
            self._effective_reflex_scale(eval_reflex_scale)
            if eval_reflex_scale is not None
            else self._effective_reflex_scale()
        )
        summary["mean_final_reflex_override_rate"] = mean_final_reflex_override_rate
        summary["mean_reflex_dominance"] = mean_reflex_dominance
        summary["mean_reward"] = mean_reward_from_behavior_payload(payload)
        candidate = {
            "name": checkpoint_name,
            "episode": int(episode),
            "path": checkpoint_path,
            "metric": str(metric),
            "eval_reflex_scale": float(summary["eval_reflex_scale"]),
            "evaluation_summary": summary,
            "scenario_success_rate": float(summary.get("scenario_success_rate", 0.0)),
            "episode_success_rate": float(summary.get("episode_success_rate", 0.0)),
            "mean_reward": float(summary["mean_reward"]),
        }
        return candidate

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

    @staticmethod
    def _label_evaluation_summary(
        summary: Dict[str, object],
        *,
        eval_reflex_scale: float | None,
        competence_type: str | None = None,
    ) -> Dict[str, object]:
        """
        Annotates an evaluation summary with competence metadata.
        
        If `competence_type` is provided, it is normalized and used; otherwise the label is derived from `eval_reflex_scale`. The returned dictionary is a shallow copy of `summary` with the following keys added or overwritten: `competence_type` (normalized label), `is_primary_benchmark` (true when the label is `"self_sufficient"`), and `eval_reflex_scale`.
        
        Parameters:
            summary: The evaluation summary to annotate.
            eval_reflex_scale: Evaluation reflex support scale used to derive competence when `competence_type` is not provided.
            competence_type: Optional explicit competence label to use instead of deriving from `eval_reflex_scale`.
        
        Returns:
            The annotated evaluation summary dictionary.
        """
        label = normalize_competence_label(
            competence_type
            if competence_type is not None
            else competence_label_from_eval_reflex_scale(eval_reflex_scale)
        )
        result = deepcopy(summary)
        result["competence_type"] = label
        result["is_primary_benchmark"] = label == "self_sufficient"
        result["eval_reflex_scale"] = eval_reflex_scale
        return result

    @staticmethod
    def _evaluation_competence_gap(
        *,
        self_sufficient: Dict[str, object],
        scaffolded: Dict[str, object] | None,
    ) -> Dict[str, object]:
        """
        Compute the scaffolded minus self-sufficient deltas for primary evaluation metrics.
        
        Parameters:
            self_sufficient (dict): Evaluation summary for the self-sufficient (no-reflex) condition.
            scaffolded (dict | None): Evaluation summary for the scaffolded (reflex-supported) condition,
                or None to indicate no scaffolded summary is available (in which case `self_sufficient` is used).
        
        Returns:
            dict: A dictionary with keys:
                - `basis`: the string `"scaffolded_minus_self_sufficient"`.
                - `scenario_success_rate_delta`: difference in `scenario_success_rate` (scaffolded − self_sufficient),
                  rounded to 6 decimal places.
                - `episode_success_rate_delta`: difference in `episode_success_rate` (scaffolded − self_sufficient),
                  rounded to 6 decimal places.
                - `mean_reward_delta`: difference in `mean_reward` (scaffolded − self_sufficient),
                  rounded to 6 decimal places.
        """
        scaffolded_summary = scaffolded if scaffolded is not None else self_sufficient
        return {
            "basis": "scaffolded_minus_self_sufficient",
            "scenario_success_rate_delta": round(
                float(scaffolded_summary.get("scenario_success_rate", 0.0))
                - float(self_sufficient.get("scenario_success_rate", 0.0)),
                6,
            ),
            "episode_success_rate_delta": round(
                float(scaffolded_summary.get("episode_success_rate", 0.0))
                - float(self_sufficient.get("episode_success_rate", 0.0)),
                6,
            ),
            "mean_reward_delta": round(
                float(scaffolded_summary.get("mean_reward", 0.0))
                - float(self_sufficient.get("mean_reward", 0.0)),
                6,
            ),
        }

    @classmethod
    def _evaluation_summary_block(
        cls,
        *,
        primary: Dict[str, object],
        self_sufficient: Dict[str, object],
        scaffolded: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        """
        Builds a consolidated evaluation block that preserves top-level primary metrics and embeds comparison blocks.
        
        Parameters:
            primary (Dict[str, object]): Flat primary evaluation metrics to remain at the top level of the block.
            self_sufficient (Dict[str, object]): Evaluation metrics for the no-reflex / primary condition.
            scaffolded (Dict[str, object] | None): Evaluation metrics for the reflex-supported condition; when
                omitted, an empty scaffolded block is inserted.
        
        Returns:
            Dict[str, object]: A summary block containing:
                - the original `primary` fields at the top level,
                - `self_sufficient`: a copy of the self-sufficient metrics,
                - `scaffolded`: a copy of the scaffolded metrics (or an empty dict),
                - `primary_benchmark`: a duplicate of `self_sufficient` for downstream compatibility,
                - `competence_gap`: a dict of deltas between `self_sufficient` and `scaffolded` metrics.
        """
        result = deepcopy(primary)
        result["self_sufficient"] = deepcopy(self_sufficient)
        result["scaffolded"] = deepcopy(scaffolded) if scaffolded is not None else {}
        result["primary_benchmark"] = deepcopy(self_sufficient)
        result["competence_gap"] = cls._evaluation_competence_gap(
            self_sufficient=self_sufficient,
            scaffolded=scaffolded,
        )
        return result

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
        checkpoint_metric: str,
        checkpoint_interval: int,
        checkpoint_dir: str | Path | None,
        checkpoint_scenario_names: Sequence[str] | None,
        selection_scenario_episodes: int,
        checkpoint_selection_config: CheckpointSelectionConfig | None = None,
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
            checkpoint_metric (str): Primary metric label used when evaluating checkpoint candidates.
            checkpoint_interval (int): Interval (in completed episodes) at which to capture checkpoint candidates.
            checkpoint_dir (str | Path | None): Optional directory in which to persist selected best/last checkpoints.
            checkpoint_scenario_names (Sequence[str] | None): Sequence of scenario names used to evaluate checkpoint candidates; defaults to all scenarios.
            selection_scenario_episodes (int): Episodes per scenario to use when evaluating each checkpoint candidate.
            checkpoint_selection_config (CheckpointSelectionConfig | None): Optional configuration that controls ranking metric and penalty weights; when omitted a default is created from `checkpoint_metric`.
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
        selection_config = (
            checkpoint_selection_config
            if checkpoint_selection_config is not None
            else CheckpointSelectionConfig(metric=checkpoint_metric)
        )
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
                        metric=checkpoint_metric,
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
                            metric=checkpoint_metric,
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
                        metric=checkpoint_metric,
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
                "metric": str(checkpoint_metric),
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
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_scenario_names: Sequence[str] | None = None,
        selection_scenario_episodes: int = 1,
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        reflex_anneal_final_scale: float | None = None,
        curriculum_profile: str = "none",
        training_regime: str | TrainingRegimeSpec | None = None,
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
            checkpoint_metric: Primary metric label used to rank checkpoint candidates when checkpoint_selection == "best".
            checkpoint_interval: Episode interval for capturing checkpoint candidates; when None the resolved budget default is used.
            checkpoint_dir: Destination directory to persist selected "best" and "last" checkpoints; persistence is skipped when None.
            checkpoint_scenario_names: Scenario names used to evaluate checkpoint candidates; defaults are used when None.
            selection_scenario_episodes: Episodes per scenario used when evaluating checkpoint candidates.
            checkpoint_override_penalty: Penalty weight (direct mode) applied for mean final reflex override rate when scoring candidates.
            checkpoint_dominance_penalty: Penalty weight (direct mode) applied for mean reflex dominance when scoring candidates.
            checkpoint_penalty_mode: Mode controlling candidate ranking: CheckpointPenaltyMode.TIEBREAKER preserves legacy tuple ordering, CheckpointPenaltyMode.DIRECT applies a penalized composite score.
            reflex_anneal_final_scale: If provided, target reflex scale for annealing during training; when None no custom anneal is applied. Mutually exclusive with providing a TrainingRegimeSpec via training_regime.
            curriculum_profile: Curriculum profile name to use during training (e.g., "none", "ecological_v1", "ecological_v2").
            training_regime: Optional named training regime or explicit TrainingRegimeSpec that defines annealing schedule and optional late finetuning.
        
        Returns:
            summary: Consolidated training and evaluation summary suitable for serialization.
            evaluation_trace: Captured trace for the final evaluation run when capture_evaluation_trace is True; otherwise an empty list.
        
        Raises:
            ValueError: If checkpoint_selection is not one of "none" or "best", if checkpoint_metric is invalid when checkpoint_selection == "best", if curriculum_profile is not supported, or if both training_regime and reflex_anneal_final_scale are provided.
        """
        if checkpoint_selection not in {"none", "best"}:
            raise ValueError(
                "Invalid checkpoint_selection. Use 'none' or 'best'."
            )
        checkpoint_selection_config: CheckpointSelectionConfig | None = None
        if training_regime is not None and reflex_anneal_final_scale is not None:
            raise ValueError(
                "training_regime and reflex_anneal_final_scale are mutually exclusive."
            )
        training_regime_spec: TrainingRegimeSpec | None = None
        if isinstance(training_regime, TrainingRegimeSpec):
            training_regime_spec = training_regime
        elif training_regime is not None:
            training_regime_spec = resolve_training_regime(str(training_regime))
        curriculum_profile = validate_curriculum_profile(curriculum_profile)
        if checkpoint_selection == "best":
            checkpoint_selection_config = CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            )
            # Intentionally call checkpoint_candidate_sort_key() with an empty
            # candidate to validate checkpoint_metric early; invalid metrics
            # raise ValueError here before any training or checkpoint work runs.
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
                    checkpoint_metric=checkpoint_metric,
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

    def run_scenarios(
        self,
        names: List[str] | None = None,
        *,
        episodes_per_scenario: int = 1,
        capture_trace: bool = False,
        debug_trace: bool = False,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Execute each scenario in `names` (or all known scenarios if `None`) for the specified number of episodes and aggregate episode statistics per scenario.
        
        Parameters:
            names (List[str] | None): Scenario names to run; if `None`, runs the default set of scenarios.
            episodes_per_scenario (int): Number of episodes to run for each scenario.
            capture_trace (bool): If true, capture a public trace from the final run of each scenario.
            debug_trace (bool): If true, include additional debug information in captured traces.
        
        Returns:
            tuple:
                results (Dict[str, object]): Mapping from scenario name to aggregated episode metrics (legacy aggregate format).
                trace (List[Dict[str, object]]): The captured trace from the final public run (empty if none captured).
        """
        stats_histories, _, trace = self._execute_behavior_suite(
            names=names,
            episodes_per_scenario=episodes_per_scenario,
            capture_trace=capture_trace,
            debug_trace=debug_trace,
        )
        results: Dict[str, object] = {
            name: self._aggregate_group(history)
            for name, history in stats_histories.items()
        }
        return results, trace

    def evaluate_behavior_suite(
        self,
        names: List[str] | None = None,
        *,
        episodes_per_scenario: int = 1,
        capture_trace: bool = False,
        debug_trace: bool = False,
        eval_reflex_scale: float | None = None,
        policy_mode: str = "normal",
        extra_row_metadata: Dict[str, object] | None = None,
        summary_only: bool = False,
    ) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
        """
        Evaluate the specified behavior scenarios and produce aggregated metrics, an optional execution trace, and flattened CSV-ready rows.
        
        Parameters:
            names (List[str] | None): Scenario names to evaluate; evaluates the simulation's default set when None.
            episodes_per_scenario (int): Number of episodes to run per scenario.
            capture_trace (bool): If True, return the public execution trace from the final run (the last episode of the last scenario).
            debug_trace (bool): If True and a trace is captured, include debug-level details in trace items.
            eval_reflex_scale (float | None): If provided, temporarily override the brain's runtime reflex scale for the evaluation; the previous scale is restored after evaluation.
            policy_mode (str): Inference mode forwarded to run_episode() for evaluation runs (e.g., "normal" or "reflex_only").
            extra_row_metadata (Dict[str, object] | None): Metadata merged into every flattened CSV row after standard simulation annotations.
            summary_only (bool): If True, skip construction of flattened rows and return an empty rows list.
        
        Returns:
            payload (Dict[str, object]): Behavior-suite payload containing:
                - `suite`: per-scenario behavioral summaries,
                - `summary`: aggregated suite summary (includes `eval_reflex_scale` and competence labeling),
                - `legacy_scenarios`: legacy aggregated episode statistics per scenario.
            trace (List[Dict[str, object]]): Captured public trace from the final run when `capture_trace` is True; empty list otherwise.
            rows (List[Dict[str, object]]): Flattened per-episode behavior rows annotated with evaluation/reflex metadata and merged `extra_row_metadata`; empty when `summary_only` is True.
        """
        previous_reflex_scale = float(self.brain.current_reflex_scale)
        effective_eval_reflex_scale = self._effective_reflex_scale(
            eval_reflex_scale if eval_reflex_scale is not None else previous_reflex_scale
        )
        competence_type = competence_label_from_eval_reflex_scale(
            effective_eval_reflex_scale
        )
        if eval_reflex_scale is not None:
            self.brain.set_runtime_reflex_scale(eval_reflex_scale)
        try:
            stats_histories, behavior_histories, trace = self._execute_behavior_suite(
                names=names,
                episodes_per_scenario=episodes_per_scenario,
                capture_trace=capture_trace,
                debug_trace=debug_trace,
                policy_mode=policy_mode,
            )
        finally:
            self.brain.set_runtime_reflex_scale(previous_reflex_scale)
        payload = self._build_behavior_payload(
            stats_histories=stats_histories,
            behavior_histories=behavior_histories,
            competence_label=competence_type,
        )
        payload["summary"]["eval_reflex_scale"] = effective_eval_reflex_scale
        if summary_only:
            return payload, trace, []
        rows: List[Dict[str, object]] = []
        for name, score_group in behavior_histories.items():
            scenario = get_scenario(name)
            rows.extend(
                flatten_behavior_rows(
                    score_group,
                    reward_profile=self.world.reward_profile,
                    scenario_map=scenario.map_template,
                    simulation_seed=self.seed,
                    scenario_description=scenario.description,
                    scenario_objective=scenario.objective,
                    scenario_focus=scenario.diagnostic_focus,
                    evaluation_map=self.default_map_template,
                    eval_reflex_scale=effective_eval_reflex_scale,
                    competence_label=competence_type,
                )
            )
        return payload, trace, self._annotate_behavior_rows(
            rows,
            eval_reflex_scale=eval_reflex_scale,
            extra_metadata=extra_row_metadata,
        )

    @contextmanager
    def _swap_eval_noise_profile(
        self,
        noise_profile: str | NoiseConfig | None,
    ) -> Iterator[None]:
        """
        Temporarily replace the simulation world's noise profile for the duration of a context and restore the previous profile on exit.
        
        Parameters:
            noise_profile (str | NoiseConfig | None): Noise profile name, a NoiseConfig instance, or `None`; resolved via `resolve_noise_profile` before being applied to `self.world.noise_profile`.
        """
        resolved_eval_noise_profile = resolve_noise_profile(noise_profile)
        previous_noise_profile = self.world.noise_profile
        self.world.noise_profile = resolved_eval_noise_profile
        try:
            yield
        finally:
            self.world.noise_profile = previous_noise_profile

    def _execute_behavior_suite(
        self,
        *,
        names: List[str] | None,
        episodes_per_scenario: int,
        capture_trace: bool,
        debug_trace: bool,
        base_index: int = 100_000,
        policy_mode: str = "normal",
    ) -> tuple[
        Dict[str, List[EpisodeStats]],
        Dict[str, List[BehavioralEpisodeScore]],
        List[Dict[str, object]],
    ]:
        """
        Run the configured scenarios for a behavior suite and collect per-episode stats, behavioral scores, and an optional public trace.
        
        Parameters:
            names (List[str] | None): Scenario names to run; if None, the default scenario set is used.
            episodes_per_scenario (int): Number of episodes to run for each scenario (minimum 1).
            capture_trace (bool): If True, capture and return the public trace from the final run of the last scenario; otherwise no trace is returned.
            debug_trace (bool): If True and a public trace is being captured, include debug-level details in that trace.
            base_index (int): Base episode index used to derive per-episode identifiers/seeds.
            policy_mode (str): Inference mode forwarded to run_episode() for each scenario rollout.
        
        Returns:
            stats_histories (Dict[str, List[EpisodeStats]]): Mapping from scenario name to the list of EpisodeStats for each run.
            behavior_histories (Dict[str, List[BehavioralEpisodeScore]]): Mapping from scenario name to the list of per-episode behavioral scores.
            trace (List[Dict[str, object]]): Captured public trace from the final run of the last scenario when `capture_trace` is True; empty list otherwise.
        """
        scenario_names = names if names is not None else list(SCENARIO_NAMES)
        stats_histories: Dict[str, List[EpisodeStats]] = {}
        behavior_histories: Dict[str, List[BehavioralEpisodeScore]] = {}
        trace: List[Dict[str, object]] = []
        run_count = max(1, int(episodes_per_scenario))
        for offset, name in enumerate(scenario_names):
            scenario = get_scenario(name)
            stats_group: List[EpisodeStats] = []
            score_group: List[BehavioralEpisodeScore] = []
            for run_idx in range(run_count):
                capture_public_trace = capture_trace and run_idx == run_count - 1
                stats, run_trace = self.run_episode(
                    base_index + offset * run_count + run_idx,
                    training=False,
                    sample=False,
                    capture_trace=True,
                    scenario_name=name,
                    debug_trace=debug_trace and capture_public_trace,
                    policy_mode=policy_mode,
                )
                stats_group.append(stats)
                score = scenario.score_episode(stats, run_trace)
                score.behavior_metrics.update(
                    self._episode_stats_behavior_metrics(stats)
                )
                score_group.append(score)
                if capture_public_trace:
                    trace = run_trace
            stats_histories[name] = stats_group
            behavior_histories[name] = score_group
        return stats_histories, behavior_histories, trace

    def _build_behavior_payload(
        self,
        *,
        stats_histories: Dict[str, List[EpisodeStats]],
        behavior_histories: Dict[str, List[BehavioralEpisodeScore]],
        competence_label: str = "mixed",
    ) -> Dict[str, object]:
        """
        Constructs a behavior-suite payload summarizing per-scenario behavioral scores and their corresponding legacy episode metrics.
        
        Parameters:
            stats_histories (Dict[str, List[EpisodeStats]]): Mapping from scenario name to legacy episode statistics for that scenario.
            behavior_histories (Dict[str, List[BehavioralEpisodeScore]]): Mapping from scenario name to per-episode behavioral score objects for that scenario.
            competence_label (str): Label describing the competence context used when summarizing the suite (e.g., "mixed", "self_sufficient", "scaffolded").
        
        Returns:
            Dict[str, object]: Payload with three keys:
                - "suite": dict mapping each scenario name to its aggregated behavioral scoring summary (includes description, objective, checks, and attached legacy metrics).
                - "summary": overall suite summary produced by summarize_behavior_suite, annotated with the provided competence label.
                - "legacy_scenarios": dict mapping each scenario name to its legacy aggregated episode metrics (suitable for compaction via compact_aggregate).
        """
        suite: Dict[str, object] = {}
        legacy_scenarios: Dict[str, object] = {}
        for name, score_group in behavior_histories.items():
            scenario = get_scenario(name)
            legacy_scenarios[name] = self._aggregate_group(stats_histories.get(name, []))
            suite_entry = aggregate_behavior_scores(
                score_group,
                scenario=name,
                description=scenario.description,
                objective=scenario.objective,
                check_specs=scenario.behavior_checks,
                diagnostic_focus=scenario.diagnostic_focus,
                success_interpretation=scenario.success_interpretation,
                failure_interpretation=scenario.failure_interpretation,
                budget_note=scenario.budget_note,
                legacy_metrics=legacy_scenarios[name],
            )
            probe_metadata = _probe_metadata_for_scenario(name)
            suite_entry["is_capability_probe"] = (
                probe_metadata["probe_type"] == ProbeType.CAPABILITY_PROBE.value
            )
            suite_entry.update(probe_metadata)
            suite[name] = suite_entry
        return {
            "suite": suite,
            "summary": summarize_behavior_suite(
                suite,
                competence_label=competence_label,
            ),
            "legacy_scenarios": legacy_scenarios,
        }

    def _aggregate_group(self, history: List[EpisodeStats]) -> Dict[str, object]:
        """
        Aggregate a list of episode statistics into a group-level summary.
        
        When `history` contains entries, returns an aggregated statistics dictionary computed from those episodes. When `history` is empty, returns a fully populated summary dictionary with zeroed numeric metrics, empty `episodes_detail`, and sensible defaults for role/state distributions (all zeros) and `dominant_predator_state` set to `"PATROL"`.
        
        Parameters:
        	history (List[EpisodeStats]): List of per-episode statistics to aggregate.
        
        Returns:
        	Dict[str, object]: A dictionary of aggregated group statistics. Contains keys such as
        	`episodes`, `mean_reward`, various `mean_*` metrics (food, sleep, predator contacts/escapes,
        	night-role metrics, predator response metrics, distance deltas, survival_rate),
        	`mean_reward_components` (per REWARD_COMPONENT_NAMES), `mean_predator_state_ticks`,
        	`mean_predator_state_occupancy` (per PREDATOR_STATES), `mean_predator_mode_transitions`,
        	`dominant_predator_state`, and `episodes_detail`.
        """
        if history:
            return aggregate_episode_stats(history)
        return {
            "episodes": 0,
            "mean_reward": 0.0,
            "mean_food": 0.0,
            "mean_sleep": 0.0,
            "mean_predator_contacts": 0.0,
            "mean_predator_escapes": 0.0,
            "mean_predator_contacts_by_type": {
                name: 0.0 for name in PREDATOR_TYPE_NAMES
            },
            "mean_predator_escapes_by_type": {
                name: 0.0 for name in PREDATOR_TYPE_NAMES
            },
            "mean_predator_response_latency_by_type": {
                name: 0.0 for name in PREDATOR_TYPE_NAMES
            },
            "mean_night_shelter_occupancy_rate": 0.0,
            "mean_night_stillness_rate": 0.0,
            "mean_night_role_ticks": {
                role: 0.0 for role in ("outside", "entrance", "inside", "deep")
            },
            "mean_night_role_distribution": {
                role: 0.0 for role in ("outside", "entrance", "inside", "deep")
            },
            "mean_predator_response_events": 0.0,
            "mean_predator_response_latency": 0.0,
            "mean_sleep_debt": 0.0,
            "mean_food_distance_delta": 0.0,
            "mean_shelter_distance_delta": 0.0,
            "survival_rate": 0.0,
            "mean_reward_components": {
                name: 0.0 for name in REWARD_COMPONENT_NAMES
            },
            "mean_predator_state_ticks": {
                name: 0.0 for name in PREDATOR_STATES
            },
            "mean_predator_state_occupancy": {
                name: 0.0 for name in PREDATOR_STATES
            },
            "mean_predator_mode_transitions": 0.0,
            "dominant_predator_state": "PATROL",
            "mean_reflex_usage_rate": 0.0,
            "mean_final_reflex_override_rate": 0.0,
            "mean_reflex_dominance": 0.0,
            "mean_module_reflex_usage_rate": {
                interface.name: 0.0 for interface in MODULE_INTERFACES
            },
            "mean_module_reflex_override_rate": {
                interface.name: 0.0 for interface in MODULE_INTERFACES
            },
            "mean_module_reflex_dominance": {
                interface.name: 0.0 for interface in MODULE_INTERFACES
            },
            "mean_module_contribution_share": {
                name: 0.0 for name in PROPOSAL_SOURCE_NAMES
            },
            "mean_module_response_by_predator_type": {
                predator_type: {
                    name: 0.0 for name in PROPOSAL_SOURCE_NAMES
                }
                for predator_type in PREDATOR_TYPE_NAMES
            },
            "mean_module_credit_weights": {
                name: 0.0 for name in PROPOSAL_SOURCE_NAMES
            },
            "module_gradient_norm_means": {
                name: 0.0 for name in PROPOSAL_SOURCE_NAMES
            },
            "mean_motor_slip_rate": 0.0,
            "mean_orientation_alignment": 0.0,
            "mean_terrain_difficulty": 0.0,
            "mean_terrain_slip_rates": {},
            "dominant_module": "",
            "dominant_module_distribution": {
                name: 0.0 for name in PROPOSAL_SOURCE_NAMES
            },
            "mean_dominant_module_share": 0.0,
            "mean_effective_module_count": 0.0,
            "mean_module_agreement_rate": 0.0,
            "mean_module_disagreement_rate": 0.0,
            "episodes_detail": [],
        }

    def _build_summary(self, training_history: List[EpisodeStats], evaluation_history: List[EpisodeStats]) -> Dict[str, object]:
        """
        Construct a consolidated summary of the simulation configuration, training progress, and evaluation results.
        
        The returned dictionary contains metadata about the run and aggregated metrics, including:
        - "is_experiment_of_record": whether the run's training regime matches the experiment-of-record.
        - "config": snapshot of run configuration (seed, max_steps, architecture fingerprint/version, brain and world summaries, learning hyperparameters, operational/noise profiles, budget, and training_regime); may include "reflex_schedule" when available.
        - "reward_audit": reward-audit summary for the current reward profile.
        - "training": aggregated metrics for the full training history.
        - "training_last_window": aggregated metrics for up to the last 10 training episodes.
        - "evaluation": structured evaluation block containing the primary evaluation and nested `self_sufficient` and optional `scaffolded` evaluations along with computed competence metadata.
        - "motor_stage": motor-stage summaries for training, training_last_window, and evaluation.
        - "parameter_norms": model parameter norm statistics.
        - Optional top-level entries when available: "checkpointing", "evaluation_without_reflex_support", and "curriculum".
        
        Returns:
            summary (dict): A JSON-serializable mapping with the keys described above.
        """
        from .comparison import (
            austere_survival_gate_passed,
            build_reward_audit,
            episode_history_reward_payload,
            shaping_reduction_status,
        )

        last_window = training_history[-min(10, len(training_history)) :]
        raw_evaluation_summary = self._aggregate_group(evaluation_history)
        primary_eval_reflex_scale = self._effective_reflex_scale()
        primary_competence_type = competence_label_from_eval_reflex_scale(
            primary_eval_reflex_scale
        )
        primary_evaluation_summary = self._label_evaluation_summary(
            raw_evaluation_summary,
            eval_reflex_scale=primary_eval_reflex_scale,
            competence_type=primary_competence_type,
        )
        if primary_competence_type == "self_sufficient":
            self_sufficient_evaluation = primary_evaluation_summary
            scaffolded_evaluation = None
        elif primary_competence_type == "scaffolded":
            self_sufficient_evaluation = self._label_evaluation_summary(
                self._aggregate_group([]),
                eval_reflex_scale=0.0,
                competence_type="self_sufficient",
            )
            scaffolded_evaluation = primary_evaluation_summary
        else:
            self_sufficient_evaluation = self._label_evaluation_summary(
                self._aggregate_group([]),
                eval_reflex_scale=0.0,
                competence_type="self_sufficient",
            )
            scaffolded_evaluation = None
        training_regime_summary = deepcopy(self._latest_training_regime_summary)
        training_regime_summary["is_experiment_of_record"] = (
            str(training_regime_summary.get("name", "baseline"))
            == EXPERIMENT_OF_RECORD_REGIME
        )
        comparison_history = (
            evaluation_history
            if evaluation_history
            else (last_window if last_window else training_history)
        )
        reward_audit = build_reward_audit(
            current_profile=self.world.reward_profile,
            comparison_payload={
                "reward_profiles": {
                    self.world.reward_profile: episode_history_reward_payload(
                        comparison_history
                    )
                }
            },
        )
        summary = {
            "is_experiment_of_record": bool(
                training_regime_summary["is_experiment_of_record"]
            ),
            "config": {
                "seed": self.seed,
                "max_steps": self.max_steps,
                "architecture_version": self.brain.ARCHITECTURE_VERSION,
                "architecture_fingerprint": self.brain._architecture_fingerprint(),
                "brain": self.brain.config.to_summary(),
                "world": {
                    "width": self.world.width,
                    "height": self.world.height,
                    "food_count": self.world.food_count,
                    "day_length": self.world.day_length,
                    "night_length": self.world.night_length,
                    "vision_range": self.world.vision_range,
                    "food_smell_range": self.world.food_smell_range,
                    "predator_smell_range": self.world.predator_smell_range,
                    "lizard_vision_range": self.world.lizard_vision_range,
                    "lizard_move_interval": self.world.lizard_move_interval,
                    "reward_profile": self.world.reward_profile,
                    "map_template": self.world.map_template_name,
                },
                "learning": {
                    "gamma": self.brain.gamma,
                    "module_lr": self.brain.module_lr,
                    "arbitration_lr": self.brain.arbitration_lr,
                    "arbitration_regularization_weight": self.brain.arbitration_regularization_weight,
                    "arbitration_valence_regularization_weight": self.brain.arbitration_valence_regularization_weight,
                    "motor_lr": self.brain.motor_lr,
                    "module_dropout": self.brain.module_dropout,
                },
                "operational_profile": self.operational_profile.to_summary(),
                "noise_profile": self.noise_profile.to_summary(),
                "budget": deepcopy(self.budget_summary),
                "training_regime": training_regime_summary,
            },
            "reward_audit": reward_audit,
            "shaping_reduction_status": shaping_reduction_status(reward_audit),
            "training": self._aggregate_group(training_history),
            "training_last_window": self._aggregate_group(last_window),
            "evaluation": self._evaluation_summary_block(
                primary=primary_evaluation_summary,
                self_sufficient=self_sufficient_evaluation,
                scaffolded=scaffolded_evaluation,
            ),
            "motor_stage": {
                "training": self._motor_stage_summary(training_history),
                "training_last_window": self._motor_stage_summary(last_window),
                "evaluation": self._motor_stage_summary(evaluation_history),
            },
            "parameter_norms": self.brain.parameter_norms(),
        }
        reward_audit_comparison = reward_audit.get("comparison")
        austere_gate_passed = austere_survival_gate_passed(
            reward_audit_comparison
            if isinstance(reward_audit_comparison, dict)
            else None
        )
        if austere_gate_passed is not None:
            summary["austere_survival_gate_passed"] = austere_gate_passed
        if self._latest_reflex_schedule_summary is not None:
            summary["config"]["reflex_schedule"] = deepcopy(
                self._latest_reflex_schedule_summary
            )
        if self._latest_checkpointing_summary is not None:
            summary["checkpointing"] = deepcopy(self._latest_checkpointing_summary)
        if self._latest_evaluation_without_reflex_support is not None:
            summary["evaluation_without_reflex_support"] = deepcopy(
                self._latest_evaluation_without_reflex_support
            )
        if self._latest_curriculum_summary is not None:
            summary["curriculum"] = deepcopy(self._latest_curriculum_summary)
        return summary

    def _annotate_behavior_rows(
        self,
        rows: List[Dict[str, object]],
        *,
        eval_reflex_scale: float | None = None,
        train_noise_profile: str | NoiseConfig | None = None,
        extra_metadata: Dict[str, object] | None = None,
    ) -> List[Dict[str, object]]:
        """
        Annotates flattened behavior rows with simulation and configuration metadata.
        
        Parameters:
            rows (List[Dict[str, object]]): Flattened behavior rows to annotate.
            eval_reflex_scale (float | None): Reflex runtime scale to record for these rows; when None the simulation's effective runtime reflex scale is used.
            train_noise_profile (str | NoiseConfig | None): Optional training-time noise profile to record alongside evaluation noise metadata.
            extra_metadata (Dict[str, object] | None): Additional key/value pairs merged into each annotated row after the standard fields.
        
        Returns:
            List[Dict[str, object]]: Shallow copies of the input rows augmented with simulation/configuration fields such as:
                ablation_variant, ablation_architecture, budget_profile, benchmark_strength,
                architecture_version, architecture_fingerprint, operational_profile,
                operational_profile_version, train_noise_profile and its config string,
                eval_noise_profile and its config string, checkpoint_source,
                reflex_scale, reflex_anneal_final_scale, eval_reflex_scale,
                competence_type, is_primary_benchmark, is_capability_probe, probe
                metadata, plus
                training-regime/curriculum metadata and any keys from `extra_metadata`.
        """
        from .comparison import noise_profile_csv_value

        annotated: List[Dict[str, object]] = []
        architecture_fingerprint = self.brain._architecture_fingerprint()
        eval_noise_profile = resolve_noise_profile(self.world.noise_profile)
        eval_noise_profile_config = noise_profile_csv_value(eval_noise_profile)
        resolved_train_noise_profile = (
            resolve_noise_profile(train_noise_profile)
            if train_noise_profile is not None
            else None
        )
        train_noise_profile_config = (
            noise_profile_csv_value(resolved_train_noise_profile)
            if resolved_train_noise_profile is not None
            else None
        )
        for row in rows:
            item = dict(row)
            item["ablation_variant"] = self.brain.config.name
            item["ablation_architecture"] = self.brain.config.architecture
            item["budget_profile"] = self.budget_profile_name
            item["benchmark_strength"] = self.benchmark_strength
            item["architecture_version"] = self.brain.ARCHITECTURE_VERSION
            item["architecture_fingerprint"] = architecture_fingerprint
            item["operational_profile"] = self.operational_profile.name
            item["operational_profile_version"] = self.operational_profile.version
            if resolved_train_noise_profile is not None:
                item["train_noise_profile"] = resolved_train_noise_profile.name
                item["train_noise_profile_config"] = train_noise_profile_config
            item["eval_noise_profile"] = eval_noise_profile.name
            item["eval_noise_profile_config"] = eval_noise_profile_config
            item["noise_profile"] = eval_noise_profile.name
            item["noise_profile_config"] = eval_noise_profile_config
            item["checkpoint_source"] = self.checkpoint_source
            item["reflex_scale"] = float(self.brain.config.reflex_scale)
            item["reflex_anneal_final_scale"] = (
                float(self._latest_reflex_schedule_summary["final_scale"])
                if self._latest_reflex_schedule_summary is not None
                else float(self.brain.config.reflex_scale)
            )
            effective_eval_reflex_scale = self._effective_reflex_scale(
                eval_reflex_scale
                if eval_reflex_scale is not None
                else self.brain.current_reflex_scale
            )
            competence_type = competence_label_from_eval_reflex_scale(
                effective_eval_reflex_scale
            )
            row_competence_type = item.get("competence_type")
            if row_competence_type is not None:
                explicit_type = normalize_competence_label(str(row_competence_type))
                if explicit_type != "mixed":
                    competence_type = explicit_type
            item["eval_reflex_scale"] = effective_eval_reflex_scale
            item["competence_type"] = competence_type
            item["is_primary_benchmark"] = competence_type == "self_sufficient"
            probe_metadata = _probe_metadata_for_scenario(
                str(item.get("scenario", ""))
            )
            item["is_capability_probe"] = (
                probe_metadata["probe_type"] == ProbeType.CAPABILITY_PROBE.value
            )
            item.update(probe_metadata)
            item.update(
                regime_row_metadata_from_summary(
                    self._latest_training_regime_summary,
                    self._latest_curriculum_summary,
                )
            )
            if extra_metadata:
                item.update(extra_metadata)
            annotated.append(item)
        return annotated
