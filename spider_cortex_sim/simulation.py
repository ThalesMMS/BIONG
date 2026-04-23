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
from dataclasses import replace
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
from .capacity_profiles import CapacityProfile, resolve_capacity_profile
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

from .simulation_checkpoints import SimulationCheckpointMixin
from .simulation_episode import SimulationEpisodeMixin
from .simulation_evaluation import SimulationEvaluationMixin
from .simulation_helpers import EXPERIMENT_OF_RECORD_REGIME, CAPABILITY_PROBE_SCENARIOS, default_behavior_evaluation, ensure_behavior_evaluation, is_capability_probe, _probe_metadata_for_scenario
from .simulation_summary import SimulationSummaryMixin
from .simulation_training import SimulationTrainingMixin
from .simulation_training_schedule import SimulationTrainingScheduleMixin


class SpiderSimulation(SimulationEpisodeMixin, SimulationCheckpointMixin, SimulationTrainingScheduleMixin, SimulationTrainingMixin, SimulationEvaluationMixin, SimulationSummaryMixin):
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
        capacity_profile: str | CapacityProfile | None = None,
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
        resolved_capacity_profile_spec = (
            capacity_profile
            if capacity_profile is not None
            else (
                self.brain_config.capacity_profile
                if self.brain_config.capacity_profile is not None
                else self.brain_config.capacity_profile_name
            )
        )
        self.capacity_profile = resolve_capacity_profile(resolved_capacity_profile_spec)
        replacement_fields = {
            "capacity_profile": self.capacity_profile,
            "capacity_profile_name": self.capacity_profile.name,
            "module_hidden_dims": dict(self.capacity_profile.module_hidden_dims),
            "integration_hidden_dim": self.capacity_profile.integration_hidden_dim,
        }
        self.brain_config = replace(self.brain_config, **replacement_fields)
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
            capacity_profile=self.capacity_profile,
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
        self._latest_distillation_summary: Dict[str, object] | None = None
        self._latest_training_regime_summary: Dict[str, object] = {
            "name": "baseline",
            "distillation_enabled": False,
            "mode": "flat",
            "curriculum_profile": "none",
            "annealing_schedule": AnnealingSchedule.NONE.value,
            "anneal_target_scale": float(self.brain.config.reflex_scale),
            "anneal_warmup_fraction": 0.0,
            "finetuning_episodes": 0,
            "finetuning_reflex_scale": 0.0,
            "loss_override_penalty_weight": 0.0,
            "loss_dominance_penalty_weight": 0.0,
            "distillation_epochs": 0,
            "distillation_temperature": 1.0,
            "distillation_lr": 0.0,
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
