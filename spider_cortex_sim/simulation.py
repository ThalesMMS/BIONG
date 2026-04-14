from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import re
import shutil
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Sequence

from .ablations import (
    BrainAblationConfig,
    PROPOSAL_SOURCE_NAMES,
    compare_predator_type_ablation_performance,
    default_brain_config,
    resolve_ablation_configs,
)
from .agent import BrainStep, SpiderBrain
from .budget_profiles import BudgetProfile, resolve_budget
from .learning_evidence import (
    resolve_learning_evidence_conditions,
)
from .bus import MessageBus
from .interfaces import MODULE_INTERFACES
from .maps import MAP_TEMPLATE_NAMES
from .memory import memory_leakage_audit
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
from .claim_tests import (
    ClaimTestSpec,
    primary_claim_test_names,
    resolve_claim_tests,
)
from .noise import (
    NoiseConfig,
    RobustnessMatrixSpec,
    canonical_robustness_matrix,
    resolve_noise_profile,
)
from .operational_profiles import OperationalProfile, resolve_operational_profile
from .perception import observation_leakage_audit
from .predator import PREDATOR_STATES
from .reward import REWARD_PROFILES, reward_component_audit, reward_profile_audit
from .scenarios import SCENARIO_NAMES, get_scenario
from .training_regimes import (
    AnnealingSchedule,
    TrainingRegimeSpec,
    resolve_training_regime,
)
from .world import ACTIONS, REWARD_COMPONENT_NAMES, SpiderWorld


CURRICULUM_PROFILE_NAMES: tuple[str, ...] = (
    "none",
    "ecological_v1",
    "ecological_v2",
)
EXPERIMENT_OF_RECORD_REGIME = "late_finetuning"
CURRICULUM_COLUMNS: tuple[str, ...] = (
    "training_regime",
    "training_regime_name",
    "curriculum_profile",
    "curriculum_phase",
    "curriculum_skill",
    "curriculum_phase_status",
    "curriculum_promotion_reason",
)
CURRICULUM_FOCUS_SCENARIOS: tuple[str, ...] = (
    "open_field_foraging",
    "corridor_gauntlet",
    "exposed_day_foraging",
    "food_deprivation",
)
MINIMAL_SHAPING_SURVIVAL_THRESHOLD: float = 0.50
SPECIALIZATION_ENGAGEMENT_CHECKS: dict[str, str] = {
    "visual_olfactory_pincer": "type_specific_response",
    "olfactory_ambush": "sensory_cortex_engaged",
    "visual_hunter_open_field": "visual_cortex_engaged",
}
CHECKPOINT_PENALTY_MODE_NAMES: tuple[str, ...] = ("tiebreaker", "direct")
CHECKPOINT_METRIC_ORDER: tuple[str, ...] = (
    "scenario_success_rate",
    "episode_success_rate",
    "mean_reward",
)


class CheckpointPenaltyMode(str, Enum):
    TIEBREAKER = "tiebreaker"
    DIRECT = "direct"


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    metric: str
    override_penalty_weight: float = 0.0
    dominance_penalty_weight: float = 0.0
    penalty_mode: CheckpointPenaltyMode | str = CheckpointPenaltyMode.TIEBREAKER

    def __post_init__(self) -> None:
        """
        Validate and normalize CheckpointSelectionConfig fields after initialization.
        
        Performs validation that `metric` is one of the allowed checkpoint metrics, coerces
        `penalty_mode` to a `CheckpointPenaltyMode`, ensures both penalty weights are
        finite and greater than or equal to zero, and writes the normalized values back
        onto the instance.
        
        Raises:
            ValueError: if `metric` is not in the allowed metric order, if `penalty_mode`
                is not a valid mode, or if either penalty weight is not finite or is
                negative.
        """
        metric = str(self.metric)
        if metric not in CHECKPOINT_METRIC_ORDER:
            raise ValueError(
                "Invalid checkpoint_metric. Use 'scenario_success_rate', "
                "'episode_success_rate' or 'mean_reward'."
            )
        try:
            mode = CheckpointPenaltyMode(self.penalty_mode)
        except ValueError as exc:
            available = ", ".join(repr(item) for item in CHECKPOINT_PENALTY_MODE_NAMES)
            raise ValueError(
                f"Invalid checkpoint_penalty_mode. Available modes: {available}."
            ) from exc
        override_weight = self._finite_non_negative(
            self.override_penalty_weight,
            "override_penalty_weight",
        )
        dominance_weight = self._finite_non_negative(
            self.dominance_penalty_weight,
            "dominance_penalty_weight",
        )
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "penalty_mode", mode)
        object.__setattr__(self, "override_penalty_weight", override_weight)
        object.__setattr__(self, "dominance_penalty_weight", dominance_weight)

    @staticmethod
    def _finite_non_negative(value: float, field_name: str) -> float:
        """
        Validate and coerce a numeric input to a finite, non-negative float.
        
        Parameters:
            value (float): The value to coerce to float and validate.
            field_name (str): Name used in error messages when validation fails.
        
        Returns:
            numeric (float): The input coerced to float, guaranteed finite and >= 0.0.
        
        Raises:
            ValueError: If the value is not finite or is less than 0.0.
        """
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite.")
        if numeric < 0.0:
            raise ValueError(f"{field_name} must be non-negative.")
        return numeric

    def to_summary(self) -> dict[str, object]:
        """
        Return a compact JSON-serializable summary of this checkpoint selection configuration.
        
        @returns
            A dict with the following keys:
            - `metric`: primary metric name used for ranking.
            - `override_penalty_weight`: numeric override penalty weight (float).
            - `dominance_penalty_weight`: numeric dominance penalty weight (float).
            - `penalty_mode`: penalty mode name as a string.
        """
        return {
            "metric": self.metric,
            "override_penalty_weight": float(self.override_penalty_weight),
            "dominance_penalty_weight": float(self.dominance_penalty_weight),
            "penalty_mode": self.penalty_mode.value,
        }


def _curriculum_profile_error_message() -> str:
    """
    Constructs an error message listing the valid curriculum profile names.
    
    Returns:
        str: Error message text stating the curriculum_profile is invalid and enumerating available profiles.
    """
    available = ", ".join(repr(name) for name in CURRICULUM_PROFILE_NAMES)
    return f"Invalid curriculum_profile. Available profiles: {available}."


def _validate_curriculum_profile(curriculum_profile: str) -> str:
    profile_name = str(curriculum_profile)
    if profile_name not in CURRICULUM_PROFILE_NAMES:
        raise ValueError(_curriculum_profile_error_message())
    return profile_name


@dataclass(frozen=True)
class PromotionCheckCriteria:
    """Promotion gate for a named behavior check.

    `aggregation` controls how a phase combines its check specs: "all" requires
    every spec to pass, while "any" promotes when at least one spec passes.
    """

    scenario: str
    check_name: str
    required_pass_rate: float
    aggregation: str = "all"


@dataclass(frozen=True)
class CurriculumPhaseDefinition:
    name: str
    training_scenarios: tuple[str, ...]
    promotion_scenarios: tuple[str, ...]
    success_threshold: float
    max_episodes: int
    min_episodes: int
    skill_name: str = ""
    promotion_check_specs: tuple[PromotionCheckCriteria, ...] = ()


SUBSKILL_CHECK_MAPPINGS: Dict[str, tuple[PromotionCheckCriteria, ...]] = {
    "shelter_exit": (
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="commits_to_foraging",
            required_pass_rate=1.0,
        ),
    ),
    "food_approach": (
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="approaches_food",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="open_field_foraging",
            check_name="made_food_progress",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="exposed_day_foraging",
            check_name="day_food_progress",
            required_pass_rate=1.0,
        ),
    ),
    "predator_response": (
        PromotionCheckCriteria(
            scenario="predator_edge",
            check_name="predator_detected",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="predator_edge",
            check_name="predator_reacted",
            required_pass_rate=1.0,
        ),
    ),
    "corridor_navigation": (
        PromotionCheckCriteria(
            scenario="corridor_gauntlet",
            check_name="corridor_survives",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="corridor_gauntlet",
            check_name="corridor_food_progress",
            required_pass_rate=1.0,
        ),
    ),
    "hunger_commitment": (
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="hunger_reduced",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="survives_deprivation",
            required_pass_rate=1.0,
        ),
    ),
}


class SpiderSimulation:
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
        self.operational_profile = resolve_operational_profile(operational_profile)
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
    def _resolve_curriculum_phase_budgets(total_episodes: int) -> list[int]:
        """Resolve per-phase budgets without exceeding the total episode count."""
        total = max(0, int(total_episodes))
        if total <= 0:
            return [0, 0, 0, 0]
        remaining = total
        phase_1 = min(remaining, max(1, total // 6))
        remaining -= phase_1
        phase_2 = min(remaining, max(1, total // 6))
        remaining -= phase_2
        phase_3 = min(remaining, max(1, total // 3))
        remaining -= phase_3
        phase_4 = remaining
        return [phase_1, phase_2, phase_3, phase_4]

    @classmethod
    def _resolve_curriculum_profile(
        cls,
        *,
        curriculum_profile: str,
        total_episodes: int,
    ) -> list[CurriculumPhaseDefinition]:
        """
        Resolve an ordered list of curriculum phases for the given profile and total episode budget.
        
        For "none" returns an empty list. For supported profiles builds per-phase CurriculumPhaseDefinition entries whose
        training and promotion scenario sets, success thresholds, skill name, and optional promotion check specs are
        profile-specific; per-phase `max_episodes` are derived by splitting `total_episodes` across phases and `min_episodes`
        is set to half of the phase budget (floor) with a minimum of 1 when the phase budget is greater than zero.
        Raises ValueError if `curriculum_profile` is not one of the supported names.
        
        Parameters:
            curriculum_profile (str): Curriculum identifier ("none", "ecological_v1", or "ecological_v2").
            total_episodes (int): Total training episodes to distribute across curriculum phases.
        
        Returns:
            list[CurriculumPhaseDefinition]: Ordered phase definitions populated with `name`, `skill_name`,
            `training_scenarios`, `promotion_scenarios`, `success_threshold`, `max_episodes`, `min_episodes`,
            and `promotion_check_specs` where applicable.
        """
        profile_name = _validate_curriculum_profile(curriculum_profile)
        if profile_name == "none":
            return []
        budgets = cls._resolve_curriculum_phase_budgets(total_episodes)

        if profile_name == "ecological_v1":
            phase_specs = (
                (
                    "phase_1_night_rest_predator_edge",
                    "predator_response",
                    ("night_rest", "predator_edge"),
                    ("night_rest", "predator_edge"),
                    1.0,
                    (),
                ),
                (
                    "phase_2_entrance_ambush_shelter_blockade",
                    "shelter_exit",
                    ("entrance_ambush", "shelter_blockade"),
                    ("entrance_ambush", "shelter_blockade"),
                    1.0,
                    (),
                ),
                (
                    "phase_3_open_field_exposed_day",
                    "food_approach",
                    ("open_field_foraging", "exposed_day_foraging"),
                    ("open_field_foraging", "exposed_day_foraging"),
                    0.5,
                    (),
                ),
                (
                    "phase_4_corridor_food_deprivation",
                    "corridor_gauntlet+food_deprivation",
                    ("corridor_gauntlet", "food_deprivation"),
                    ("corridor_gauntlet", "food_deprivation"),
                    0.5,
                    (),
                ),
            )
        else:
            phase_specs = (
                (
                    "phase_1_shelter_safety_predator_awareness",
                    "predator_response",
                    ("night_rest", "predator_edge", "entrance_ambush"),
                    ("predator_edge",),
                    1.0,
                    tuple(SUBSKILL_CHECK_MAPPINGS["predator_response"]),
                ),
                (
                    "phase_2_shelter_exit_commitment",
                    "shelter_exit",
                    ("entrance_ambush", "shelter_blockade", "food_deprivation"),
                    ("food_deprivation",),
                    1.0,
                    tuple(SUBSKILL_CHECK_MAPPINGS["shelter_exit"]),
                ),
                (
                    "phase_3_food_approach_under_exposure",
                    "food_approach",
                    (
                        "food_deprivation",
                        "open_field_foraging",
                        "exposed_day_foraging",
                    ),
                    (
                        "food_deprivation",
                        "open_field_foraging",
                        "exposed_day_foraging",
                    ),
                    1.0,
                    tuple(SUBSKILL_CHECK_MAPPINGS["food_approach"]),
                ),
                (
                    "phase_4_corridor_navigation_hunger_survival",
                    "corridor_navigation+hunger_commitment",
                    ("corridor_gauntlet", "food_deprivation"),
                    ("corridor_gauntlet", "food_deprivation"),
                    1.0,
                    (
                        *SUBSKILL_CHECK_MAPPINGS["corridor_navigation"],
                        *SUBSKILL_CHECK_MAPPINGS["hunger_commitment"],
                    ),
                ),
            )

        phases: list[CurriculumPhaseDefinition] = []
        for (
            budget,
            (
                name,
                skill_name,
                training_scenarios,
                promotion_scenarios,
                threshold,
                promotion_check_specs,
            ),
        ) in zip(
            budgets, phase_specs, strict=True
        ):
            phases.append(
                CurriculumPhaseDefinition(
                    name=name,
                    training_scenarios=tuple(training_scenarios),
                    promotion_scenarios=tuple(promotion_scenarios),
                    success_threshold=float(threshold),
                    max_episodes=int(budget),
                    min_episodes=max(1, int(budget) // 2) if int(budget) > 0 else 0,
                    skill_name=str(skill_name),
                    promotion_check_specs=tuple(promotion_check_specs),
                )
            )
        return phases

    @staticmethod
    def _promotion_check_spec_records(
        specs: Sequence[PromotionCheckCriteria],
    ) -> list[Dict[str, object]]:
        """
        Serialize promotion check criteria into plain dict records for inclusion in curriculum summaries.
        
        Parameters:
            specs (Sequence[PromotionCheckCriteria]): Sequence of promotion check criteria to serialize.
        
        Returns:
            list[dict]: A list of records, each containing the keys `"scenario"`, `"check_name"`, `"required_pass_rate"`, and `"aggregation"`.
        """
        return [
            {
                "scenario": str(spec.scenario),
                "check_name": str(spec.check_name),
                "required_pass_rate": float(spec.required_pass_rate),
                "aggregation": str(spec.aggregation),
            }
            for spec in specs
        ]

    @staticmethod
    def _evaluate_promotion_check_specs(
        payload: Dict[str, object],
        specs: Sequence[PromotionCheckCriteria],
    ) -> tuple[Dict[str, Dict[str, Dict[str, object]]], bool, str]:
        """
        Evaluate promotion check criteria against a behavior-suite payload.

        The `aggregation` field controls how individual check results are
        combined: "all" requires every check to pass, while "any" accepts at
        least one passing check. Missing or malformed pass rates count as 0.0.
        """
        if not specs:
            return {}, False, "no_checks_specified"
        suite = payload.get("suite", {})
        if not isinstance(suite, dict):
            suite = {}
        results: Dict[str, Dict[str, Dict[str, object]]] = {}
        first_failed_check = ""
        aggregations = {str(spec.aggregation).lower() for spec in specs}
        invalid_aggregations = aggregations - {"all", "any"}
        if invalid_aggregations:
            invalid = sorted(invalid_aggregations)[0]
            raise ValueError(
                "Invalid promotion check aggregation. Use 'all' or 'any': "
                f"{invalid!r}."
            )
        if len(aggregations) > 1:
            raise ValueError(
                "Mixed promotion check aggregations are not supported within "
                "one phase."
            )
        aggregation = next(iter(aggregations), "all")
        seen_specs: set[tuple[str, str]] = set()
        for spec in specs:
            scenario_name = str(spec.scenario)
            check_name = str(spec.check_name)
            spec_key = (scenario_name, check_name)
            if spec_key in seen_specs:
                raise ValueError(
                    "Duplicate promotion check spec for scenario "
                    f"{scenario_name!r} check {check_name!r}."
                )
            seen_specs.add(spec_key)
            try:
                required = float(spec.required_pass_rate)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid promotion check required_pass_rate for scenario "
                    f"{scenario_name!r} check {check_name!r}: "
                    f"{spec.required_pass_rate!r}."
                ) from exc
            if not 0.0 <= required <= 1.0:
                raise ValueError(
                    "Promotion check required_pass_rate must be between 0.0 "
                    f"and 1.0 for scenario {scenario_name!r} check "
                    f"{check_name!r}: {required!r}."
                )
            pass_rate = 0.0
            scenario_data = suite.get(scenario_name)
            checks = (
                scenario_data.get("checks", {})
                if isinstance(scenario_data, dict)
                else {}
            )
            check_data = checks.get(check_name) if isinstance(checks, dict) else None
            if isinstance(check_data, dict) and "pass_rate" in check_data:
                try:
                    pass_rate = float(check_data.get("pass_rate", 0.0))
                except (TypeError, ValueError):
                    pass_rate = 0.0
            passed = bool(pass_rate >= required)
            if not passed and not first_failed_check:
                first_failed_check = check_name
            scenario_results = results.setdefault(scenario_name, {})
            scenario_results[check_name] = {
                "scenario": scenario_name,
                "pass_rate": pass_rate,
                "required": required,
                "passed": passed,
            }
        passed_values = [
            bool(result["passed"])
            for scenario_results in results.values()
            for result in scenario_results.values()
        ]
        if aggregation == "any":
            promotion_passed = any(passed_values)
        else:
            promotion_passed = bool(passed_values) and all(passed_values)
        if promotion_passed:
            reason = (
                "any_check_passed"
                if aggregation == "any"
                else "all_checks_passed"
            )
        else:
            reason = f"check_failed:{first_failed_check}"
        return results, promotion_passed, reason

    @staticmethod
    def _empty_curriculum_summary(
        curriculum_profile: str,
        total_episodes: int,
    ) -> Dict[str, object]:
        """Build an empty curriculum summary for metadata seeding and zero-budget runs."""
        resolved_episodes = max(0, int(total_episodes))
        return {
            "profile": str(curriculum_profile),
            "total_training_episodes": resolved_episodes,
            "executed_training_episodes": 0,
            "status": "not_started" if resolved_episodes > 0 else "no_training_budget",
            "phases": [],
        }

    @staticmethod
    def _regime_row_metadata_from_summary(
        training_regime: Dict[str, object],
        curriculum_summary: Dict[str, object] | None,
    ) -> Dict[str, object]:
        """
        Build a flat metadata mapping of the current training regime and the most recently executed curriculum phase for CSV export.
        
        Parameters:
            training_regime (dict): Regime summary containing at least `mode` and optionally `name` and `curriculum_profile`.
            curriculum_summary (dict | None): Curriculum summary produced during training, or `None` when no curriculum was used.
        
        Returns:
            dict: Mapping with keys used for CSV rows:
                - `training_regime`: regime mode (e.g., `"flat"` or `"curriculum"`).
                - `training_regime_name`: regime name or `"baseline"`.
                - `curriculum_profile`: curriculum profile name or `"none"`.
                - `curriculum_phase`: most recently executed phase name or empty string.
                - `curriculum_skill`: skill name for the most recently executed phase or empty string.
                - `curriculum_phase_status`: status of the most recently executed phase or empty string.
                - `curriculum_promotion_reason`: promotion reason recorded for the most recently executed phase or empty string.
        """
        latest_phase: Dict[str, object] | None = None
        if isinstance(curriculum_summary, dict):
            try:
                executed_training_episodes = int(
                    curriculum_summary.get("executed_training_episodes", 0)
                )
            except (TypeError, ValueError):
                executed_training_episodes = 0
            if executed_training_episodes > 0:
                phases = curriculum_summary.get("phases", [])
                if isinstance(phases, list) and phases:
                    for phase in reversed(phases):
                        if not isinstance(phase, dict):
                            continue
                        if int(phase.get("episodes_executed", 0)) > 0:
                            latest_phase = phase
                            break

        def _phase_str(key: str) -> str:
            return str(latest_phase.get(key, "")) if latest_phase is not None else ""

        return {
            "training_regime": str(training_regime.get("mode", "flat")),
            "training_regime_name": str(training_regime.get("name", "baseline")),
            "curriculum_profile": str(
                training_regime.get("curriculum_profile", "none")
            ),
            "curriculum_phase": _phase_str("name"),
            "curriculum_skill": _phase_str("skill_name"),
            "curriculum_phase_status": _phase_str("status"),
            "curriculum_promotion_reason": _phase_str("promotion_reason"),
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
        profile_name = _validate_curriculum_profile(curriculum_profile)
        phase_budgets: list[int] = []
        if profile_name != "none":
            phase_budgets = self._resolve_curriculum_phase_budgets(episodes)
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
                    item["observation"] = self._jsonify_observation(observation)
                    item["next_observation"] = self._jsonify_observation(next_observation)
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

    @staticmethod
    def _jsonify(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {
                str(key): SpiderSimulation._jsonify(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [SpiderSimulation._jsonify(item) for item in value]
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _noise_profile_metadata(noise_profile: NoiseConfig) -> Dict[str, object]:
        """Return aggregate-safe metadata for a resolved noise profile."""
        return {
            "noise_profile": noise_profile.name,
            "noise_profile_config": noise_profile.to_summary(),
        }

    @staticmethod
    def _noise_profile_csv_value(noise_profile: NoiseConfig) -> str:
        """Serialize a resolved noise profile as stable JSON for CSV exports."""
        return json.dumps(
            noise_profile.to_summary(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _with_noise_profile_metadata(
        cls,
        payload: Dict[str, object],
        noise_profile: NoiseConfig,
    ) -> Dict[str, object]:
        """
        Merge noise-profile metadata into a payload.
        
        Parameters:
            payload: Base payload to augment.
            noise_profile: Resolved noise profile to attach.
        
        Returns:
            A shallow copy with noise-profile metadata.
        """
        enriched = dict(payload)
        enriched.update(cls._noise_profile_metadata(noise_profile))
        return enriched

    @staticmethod
    def _robustness_matrix_metadata(
        robustness_matrix: RobustnessMatrixSpec,
    ) -> Dict[str, object]:
        """
        Build serializable robustness-matrix metadata.
        
        Parameters:
            robustness_matrix: Matrix specification to summarize.
        
        Returns:
            A mapping containing the `"matrix_spec"` summary.
        """
        return {
            "matrix_spec": robustness_matrix.to_summary(),
        }

    @staticmethod
    def _matrix_cell_success_rate(payload: Dict[str, object] | None) -> float:
        """
        Extracts the scenario success rate from a compact matrix-cell payload.
        
        Parameters:
            payload (dict | None): Compact matrix-cell payload (or None) produced by robustness matrix evaluations.
        
        Returns:
            float: The `scenario_success_rate` value from the payload, or `0.0` if the payload is None or the field is missing.
        """
        return SpiderSimulation._condition_compact_summary(payload).get(
            "scenario_success_rate",
            0.0,
        )

    @classmethod
    def _robustness_aggregate_metrics(
        cls,
        matrix_payloads: Dict[str, Dict[str, object]],
        *,
        robustness_matrix: RobustnessMatrixSpec,
    ) -> Dict[str, object]:
        """
        Compute aggregate scores for a train x eval noise matrix.
        
        Parameters:
            matrix_payloads: Nested train->eval matrix-cell payloads.
            robustness_matrix: Matrix axes and iteration order.
        
        Returns:
            Train/eval marginals and overall, diagonal, and off-diagonal scores.
        """
        train_marginals: Dict[str, float] = {}
        eval_marginals: Dict[str, float] = {}
        all_scores: List[float] = []
        diagonal_scores: List[float] = []
        off_diagonal_scores: List[float] = []

        for train_condition in robustness_matrix.train_conditions:
            train_scores = [
                cls._matrix_cell_success_rate(
                    matrix_payloads.get(train_condition, {}).get(eval_condition)
                )
                for eval_condition in robustness_matrix.eval_conditions
            ]
            train_marginals[train_condition] = cls._safe_float(
                sum(train_scores) / len(train_scores) if train_scores else 0.0
            )

        for eval_condition in robustness_matrix.eval_conditions:
            eval_scores = [
                cls._matrix_cell_success_rate(
                    matrix_payloads.get(train_condition, {}).get(eval_condition)
                )
                for train_condition in robustness_matrix.train_conditions
            ]
            eval_marginals[eval_condition] = cls._safe_float(
                sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
            )

        for train_condition, eval_condition in robustness_matrix.cells():
            score = cls._matrix_cell_success_rate(
                matrix_payloads.get(train_condition, {}).get(eval_condition)
            )
            all_scores.append(score)
            if train_condition == eval_condition:
                diagonal_scores.append(score)
            else:
                off_diagonal_scores.append(score)

        return {
            "train_marginals": train_marginals,
            "eval_marginals": eval_marginals,
            "robustness_score": cls._safe_float(
                sum(all_scores) / len(all_scores) if all_scores else 0.0
            ),
            "diagonal_score": cls._safe_float(
                sum(diagonal_scores) / len(diagonal_scores)
                if diagonal_scores
                else 0.0
            ),
            "off_diagonal_score": cls._safe_float(
                sum(off_diagonal_scores) / len(off_diagonal_scores)
                if off_diagonal_scores
                else 0.0
            ),
        }

    @staticmethod
    def _resolve_checkpoint_load_dir(
        checkpoint_dir: str | Path | None,
        *,
        checkpoint_selection: str,
    ) -> Path | None:
        """
        Selects an existing checkpoint directory under the given run root according to the requested selection preference.
        
        Parameters:
            checkpoint_dir (str | Path | None): Root path containing checkpoint subdirectories; if `None`, no lookup is performed.
            checkpoint_selection (str): Selection preference, either `"best"` to prefer the `best/` candidate (falling back to `last/` then root) or `"none"` to disable loading entirely. Public entry points validate this value before calling the helper.
        
        Returns:
            Path | None: Path to the first valid checkpoint directory that contains a `metadata.json` file, or `None` if no valid candidate exists or `checkpoint_dir` is `None`.
        """
        if checkpoint_dir is None or checkpoint_selection == "none":
            return None
        root = Path(checkpoint_dir)
        candidate_dirs = (
            [root / "best", root / "last", root]
            if checkpoint_selection == "best"
            else [root / "last", root / "best", root]
        )
        for candidate_dir in candidate_dirs:
            if candidate_dir.is_dir() and (candidate_dir / "metadata.json").exists():
                return candidate_dir
        return None

    @classmethod
    def _checkpoint_run_fingerprint(cls, payload: Dict[str, object]) -> str:
        """Return a short stable fingerprint for checkpoint-compatible run settings."""
        stable_payload = cls._jsonify(payload)
        serialized = json.dumps(
            stable_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Return the SHA-256 digest for a file."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _checkpoint_preload_fingerprint(
        cls,
        load_brain: str | Path | None,
        load_modules: Sequence[str] | None = None,
    ) -> Dict[str, object]:
        """Return content identifiers for a preloaded brain artifact."""
        if load_brain is None:
            return {
                "load_brain": None,
                "metadata_sha256": None,
                "load_modules": None,
                "module_sha256": None,
            }

        root = Path(load_brain)
        if root.is_file():
            normalized_modules = (
                sorted({str(module_name) for module_name in load_modules})
                if load_modules is not None
                else None
            )
            return {
                "load_brain": str(root),
                "artifact_sha256": cls._file_sha256(root),
                "load_modules": normalized_modules,
                "module_sha256": None,
            }

        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_modules = metadata.get("modules", {})
        if not isinstance(saved_modules, dict):
            saved_modules = {}
        normalized_modules = (
            sorted({str(module_name) for module_name in load_modules})
            if load_modules is not None
            else sorted(str(module_name) for module_name in saved_modules)
        )
        module_sha256 = {
            module_name: cls._file_sha256(root / f"{module_name}.npz")
            for module_name in normalized_modules
        }
        return {
            "load_brain": str(root),
            "metadata_sha256": cls._file_sha256(metadata_path),
            "load_modules": normalized_modules,
            "module_sha256": module_sha256,
        }

    @classmethod
    def _jsonify_observation(cls, observation: Dict[str, object]) -> Dict[str, object]:
        """
        Convert all values in an observation mapping to JSON-safe representations.
        
        Parameters:
            observation (Dict[str, object]): Mapping of observation keys to values to be converted.
        
        Returns:
            Dict[str, object]: New mapping with the same keys where each value has been converted into a JSON-safe form (primitives, lists, dicts, or strings).
        """
        return {
            key: cls._jsonify(value)
            for key, value in observation.items()
        }

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

    @staticmethod
    def _mean_reward_from_behavior_payload(payload: Dict[str, object]) -> float:
        """
        Compute the average `mean_reward` across entries in a behavior payload's `legacy_scenarios`.
        
        Parameters:
            payload (Dict[str, object]): Behavior payload expected to contain a `"legacy_scenarios"` mapping of
                scenario names to dictionaries that may include a numeric `"mean_reward"` field.
        
        Returns:
            float: The arithmetic mean of all present `mean_reward` values converted to float, or `0.0` if no valid values are found.
        """
        legacy = payload.get("legacy_scenarios", {})
        if not isinstance(legacy, dict) or not legacy:
            return 0.0
        values = [
            float(data.get("mean_reward", 0.0))
            for data in legacy.values()
            if isinstance(data, dict)
        ]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

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
        summary["mean_reward"] = self._mean_reward_from_behavior_payload(payload)
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

    @staticmethod
    def _checkpoint_candidate_sort_key(
        candidate: Dict[str, object],
        *,
        primary_metric: str | None = None,
        selection_config: CheckpointSelectionConfig | None = None,
    ) -> tuple[float, ...]:
        """
        Builds a sort key for ranking checkpoint candidates according to the configured primary metric and optional reflex-penalty mode.
        
        If `selection_config.penalty_mode` is `TIEBREAKER`, the returned key is the legacy six-tuple:
        (primary_metric, secondary_metric, tertiary_metric, -mean_final_reflex_override_rate, -mean_reflex_dominance, episode).
        If `DIRECT`, a penalized composite score (primary minus weighted penalties) is prepended to that tuple.
        
        Parameters:
            candidate (Dict[str, object]): Candidate metadata; expected to contain metric fields named in CHECKPOINT_METRIC_ORDER and an optional
                `"evaluation_summary"` dict with `"mean_final_reflex_override_rate"` and `"mean_reflex_dominance"`, and an `"episode"` index.
            primary_metric (str | None): Backward-compatible primary metric name used when `selection_config` is not provided.
            selection_config (CheckpointSelectionConfig | None): When provided, controls the primary metric, penalty weights, and penalty mode.
        
        Returns:
            tuple[float, ...]: A numeric sort key suitable for ordering candidates. Numeric fields default to 0.0 and episode defaults to 0.
                - TIEBREAKER mode: (primary, secondary, tertiary, -override_rate, -dominance, episode)
                - DIRECT mode: (composite_score, primary, secondary, tertiary, -override_rate, -dominance, episode)
        
        Raises:
            ValueError: If neither `primary_metric` nor `selection_config` supplies a primary metric, or if `primary_metric` conflicts with
                `selection_config.metric`.
        """
        if selection_config is None:
            if primary_metric is None:
                raise ValueError(
                    "checkpoint selection requires a primary metric or config."
                )
            selection_config = CheckpointSelectionConfig(metric=primary_metric)
        elif primary_metric is not None and primary_metric != selection_config.metric:
            raise ValueError(
                "primary_metric must match selection_config.metric when both are provided."
            )
        ordered_metrics = [selection_config.metric] + [
            metric_name
            for metric_name in CHECKPOINT_METRIC_ORDER
            if metric_name != selection_config.metric
        ]
        evaluation_summary = candidate.get("evaluation_summary", {})
        if not isinstance(evaluation_summary, dict):
            evaluation_summary = {}
        override_rate = float(
            evaluation_summary.get("mean_final_reflex_override_rate", 0.0)
        )
        dominance = float(evaluation_summary.get("mean_reflex_dominance", 0.0))
        legacy_key = (
            float(candidate.get(ordered_metrics[0], 0.0)),
            float(candidate.get(ordered_metrics[1], 0.0)),
            float(candidate.get(ordered_metrics[2], 0.0)),
            -override_rate,
            -dominance,
            int(candidate.get("episode", 0)),
        )
        if selection_config.penalty_mode is CheckpointPenaltyMode.TIEBREAKER:
            return legacy_key
        composite_score = (
            legacy_key[0]
            - float(selection_config.override_penalty_weight) * override_rate
            - float(selection_config.dominance_penalty_weight) * dominance
        )
        return (float(composite_score), *legacy_key)

    @staticmethod
    def _checkpoint_candidate_composite_score(
        candidate: Dict[str, object],
        selection_config: CheckpointSelectionConfig,
    ) -> float:
        """
        Compute the composite selection score for a checkpoint candidate using direct penalty mode.
        
        Parameters:
            candidate (dict): Checkpoint metadata containing evaluation metrics and diagnostic fields used for ranking.
            selection_config (CheckpointSelectionConfig): Selection parameters whose `metric`, `override_penalty_weight`, and `dominance_penalty_weight` determine the penalized composite score.
        
        Returns:
            float: The composite score (higher is better) produced by applying the configured penalties to the primary metric.
        """
        direct_config = CheckpointSelectionConfig(
            metric=selection_config.metric,
            override_penalty_weight=selection_config.override_penalty_weight,
            dominance_penalty_weight=selection_config.dominance_penalty_weight,
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )
        return float(
            SpiderSimulation._checkpoint_candidate_sort_key(
                candidate,
                selection_config=direct_config,
            )[0]
        )

    @staticmethod
    def _persist_checkpoint_pair(
        *,
        checkpoint_dir: str | Path | None,
        best_candidate: Dict[str, object],
        last_candidate: Dict[str, object],
    ) -> Dict[str, str]:
        """
        Persist the selected checkpoint candidate directories into a checkpoint root as 'best' and 'last'.
        
        Parameters:
            checkpoint_dir (str | Path | None): Destination root for persisted checkpoints. If `None`, no action is taken and an empty dict is returned.
            best_candidate (Dict[str, object]): Candidate metadata for the best checkpoint; must include a `"path"` key pointing to the source directory to copy.
            last_candidate (Dict[str, object]): Candidate metadata for the last checkpoint; must include a `"path"` key pointing to the source directory to copy.
        
        Returns:
            Dict[str, str]: Mapping of labels to persisted directory paths, e.g. `{"best": "<path>", "last": "<path>"}`. Returns an empty dict if `checkpoint_dir` is `None`.
        
        Notes:
            Existing target directories named "best" or "last" under `checkpoint_dir` will be removed before copying.
        """
        if checkpoint_dir is None:
            return {}
        destination_root = Path(checkpoint_dir)
        destination_root.mkdir(parents=True, exist_ok=True)
        persisted: Dict[str, str] = {}
        for label, candidate in (("best", best_candidate), ("last", last_candidate)):
            source = Path(candidate["path"])
            target = destination_root / label
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            persisted[label] = str(target)
        return persisted

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
        profile_name = _validate_curriculum_profile(curriculum_profile)

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
        curriculum_summary = self._empty_curriculum_summary(
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
                "promotion_check_specs": self._promotion_check_spec_records(
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
                    "mean_reward": self._mean_reward_from_behavior_payload(
                        microeval_payload
                    ),
                }
                if phase.promotion_check_specs:
                    check_results, promotion_passed, promotion_reason = (
                        self._evaluate_promotion_check_specs(
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
                key=lambda item: self._checkpoint_candidate_sort_key(
                    item,
                    selection_config=selection_config,
                ),
            )
            last_candidate = max(candidates, key=lambda item: int(item.get("episode", 0)))
            self.brain.load(best_candidate["path"])
            persisted = self._persist_checkpoint_pair(
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
                        self._checkpoint_candidate_composite_score(
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
        curriculum_profile = _validate_curriculum_profile(curriculum_profile)
        if checkpoint_selection == "best":
            checkpoint_selection_config = CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            )
            # Intentionally call _checkpoint_candidate_sort_key() with an empty
            # candidate to validate checkpoint_metric early; invalid metrics
            # raise ValueError here before any training or checkpoint work runs.
            self._checkpoint_candidate_sort_key(
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
                - "legacy_scenarios": dict mapping each scenario name to its legacy aggregated episode metrics (suitable for compaction via _compact_aggregate).
        """
        suite: Dict[str, object] = {}
        legacy_scenarios: Dict[str, object] = {}
        for name, score_group in behavior_histories.items():
            scenario = get_scenario(name)
            legacy_scenarios[name] = self._aggregate_group(stats_histories.get(name, []))
            suite[name] = aggregate_behavior_scores(
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
            "reward_audit": self._build_reward_audit(
                current_profile=self.world.reward_profile
            ),
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
                competence_type, is_primary_benchmark, plus training-regime/curriculum metadata
                and any keys from `extra_metadata`.
        """
        annotated: List[Dict[str, object]] = []
        architecture_fingerprint = self.brain._architecture_fingerprint()
        eval_noise_profile = resolve_noise_profile(self.world.noise_profile)
        eval_noise_profile_config = self._noise_profile_csv_value(eval_noise_profile)
        resolved_train_noise_profile = (
            resolve_noise_profile(train_noise_profile)
            if train_noise_profile is not None
            else None
        )
        train_noise_profile_config = (
            self._noise_profile_csv_value(resolved_train_noise_profile)
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
            item.update(
                self._regime_row_metadata_from_summary(
                    self._latest_training_regime_summary,
                    self._latest_curriculum_summary,
                )
            )
            if extra_metadata:
                item.update(extra_metadata)
            annotated.append(item)
        return annotated

    @classmethod
    def _profile_comparison_metrics(
        cls,
        payload: Dict[str, object] | None,
    ) -> Dict[str, float]:
        """
        Extract three numeric comparison metrics from a behavior-suite or aggregate comparison payload.
        
        Parameters:
            payload (dict | None): A behavior-suite payload (with a top-level "summary") or an aggregate comparison payload; may be None or malformed.
        
        Returns:
            dict: A mapping with keys:
                - "scenario_success_rate": float, success rate aggregated per scenario.
                - "episode_success_rate": float, success rate aggregated per episode.
                - "mean_reward": float, mean reward (computed from payload summary or legacy scenario entries).
        """
        if not isinstance(payload, dict):
            return {
                "scenario_success_rate": 0.0,
                "episode_success_rate": 0.0,
                "mean_reward": 0.0,
            }
        summary = payload.get("summary")
        if isinstance(summary, dict):
            return {
                "scenario_success_rate": cls._safe_float(
                    summary.get("scenario_success_rate")
                ),
                "episode_success_rate": cls._safe_float(
                    summary.get("episode_success_rate")
                ),
                "mean_reward": cls._condition_mean_reward(payload),
            }
        return {
            "scenario_success_rate": cls._safe_float(
                payload.get("scenario_success_rate")
            ),
            "episode_success_rate": cls._safe_float(
                payload.get("episode_success_rate")
            ),
            "mean_reward": cls._safe_float(payload.get("mean_reward")),
        }

    @classmethod
    def _build_reward_audit_comparison(
        cls,
        comparison_payload: Dict[str, object] | None,
    ) -> Dict[str, object] | None:
        """
        Summarize reward-profile metrics and compute per-profile deltas relative to the "austere" baseline when available.
        
        Parameters:
            comparison_payload (dict | None): A payload containing a "reward_profiles" mapping from profile
                name to per-profile payloads. Each per-profile payload may include a "suite" mapping of
                per-scenario aggregates and summary/metrics used for comparison.
        
        Returns:
            dict | None: A compact comparison dictionary, or `None` if input is missing or invalid. When
            returned the dict contains:
              - "minimal_profile": name of the austere baseline profile if present, else `None`.
              - "profiles": mapping profile_name -> metrics (`scenario_success_rate`,
                `episode_success_rate`, `mean_reward`) with values rounded to 6 decimals.
              - "deltas_vs_minimal": mapping profile_name -> deltas vs the minimal profile for the same
                three metrics (rounded to 6 decimals). Empty if no austere baseline is available.
              - "behavior_survival": availability flag, minimal_profile name, configured survival threshold,
                per-scenario austere success rates, per-scenario episode counts, and per-scenario boolean
                `survives` indicating whether austere success_rate >= threshold.
              - "survival_rate": overall fraction of scenarios that survive under the austere profile
                (rounded to 6 decimals).
              - "notes": brief notes describing how metrics and survival were derived.
        """
        if not isinstance(comparison_payload, dict):
            return None
        profile_payloads = comparison_payload.get("reward_profiles")
        if not isinstance(profile_payloads, dict) or not profile_payloads:
            return None
        profiles = {
            name: cls._profile_comparison_metrics(dict(payload))
            for name, payload in profile_payloads.items()
            if isinstance(payload, dict)
        }
        if not profiles:
            return None
        minimal_profile = "austere" if "austere" in profiles else None
        deltas_vs_minimal: Dict[str, object] = {}
        if minimal_profile is not None:
            minimal_metrics = profiles[minimal_profile]
            for name, metrics in profiles.items():
                deltas_vs_minimal[name] = {
                    "scenario_success_rate_delta": round(
                        float(metrics["scenario_success_rate"])
                        - float(minimal_metrics["scenario_success_rate"]),
                        6,
                    ),
                    "episode_success_rate_delta": round(
                        float(metrics["episode_success_rate"])
                        - float(minimal_metrics["episode_success_rate"]),
                        6,
                    ),
                    "mean_reward_delta": round(
                        float(metrics["mean_reward"])
                        - float(minimal_metrics["mean_reward"]),
                        6,
                    ),
                }
        behavior_survival: Dict[str, object] = {
            "available": False,
            "minimal_profile": minimal_profile,
            "survival_threshold": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
            "scenario_count": 0,
            "surviving_scenario_count": 0,
            "survival_rate": 0.0,
            "scenarios": {},
        }
        if minimal_profile is not None:
            minimal_payload = profile_payloads.get(minimal_profile, {})
            minimal_suite = (
                minimal_payload.get("suite", {})
                if isinstance(minimal_payload, dict)
                else {}
            )
            if isinstance(minimal_suite, dict) and minimal_suite:
                scenario_names = sorted(
                    str(scenario_name)
                    for scenario_name in minimal_suite.keys()
                )
                scenario_payloads: Dict[str, object] = {}
                for scenario_name in scenario_names:
                    minimal_scenario = minimal_suite.get(scenario_name, {})
                    if not isinstance(minimal_scenario, dict):
                        minimal_scenario = {}
                    austere_success_rate = cls._safe_float(
                        minimal_scenario.get("success_rate")
                    )
                    scenario_payloads[scenario_name] = {
                        "austere_success_rate": round(austere_success_rate, 6),
                        "survives": austere_success_rate >= MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                        "episodes": int(
                            cls._safe_float(minimal_scenario.get("episodes"))
                        ),
                    }
                surviving_count = sum(
                    1
                    for payload in scenario_payloads.values()
                    if payload.get("survives")
                )
                scenario_count = len(scenario_payloads)
                behavior_survival = {
                    "available": scenario_count > 0,
                    "minimal_profile": minimal_profile,
                    "survival_threshold": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                    "scenario_count": scenario_count,
                    "surviving_scenario_count": surviving_count,
                    "survival_rate": round(
                        float(surviving_count / scenario_count)
                        if scenario_count
                        else 0.0,
                        6,
                    ),
                    "scenarios": scenario_payloads,
                }
        return {
            "minimal_profile": minimal_profile,
            "profiles": {
                name: {
                    key: round(float(value), 6)
                    for key, value in sorted(metrics.items())
                }
                for name, metrics in sorted(profiles.items())
            },
            "deltas_vs_minimal": deltas_vs_minimal,
            "behavior_survival": behavior_survival,
            "survival_rate": behavior_survival["survival_rate"],
            "notes": [
                "scenario_success_rate and episode_success_rate mirror the existing comparison payload.",
                "mean_reward is derived from the summary when available or from the corresponding compact aggregate.",
                "behavior_survival treats a scenario as surviving minimal shaping when austere success_rate reaches the configured survival threshold.",
            ],
        }

    @classmethod
    def _build_reward_audit(
        cls,
        *,
        current_profile: str | None = None,
        comparison_payload: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        """
        Build a structured audit of reward shaping and leakage signals for the simulation.
        
        This produces a dictionary summarizing the current reward profile, a minimal baseline
        marker (when available), per-component reward audits, observation- and memory-leakage
        signals, per-profile audits for every known reward profile, and human-readable notes.
        If a comparison payload is supplied, a `comparison` block with computed deltas is
        included.
        
        Parameters:
            current_profile (str | None): Optional name of the currently selected reward profile
                to record in the audit. May be None.
            comparison_payload (dict | None): Optional payload used to compute comparison deltas.
                When provided, a `comparison` entry will be added to the returned audit.
        
        Returns:
            dict: Audit payload containing keys:
              - "current_profile": the provided `current_profile` value
              - "minimal_profile": name of an austere baseline when available, else None
              - "reward_components": per-component audit results
              - "observation_signals": observation-leakage audit results
              - "memory_signals": memory-leakage audit results
              - "reward_profiles": per-profile audit entries for all known profiles
              - "notes": explanatory notes
              - "comparison" (optional): computed comparison block when `comparison_payload` was provided
        """
        audit = {
            "current_profile": current_profile,
            "minimal_profile": None,
            "reward_components": reward_component_audit(),
            "observation_signals": observation_leakage_audit(),
            "memory_signals": memory_leakage_audit(),
            "reward_profiles": {
                name: reward_profile_audit(name)
                for name in sorted(REWARD_PROFILES)
            },
            "notes": [
                "The audit distinguishes event shaping, progress shaping, and internal pressure.",
                "Observation and memory signals classified as privileged/world_derived should be interpreted cautiously in benchmarks.",
            ],
        }
        comparison = cls._build_reward_audit_comparison(comparison_payload)
        if comparison is not None:
            audit["minimal_profile"] = comparison.get("minimal_profile")
            audit["comparison"] = comparison
        elif "austere" in REWARD_PROFILES:
            audit["minimal_profile"] = "austere"
        return audit

    @staticmethod
    def _safe_float(value: object) -> float:
        """
        Safely convert a value to float, returning 0.0 for invalid inputs.

        Parameters:
            value: Any value to attempt conversion to float.

        Returns:
            float: The converted float value, or 0.0 if conversion fails.
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _condition_compact_summary(
        payload: Dict[str, object] | None,
    ) -> Dict[str, float]:
        """
        Extract a compact numeric summary of scenario/episode success rates and mean reward from a behavior-suite payload.

        Parameters:
            payload (Dict[str, object] | None): A behavior-suite payload (expected to contain a top-level "summary" mapping) or None.

        Returns:
            Dict[str, float]: A mapping with keys `"scenario_success_rate"`, `"episode_success_rate"`, and `"mean_reward"`, each cast to float. Missing or malformed input yields zeros for all three fields.
        """
        if not isinstance(payload, dict):
            return {
                "scenario_success_rate": 0.0,
                "episode_success_rate": 0.0,
                "mean_reward": 0.0,
            }
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            return {
                "scenario_success_rate": 0.0,
                "episode_success_rate": 0.0,
                "mean_reward": 0.0,
            }
        return {
            "scenario_success_rate": SpiderSimulation._safe_float(
                summary.get("scenario_success_rate", 0.0)
            ),
            "episode_success_rate": SpiderSimulation._safe_float(
                summary.get("episode_success_rate", 0.0)
            ),
            "mean_reward": SpiderSimulation._condition_mean_reward(payload),
        }

    @staticmethod
    def _condition_mean_reward(payload: Dict[str, object] | None) -> float:
        """
        Resolve a condition payload's mean reward, preferring `summary["mean_reward"]`
        and falling back to the compact payload's `legacy_scenarios` aggregate.
        """
        if not isinstance(payload, dict):
            return 0.0
        summary = payload.get("summary", {})
        if isinstance(summary, dict):
            summary_mean_reward = summary.get("mean_reward")
            if summary_mean_reward is not None:
                try:
                    return float(summary_mean_reward)
                except (TypeError, ValueError):
                    pass
        try:
            return float(SpiderSimulation._mean_reward_from_behavior_payload(payload))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_learning_evidence_deltas(
        conditions: Dict[str, Dict[str, object]],
        *,
        reference_condition: str,
        scenario_names: Sequence[str],
    ) -> Dict[str, object]:
        """
        Compute numeric deltas for each learning-evidence condition relative to a reference condition.

        Calculates deltas for overall suite summary fields (`scenario_success_rate`, `episode_success_rate`, `mean_reward`)
        and per-scenario `success_rate`. All numeric deltas are (condition - reference) and rounded to 6 decimal places.
        Conditions marked as skipped (or missing a `summary`) are represented with `{"skipped": True, "reason": <str>}`.

        Parameters:
            conditions (Dict[str, Dict[str, object]]): Mapping from condition name to its evaluation payload. Each payload
                is expected to contain a `summary` dict with top-level metrics and a `suite` dict keyed by scenario name.
            reference_condition (str): Name of the condition to use as the reference baseline for delta computation.
            scenario_names (Sequence[str]): Ordered list of scenario names to include for per-scenario `success_rate` deltas.
        
        Returns:
            Dict[str, object]: Mapping from condition name to a delta payload. For non-skipped conditions the payload has:
                {
                    "summary": {
                        "scenario_success_rate_delta": float,
                        "episode_success_rate_delta": float,
                        "mean_reward_delta": float
                    },
                    "scenarios": {
                        "<scenario_name>": {"success_rate_delta": float}, ...
                    }
                }
            Skipped conditions return:
                {"skipped": True, "reason": <string>}
        """
        deltas: Dict[str, object] = {}
        reference = conditions.get(reference_condition)
        reference_missing_or_skipped = (
            not isinstance(reference, dict) or bool(reference.get("skipped"))
        )
        if reference_missing_or_skipped:
            for condition_name in conditions:
                deltas[condition_name] = {
                    "skipped": True,
                    "reason": (
                        f"reference condition {reference_condition!r} missing or skipped"
                    ),
                }
            return deltas
        reference_summary = dict(reference.get("summary", {}))
        reference_suite = dict(reference.get("suite", {}))
        reference_mean_reward = SpiderSimulation._condition_mean_reward(reference)
        for condition_name, payload in conditions.items():
            if bool(payload.get("skipped")) or "summary" not in payload:
                deltas[condition_name] = {
                    "skipped": True,
                    "reason": str(payload.get("reason", "")),
                }
                continue
            summary = dict(payload.get("summary", {}))
            suite = dict(payload.get("suite", {}))
            mean_reward = SpiderSimulation._condition_mean_reward(payload)
            deltas[condition_name] = {
                "summary": {
                    "scenario_success_rate_delta": round(
                        float(summary.get("scenario_success_rate", 0.0))
                        - float(reference_summary.get("scenario_success_rate", 0.0)),
                        6,
                    ),
                    "episode_success_rate_delta": round(
                        float(summary.get("episode_success_rate", 0.0))
                        - float(reference_summary.get("episode_success_rate", 0.0)),
                        6,
                    ),
                    "mean_reward_delta": round(
                        float(mean_reward) - float(reference_mean_reward),
                        6,
                    ),
                },
                "scenarios": {
                    scenario_name: {
                        "success_rate_delta": round(
                            float(
                                dict(suite.get(scenario_name, {})).get("success_rate", 0.0)
                            )
                            - float(
                                dict(reference_suite.get(scenario_name, {})).get(
                                    "success_rate",
                                    0.0,
                                )
                            ),
                            6,
                        )
                    }
                    for scenario_name in scenario_names
                },
            }
        return deltas

    @classmethod
    def _build_learning_evidence_summary(
        cls,
        conditions: Dict[str, Dict[str, object]],
        *,
        reference_condition: str,
    ) -> Dict[str, object]:
        """
        Build a compact summary comparing learning-evidence conditions against a reference condition.
        
        Parameters:
            conditions (Dict[str, Dict[str, object]]): Mapping from condition name to its compact payload; a payload may include a `"skipped"` flag.
            reference_condition (str): Name of the condition to use as the primary trained reference. For the canonical workflow this is `trained_without_reflex_support`.
        
        Returns:
            Dict[str, object]: Summary containing:
                - reference_condition: the provided reference name.
                - primary_gate_metric: the metric used for the primary evidence gate ("scenario_success_rate").
                - supports_primary_evidence: `true` if `reference_condition`, `random_init`, and `reflex_only` are present and not skipped.
                - has_learning_evidence: `true` if `scenario_success_rate` for the no-reflex reference exceeds both `random_init` and `reflex_only`.
                - trained_final, random_init, reflex_only, trained_without_reflex_support: compact summaries for each condition (zeroed defaults when missing).
                - trained_vs_random_init, trained_vs_reflex_only: per-metric deltas (`scenario_success_rate`, `episode_success_rate`, `mean_reward`) computed as no-reflex reference minus comparator and rounded to 6 decimals.
                - notes: list of explanatory messages about gating, supporting metrics, and reflex-only availability.
        """
        reference = cls._condition_compact_summary(conditions.get(reference_condition))
        trained_final = cls._condition_compact_summary(conditions.get("trained_final"))
        random_init = cls._condition_compact_summary(conditions.get("random_init"))
        reflex_only = cls._condition_compact_summary(conditions.get("reflex_only"))
        trained_without_reflex_support = cls._condition_compact_summary(
            conditions.get("trained_without_reflex_support")
        )
        reference_available = (
            reference_condition in conditions
            and not bool(conditions[reference_condition].get("skipped"))
        )
        random_init_available = (
            "random_init" in conditions
            and not bool(conditions["random_init"].get("skipped"))
        )
        reflex_only_available = (
            "reflex_only" in conditions
            and not bool(conditions["reflex_only"].get("skipped"))
        )
        primary_supported = (
            reference_available
            and random_init_available
            and reflex_only_available
        )
        trained_vs_random = {
            "scenario_success_rate_delta": round(
                reference["scenario_success_rate"]
                - random_init["scenario_success_rate"],
                6,
            ),
            "episode_success_rate_delta": round(
                reference["episode_success_rate"]
                - random_init["episode_success_rate"],
                6,
            ),
            "mean_reward_delta": round(
                reference["mean_reward"] - random_init["mean_reward"],
                6,
            ),
        }
        trained_vs_reflex = {
            "scenario_success_rate_delta": round(
                reference["scenario_success_rate"]
                - reflex_only["scenario_success_rate"],
                6,
            ),
            "episode_success_rate_delta": round(
                reference["episode_success_rate"]
                - reflex_only["episode_success_rate"],
                6,
            ),
            "mean_reward_delta": round(
                reference["mean_reward"] - reflex_only["mean_reward"],
                6,
            ),
        }
        notes = [
            f"Primary evidence uses {reference_condition} and only scenario_success_rate as the gate.",
            "episode_success_rate and mean_reward are included only as supporting documentation.",
            "trained_final is retained as a default-runtime diagnostic and does not drive the primary gate.",
        ]
        if not reflex_only_available:
            notes.append(
                "The reflex_only condition is not available for the current architecture."
            )
        has_learning_evidence = (
            primary_supported
            and reference["scenario_success_rate"] > random_init["scenario_success_rate"]
            and reference["scenario_success_rate"] > reflex_only["scenario_success_rate"]
        )
        return {
            "reference_condition": reference_condition,
            "primary_gate_metric": "scenario_success_rate",
            "supports_primary_evidence": bool(primary_supported),
            "has_learning_evidence": bool(has_learning_evidence),
            "primary_condition": reference,
            "trained_final": trained_final,
            "random_init": random_init,
            "reflex_only": reflex_only,
            "trained_without_reflex_support": trained_without_reflex_support,
            "trained_vs_random_init": trained_vs_random,
            "trained_vs_reflex_only": trained_vs_reflex,
            "notes": notes,
        }

    @staticmethod
    def _build_ablation_deltas(
        variants: Dict[str, Dict[str, object]],
        *,
        reference_variant: str,
        scenario_names: Sequence[str],
    ) -> Dict[str, object]:
        """
        Compute per-variant deltas of success rates against a reference ablation variant.
        
        Parameters:
            variants (Dict[str, Dict[str, object]]): Mapping from variant name to its behavior payload containing at least `"summary"` and `"suite"` entries.
            reference_variant (str): Name of the variant to use as the reference baseline.
            scenario_names (Sequence[str]): Sequence of scenario names for which per-scenario deltas will be computed.
        
        Returns:
            Dict[str, object]: Mapping from variant name to a dict with two keys:
                - "summary": contains
                    - "scenario_success_rate_delta": difference between the variant's overall scenario success rate and the reference's, rounded to 6 decimals.
                    - "episode_success_rate_delta": difference between the variant's episode success rate and the reference's, rounded to 6 decimals.
                - "scenarios": mapping of each scenario name to
                    - "success_rate_delta": difference between the variant's per-scenario success rate and the reference's, rounded to 6 decimals.
        """
        reference = variants[reference_variant]
        reference_summary = reference["summary"]
        reference_suite = reference["suite"]
        deltas: Dict[str, object] = {}
        for variant_name, payload in variants.items():
            summary = payload["summary"]
            suite = payload["suite"]
            deltas[variant_name] = {
                "summary": {
                    "scenario_success_rate_delta": round(
                        float(summary["scenario_success_rate"] - reference_summary["scenario_success_rate"]),
                        6,
                    ),
                    "episode_success_rate_delta": round(
                        float(summary["episode_success_rate"] - reference_summary["episode_success_rate"]),
                        6,
                    ),
                },
                "scenarios": {
                    scenario_name: {
                        "success_rate_delta": round(
                            float(suite[scenario_name]["success_rate"] - reference_suite[scenario_name]["success_rate"]),
                            6,
                        )
                    }
                    for scenario_name in scenario_names
                },
            }
        return deltas

    @classmethod
    def compare_configurations(
        cls,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile: str | BudgetProfile | None = None,
        reward_profiles: Sequence[str] | None = None,
        map_templates: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
    ) -> Dict[str, object]:
        """
        Compare agent performance across reward profiles and map templates by training and evaluating simulations and aggregating episode statistics.
        
        Runs independent simulation instances for each (reward_profile, map_template) pair across the provided seeds using a resolved budget (episodes/evaluation_episodes/max_steps). Each instance is trained and then evaluated; per-seed results are aggregated into compact episode-stat summaries.
        
        Parameters:
            episodes (int | None): Training episodes override used to resolve the runtime budget; if None the budget profile or defaults are used.
            evaluation_episodes (int | None): Evaluation episodes override used to resolve the runtime budget.
            operational_profile (str | OperationalProfile | None): Operational profile or its name applied to each simulation.
            noise_profile (str | NoiseConfig | None): Noise profile or its name applied to each simulation.
            reward_profiles (Sequence[str] | None): Iterable of reward profile names to compare; defaults to ("classic",) when None.
            map_templates (Sequence[str] | None): Iterable of map template names to compare; defaults to ("central_burrow",) when None.
            seeds (Sequence[int] | None): Random seeds used to instantiate independent simulation runs; if None the budget's comparison seeds are used.
            budget_profile (str | BudgetProfile | None): Budget profile or its name used to resolve episodes/eval/max_steps when overrides are not provided.
            width, height, food_count, day_length, night_length, max_steps, gamma, module_lr, motor_lr, module_dropout: World and agent hyperparameters used to construct each simulation instance.
        
        Returns:
            Dict[str, object]: A mapping containing:
              - "budget_profile": resolved budget profile name.
              - "benchmark_strength": resolved benchmark strength from the budget.
              - "seeds": list of seeds actually used.
              - "reward_profiles": mapping from reward profile name to compact aggregated episode statistics across all maps and seeds.
              - "map_templates": mapping from map template name to compact aggregated episode statistics across all profiles and seeds.
              - "matrix": nested mapping matrix[profile][map_name] containing compact aggregated episode statistics for each (profile, map) pair.
        """
        resolved_noise_profile = resolve_noise_profile(noise_profile)
        budget = resolve_budget(
            profile=budget_profile,
            episodes=episodes,
            eval_episodes=evaluation_episodes,
            max_steps=max_steps,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        profile_names = list(reward_profiles or ("classic",))
        map_names = list(map_templates or ("central_burrow",))
        seed_values = tuple(seeds) if seeds is not None else budget.comparison_seeds
        eval_runs = max(1, int(budget.eval_episodes))
        matrix: Dict[str, Dict[str, object]] = {}
        profile_histories: Dict[str, List[EpisodeStats]] = {
            profile: [] for profile in profile_names
        }
        map_histories: Dict[str, List[EpisodeStats]] = {
            map_name: [] for map_name in map_names
        }

        for profile in profile_names:
            matrix[profile] = {}
            for map_name in map_names:
                combined_history: List[EpisodeStats] = []
                for seed in seed_values:
                    sim = cls(
                        width=width,
                        height=height,
                        food_count=food_count,
                        day_length=day_length,
                        night_length=night_length,
                        max_steps=budget.max_steps,
                        seed=seed,
                        gamma=gamma,
                        module_lr=module_lr,
                        motor_lr=motor_lr,
                        module_dropout=module_dropout,
                        operational_profile=operational_profile,
                        noise_profile=resolved_noise_profile,
                        reward_profile=profile,
                        map_template=map_name,
                    )
                    _, evaluation_history, _ = sim._train_histories(
                        episodes=budget.episodes,
                        evaluation_episodes=eval_runs,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                    )
                    combined_history.extend(evaluation_history)
                matrix[profile][map_name] = cls._with_noise_profile_metadata(
                    cls._compact_aggregate(aggregate_episode_stats(combined_history)),
                    resolved_noise_profile,
                )
                profile_histories[profile].extend(combined_history)
                map_histories[map_name].extend(combined_history)

        return {
            "budget_profile": budget.profile,
            "benchmark_strength": budget.benchmark_strength,
            "seeds": list(seed_values),
            "reward_profiles": {
                profile: cls._with_noise_profile_metadata(
                    cls._compact_aggregate(aggregate_episode_stats(history)),
                    resolved_noise_profile,
                )
                for profile, history in profile_histories.items()
            },
            "map_templates": {
                map_name: cls._with_noise_profile_metadata(
                    cls._compact_aggregate(aggregate_episode_stats(history)),
                    resolved_noise_profile,
                )
                for map_name, history in map_histories.items()
            },
            "matrix": matrix,
            "reward_audit": cls._build_reward_audit(
                current_profile=profile_names[0] if profile_names else None,
                comparison_payload={"reward_profiles": {
                    profile: cls._with_noise_profile_metadata(
                        cls._compact_aggregate(aggregate_episode_stats(history)),
                        resolved_noise_profile,
                    )
                    for profile, history in profile_histories.items()
                }},
            ),
            **cls._noise_profile_metadata(resolved_noise_profile),
        }

    @classmethod
    def compare_behavior_suite(
        cls,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile: str | BudgetProfile | None = None,
        reward_profiles: Sequence[str] | None = None,
        map_templates: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        episodes_per_scenario: int | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Compare behavior-suite performance across reward profiles, map templates, and random seeds.
        
        Runs (optionally trains) simulations for each combination of reward profile, map template, and seed, executes the behavior suite per scenario, aggregates behavioral scores and legacy episode statistics into a compact payload matrix, and collects flattened per-episode behavior rows suitable for CSV export.
        
        Parameters:
            width (int): World width in grid cells.
            height (int): World height in grid cells.
            food_count (int): Number of food items placed in the world.
            day_length (int): Duration of daytime ticks.
            night_length (int): Duration of nighttime ticks.
            max_steps (int): Maximum steps per episode.
            episodes (int): Number of training episodes to run before behavior evaluation for each simulation.
            evaluation_episodes (int): Number of evaluation episodes (performed during training phase) to run.
            gamma (float): Discount factor for the agent's learning configuration.
            module_lr (float): Module learning rate passed to the agent.
            motor_lr (float): Motor learning rate passed to the agent.
            module_dropout (float): Module dropout probability used when constructing simulations.
            operational_profile (str | OperationalProfile | None): Registered operational profile name, explicit
                `OperationalProfile` instance, or `None` to use the default profile for each simulation run.
                Invalid profile names raise `ValueError` during resolution.
            reward_profiles (Sequence[str] | None): Iterable of reward profile names to evaluate; defaults to ("classic",) when None.
            map_templates (Sequence[str] | None): Iterable of map template names to evaluate; defaults to ("central_burrow",) when None.
            seeds (Sequence[int]): Sequence of RNG seeds to instantiate independent simulations.
            names (Sequence[str] | None): Sequence of scenario names to run; defaults to the global SCENARIO_NAMES when None.
            episodes_per_scenario (int): Number of runs to execute per scenario (controls how many episodes are aggregated per scenario).
        
        Returns:
            tuple:
                - payload (Dict[str, object]): A dictionary containing:
                    - "seeds": list of seeds used,
                    - "scenario_names": list of scenario names evaluated,
                    - "episodes_per_scenario": number of runs per scenario,
                    - "reward_profiles": per-profile compact behavior payloads,
                    - "map_templates": per-map compact behavior payloads,
                    - "matrix": nested mapping reward_profile -> map_template -> compact behavior payload.
                - rows (List[Dict[str, object]]): Flattened per-episode behavior rows collected across all runs, suitable for CSV export.
        """
        resolved_noise_profile = resolve_noise_profile(noise_profile)
        budget = resolve_budget(
            profile=budget_profile,
            episodes=episodes,
            eval_episodes=evaluation_episodes,
            max_steps=max_steps,
            scenario_episodes=episodes_per_scenario,
            checkpoint_interval=checkpoint_interval,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        scenario_names = list(names or SCENARIO_NAMES)
        profile_names = list(reward_profiles or ("classic",))
        map_names = list(map_templates or ("central_burrow",))
        run_count = max(1, int(budget.scenario_episodes))
        seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
        matrix: Dict[str, Dict[str, object]] = {}
        rows: List[Dict[str, object]] = []
        profile_stats = {
            profile: {name: [] for name in scenario_names}
            for profile in profile_names
        }
        profile_scores = {
            profile: {name: [] for name in scenario_names}
            for profile in profile_names
        }
        map_stats = {
            map_name: {name: [] for name in scenario_names}
            for map_name in map_names
        }
        map_scores = {
            map_name: {name: [] for name in scenario_names}
            for map_name in map_names
        }

        for profile in profile_names:
            matrix[profile] = {}
            for map_name in map_names:
                combined_stats = {
                    name: [] for name in scenario_names
                }
                combined_scores = {
                    name: [] for name in scenario_names
                }
                sim: SpiderSimulation | None = None
                for seed in seed_values:
                    sim_budget = budget.to_summary()
                    sim_budget["resolved"]["scenario_episodes"] = run_count
                    sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
                    sim_budget["resolved"]["ablation_seeds"] = list(budget.ablation_seeds)
                    sim = cls(
                        width=width,
                        height=height,
                        food_count=food_count,
                        day_length=day_length,
                        night_length=night_length,
                        max_steps=budget.max_steps,
                        seed=seed,
                        gamma=gamma,
                        module_lr=module_lr,
                        motor_lr=motor_lr,
                        module_dropout=module_dropout,
                        operational_profile=operational_profile,
                        noise_profile=resolved_noise_profile,
                        reward_profile=profile,
                        map_template=map_name,
                        budget_profile_name=budget.profile,
                        benchmark_strength=budget.benchmark_strength,
                        budget_summary=sim_budget,
                    )
                    if (
                        checkpoint_selection == "best"
                        or budget.episodes > 0
                        or budget.eval_episodes > 0
                    ):
                        run_checkpoint_dir = None
                        if checkpoint_dir is not None:
                            run_checkpoint_dir = (
                                Path(checkpoint_dir)
                                / "behavior_compare"
                                / f"{profile}__{map_name}__seed_{seed}"
                            )
                        sim.train(
                            budget.episodes,
                            evaluation_episodes=0,
                            render_last_evaluation=False,
                            capture_evaluation_trace=False,
                            debug_trace=False,
                            checkpoint_selection=checkpoint_selection,
                            checkpoint_metric=checkpoint_metric,
                            checkpoint_override_penalty=checkpoint_override_penalty,
                            checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                            checkpoint_penalty_mode=checkpoint_penalty_mode,
                            checkpoint_interval=budget.checkpoint_interval,
                            checkpoint_dir=run_checkpoint_dir,
                            checkpoint_scenario_names=scenario_names,
                            selection_scenario_episodes=budget.selection_scenario_episodes,
                        )
                    stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                        names=scenario_names,
                        episodes_per_scenario=run_count,
                        capture_trace=False,
                        debug_trace=False,
                        base_index=200_000,
                    )
                    for name in scenario_names:
                        combined_stats[name].extend(stats_histories[name])
                        combined_scores[name].extend(behavior_histories[name])
                        profile_stats[profile][name].extend(stats_histories[name])
                        profile_scores[profile][name].extend(behavior_histories[name])
                        map_stats[map_name][name].extend(stats_histories[name])
                        map_scores[map_name][name].extend(behavior_histories[name])
                        rows.extend(
                            sim._annotate_behavior_rows(
                                flatten_behavior_rows(
                                    behavior_histories[name],
                                    reward_profile=profile,
                                    scenario_map=get_scenario(name).map_template,
                                    simulation_seed=seed,
                                    scenario_description=get_scenario(name).description,
                                    scenario_objective=get_scenario(name).objective,
                                    scenario_focus=get_scenario(name).diagnostic_focus,
                                    evaluation_map=map_name,
                                )
                            )
                        )
                if sim is None:
                    continue
                matrix[profile][map_name] = cls._with_noise_profile_metadata(
                    cls._compact_behavior_payload(
                        sim._build_behavior_payload(
                            stats_histories=combined_stats,
                            behavior_histories=combined_scores,
                        )
                    ),
                    resolved_noise_profile,
                )

        reward_profile_payloads: Dict[str, object] = {}
        for profile in profile_names:
            sim = cls(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=budget.max_steps,
                seed=seed_values[0] if seed_values else 0,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                operational_profile=operational_profile,
                noise_profile=resolved_noise_profile,
                reward_profile=profile,
                map_template=map_names[0] if map_names else "central_burrow",
            )
            reward_profile_payloads[profile] = cls._with_noise_profile_metadata(
                cls._compact_behavior_payload(
                    sim._build_behavior_payload(
                        stats_histories=profile_stats[profile],
                        behavior_histories=profile_scores[profile],
                    )
                ),
                resolved_noise_profile,
            )

        map_payloads: Dict[str, object] = {}
        for map_name in map_names:
            sim = cls(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=budget.max_steps,
                seed=seed_values[0] if seed_values else 0,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                operational_profile=operational_profile,
                noise_profile=resolved_noise_profile,
                reward_profile=profile_names[0] if profile_names else "classic",
                map_template=map_name,
            )
            map_payloads[map_name] = cls._with_noise_profile_metadata(
                cls._compact_behavior_payload(
                    sim._build_behavior_payload(
                        stats_histories=map_stats[map_name],
                        behavior_histories=map_scores[map_name],
                    )
                ),
                resolved_noise_profile,
            )

        return {
            "budget_profile": budget.profile,
            "benchmark_strength": budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_penalty_config": CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            ).to_summary(),
            "seeds": list(seed_values),
            "scenario_names": scenario_names,
            "episodes_per_scenario": run_count,
            "reward_profiles": reward_profile_payloads,
            "map_templates": map_payloads,
            "matrix": matrix,
            "reward_audit": cls._build_reward_audit(
                current_profile=profile_names[0] if profile_names else None,
                comparison_payload={"reward_profiles": reward_profile_payloads},
            ),
            **cls._noise_profile_metadata(resolved_noise_profile),
        }, rows

    @classmethod
    def _compare_named_training_regimes(
        cls,
        *,
        regime_names: Sequence[str],
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile: str | BudgetProfile | None = None,
        seeds: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        episodes_per_scenario: int | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Compare specified training regimes by training and evaluating each under both self-sufficient (no reflex support) and scaffolded (reflex-supported) conditions.
        
        For each regime and seed this runs training according to the resolved budget and regime spec, collects per-scenario episode statistics and behavior scores for scaffolded and self-sufficient evaluations, aggregates results across seeds, computes competence gaps (scaffolded vs self-sufficient), and flattens per-episode/seed rows suitable for CSV export.
        
        Parameters:
            regime_names (Sequence[str]): Names of training regimes to compare. `"baseline"` will be ensured as the first regime if not present.
            width, height, food_count, day_length, night_length, max_steps: Environment layout and episode length overrides used to construct each simulation.
            episodes (int | None): Total training episodes (resolved via the budget profile if None).
            evaluation_episodes (int | None): Evaluation episodes per evaluation pass (resolved via the budget profile if None).
            seeds (Sequence[int] | None): Random seeds to run; at least one seed is required.
            names (Sequence[str] | None): Scenario names to evaluate; defaults to all known scenarios.
            episodes_per_scenario (int | None): Number of evaluation episodes per scenario (overrides budget per-scenario setting).
            checkpoint_selection (str): One of `"none"` or `"best"`, controls whether checkpoints are captured/selected during training.
            checkpoint_metric (str): Primary metric label used when ranking checkpoint candidates.
            checkpoint_override_penalty (float): Weight applied to override-rate penalty when using direct composite scoring.
            checkpoint_dominance_penalty (float): Weight applied to reflex dominance penalty when using direct composite scoring.
            checkpoint_penalty_mode (CheckpointPenaltyMode | str): Penalty interpretation mode; `TIEBREAKER` preserves legacy tuple ordering, `DIRECT` uses a penalized composite score.
            checkpoint_interval, checkpoint_dir: Checkpoint capture cadence and optional persistence directory.
        
        Returns:
            tuple[Dict[str, object], List[Dict[str, object]]]: A pair (payload, rows) where:
              - payload: a comparison summary containing per-regime payloads (`regimes`), competence gaps, deltas vs baseline, checkpoint penalty configuration, and noise/budget metadata.
              - rows: flattened annotated behavior rows (one row per evaluated episode/scenario/seed) suitable for CSV export.
        """
        requested_regime_names = [str(name) for name in regime_names]
        if not requested_regime_names:
            requested_regime_names = [
                "baseline",
                "reflex_annealed",
                EXPERIMENT_OF_RECORD_REGIME,
            ]
        if "baseline" not in requested_regime_names:
            requested_regime_names.insert(0, "baseline")
        deduped_regime_names = list(dict.fromkeys(requested_regime_names))
        regime_specs = {
            name: resolve_training_regime(name)
            for name in deduped_regime_names
        }
        resolved_noise_profile = resolve_noise_profile(noise_profile)
        budget = resolve_budget(
            profile=budget_profile,
            episodes=episodes,
            eval_episodes=evaluation_episodes,
            max_steps=max_steps,
            scenario_episodes=episodes_per_scenario,
            checkpoint_interval=checkpoint_interval,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        scenario_names = list(names or SCENARIO_NAMES)
        run_count = max(1, int(budget.scenario_episodes))
        seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
        if not seed_values:
            raise ValueError(
                "compare_training_regimes() requires at least one seed."
            )

        rows: List[Dict[str, object]] = []
        regime_payloads: Dict[str, Dict[str, object]] = {}

        for regime_index, (regime_name, regime_spec) in enumerate(regime_specs.items()):
            self_sufficient_stats = {name: [] for name in scenario_names}
            self_sufficient_scores = {name: [] for name in scenario_names}
            scaffolded_stats = {name: [] for name in scenario_names}
            scaffolded_scores = {name: [] for name in scenario_names}
            training_summaries: list[Dict[str, object]] = []
            training_metadata: list[Dict[str, object]] = []
            exemplar_sim: SpiderSimulation | None = None

            for seed in seed_values:
                sim_budget = budget.to_summary()
                sim_budget["resolved"]["scenario_episodes"] = run_count
                sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
                sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
                sim = cls(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=budget.max_steps,
                    seed=seed,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    operational_profile=operational_profile,
                    noise_profile=resolved_noise_profile,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    budget_profile_name=budget.profile,
                    benchmark_strength=budget.benchmark_strength,
                    budget_summary=sim_budget,
                )
                exemplar_sim = sim
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = (
                        Path(checkpoint_dir)
                        / "training_regime_compare"
                        / f"{regime_name}__seed_{seed}"
                    )
                training_summary, _ = sim.train(
                    budget.episodes,
                    evaluation_episodes=budget.eval_episodes,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=budget.selection_scenario_episodes,
                    training_regime=regime_spec,
                )
                training_summaries.append(deepcopy(training_summary))
                seed_training_metadata = deepcopy(sim._latest_training_regime_summary)
                seed_training_metadata["seed"] = int(seed)
                training_metadata.append(seed_training_metadata)

                scaffolded_eval_scale = sim._effective_reflex_scale(
                    sim.brain.current_reflex_scale
                )
                previous_reflex_scale = float(sim.brain.current_reflex_scale)
                sim.brain.set_runtime_reflex_scale(scaffolded_eval_scale)
                try:
                    scaffolded_stats_histories, scaffolded_behavior_histories, _ = (
                        sim._execute_behavior_suite(
                            names=scenario_names,
                            episodes_per_scenario=run_count,
                            capture_trace=False,
                            debug_trace=False,
                            base_index=600_000 + regime_index * 20_000,
                        )
                    )
                finally:
                    sim.brain.set_runtime_reflex_scale(previous_reflex_scale)

                previous_reflex_scale = float(sim.brain.current_reflex_scale)
                sim.brain.set_runtime_reflex_scale(0.0)
                try:
                    self_stats_histories, self_behavior_histories, _ = (
                        sim._execute_behavior_suite(
                            names=scenario_names,
                            episodes_per_scenario=run_count,
                            capture_trace=False,
                            debug_trace=False,
                            base_index=600_000 + regime_index * 20_000,
                        )
                    )
                finally:
                    sim.brain.set_runtime_reflex_scale(previous_reflex_scale)

                for scenario_name in scenario_names:
                    scenario = get_scenario(scenario_name)
                    scaffolded_stats[scenario_name].extend(
                        scaffolded_stats_histories[scenario_name]
                    )
                    scaffolded_scores[scenario_name].extend(
                        scaffolded_behavior_histories[scenario_name]
                    )
                    self_sufficient_stats[scenario_name].extend(
                        self_stats_histories[scenario_name]
                    )
                    self_sufficient_scores[scenario_name].extend(
                        self_behavior_histories[scenario_name]
                    )
                    rows.extend(
                        sim._annotate_behavior_rows(
                            flatten_behavior_rows(
                                scaffolded_behavior_histories[scenario_name],
                                reward_profile=reward_profile,
                                scenario_map=scenario.map_template,
                                simulation_seed=seed,
                                scenario_description=scenario.description,
                                scenario_objective=scenario.objective,
                                scenario_focus=scenario.diagnostic_focus,
                                evaluation_map=map_template,
                                eval_reflex_scale=scaffolded_eval_scale,
                                competence_label="scaffolded",
                            ),
                            eval_reflex_scale=scaffolded_eval_scale,
                        )
                    )
                    rows.extend(
                        sim._annotate_behavior_rows(
                            flatten_behavior_rows(
                                self_behavior_histories[scenario_name],
                                reward_profile=reward_profile,
                                scenario_map=scenario.map_template,
                                simulation_seed=seed,
                                scenario_description=scenario.description,
                                scenario_objective=scenario.objective,
                                scenario_focus=scenario.diagnostic_focus,
                                evaluation_map=map_template,
                                eval_reflex_scale=0.0,
                                competence_label="self_sufficient",
                            ),
                            eval_reflex_scale=0.0,
                        )
                    )

            if exemplar_sim is None:
                continue

            self_payload = cls._with_noise_profile_metadata(
                cls._compact_behavior_payload(
                    exemplar_sim._build_behavior_payload(
                        stats_histories=self_sufficient_stats,
                        behavior_histories=self_sufficient_scores,
                        competence_label="self_sufficient",
                    )
                ),
                resolved_noise_profile,
            )
            self_payload["summary"]["eval_reflex_scale"] = 0.0
            self_payload["eval_reflex_scale"] = 0.0
            self_payload["competence_type"] = "self_sufficient"

            scaffolded_payload = cls._with_noise_profile_metadata(
                cls._compact_behavior_payload(
                    exemplar_sim._build_behavior_payload(
                        stats_histories=scaffolded_stats,
                        behavior_histories=scaffolded_scores,
                        competence_label="scaffolded",
                    )
                ),
                resolved_noise_profile,
            )
            scaffolded_eval_scale = exemplar_sim._effective_reflex_scale(
                exemplar_sim.brain.current_reflex_scale
            )
            scaffolded_payload["summary"]["eval_reflex_scale"] = scaffolded_eval_scale
            scaffolded_payload["eval_reflex_scale"] = scaffolded_eval_scale
            scaffolded_payload["competence_type"] = "scaffolded"

            competence_gap = cls._evaluation_competence_gap(
                self_sufficient=self_payload["summary"],
                scaffolded=scaffolded_payload["summary"],
            )
            regime_summary = regime_spec.to_summary()
            regime_summary["is_experiment_of_record"] = (
                regime_name == EXPERIMENT_OF_RECORD_REGIME
            )
            regime_payloads[regime_name] = {
                "regime": regime_name,
                "training_regime": regime_summary,
                "is_experiment_of_record": regime_name == EXPERIMENT_OF_RECORD_REGIME,
                "training_regimes": training_metadata,
                "training_summaries": training_summaries,
                "primary_evaluation": "self_sufficient",
                "summary": deepcopy(self_payload["summary"]),
                "success_rates": {
                    "self_sufficient": float(
                        self_payload["summary"].get("scenario_success_rate", 0.0)
                    ),
                    "scaffolded": float(
                        scaffolded_payload["summary"].get(
                            "scenario_success_rate",
                            0.0,
                        )
                    ),
                },
                "episode_success_rates": {
                    "self_sufficient": float(
                        self_payload["summary"].get("episode_success_rate", 0.0)
                    ),
                    "scaffolded": float(
                        scaffolded_payload["summary"].get(
                            "episode_success_rate",
                            0.0,
                        )
                    ),
                },
                "competence_gap": competence_gap,
                "self_sufficient": self_payload,
                "scaffolded": scaffolded_payload,
                "primary_benchmark": self_payload,
                "episode_allocation": {
                    "main_training_episodes": int(budget.episodes),
                    "evaluation_episodes": int(budget.eval_episodes),
                    "episodes_per_scenario": int(run_count),
                },
                **cls._noise_profile_metadata(resolved_noise_profile),
            }

        baseline_payload = regime_payloads.get("baseline", {})
        baseline_self = baseline_payload.get("self_sufficient", {})
        baseline_scaffolded = baseline_payload.get("scaffolded", {})
        baseline_self_summary = (
            baseline_self.get("summary", {})
            if isinstance(baseline_self, dict)
            else {}
        )
        baseline_scaffolded_summary = (
            baseline_scaffolded.get("summary", {})
            if isinstance(baseline_scaffolded, dict)
            else {}
        )
        baseline_gap = baseline_payload.get("competence_gap", {})
        deltas_vs_baseline: Dict[str, Dict[str, float]] = {}
        for regime_name, payload in regime_payloads.items():
            self_summary = payload["self_sufficient"]["summary"]
            scaffolded_summary = payload["scaffolded"]["summary"]
            competence_gap = payload["competence_gap"]
            deltas_vs_baseline[regime_name] = {
                "scenario_success_rate_delta": round(
                    float(self_summary.get("scenario_success_rate", 0.0))
                    - float(baseline_self_summary.get("scenario_success_rate", 0.0)),
                    6,
                ),
                "episode_success_rate_delta": round(
                    float(self_summary.get("episode_success_rate", 0.0))
                    - float(baseline_self_summary.get("episode_success_rate", 0.0)),
                    6,
                ),
                "scaffolded_scenario_success_rate_delta": round(
                    float(scaffolded_summary.get("scenario_success_rate", 0.0))
                    - float(
                        baseline_scaffolded_summary.get(
                            "scenario_success_rate",
                            0.0,
                        )
                    ),
                    6,
                ),
                "competence_gap_delta": round(
                    float(competence_gap.get("scenario_success_rate_delta", 0.0))
                    - float(baseline_gap.get("scenario_success_rate_delta", 0.0)),
                    6,
                ),
            }

        return {
            "comparison_type": "training_regimes",
            "budget_profile": budget.profile,
            "benchmark_strength": budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_penalty_config": CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            ).to_summary(),
            "reference_regime": "baseline",
            "experiment_of_record_regime": EXPERIMENT_OF_RECORD_REGIME,
            "regime_names": deduped_regime_names,
            "seeds": list(seed_values),
            "scenario_names": scenario_names,
            "episodes_per_scenario": run_count,
            "regimes": regime_payloads,
            "competence_gaps": {
                regime_name: payload["competence_gap"]
                for regime_name, payload in regime_payloads.items()
            },
            "deltas_vs_baseline": deltas_vs_baseline,
            **cls._noise_profile_metadata(resolved_noise_profile),
        }, rows

    @classmethod
    def compare_training_regimes(
        cls,
        regime_names: Sequence[str] | None = None,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile: str | BudgetProfile | None = None,
        seeds: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        episodes_per_scenario: int | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
        curriculum_profile: str = "ecological_v1",
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Compare 'flat' and curriculum training regimes by training and evaluating both under a shared budget and seed set.
        
        Trains a flat regime and a curriculum regime (using the provided curriculum_profile) for each seed, optionally capturing checkpoints per the checkpoint parameters, evaluates both regimes across the same scenario suite, and aggregates per-scenario episode statistics, behavioral scores, training/curriculum metadata, and computed deltas comparing curriculum versus flat.
        
        Returns:
            result_payload (Dict[str, object]): Aggregated comparison payload containing budget and seed metadata, per-regime compact behavior payloads under "regimes", computed deltas versus the flat reference under "deltas_vs_flat", focus summaries, and noise profile metadata.
            rows (List[Dict[str, object]]): A flattened, annotated list of per-episode/behavior rows suitable for CSV export.
        """
        if regime_names is not None:
            return cls._compare_named_training_regimes(
                regime_names=regime_names,
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                budget_profile=budget_profile,
                seeds=seeds,
                names=names,
                episodes_per_scenario=episodes_per_scenario,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        profile_name = _validate_curriculum_profile(curriculum_profile)
        if profile_name == "none":
            raise ValueError(
                "compare_training_regimes() requires an active curriculum."
            )
        resolved_noise_profile = resolve_noise_profile(noise_profile)
        budget = resolve_budget(
            profile=budget_profile,
            episodes=episodes,
            eval_episodes=evaluation_episodes,
            max_steps=max_steps,
            scenario_episodes=episodes_per_scenario,
            checkpoint_interval=checkpoint_interval,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        scenario_names = list(names or SCENARIO_NAMES)
        focus_scenarios = [
            name for name in CURRICULUM_FOCUS_SCENARIOS if name in scenario_names
        ]
        run_count = max(1, int(budget.scenario_episodes))
        seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
        if not seed_values:
            raise ValueError(
                "compare_training_regimes() requires at least one seed."
            )
        regimes = ("flat", "curriculum")
        regime_stats = {
            regime: {name: [] for name in scenario_names}
            for regime in regimes
        }
        regime_scores = {
            regime: {name: [] for name in scenario_names}
            for regime in regimes
        }
        regime_training_metadata: Dict[str, list[Dict[str, object]]] = {
            regime: [] for regime in regimes
        }
        regime_curriculum_metadata: Dict[str, list[Dict[str, object]]] = {
            regime: [] for regime in regimes
        }
        rows: List[Dict[str, object]] = []
        sim_budget = budget.to_summary()
        sim_budget["resolved"]["scenario_episodes"] = run_count
        sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
        sim_budget["resolved"]["ablation_seeds"] = list(budget.ablation_seeds)

        for regime in regimes:
            for seed in seed_values:
                sim = cls(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=budget.max_steps,
                    seed=seed,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    operational_profile=operational_profile,
                    noise_profile=resolved_noise_profile,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    budget_profile_name=budget.profile,
                    benchmark_strength=budget.benchmark_strength,
                    budget_summary=sim_budget,
                )
                if regime == "curriculum":
                    sim._set_training_regime_metadata(
                        curriculum_profile=profile_name,
                        episodes=int(budget.episodes),
                        curriculum_summary=cls._empty_curriculum_summary(
                            profile_name,
                            int(budget.episodes),
                        ) if int(budget.episodes) <= 0 else None,
                    )
                else:
                    sim._set_training_regime_metadata(
                        curriculum_profile="none",
                        episodes=int(budget.episodes),
                    )
                if (
                    checkpoint_selection == "best"
                    or budget.episodes > 0
                    or budget.eval_episodes > 0
                ):
                    run_checkpoint_dir = None
                    if checkpoint_dir is not None:
                        run_checkpoint_dir = (
                            Path(checkpoint_dir)
                            / "curriculum_compare"
                            / f"{regime}__seed_{seed}"
                        )
                    sim.train(
                        budget.episodes,
                        evaluation_episodes=0,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                        checkpoint_selection=checkpoint_selection,
                        checkpoint_metric=checkpoint_metric,
                        checkpoint_override_penalty=checkpoint_override_penalty,
                        checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                        checkpoint_penalty_mode=checkpoint_penalty_mode,
                        checkpoint_interval=budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=budget.selection_scenario_episodes,
                        curriculum_profile=(
                            profile_name if regime == "curriculum" else "none"
                        ),
                    )
                seed_identifier = int(getattr(sim, "seed", seed))
                training_metadata = deepcopy(sim._latest_training_regime_summary)
                training_metadata["seed"] = seed_identifier
                regime_training_metadata[regime].append(training_metadata)
                if sim._latest_curriculum_summary is not None:
                    curriculum_metadata = deepcopy(sim._latest_curriculum_summary)
                    curriculum_metadata["seed"] = seed_identifier
                    regime_curriculum_metadata[regime].append(
                        curriculum_metadata
                    )
                stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                    names=scenario_names,
                    episodes_per_scenario=run_count,
                    capture_trace=False,
                    debug_trace=False,
                    base_index=300_000,
                )
                for name in scenario_names:
                    regime_stats[regime][name].extend(stats_histories[name])
                    regime_scores[regime][name].extend(behavior_histories[name])
                    rows.extend(
                        sim._annotate_behavior_rows(
                            flatten_behavior_rows(
                                behavior_histories[name],
                                reward_profile=reward_profile,
                                scenario_map=get_scenario(name).map_template,
                                simulation_seed=seed,
                                scenario_description=get_scenario(name).description,
                                scenario_objective=get_scenario(name).objective,
                                scenario_focus=get_scenario(name).diagnostic_focus,
                                evaluation_map=map_template,
                            )
                        )
                    )

        regime_payloads = {
            regime: cls._with_noise_profile_metadata(
                cls._compact_behavior_payload(
                    sim._build_behavior_payload(
                        stats_histories=regime_stats[regime],
                        behavior_histories=regime_scores[regime],
                    )
                ),
                resolved_noise_profile,
            )
            for regime in regimes
        }
        for regime in regimes:
            regime_payloads[regime]["training_regimes"] = deepcopy(
                regime_training_metadata.get(regime, [])
            )
            latest_training_regime = (
                regime_training_metadata[regime][-1]
                if regime_training_metadata.get(regime)
                else {}
            )
            regime_payloads[regime]["training_regime"] = deepcopy(
                latest_training_regime
            )
            regime_payloads[regime]["episode_allocation"] = {
                "total_training_episodes": int(budget.episodes),
                "evaluation_episodes": int(budget.eval_episodes),
                "episodes_per_scenario": int(run_count),
            }
            curriculum_metadata = regime_curriculum_metadata.get(regime, [])
            if curriculum_metadata:
                regime_payloads[regime]["curriculum_runs"] = deepcopy(
                    curriculum_metadata
                )
                regime_payloads[regime]["curriculum"] = deepcopy(
                    curriculum_metadata[-1]
                )
        deltas = cls._build_learning_evidence_deltas(
            regime_payloads,
            reference_condition="flat",
            scenario_names=scenario_names,
        )
        return {
            "budget_profile": budget.profile,
            "benchmark_strength": budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_penalty_config": CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            ).to_summary(),
            "curriculum_profile": profile_name,
            "reference_regime": "flat",
            "seeds": list(seed_values),
            "scenario_names": scenario_names,
            "episodes_per_scenario": run_count,
            "focus_scenarios": focus_scenarios,
            "regimes": regime_payloads,
            "deltas_vs_flat": deltas,
            "focus_summary": {
                regime: {
                    name: {
                        "success_rate": float(
                            regime_payloads[regime]["suite"]
                            .get(name, {})
                            .get("success_rate", 0.0)
                        )
                    }
                    for name in focus_scenarios
                }
                for regime in regimes
            },
            **cls._noise_profile_metadata(resolved_noise_profile),
        }, rows

    @classmethod
    def compare_noise_robustness(
        cls,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        operational_profile: str | OperationalProfile | None = None,
        budget_profile: str | BudgetProfile | None = None,
        seeds: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        episodes_per_scenario: int | None = None,
        robustness_matrix: RobustnessMatrixSpec | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
        load_brain: str | Path | None = None,
        load_modules: Sequence[str] | None = None,
        save_brain: str | Path | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Train each robustness-matrix row and evaluate it across every column.

        When `robustness_matrix` is omitted, uses the canonical 4x4 protocol.
        Raises `ValueError` if no seeds are available after budget resolution.
        If `checkpoint_dir` is provided, per-train-condition checkpoints are
        stored and loaded from train/seed-specific subdirectories keyed by a
        stable config fingerprint. `checkpoint_selection` controls whether
        checkpoints are ignored (`"none"`) or selected (`"best"`) using
        `checkpoint_metric`. Returns `(payload, rows)`, where
        `payload["matrix"]` is the nested train->eval summary mapping and
        `rows` is the flattened per-episode behavior export.
        """
        if robustness_matrix is None:
            robustness_matrix = canonical_robustness_matrix()
        if checkpoint_selection not in {"none", "best"}:
            raise ValueError(
                "Invalid checkpoint_selection. Use 'none' or 'best'."
            )
        checkpoint_selection_config: CheckpointSelectionConfig | None = None
        if checkpoint_selection == "best":
            checkpoint_selection_config = CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            )
            cls._checkpoint_candidate_sort_key(
                {},
                selection_config=checkpoint_selection_config,
            )
        budget = resolve_budget(
            profile=budget_profile,
            episodes=episodes,
            eval_episodes=evaluation_episodes,
            max_steps=max_steps,
            scenario_episodes=episodes_per_scenario,
            checkpoint_interval=checkpoint_interval,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        scenario_names = list(names or SCENARIO_NAMES)
        run_count = max(1, int(budget.scenario_episodes))
        seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
        if not seed_values:
            raise ValueError(
                "compare_noise_robustness() requires at least one seed."
            )

        rows: List[Dict[str, object]] = []
        matrix_payloads: Dict[str, Dict[str, object]] = {
            train_condition: {}
            for train_condition in robustness_matrix.train_conditions
        }

        for train_index, train_condition in enumerate(
            robustness_matrix.train_conditions
        ):
            resolved_train_noise_profile = resolve_noise_profile(train_condition)
            combined_stats_by_eval = {
                eval_condition: {name: [] for name in scenario_names}
                for eval_condition in robustness_matrix.eval_conditions
            }
            combined_scores_by_eval = {
                eval_condition: {name: [] for name in scenario_names}
                for eval_condition in robustness_matrix.eval_conditions
            }

            for seed in seed_values:
                sim_budget = budget.to_summary()
                sim_budget["resolved"]["scenario_episodes"] = run_count
                sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
                sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
                fingerprint_budget_resolved = {
                    "episodes": budget.episodes,
                    "eval_episodes": budget.eval_episodes,
                    "max_steps": budget.max_steps,
                    "scenario_episodes": run_count,
                    "checkpoint_interval": budget.checkpoint_interval,
                    "selection_scenario_episodes": (
                        budget.selection_scenario_episodes
                    ),
                }
                preload_fingerprint = cls._checkpoint_preload_fingerprint(
                    load_brain,
                    load_modules,
                )
                sim = cls(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=budget.max_steps,
                    seed=seed,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    operational_profile=operational_profile,
                    noise_profile=resolved_train_noise_profile,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    budget_profile_name=budget.profile,
                    benchmark_strength=budget.benchmark_strength,
                    budget_summary=sim_budget,
                )

                brain_config_summary = sim.brain.config.to_summary()
                architecture_fingerprint = sim.brain._architecture_fingerprint()
                run_fingerprint = cls._checkpoint_run_fingerprint(
                    {
                        "workflow": "noise_robustness",
                        "scenario_names": scenario_names,
                        "episodes_per_scenario": run_count,
                        "budget_profile": budget.profile,
                        "budget_benchmark_strength": budget.benchmark_strength,
                        "budget_resolved": fingerprint_budget_resolved,
                        "world": {
                            "width": width,
                            "height": height,
                            "food_count": food_count,
                            "day_length": day_length,
                            "night_length": night_length,
                            "max_steps": budget.max_steps,
                            "reward_profile": reward_profile,
                            "map_template": map_template,
                            "train_noise_profile": resolved_train_noise_profile.name,
                        },
                        "learning": {
                            "gamma": gamma,
                            "module_lr": module_lr,
                            "motor_lr": motor_lr,
                            "module_dropout": module_dropout,
                        },
                        "operational_profile": sim.operational_profile.to_summary(),
                        "architecture": brain_config_summary,
                        "architecture_fingerprint": architecture_fingerprint,
                        "checkpoint_selection": checkpoint_selection,
                        "checkpoint_metric": checkpoint_metric,
                        "checkpoint_penalty_config": (
                            checkpoint_selection_config.to_summary()
                            if checkpoint_selection_config is not None
                            else {
                                "metric": checkpoint_metric,
                                "override_penalty_weight": float(
                                    checkpoint_override_penalty
                                ),
                                "dominance_penalty_weight": float(
                                    checkpoint_dominance_penalty
                                ),
                                "penalty_mode": (
                                    checkpoint_penalty_mode.value
                                    if isinstance(
                                        checkpoint_penalty_mode,
                                        CheckpointPenaltyMode,
                                    )
                                    else str(checkpoint_penalty_mode)
                                ),
                            }
                        ),
                        "checkpoint_interval": budget.checkpoint_interval,
                        "selection_scenario_episodes": budget.selection_scenario_episodes,
                        "preload": preload_fingerprint,
                    }
                )
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = (
                        Path(checkpoint_dir)
                        / "noise_robustness"
                        / f"{train_condition}__seed_{seed}__{run_fingerprint}"
                    )
                checkpoint_load_dir = cls._resolve_checkpoint_load_dir(
                    run_checkpoint_dir,
                    checkpoint_selection=checkpoint_selection,
                )
                if checkpoint_load_dir is not None:
                    sim.brain.load(checkpoint_load_dir)
                    sim.checkpoint_source = (
                        checkpoint_load_dir.name
                        if checkpoint_load_dir.name in {"best", "last"}
                        else "checkpoint"
                    )
                else:
                    if load_brain is not None:
                        sim.brain.load(load_brain, modules=load_modules)
                    should_train = (
                        checkpoint_selection == "best"
                        or budget.episodes > 0
                    )
                    if load_brain is not None and not should_train:
                        sim.checkpoint_source = "preloaded"
                    if should_train:
                        sim.train(
                            budget.episodes,
                            evaluation_episodes=0,
                            render_last_evaluation=False,
                            capture_evaluation_trace=False,
                            debug_trace=False,
                            checkpoint_selection=checkpoint_selection,
                            checkpoint_metric=checkpoint_metric,
                            checkpoint_override_penalty=checkpoint_override_penalty,
                            checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                            checkpoint_penalty_mode=checkpoint_penalty_mode,
                            checkpoint_interval=budget.checkpoint_interval,
                            checkpoint_dir=run_checkpoint_dir,
                            checkpoint_scenario_names=scenario_names,
                            selection_scenario_episodes=budget.selection_scenario_episodes,
                        )
                if save_brain is not None:
                    save_root = (
                        Path(save_brain)
                        / "noise_robustness"
                        / f"{train_condition}__seed_{seed}__{run_fingerprint}"
                    )
                    sim.brain.save(save_root)

                episodes_per_cell = max(
                    1,
                    run_count * max(1, len(scenario_names)),
                )
                eval_stride = episodes_per_cell
                train_stride = eval_stride * max(
                    1,
                    len(robustness_matrix.eval_conditions),
                )
                for eval_index, eval_condition in enumerate(
                    robustness_matrix.eval_conditions
                ):
                    with sim._swap_eval_noise_profile(eval_condition):
                        stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                            names=scenario_names,
                            episodes_per_scenario=run_count,
                            capture_trace=False,
                            debug_trace=False,
                            base_index=500_000
                            + train_index * train_stride
                            + eval_index * eval_stride,
                        )
                        for name in scenario_names:
                            combined_stats_by_eval[eval_condition][name].extend(
                                stats_histories[name]
                            )
                            combined_scores_by_eval[eval_condition][name].extend(
                                behavior_histories[name]
                            )
                            rows.extend(
                                sim._annotate_behavior_rows(
                                    flatten_behavior_rows(
                                        behavior_histories[name],
                                        reward_profile=reward_profile,
                                        scenario_map=get_scenario(name).map_template,
                                        simulation_seed=seed,
                                        scenario_description=get_scenario(name).description,
                                        scenario_objective=get_scenario(name).objective,
                                        scenario_focus=get_scenario(name).diagnostic_focus,
                                        evaluation_map=map_template,
                                    ),
                                    train_noise_profile=resolved_train_noise_profile,
                                )
                            )

            for eval_condition in robustness_matrix.eval_conditions:
                resolved_eval_noise_profile = resolve_noise_profile(eval_condition)
                cell_payload = cls._with_noise_profile_metadata(
                    cls._compact_behavior_payload(
                        sim._build_behavior_payload(
                            stats_histories=combined_stats_by_eval[eval_condition],
                            behavior_histories=combined_scores_by_eval[eval_condition],
                        )
                    ),
                    resolved_eval_noise_profile,
                )
                cell_payload["train_noise_profile"] = resolved_train_noise_profile.name
                cell_payload["train_noise_profile_config"] = (
                    resolved_train_noise_profile.to_summary()
                )
                cell_payload["eval_noise_profile"] = resolved_eval_noise_profile.name
                cell_payload["eval_noise_profile_config"] = (
                    resolved_eval_noise_profile.to_summary()
                )
                matrix_payloads[train_condition][eval_condition] = cell_payload

        aggregate_metrics = cls._robustness_aggregate_metrics(
            matrix_payloads,
            robustness_matrix=robustness_matrix,
        )
        return {
            "budget_profile": budget.profile,
            "benchmark_strength": budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_penalty_config": (
                checkpoint_selection_config.to_summary()
                if checkpoint_selection_config is not None
                else {
                    "metric": checkpoint_metric,
                    "override_penalty_weight": float(checkpoint_override_penalty),
                    "dominance_penalty_weight": float(checkpoint_dominance_penalty),
                    "penalty_mode": (
                        checkpoint_penalty_mode.value
                        if isinstance(checkpoint_penalty_mode, CheckpointPenaltyMode)
                        else str(checkpoint_penalty_mode)
                    ),
                }
            ),
            "reward_profile": reward_profile,
            "map_template": map_template,
            "seeds": list(seed_values),
            "scenario_names": scenario_names,
            "episodes_per_scenario": run_count,
            "matrix": matrix_payloads,
            **aggregate_metrics,
            **cls._robustness_matrix_metadata(robustness_matrix),
        }, rows

    @classmethod
    def compare_ablation_suite(
        cls,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile: str | BudgetProfile | None = None,
        seeds: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        variant_names: Sequence[str] | None = None,
        episodes_per_scenario: int | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Compare ablation variants by optionally training them and evaluating behavior suites, returning per-variant aggregated payloads and flattened CSV-ready rows.
        
        Each variant (including a default reference) is executed across the requested seeds and scenarios. For each seed the simulation may be trained (controlled by `checkpoint_selection` and budget), then evaluated twice: once with the runtime reflex support preserved (diagnostic/scaffolded) and once with reflexes disabled (primary/self-sufficient). The primary evaluation recorded in the returned payloads is the no-reflex result; reflex-enabled results are included for diagnostics and delta computations versus the reference.
        
        Returns:
            tuple:
                payload (dict): Summary payload containing keys including:
                    - "reference_variant": name of the reference variant,
                    - "scenario_names": list of evaluated scenario names,
                    - "episodes_per_scenario": number of runs per scenario,
                    - "variants": mapping from variant name to a compact behavior-suite payload where
                      the primary evaluation is the no-reflex result and reflex-enabled results appear under
                      "with_reflex_support" / "without_reflex_support",
                    - "deltas_vs_reference": per-variant delta metrics versus the reference,
                    - checkpoint selection metadata and resolved noise profile metadata.
                rows (list[dict]): Flattened, annotated rows for every evaluated episode (suitable for CSV export),
                    including ablation, reflex/evaluation metadata, seed and scenario details.
        """
        resolved_noise_profile = resolve_noise_profile(noise_profile)
        budget = resolve_budget(
            profile=budget_profile,
            episodes=episodes,
            eval_episodes=evaluation_episodes,
            max_steps=max_steps,
            scenario_episodes=episodes_per_scenario,
            checkpoint_interval=checkpoint_interval,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        scenario_names = list(names or SCENARIO_NAMES)
        requested_configs = resolve_ablation_configs(
            variant_names,
            module_dropout=module_dropout,
        )
        reference_config = default_brain_config(module_dropout=module_dropout)
        configs: List[BrainAblationConfig] = [reference_config]
        for config in requested_configs:
            if config.name != reference_config.name:
                configs.append(config)

        run_count = max(1, int(budget.scenario_episodes))
        seed_values = tuple(seeds) if seeds is not None else budget.ablation_seeds
        rows: List[Dict[str, object]] = []
        variants: Dict[str, Dict[str, object]] = {}

        for config in configs:
            combined_stats = {name: [] for name in scenario_names}
            combined_scores = {name: [] for name in scenario_names}
            combined_stats_without_reflex = {name: [] for name in scenario_names}
            combined_scores_without_reflex = {name: [] for name in scenario_names}
            sim: SpiderSimulation | None = None
            behavior_base_index = 300_000
            for seed in seed_values:
                sim_budget = budget.to_summary()
                sim_budget["resolved"]["scenario_episodes"] = run_count
                sim_budget["resolved"]["behavior_seeds"] = list(budget.behavior_seeds)
                sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
                sim = cls(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=budget.max_steps,
                    seed=seed,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=config.module_dropout,
                    operational_profile=operational_profile,
                    noise_profile=resolved_noise_profile,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    brain_config=config,
                    budget_profile_name=budget.profile,
                    benchmark_strength=budget.benchmark_strength,
                    budget_summary=sim_budget,
                )
                if (
                    checkpoint_selection == "best"
                    or budget.episodes > 0
                    or budget.eval_episodes > 0
                ):
                    run_checkpoint_dir = None
                    if checkpoint_dir is not None:
                        run_checkpoint_dir = (
                            Path(checkpoint_dir)
                            / "ablation_compare"
                            / f"{config.name}__seed_{seed}"
                        )
                    sim.train(
                        budget.episodes,
                        evaluation_episodes=0,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                        checkpoint_selection=checkpoint_selection,
                        checkpoint_metric=checkpoint_metric,
                        checkpoint_override_penalty=checkpoint_override_penalty,
                        checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                        checkpoint_penalty_mode=checkpoint_penalty_mode,
                        checkpoint_interval=budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=budget.selection_scenario_episodes,
                    )
                stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                    names=scenario_names,
                    episodes_per_scenario=run_count,
                    capture_trace=False,
                    debug_trace=False,
                    base_index=behavior_base_index,
                )
                previous_reflex_scale = float(sim.brain.current_reflex_scale)
                sim.brain.set_runtime_reflex_scale(0.0)
                try:
                    no_reflex_stats_histories, no_reflex_behavior_histories, _ = sim._execute_behavior_suite(
                        names=scenario_names,
                        episodes_per_scenario=run_count,
                        capture_trace=False,
                        debug_trace=False,
                        base_index=behavior_base_index,
                    )
                finally:
                    sim.brain.set_runtime_reflex_scale(previous_reflex_scale)
                for name in scenario_names:
                    combined_stats[name].extend(stats_histories[name])
                    combined_scores[name].extend(behavior_histories[name])
                    combined_stats_without_reflex[name].extend(no_reflex_stats_histories[name])
                    combined_scores_without_reflex[name].extend(no_reflex_behavior_histories[name])
                    rows.extend(
                        sim._annotate_behavior_rows(
                            flatten_behavior_rows(
                                behavior_histories[name],
                                reward_profile=reward_profile,
                                scenario_map=get_scenario(name).map_template,
                                simulation_seed=seed,
                                scenario_description=get_scenario(name).description,
                                scenario_objective=get_scenario(name).objective,
                                scenario_focus=get_scenario(name).diagnostic_focus,
                                evaluation_map=map_template,
                                eval_reflex_scale=previous_reflex_scale,
                                competence_label="scaffolded",
                            ),
                            eval_reflex_scale=previous_reflex_scale,
                        )
                    )
                    rows.extend(
                        sim._annotate_behavior_rows(
                            flatten_behavior_rows(
                                no_reflex_behavior_histories[name],
                                reward_profile=reward_profile,
                                scenario_map=get_scenario(name).map_template,
                                simulation_seed=seed,
                                scenario_description=get_scenario(name).description,
                                scenario_objective=get_scenario(name).objective,
                                scenario_focus=get_scenario(name).diagnostic_focus,
                                evaluation_map=map_template,
                                eval_reflex_scale=0.0,
                                competence_label="self_sufficient",
                            ),
                            eval_reflex_scale=0.0,
                        )
                    )
            if sim is None:
                continue
            with_reflex_payload = cls._compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=combined_stats,
                    behavior_histories=combined_scores,
                    competence_label="scaffolded",
                )
            )
            with_reflex_payload["summary"]["eval_reflex_scale"] = float(
                previous_reflex_scale
            )
            with_reflex_payload["summary"]["competence_type"] = "scaffolded"
            with_reflex_payload["eval_reflex_scale"] = float(previous_reflex_scale)
            with_reflex_payload["competence_type"] = "scaffolded"
            with_reflex_payload["config"] = config.to_summary()
            with_reflex_payload.update(
                cls._noise_profile_metadata(resolved_noise_profile)
            )
            without_reflex_payload = cls._with_noise_profile_metadata(
                cls._compact_behavior_payload(
                    sim._build_behavior_payload(
                        stats_histories=combined_stats_without_reflex,
                        behavior_histories=combined_scores_without_reflex,
                        competence_label="self_sufficient",
                    )
                ),
                resolved_noise_profile,
            )
            without_reflex_payload["summary"]["eval_reflex_scale"] = 0.0
            without_reflex_payload["summary"]["competence_type"] = "self_sufficient"
            without_reflex_payload["eval_reflex_scale"] = 0.0
            without_reflex_payload["competence_type"] = "self_sufficient"
            without_reflex_payload["config"] = config.to_summary()
            compact_payload = dict(without_reflex_payload)
            compact_payload["primary_evaluation"] = "without_reflex_support"
            compact_payload["with_reflex_support"] = with_reflex_payload
            compact_payload["without_reflex_support"] = without_reflex_payload
            variants[config.name] = compact_payload

        reference_variant = reference_config.name
        return {
            "budget_profile": budget.profile,
            "benchmark_strength": budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_penalty_config": CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            ).to_summary(),
            "primary_evaluation": "without_reflex_support",
            "reference_variant": reference_variant,
            "reference_eval_reflex_scale": 0.0,
            "seeds": list(seed_values),
            "scenario_names": scenario_names,
            "episodes_per_scenario": run_count,
            "variants": variants,
            "deltas_vs_reference": cls._build_ablation_deltas(
                variants,
                reference_variant=reference_variant,
                scenario_names=scenario_names,
            ),
            **cls._noise_profile_metadata(resolved_noise_profile),
        }, rows

    @classmethod
    def compare_learning_evidence(
        cls,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        brain_config: BrainAblationConfig | None = None,
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile: str | BudgetProfile | None = None,
        long_budget_profile: str | BudgetProfile | None = "report",
        seeds: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        condition_names: Sequence[str] | None = None,
        episodes_per_scenario: int | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Run a registry of learning-evidence conditions across seeds and scenarios and produce compact comparison results.
        
        For each resolved condition this method trains or initializes simulations according to the condition's specified budget (base, long, freeze_half, or initial), optionally persists checkpoints, executes the behavior suite with the condition's evaluation policy mode, and aggregates per-condition behavior-suite payloads and flattened CSV rows. Conditions that are incompatible with the base architecture are marked as skipped. The returned payload contains budget and noise metadata, per-condition compact behavior summaries under `"conditions"`, deltas versus the no-reflex reference condition (`"trained_without_reflex_support"`), and a synthesized evidence summary.
        
        Returns:
            tuple: A pair (payload, rows) where
                payload (dict): Compact comparison payload including:
                    - budget and long_budget identifiers and benchmark strengths
                    - checkpoint selection settings and reference condition
                    - list of seeds and scenario names
                    - aggregated `conditions` mapping (compact per-condition behavior payloads)
                    - `deltas_vs_reference` and `evidence_summary` computed from conditions
                    - noise profile metadata
                rows (list[dict]): Flattened, annotated per-episode CSV rows for all runs with learning-evidence metadata fields added.
        """
        resolved_noise_profile = resolve_noise_profile(noise_profile)
        base_budget = resolve_budget(
            profile=budget_profile,
            episodes=episodes,
            eval_episodes=evaluation_episodes,
            max_steps=max_steps,
            scenario_episodes=episodes_per_scenario,
            checkpoint_interval=checkpoint_interval,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        long_budget = resolve_budget(
            profile=long_budget_profile,
            episodes=None,
            eval_episodes=None,
            max_steps=max_steps if max_steps is not None else base_budget.max_steps,
            scenario_episodes=episodes_per_scenario,
            checkpoint_interval=checkpoint_interval,
            behavior_seeds=seeds,
            ablation_seeds=seeds,
        )
        scenario_names = list(names or SCENARIO_NAMES)
        condition_specs = resolve_learning_evidence_conditions(condition_names)
        primary_reference_condition = "trained_without_reflex_support"
        if condition_names is not None and not any(
            spec.name == primary_reference_condition for spec in condition_specs
        ):
            condition_specs = resolve_learning_evidence_conditions(
                (primary_reference_condition, *tuple(condition_names))
            )
        seed_values = tuple(seeds) if seeds is not None else base_budget.behavior_seeds
        run_count = max(1, int(base_budget.scenario_episodes))
        base_config = (
            brain_config
            if brain_config is not None
            else default_brain_config(module_dropout=module_dropout)
        )

        rows: List[Dict[str, object]] = []
        conditions: Dict[str, Dict[str, object]] = {}

        for condition_index, condition in enumerate(condition_specs):
            if base_config.architecture not in condition.supports_architectures:
                conditions[condition.name] = {
                    "condition": condition.name,
                    "description": condition.description,
                    "policy_mode": condition.policy_mode,
                    "train_budget": condition.train_budget,
                    "training_regime": condition.training_regime,
                    "checkpoint_source": condition.checkpoint_source,
                    "budget_profile": (
                        long_budget.profile
                        if condition.train_budget == "long"
                        else base_budget.profile
                    ),
                    "benchmark_strength": (
                        long_budget.benchmark_strength
                        if condition.train_budget == "long"
                        else base_budget.benchmark_strength
                    ),
                    "config": base_config.to_summary(),
                    "skipped": True,
                    "reason": (
                        f"Condition incompatible with architecture {base_config.architecture!r}."
                    ),
                    **cls._noise_profile_metadata(resolved_noise_profile),
                }
                continue
            if (
                condition.policy_mode in {"reflex_only", "reflex-only"}
                and not base_config.enable_reflexes
            ):
                conditions[condition.name] = {
                    "condition": condition.name,
                    "description": condition.description,
                    "policy_mode": condition.policy_mode,
                    "train_budget": condition.train_budget,
                    "training_regime": condition.training_regime,
                    "checkpoint_source": condition.checkpoint_source,
                    "budget_profile": (
                        long_budget.profile
                        if condition.train_budget == "long"
                        else base_budget.profile
                    ),
                    "benchmark_strength": (
                        long_budget.benchmark_strength
                        if condition.train_budget == "long"
                        else base_budget.benchmark_strength
                    ),
                    "config": base_config.to_summary(),
                    "skipped": True,
                    "reason": (
                        f"Condition {condition.name!r} requires reflexes to be enabled, "
                        "but the base configuration has reflexes disabled."
                    ),
                    **cls._noise_profile_metadata(resolved_noise_profile),
                }
                continue

            combined_stats = {name: [] for name in scenario_names}
            combined_scores = {name: [] for name in scenario_names}
            exemplar_sim: SpiderSimulation | None = None
            condition_budget = long_budget if condition.train_budget == "long" else base_budget
            train_episodes = 0
            frozen_after_episode: int | None = None
            observed_checkpoint_source = condition.checkpoint_source
            observed_eval_reflex_scale: float | None = None

            for seed in seed_values:
                sim_budget = condition_budget.to_summary()
                sim_budget["resolved"]["scenario_episodes"] = run_count
                sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
                sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
                sim = cls(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=condition_budget.max_steps,
                    seed=seed,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=base_config.module_dropout,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    brain_config=base_config,
                    operational_profile=operational_profile,
                    noise_profile=resolved_noise_profile,
                    budget_profile_name=condition_budget.profile,
                    benchmark_strength=condition_budget.benchmark_strength,
                    budget_summary=sim_budget,
                )
                exemplar_sim = sim

                if condition.train_budget == "base":
                    run_checkpoint_dir = None
                    if checkpoint_dir is not None:
                        run_checkpoint_dir = (
                            Path(checkpoint_dir)
                            / "learning_evidence"
                            / f"{condition.name}__seed_{seed}"
                        )
                    sim.train(
                        condition_budget.episodes,
                        evaluation_episodes=0,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                        checkpoint_selection=checkpoint_selection,
                        checkpoint_metric=checkpoint_metric,
                        checkpoint_override_penalty=checkpoint_override_penalty,
                        checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                        checkpoint_penalty_mode=checkpoint_penalty_mode,
                        checkpoint_interval=condition_budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=condition_budget.selection_scenario_episodes,
                        training_regime=condition.training_regime,
                    )
                    train_episodes = int(
                        sim._latest_training_regime_summary.get(
                            "resolved_budget",
                            {},
                        ).get(
                            "total_training_episodes",
                            condition_budget.episodes,
                        )
                    )
                    observed_checkpoint_source = str(sim.checkpoint_source)
                elif condition.train_budget == "long":
                    run_checkpoint_dir = None
                    if checkpoint_dir is not None:
                        run_checkpoint_dir = (
                            Path(checkpoint_dir)
                            / "learning_evidence"
                            / f"{condition.name}__seed_{seed}"
                        )
                    sim.train(
                        long_budget.episodes,
                        evaluation_episodes=0,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                        checkpoint_selection=checkpoint_selection,
                        checkpoint_metric=checkpoint_metric,
                        checkpoint_override_penalty=checkpoint_override_penalty,
                        checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                        checkpoint_penalty_mode=checkpoint_penalty_mode,
                        checkpoint_interval=long_budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=long_budget.selection_scenario_episodes,
                        training_regime=condition.training_regime,
                    )
                    train_episodes = int(
                        sim._latest_training_regime_summary.get(
                            "resolved_budget",
                            {},
                        ).get(
                            "total_training_episodes",
                            long_budget.episodes,
                        )
                    )
                    observed_checkpoint_source = str(sim.checkpoint_source)
                elif condition.train_budget == "freeze_half":
                    train_episodes = max(0, int(base_budget.episodes) // 2)
                    run_checkpoint_dir = None
                    if checkpoint_dir is not None:
                        run_checkpoint_dir = (
                            Path(checkpoint_dir)
                            / "learning_evidence"
                            / f"{condition.name}__seed_{seed}"
                        )
                    sim.train(
                        train_episodes,
                        evaluation_episodes=0,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                        checkpoint_selection=checkpoint_selection,
                        checkpoint_metric=checkpoint_metric,
                        checkpoint_override_penalty=checkpoint_override_penalty,
                        checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                        checkpoint_penalty_mode=checkpoint_penalty_mode,
                        checkpoint_interval=base_budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=base_budget.selection_scenario_episodes,
                        training_regime=condition.training_regime,
                    )
                    train_episodes = int(
                        sim._latest_training_regime_summary.get(
                            "resolved_budget",
                            {},
                        ).get(
                            "total_training_episodes",
                            train_episodes,
                        )
                    )
                    frozen_after_episode = train_episodes
                    observed_checkpoint_source = str(sim.checkpoint_source)
                    remaining_episodes = max(0, int(base_budget.episodes) - train_episodes)
                    if remaining_episodes:
                        sim._consume_episodes_without_learning(
                            episodes=remaining_episodes,
                            episode_start=train_episodes,
                            policy_mode="normal",
                        )
                else:
                    train_episodes = 0
                    sim.checkpoint_source = "initial"
                    observed_checkpoint_source = "initial"

                previous_reflex_scale = float(sim.brain.current_reflex_scale)
                effective_eval_reflex_scale = (
                    float(condition.eval_reflex_scale)
                    if condition.eval_reflex_scale is not None
                    else sim._effective_reflex_scale(sim.brain.current_reflex_scale)
                )
                observed_eval_reflex_scale = effective_eval_reflex_scale
                sim.brain.set_runtime_reflex_scale(effective_eval_reflex_scale)
                try:
                    stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                        names=scenario_names,
                        episodes_per_scenario=run_count,
                        capture_trace=False,
                        debug_trace=False,
                        base_index=400_000 + condition_index * 10_000,
                        policy_mode=condition.policy_mode,
                    )
                finally:
                    sim.brain.set_runtime_reflex_scale(previous_reflex_scale)

                extra_row_metadata = {
                    "learning_evidence_condition": condition.name,
                    "learning_evidence_policy_mode": condition.policy_mode,
                    "learning_evidence_training_regime": (
                        "" if condition.training_regime is None else condition.training_regime
                    ),
                    "learning_evidence_train_episodes": int(train_episodes),
                    "learning_evidence_frozen_after_episode": (
                        "" if frozen_after_episode is None else int(frozen_after_episode)
                    ),
                    "learning_evidence_checkpoint_source": (
                        condition.checkpoint_source
                        if condition.train_budget == "freeze_half"
                        else observed_checkpoint_source
                    ),
                    "learning_evidence_budget_profile": condition_budget.profile,
                    "learning_evidence_budget_benchmark_strength": condition_budget.benchmark_strength,
                }
                for name in scenario_names:
                    combined_stats[name].extend(stats_histories[name])
                    combined_scores[name].extend(behavior_histories[name])
                    rows.extend(
                        sim._annotate_behavior_rows(
                            flatten_behavior_rows(
                                behavior_histories[name],
                                reward_profile=reward_profile,
                                scenario_map=get_scenario(name).map_template,
                                simulation_seed=seed,
                                scenario_description=get_scenario(name).description,
                                scenario_objective=get_scenario(name).objective,
                                scenario_focus=get_scenario(name).diagnostic_focus,
                                evaluation_map=map_template,
                                eval_reflex_scale=effective_eval_reflex_scale,
                                competence_label=competence_label_from_eval_reflex_scale(
                                    effective_eval_reflex_scale
                                ),
                            ),
                            eval_reflex_scale=effective_eval_reflex_scale,
                            extra_metadata=extra_row_metadata,
                        )
                    )

            if exemplar_sim is None:
                continue

            compact_payload = cls._compact_behavior_payload(
                exemplar_sim._build_behavior_payload(
                    stats_histories=combined_stats,
                    behavior_histories=combined_scores,
                    competence_label=competence_label_from_eval_reflex_scale(
                        observed_eval_reflex_scale
                    ),
                )
            )
            compact_payload["summary"]["eval_reflex_scale"] = (
                observed_eval_reflex_scale
            )
            compact_payload["eval_reflex_scale"] = observed_eval_reflex_scale
            compact_payload.update(
                {
                    "condition": condition.name,
                    "description": condition.description,
                    "policy_mode": condition.policy_mode,
                    "train_budget": condition.train_budget,
                    "training_regime": condition.training_regime,
                    "train_episodes": int(train_episodes),
                    "frozen_after_episode": frozen_after_episode,
                    "checkpoint_source": (
                        condition.checkpoint_source
                        if condition.train_budget == "freeze_half"
                        else observed_checkpoint_source
                    ),
                    "budget_profile": condition_budget.profile,
                    "benchmark_strength": condition_budget.benchmark_strength,
                    "config": base_config.to_summary(),
                    "skipped": False,
                }
            )
            compact_payload.update(cls._noise_profile_metadata(resolved_noise_profile))
            conditions[condition.name] = compact_payload

        reference_condition = primary_reference_condition
        return {
            "budget_profile": base_budget.profile,
            "benchmark_strength": base_budget.benchmark_strength,
            "long_budget_profile": long_budget.profile,
            "long_budget_benchmark_strength": long_budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_penalty_config": CheckpointSelectionConfig(
                metric=checkpoint_metric,
                override_penalty_weight=checkpoint_override_penalty,
                dominance_penalty_weight=checkpoint_dominance_penalty,
                penalty_mode=checkpoint_penalty_mode,
            ).to_summary(),
            "reference_condition": reference_condition,
            "seeds": list(seed_values),
            "scenario_names": scenario_names,
            "episodes_per_scenario": run_count,
            "conditions": conditions,
            "deltas_vs_reference": cls._build_learning_evidence_deltas(
                conditions,
                reference_condition=reference_condition,
                scenario_names=scenario_names,
            ),
            "evidence_summary": cls._build_learning_evidence_summary(
                conditions,
                reference_condition=reference_condition,
            ),
            **cls._noise_profile_metadata(resolved_noise_profile),
        }, rows

    @staticmethod
    def _claim_test_source(spec: ClaimTestSpec) -> str | None:
        """
        Map a claim-test specification to the primitive payload type it requires.
        
        Parameters:
            spec (ClaimTestSpec): Claim-test specification object with a `.protocol` attribute.
        
        Returns:
            str | None: `'learning_evidence'`, `'noise_robustness'`, or `'ablation'` when the protocol name references that primitive; `None` if no known primitive is referenced.
        """
        protocol = spec.protocol.lower()
        if "learning-evidence" in protocol:
            return "learning_evidence"
        if "noise-robustness" in protocol:
            return "noise_robustness"
        if "ablation" in protocol:
            return "ablation"
        return None

    @staticmethod
    def _claim_skip_result(spec: ClaimTestSpec, reason: str) -> Dict[str, object]:
        """
        Produce a standardized "skipped" result record for a claim test.
        
        Parameters:
            spec (ClaimTestSpec): The claim test specification; used to populate primary metric, scenarios, and success criterion.
            reason (str): Human-readable explanation for why the claim test was skipped.
        
        Returns:
            Dict[str, object]: A normalized result dictionary with keys:
                - "status": "skipped"
                - "passed": False
                - "reason": the provided reason string
                - "reference_value": None
                - "comparison_values": empty dict
                - "delta": empty dict
                - "effect_size": None
                - "primary_metric": value from spec.primary_metric
                - "scenarios_evaluated": list of scenarios from spec.scenarios
                - "notes": list containing spec.success_criterion
        """
        return {
            "status": "skipped",
            "passed": False,
            "reason": str(reason),
            "reference_value": None,
            "comparison_values": {},
            "delta": {},
            "effect_size": None,
            "primary_metric": spec.primary_metric,
            "scenarios_evaluated": list(spec.scenarios),
            "notes": [spec.success_criterion],
        }

    @staticmethod
    def _claim_threshold_from_operator(
        success_criterion: str,
        operator: str,
    ) -> float | None:
        """
        Parse a numeric threshold that immediately follows a given operator token in a criterion string.
        
        Parameters:
            success_criterion (str): The string containing the success criterion to search.
            operator (str): The literal operator token to match (e.g., ">=", "<", "==").
        
        Returns:
            float | None: The parsed number (supports negative and decimal values) from the first match after the operator, or `None` if no valid number is found.
        """
        match = re.search(
            rf"{re.escape(operator)}\s*(-?\d+(?:\.\d+)?)",
            success_criterion,
        )
        if match is None:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _claim_threshold_from_phrase(
        success_criterion: str,
        phrase: str,
    ) -> float | None:
        """
        Parse and return a numeric threshold that immediately follows a given literal phrase in a criterion string.
        
        Parameters:
            success_criterion (str): The criterion text to search for a numeric value.
            phrase (str): The literal phrase to locate; the function looks for a number directly after this phrase.
        
        Returns:
            float | None: The parsed numeric threshold (supports optional leading `-` and decimal points) if found, otherwise `None`.
        """
        match = re.search(
            rf"{re.escape(phrase)}\s*(-?\d+(?:\.\d+)?)",
            success_criterion,
        )
        if match is None:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _claim_count_threshold(success_criterion: str) -> int | None:
        """
        Extract an integer threshold from a phrase of the form "at least N of the M".
        
        Parameters:
        	success_criterion (str): Input success criterion string to parse.
        
        Returns:
        	int | None: The parsed integer N when the phrase is present and valid, otherwise None.
        """
        match = re.search(r"at least\s+(\d+)\s+of\s+the\s+\d+", success_criterion)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _claim_subset_scenario_success_rate(
        cls,
        payload: Dict[str, object] | None,
        *,
        scenarios: Sequence[str],
    ) -> tuple[float | None, str | None]:
        """
        Compute the mean scenario success rate across a required subset of scenarios from a behavior-suite payload.
        
        Parameters:
            payload (Dict[str, object] | None): Behavior payload expected to contain a top-level "suite" mapping where each scenario maps to a dict with a numeric "success_rate" key.
            scenarios (Sequence[str]): Sequence of scenario names required for the subset.
        
        Returns:
            tuple[float | None, str | None]: A pair (score, error). `score` is the mean of the per-scenario
            `success_rate` values rounded to six decimal places when all required scenarios are present; `None`
            if the payload is missing or invalid. `error` is a human-readable error message when `score` is
            `None`, otherwise `None`.
        """
        if not isinstance(payload, dict):
            return None, "Missing behavior payload."
        suite = payload.get("suite", {})
        if not isinstance(suite, dict):
            return None, "Behavior payload is missing suite results."
        if not scenarios:
            return 0.0, None
        missing = [name for name in scenarios if name not in suite]
        if missing:
            return None, f"Missing required scenarios: {missing}."
        scenario_passes = [
            cls._safe_float(dict(suite[name]).get("success_rate", 0.0))
            for name in scenarios
        ]
        return round(sum(scenario_passes) / len(scenarios), 6), None

    @classmethod
    def _claim_registry_entry(
        cls,
        payload: Dict[str, object] | None,
        *,
        registry_key: str,
        entry_name: str,
    ) -> tuple[Dict[str, object] | None, str | None]:
        """
        Locate a named entry in a registry payload and verify it was evaluated.
        
        Parameters:
            registry_key (str): Top-level key in `payload` expected to contain the registry mapping.
            entry_name (str): Name of the entry to fetch from the registry.
        
        Returns:
            tuple[Dict[str, object] | None, str | None]: A pair `(entry, error)`. `entry` is the found registry entry dict when present and not marked skipped; otherwise `None`. `error` is a human-readable message when the registry, entry, or evaluation result is missing or the entry was skipped; otherwise `None`.
        """
        if not isinstance(payload, dict):
            return None, f"Missing {registry_key} payload."
        registry = payload.get(registry_key, {})
        if not isinstance(registry, dict):
            return None, f"{registry_key!r} payload is missing its registry."
        entry = registry.get(entry_name)
        if not isinstance(entry, dict):
            return None, f"Missing {registry_key[:-1]} {entry_name!r}."
        if bool(entry.get("skipped")):
            return None, str(
                entry.get("reason", f"{registry_key[:-1].capitalize()} {entry_name!r} was skipped.")
            )
        return entry, None

    @staticmethod
    def _claim_leakage_audit_summary() -> Dict[str, object]:
        """
        Summarizes unresolved privileged or world-derived leakage findings.
        
        Returns:
            summary (dict): A mapping with:
                - finding_count (int): Number of unresolved findings.
                - findings (list[str]): List of findings formatted as "<audit_name>:<signal_name>".
        """
        findings: list[str] = []
        for audit_name, audit_entries in (
            ("observation", observation_leakage_audit()),
            ("memory", memory_leakage_audit()),
        ):
            for signal_name, metadata in audit_entries.items():
                classification = str(metadata.get("classification", ""))
                risk = str(metadata.get("risk", ""))
                if classification in {
                    "privileged_world_signal",
                    "world_derived_navigation_hint",
                } and risk != "resolved":
                    findings.append(f"{audit_name}:{signal_name}")
        return {
            "finding_count": len(findings),
            "findings": findings,
        }

    @classmethod
    def _claim_noise_subset_scores(
        cls,
        payload: Dict[str, object] | None,
        *,
        scenarios: Sequence[str],
    ) -> tuple[float | None, float | None, str | None]:
        """
        Compute mean diagonal and off-diagonal subset success rates for a noise-robustness matrix.
        
        Parameters:
            payload (dict | None): Robustness comparison payload containing a "matrix" mapping
                train_condition -> eval_condition -> cell payload. If missing or malformed,
                an error reason is returned.
            scenarios (Sequence[str]): Sequence of scenario names to include when computing
                per-cell subset success rates.
        
        Returns:
            tuple[float | None, float | None, str | None]:
                diagonal_mean: Mean subset success rate across cells where train_condition == eval_condition,
                    rounded to 6 decimal places, or `None` if unavailable.
                off_diagonal_mean: Mean subset success rate across cells where train_condition != eval_condition,
                    rounded to 6 decimal places, or `None` if unavailable.
                error_reason: `None` on success; otherwise a short string explaining why the computation
                    could not be completed (e.g., missing matrix, malformed rows, or missing cell data).
        """
        if not isinstance(payload, dict):
            return None, None, "Missing noise_robustness payload."
        matrix = payload.get("matrix", {})
        if not isinstance(matrix, dict):
            return None, None, "Noise robustness payload is missing matrix results."
        diagonal_scores: list[float] = []
        off_diagonal_scores: list[float] = []
        for train_condition, row in matrix.items():
            if not isinstance(row, dict):
                return None, None, f"Noise robustness row {train_condition!r} is malformed."
            for eval_condition, cell_payload in row.items():
                metric_value, reason = cls._claim_subset_scenario_success_rate(
                    cell_payload,
                    scenarios=scenarios,
                )
                if metric_value is None:
                    return None, None, (
                        f"Missing data for noise cell {train_condition!r} -> "
                        f"{eval_condition!r}: {reason}"
                    )
                if train_condition == eval_condition:
                    diagonal_scores.append(metric_value)
                else:
                    off_diagonal_scores.append(metric_value)
        if not diagonal_scores:
            return None, None, "Noise robustness payload has no diagonal cells."
        if not off_diagonal_scores:
            return None, None, "Noise robustness payload has no off-diagonal cells."
        return (
            round(sum(diagonal_scores) / len(diagonal_scores), 6),
            round(sum(off_diagonal_scores) / len(off_diagonal_scores), 6),
            None,
        )

    @classmethod
    def _claim_specialization_engagement(
        cls,
        payload: Dict[str, object] | None,
        *,
        variant_name: str,
        scenarios: Sequence[str],
    ) -> tuple[int | None, Dict[str, float] | None, str | None]:
        """
        Determine how many scenarios for a variant show full specialization engagement according to type-specific checks.
        
        Examines the `variants` registry entry in `payload` for `variant_name`, reads each `scenario`'s check named in SPECIALIZATION_ENGAGEMENT_CHECKS, and records each check's pass rate. A scenario is counted as "engaged" when its pass rate is greater than or equal to 1.0.
        
        Parameters:
            payload (dict | None): Comparison payload containing a `variants` registry (may be None).
            variant_name (str): Name of the variant whose suite to inspect.
            scenarios (Sequence[str]): Sequence of scenario names to evaluate for specialization engagement.
        
        Returns:
            tuple:
                engaged_count (int | None): Number of scenarios with pass rate >= 1.0, or `None` if the check could not be performed.
                pass_rates (dict[str, float] | None): Mapping from scenario name to its recorded `pass_rate` (rounded to 6 decimals), or `None` when unavailable.
                reason (str | None): `None` on success; otherwise a short explanatory message describing why the result is unavailable (missing registry entry, missing scenario/check, or unregistered engagement check).
        """
        variant_payload, reason = cls._claim_registry_entry(
            payload,
            registry_key="variants",
            entry_name=variant_name,
        )
        if variant_payload is None:
            return None, None, reason
        suite = variant_payload.get("suite", {})
        if not isinstance(suite, dict):
            return None, None, f"Variant {variant_name!r} is missing suite results."
        pass_rates: Dict[str, float] = {}
        engaged_count = 0
        for scenario_name in scenarios:
            scenario_payload = suite.get(scenario_name)
            if not isinstance(scenario_payload, dict):
                return None, None, (
                    f"Variant {variant_name!r} is missing scenario {scenario_name!r}."
                )
            checks = scenario_payload.get("checks", {})
            if not isinstance(checks, dict):
                return None, None, (
                    f"Variant {variant_name!r} scenario {scenario_name!r} is missing checks."
                )
            check_name = SPECIALIZATION_ENGAGEMENT_CHECKS.get(scenario_name)
            if check_name is None:
                return None, None, (
                    f"No specialization engagement check is registered for scenario {scenario_name!r}."
                )
            check_payload = checks.get(check_name)
            if not isinstance(check_payload, dict):
                return None, None, (
                    f"Variant {variant_name!r} scenario {scenario_name!r} is missing "
                    f"check {check_name!r}."
                )
            pass_rate = round(
                cls._safe_float(check_payload.get("pass_rate", 0.0)),
                6,
            )
            pass_rates[scenario_name] = pass_rate
            if pass_rate >= 1.0:
                engaged_count += 1
        return engaged_count, pass_rates, None

    @classmethod
    def _evaluate_claim_test(
        cls,
        spec: ClaimTestSpec,
        payloads: Dict[str, Dict[str, object]],
    ) -> Dict[str, object]:
        """Evaluate a single canonical claim test against the available primitive payloads."""
        source = cls._claim_test_source(spec)
        if source is None:
            return cls._claim_skip_result(
                spec,
                f"Could not determine a primitive payload source from protocol {spec.protocol!r}.",
            )

        if source == "learning_evidence":
            payload = payloads.get("learning_evidence")
            reference_payload, reason = cls._claim_registry_entry(
                payload,
                registry_key="conditions",
                entry_name=spec.reference_condition,
            )
            if reference_payload is None:
                return cls._claim_skip_result(spec, str(reason))
            reference_value, reason = cls._claim_subset_scenario_success_rate(
                reference_payload,
                scenarios=spec.scenarios,
            )
            if reference_value is None:
                return cls._claim_skip_result(spec, str(reason))
            comparison_values: Dict[str, float] = {}
            deltas: Dict[str, float] = {}
            for comparison_name in spec.comparison_conditions:
                comparison_payload, reason = cls._claim_registry_entry(
                    payload,
                    registry_key="conditions",
                    entry_name=comparison_name,
                )
                if comparison_payload is None:
                    return cls._claim_skip_result(spec, str(reason))
                comparison_value, reason = cls._claim_subset_scenario_success_rate(
                    comparison_payload,
                    scenarios=spec.scenarios,
                )
                if comparison_value is None:
                    return cls._claim_skip_result(spec, str(reason))
                comparison_values[comparison_name] = comparison_value
                deltas[comparison_name] = round(comparison_value - reference_value, 6)

            if spec.name == "learning_without_privileged_signals":
                delta_threshold = cls._claim_threshold_from_phrase(
                    spec.success_criterion,
                    "by at least",
                )
                if delta_threshold is None:
                    return cls._claim_skip_result(
                        spec,
                        f"Could not parse success criterion {spec.success_criterion!r}.",
                    )
                leakage_audit = cls._claim_leakage_audit_summary()
                trained_value = comparison_values.get("trained_without_reflex_support")
                trained_delta = deltas.get("trained_without_reflex_support")
                if trained_value is None or trained_delta is None:
                    return cls._claim_skip_result(
                        spec,
                        "Missing trained_without_reflex_support comparison data.",
                    )
                passed = bool(
                    trained_delta >= delta_threshold
                    and int(leakage_audit["finding_count"]) == 0
                )
                return {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "reference_value": reference_value,
                    "comparison_values": comparison_values,
                    "delta": deltas,
                    "effect_size": dict(deltas),
                    "primary_metric": spec.primary_metric,
                    "scenarios_evaluated": list(spec.scenarios),
                    "notes": [
                        spec.success_criterion,
                        f"Leakage audit unresolved findings: {leakage_audit['finding_count']}.",
                    ],
                }

            if spec.name == "escape_without_reflex_support":
                minimum_success = cls._claim_threshold_from_operator(
                    spec.success_criterion,
                    ">=",
                )
                delta_threshold = cls._claim_threshold_from_phrase(
                    spec.success_criterion,
                    "by at least",
                )
                if minimum_success is None or delta_threshold is None:
                    return cls._claim_skip_result(
                        spec,
                        f"Could not parse success criterion {spec.success_criterion!r}.",
                    )
                trained_value = comparison_values.get("trained_without_reflex_support")
                trained_delta = deltas.get("trained_without_reflex_support")
                if trained_value is None or trained_delta is None:
                    return cls._claim_skip_result(
                        spec,
                        "Missing trained_without_reflex_support comparison data.",
                    )
                passed = bool(
                    trained_value >= minimum_success
                    and trained_delta >= delta_threshold
                )
                return {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "reference_value": reference_value,
                    "comparison_values": comparison_values,
                    "delta": deltas,
                    "effect_size": dict(deltas),
                    "primary_metric": spec.primary_metric,
                    "scenarios_evaluated": list(spec.scenarios),
                    "notes": [spec.success_criterion],
                }

            return cls._claim_skip_result(
                spec,
                f"Unsupported learning-evidence claim test {spec.name!r}.",
            )

        if source == "ablation":
            payload = payloads.get("ablation")
            reference_payload, reason = cls._claim_registry_entry(
                payload,
                registry_key="variants",
                entry_name=spec.reference_condition,
            )
            if reference_payload is None:
                return cls._claim_skip_result(spec, str(reason))

            if spec.name == "memory_improves_shelter_return":
                reference_value, reason = cls._claim_subset_scenario_success_rate(
                    reference_payload,
                    scenarios=spec.scenarios,
                )
                if reference_value is None:
                    return cls._claim_skip_result(spec, str(reason))
                comparison_values: Dict[str, float] = {}
                deltas: Dict[str, float] = {}
                for comparison_name in spec.comparison_conditions:
                    comparison_payload, reason = cls._claim_registry_entry(
                        payload,
                        registry_key="variants",
                        entry_name=comparison_name,
                    )
                    if comparison_payload is None:
                        return cls._claim_skip_result(spec, str(reason))
                    comparison_value, reason = cls._claim_subset_scenario_success_rate(
                        comparison_payload,
                        scenarios=spec.scenarios,
                    )
                    if comparison_value is None:
                        return cls._claim_skip_result(spec, str(reason))
                    comparison_values[comparison_name] = comparison_value
                    deltas[comparison_name] = round(comparison_value - reference_value, 6)
                delta_threshold = cls._claim_threshold_from_phrase(
                    spec.success_criterion,
                    "by at least",
                )
                if delta_threshold is None:
                    return cls._claim_skip_result(
                        spec,
                        f"Could not parse success criterion {spec.success_criterion!r}.",
                    )
                comparison_name = spec.comparison_conditions[0]
                passed = bool(deltas.get(comparison_name, 0.0) >= delta_threshold)
                return {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "reference_value": reference_value,
                    "comparison_values": comparison_values,
                    "delta": deltas,
                    "effect_size": dict(deltas),
                    "primary_metric": spec.primary_metric,
                    "scenarios_evaluated": list(spec.scenarios),
                    "notes": [spec.success_criterion],
                }

            if spec.name == "specialization_emerges_with_multiple_predators":
                comparison_summary = compare_predator_type_ablation_performance(
                    payload or {},
                    variant_names=(spec.reference_condition, *spec.comparison_conditions),
                )
                comparisons = comparison_summary.get("comparisons", {})
                if not isinstance(comparisons, dict):
                    return cls._claim_skip_result(
                        spec,
                        "Predator-type ablation comparison did not return comparison rows.",
                    )
                reference_comparison = comparisons.get(spec.reference_condition)
                if not isinstance(reference_comparison, dict):
                    return cls._claim_skip_result(
                        spec,
                        f"Predator-type comparison is missing reference variant {spec.reference_condition!r}.",
                    )
                reference_value = reference_comparison.get(
                    "visual_minus_olfactory_success_rate"
                )
                if reference_value is None:
                    return cls._claim_skip_result(
                        spec,
                        f"Reference variant {spec.reference_condition!r} is missing "
                        "visual_minus_olfactory_success_rate.",
                    )
                comparison_values: Dict[str, float] = {}
                deltas: Dict[str, float] = {}
                effect_sizes: Dict[str, float | None] = {}
                for comparison_name in spec.comparison_conditions:
                    comparison_payload = comparisons.get(comparison_name)
                    if not isinstance(comparison_payload, dict):
                        return cls._claim_skip_result(
                            spec,
                            f"Predator-type comparison is missing {comparison_name!r}.",
                        )
                    raw_value = comparison_payload.get("visual_minus_olfactory_success_rate")
                    if raw_value is None:
                        return cls._claim_skip_result(
                            spec,
                            f"Comparison {comparison_name!r} is missing "
                            "visual_minus_olfactory_success_rate.",
                        )
                    comparison_value = round(float(raw_value), 6)
                    comparison_values[comparison_name] = comparison_value
                    deltas[comparison_name] = round(
                        comparison_value - float(reference_value),
                        6,
                    )
                    raw_effect_size = comparison_payload.get(
                        "visual_minus_olfactory_success_rate_delta"
                    )
                    effect_sizes[comparison_name] = (
                        round(float(raw_effect_size), 6)
                        if raw_effect_size is not None
                        else None
                    )
                engagement_threshold = cls._claim_count_threshold(spec.success_criterion)
                negative_threshold = cls._claim_threshold_from_operator(
                    spec.success_criterion,
                    "<=",
                )
                positive_threshold = cls._claim_threshold_from_operator(
                    spec.success_criterion,
                    ">=",
                )
                if (
                    engagement_threshold is None
                    or negative_threshold is None
                    or positive_threshold is None
                ):
                    return cls._claim_skip_result(
                        spec,
                        f"Could not parse success criterion {spec.success_criterion!r}.",
                    )
                engagement_count, engagement_pass_rates, reason = (
                    cls._claim_specialization_engagement(
                        payload,
                        variant_name=spec.reference_condition,
                        scenarios=spec.scenarios,
                    )
                )
                if engagement_count is None or engagement_pass_rates is None:
                    return cls._claim_skip_result(spec, str(reason))
                visual_drop = comparison_values.get("drop_visual_cortex")
                sensory_drop = comparison_values.get("drop_sensory_cortex")
                if visual_drop is None or sensory_drop is None:
                    return cls._claim_skip_result(
                        spec,
                        "Missing drop_visual_cortex or drop_sensory_cortex comparison data.",
                    )
                passed = bool(
                    visual_drop <= negative_threshold
                    and sensory_drop >= positive_threshold
                    and engagement_count >= engagement_threshold
                )
                return {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "reference_value": {
                        "visual_minus_olfactory_success_rate": round(
                            float(reference_value),
                            6,
                        ),
                        "type_specific_cortex_engagement_count": engagement_count,
                        "type_specific_cortex_engagement_pass_rates": engagement_pass_rates,
                    },
                    "comparison_values": comparison_values,
                    "delta": deltas,
                    "effect_size": effect_sizes,
                    "primary_metric": spec.primary_metric,
                    "scenarios_evaluated": list(spec.scenarios),
                    "notes": [spec.success_criterion],
                }

            return cls._claim_skip_result(
                spec,
                f"Unsupported ablation-backed claim test {spec.name!r}.",
            )

        if source == "noise_robustness":
            payload = payloads.get("noise_robustness")
            diagonal_score, off_diagonal_score, reason = cls._claim_noise_subset_scores(
                payload,
                scenarios=spec.scenarios,
            )
            if diagonal_score is None or off_diagonal_score is None:
                return cls._claim_skip_result(spec, str(reason))
            minimum_off_diagonal = cls._claim_threshold_from_operator(
                spec.success_criterion,
                ">=",
            )
            maximum_gap = cls._claim_threshold_from_operator(
                spec.success_criterion,
                "<=",
            )
            if minimum_off_diagonal is None or maximum_gap is None:
                return cls._claim_skip_result(
                    spec,
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            effect_size = round(diagonal_score - off_diagonal_score, 6)
            passed = bool(
                off_diagonal_score >= minimum_off_diagonal
                and effect_size <= maximum_gap
            )
            return {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": diagonal_score,
                "comparison_values": {"off_diagonal": off_diagonal_score},
                "delta": {"off_diagonal": round(off_diagonal_score - diagonal_score, 6)},
                "effect_size": effect_size,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": [spec.success_criterion],
            }

        return cls._claim_skip_result(
            spec,
            f"Unsupported claim-test source {source!r}.",
        )

    @staticmethod
    def _build_claim_test_summary(
        claim_results: Dict[str, Dict[str, object]],
    ) -> Dict[str, object]:
        """
        Summarizes a collection of claim-test results into counts and primary-claim pass status.
        
        Parameters:
            claim_results (Dict[str, Dict[str, object]]): Mapping from claim-test name to its result record.
                Each result record is expected to include a `status` field (commonly `"passed"`, `"failed"`, or `"skipped"`)
                and may include a boolean `passed` field used for primary-claim membership checks.
        
        Returns:
            Dict[str, object]: Summary dictionary with the following keys:
                - `claim_count`: total number of claim tests processed.
                - `claims_passed`: number of tests whose `status` equals `"passed"`.
                - `claims_failed`: number of tests whose `status` equals `"failed"`.
                - `claims_skipped`: number of tests whose `status` equals `"skipped"`.
                - `all_primary_claims_passed`: `true` if every executed primary claim has a truthy `passed` value in
                  `claim_results`, `false` otherwise.
                - `primary_claims`: list of canonical primary claim names derived from the claim registry.
        """
        claims_passed = sum(
            1
            for result in claim_results.values()
            if str(result.get("status")) == "passed"
        )
        claims_failed = sum(
            1
            for result in claim_results.values()
            if str(result.get("status")) == "failed"
        )
        claims_skipped = sum(
            1
            for result in claim_results.values()
            if str(result.get("status")) == "skipped"
        )
        primary_claims = primary_claim_test_names()
        executed_primary_claims = [
            name for name in primary_claims if name in claim_results
        ]
        all_primary_claims_passed = bool(executed_primary_claims) and all(
            bool(claim_results[name].get("passed", False))
            for name in executed_primary_claims
        )
        return {
            "claim_count": len(claim_results),
            "claims_passed": claims_passed,
            "claims_failed": claims_failed,
            "claims_skipped": claims_skipped,
            "all_primary_claims_passed": bool(all_primary_claims_passed),
            "primary_claims": list(primary_claims),
        }

    @classmethod
    def run_claim_test_suite(
        cls,
        *,
        claim_tests: Sequence[str] | None = None,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int | None = None,
        episodes: int | None = None,
        evaluation_episodes: int | None = None,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        brain_config: BrainAblationConfig | None = None,
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
        budget_profile: str | BudgetProfile | None = None,
        long_budget_profile: str | BudgetProfile | None = "report",
        seeds: Sequence[int] | None = None,
        episodes_per_scenario: int | None = None,
        robustness_matrix: RobustnessMatrixSpec | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_override_penalty: float = 0.0,
        checkpoint_dominance_penalty: float = 0.0,
        checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
            CheckpointPenaltyMode.TIEBREAKER
        ),
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
        ablation_payload: Dict[str, object] | None = None,
        learning_evidence_payload: Dict[str, object] | None = None,
        noise_robustness_payload: Dict[str, object] | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Run selected claim tests by synthesizing or reusing primitive comparison payloads.
        
        This method resolves requested claim-test specs, ensures required primitive comparison data (learning evidence, ablation, noise robustness) are available by reusing provided payloads or invoking the corresponding comparison routines, evaluates each claim test to produce structured pass/skip/fail results, and returns a combined claims payload plus CSV-like row records suitable for export.
        
        Parameters:
            claim_tests: Optional sequence of claim-test identifiers or specs to run; if None all canonical claim tests are evaluated.
            width, height, food_count, day_length, night_length, max_steps:
                Environment layout and step-limit overrides for any generated comparison runs.
            episodes, evaluation_episodes, episodes_per_scenario:
                Training / evaluation budget overrides used for generated primitive comparisons.
            gamma, module_lr, motor_lr, module_dropout:
                Learning hyperparameter overrides applied when running training comparisons.
            reward_profile, map_template, brain_config, operational_profile, noise_profile:
                Configuration overrides used when generating comparison payloads.
            budget_profile, long_budget_profile:
                Budget profile names for base and long-form comparisons where applicable.
            seeds:
                Sequence of RNG seeds to use for generated comparisons; when omitted comparison helpers choose defaults.
            robustness_matrix:
                Optional robustness-matrix spec to use for noise-robustness comparisons.
            checkpoint_selection, checkpoint_metric, checkpoint_interval, checkpoint_dir:
                Checkpointing controls passed through to comparison runs that support candidate selection.
            ablation_payload, learning_evidence_payload, noise_robustness_payload:
                Optional precomputed primitive payloads to reuse; when provided the method will not regenerate that source.
        
        Returns:
            tuple:
                - claims_payload (dict): A mapping with keys "claims" (per-claim result dicts), "summary" (aggregate pass/skip/fail counts and primary-claims gating), and "metadata" (requested tests, required sources, per-source metadata, seeds, noise profile mapping, and leakage-audit summary).
                - rows (list[dict]): CSV-ready row dictionaries, one per evaluated claim test, containing serialized reference/comparison values, deltas, effect sizes, evaluated scenarios, status, reason, and notes.
        """
        resolved_claim_tests = resolve_claim_tests(claim_tests)
        required_sources = {
            source
            for spec in resolved_claim_tests
            if (source := cls._claim_test_source(spec)) is not None
        }
        learning_scenarios = list(
            dict.fromkeys(
                scenario_name
                for spec in resolved_claim_tests
                if cls._claim_test_source(spec) == "learning_evidence"
                for scenario_name in spec.scenarios
            )
        )
        learning_conditions = list(
            dict.fromkeys(
                condition_name
                for spec in resolved_claim_tests
                if cls._claim_test_source(spec) == "learning_evidence"
                for condition_name in (spec.reference_condition, *spec.comparison_conditions)
            )
        )
        ablation_scenarios = list(
            dict.fromkeys(
                scenario_name
                for spec in resolved_claim_tests
                if cls._claim_test_source(spec) == "ablation"
                for scenario_name in spec.scenarios
            )
        )
        ablation_variants = list(
            dict.fromkeys(
                variant_name
                for spec in resolved_claim_tests
                if cls._claim_test_source(spec) == "ablation"
                for variant_name in (spec.reference_condition, *spec.comparison_conditions)
            )
        )
        noise_scenarios = list(
            dict.fromkeys(
                scenario_name
                for spec in resolved_claim_tests
                if cls._claim_test_source(spec) == "noise_robustness"
                for scenario_name in spec.scenarios
            )
        )

        payloads: Dict[str, Dict[str, object]] = {}
        source_reused = {
            "ablation": ablation_payload is not None,
            "learning_evidence": learning_evidence_payload is not None,
            "noise_robustness": noise_robustness_payload is not None,
        }
        if "learning_evidence" in required_sources:
            if learning_evidence_payload is None:
                learning_evidence_payload, _ = cls.compare_learning_evidence(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=max_steps,
                    episodes=episodes,
                    evaluation_episodes=evaluation_episodes,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    brain_config=brain_config,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    budget_profile=budget_profile,
                    long_budget_profile=long_budget_profile,
                    seeds=seeds,
                    names=learning_scenarios or None,
                    condition_names=learning_conditions or None,
                    episodes_per_scenario=episodes_per_scenario,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                )
            payloads["learning_evidence"] = learning_evidence_payload
        if "ablation" in required_sources:
            if ablation_payload is None:
                ablation_payload, _ = cls.compare_ablation_suite(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=max_steps,
                    episodes=episodes,
                    evaluation_episodes=evaluation_episodes,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    budget_profile=budget_profile,
                    seeds=seeds,
                    names=ablation_scenarios or None,
                    variant_names=ablation_variants or None,
                    episodes_per_scenario=episodes_per_scenario,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                )
            payloads["ablation"] = ablation_payload
        if "noise_robustness" in required_sources:
            if noise_robustness_payload is None:
                noise_robustness_payload, _ = cls.compare_noise_robustness(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=max_steps,
                    episodes=episodes,
                    evaluation_episodes=evaluation_episodes,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    operational_profile=operational_profile,
                    budget_profile=budget_profile,
                    seeds=seeds,
                    names=noise_scenarios or None,
                    episodes_per_scenario=episodes_per_scenario,
                    robustness_matrix=robustness_matrix,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=checkpoint_interval,
                    checkpoint_dir=checkpoint_dir,
                )
            payloads["noise_robustness"] = noise_robustness_payload

        claim_results: Dict[str, Dict[str, object]] = {
            spec.name: cls._evaluate_claim_test(spec, payloads)
            for spec in resolved_claim_tests
        }
        rows: List[Dict[str, object]] = []
        for spec in resolved_claim_tests:
            result = claim_results[spec.name]
            row = {
                "claim_test": spec.name,
                "claim_test_status": result.get("status"),
                "claim_test_passed": bool(result.get("passed", False)),
                "claim_test_primary_metric": result.get("primary_metric"),
                "claim_test_reference_condition": spec.reference_condition,
                "claim_test_comparison_conditions": json.dumps(
                    list(spec.comparison_conditions),
                    sort_keys=True,
                ),
                "claim_test_reference_value": json.dumps(
                    result.get("reference_value"),
                    sort_keys=True,
                ),
                "claim_test_comparison_values": json.dumps(
                    result.get("comparison_values", {}),
                    sort_keys=True,
                ),
                "claim_test_delta": json.dumps(
                    result.get("delta", {}),
                    sort_keys=True,
                ),
                "claim_test_effect_size": json.dumps(
                    result.get("effect_size"),
                    sort_keys=True,
                ),
                "claim_test_scenarios": json.dumps(
                    result.get("scenarios_evaluated", []),
                    sort_keys=True,
                ),
                "claim_test_reason": str(result.get("reason", "")),
                "claim_test_notes": json.dumps(result.get("notes", []), sort_keys=True),
            }
            rows.append(row)

        def _metadata_sequence_or_empty(value: object) -> list[object]:
            if isinstance(value, (list, tuple)):
                return list(value)
            return []

        source_metadata: Dict[str, object] = {}
        for source_name, payload in payloads.items():
            source_metadata[source_name] = {
                "reused": bool(source_reused.get(source_name, False)),
                "budget_profile": payload.get("budget_profile"),
                "benchmark_strength": payload.get("benchmark_strength"),
                "noise_profile": payload.get("noise_profile"),
                "seeds": _metadata_sequence_or_empty(payload.get("seeds")),
                "scenario_names": _metadata_sequence_or_empty(
                    payload.get("scenario_names")
                ),
                "episodes_per_scenario": payload.get("episodes_per_scenario"),
            }
        metadata = {
            "requested_claim_tests": [spec.name for spec in resolved_claim_tests],
            "required_sources": sorted(required_sources),
            "sources": source_metadata,
            "seeds": sorted(
                {
                    int(seed)
                    for payload in payloads.values()
                    for seed in _metadata_sequence_or_empty(payload.get("seeds"))
                    if isinstance(seed, int)
                }
            ),
            "noise_profiles": {
                source_name: payload.get("noise_profile")
                for source_name, payload in payloads.items()
                if payload.get("noise_profile") is not None
            },
            "leakage_audit": cls._claim_leakage_audit_summary(),
        }
        return {
            "claims": claim_results,
            "summary": cls._build_claim_test_summary(claim_results),
            "metadata": metadata,
        }, rows

    @staticmethod
    def _compact_aggregate(data: Dict[str, object]) -> Dict[str, object]:
        """
        Create a shallow copy of an aggregate metrics dictionary with per-episode details removed.
        
        Parameters:
            data (Dict[str, object]): Aggregate metrics that may contain an "episodes_detail" key.
        
        Returns:
            Dict[str, object]: A shallow copy of `data` with the "episodes_detail" entry removed if present.
        """
        compact = dict(data)
        compact.pop("episodes_detail", None)
        return compact

    @classmethod
    def _compact_behavior_payload(cls, payload: Dict[str, object]) -> Dict[str, object]:
        """
        Create a compacted version of a behavior-suite payload by removing per-episode details and compacting nested legacy metrics.
        
        Parameters:
            payload (Dict[str, object]): Behavior-suite payload containing keys "summary", "suite", and "legacy_scenarios" produced by _build_behavior_payload.
        
        Returns:
            compact (Dict[str, object]): A payload dict with:
                - "summary": shallow copy of the original summary,
                - "suite": per-scenario entries with "episodes_detail" removed and "legacy_metrics" compacted,
                - "legacy_scenarios": per-scenario legacy aggregates compacted via _compact_aggregate.
        """
        compact = {
            "summary": dict(payload.get("summary", {})),
            "suite": {},
            "legacy_scenarios": {},
        }
        suite = payload.get("suite", {})
        if isinstance(suite, dict):
            for name, data in suite.items():
                item = dict(data)
                item.pop("episodes_detail", None)
                legacy_metrics = item.get("legacy_metrics", {})
                if isinstance(legacy_metrics, dict):
                    item["legacy_metrics"] = cls._compact_aggregate(legacy_metrics)
                compact["suite"][name] = item
        legacy_scenarios = payload.get("legacy_scenarios", {})
        if isinstance(legacy_scenarios, dict):
            for name, data in legacy_scenarios.items():
                compact["legacy_scenarios"][name] = cls._compact_aggregate(dict(data))
        return compact

    @staticmethod
    def save_summary(summary: Dict[str, object], path: str | Path) -> None:
        """
        Write the simulation summary to a file as pretty-printed JSON.
        
        Parameters:
            summary (Dict[str, object]): Summary data to serialize.
            path (str | Path): Destination file path; the JSON is written with indent=2, ensure_ascii=False, using UTF-8 encoding.
        """
        path = Path(path)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def save_trace(trace: List[Dict[str, object]], path: str | Path) -> None:
        """
        Write a trace (list of dictionaries) to a file as one JSON object per line.
        
        Parameters:
        	trace (List[Dict[str, object]]): Sequence of trace items; each item will be serialized as a single JSON object on its own line.
        	path (str | Path): Filesystem path to write; file is opened with UTF-8 encoding and overwritten if it exists.
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for item in trace:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def save_behavior_csv(rows: List[Dict[str, object]], path: str | Path) -> None:
        """
        Write flattened behavior rows to a CSV file using a preferred column ordering.
        
        Preferred columns (written first) include reward/profile/map, ablation and operational metadata, reflex fields, checkpoint/source, seeds, scenario identifiers, and basic success/failure metrics; any additional keys found in `rows` are appended as extra columns in sorted order. The CSV is written as UTF-8 text and includes a header row.
        
        Parameters:
            rows (List[Dict[str, object]]): Sequence of flattened behavior rows; each dict represents one CSV row.
            path (str | Path): Destination filesystem path for the CSV file.
        """
        path = Path(path)
        preferred = [
            "reward_profile",
            "scenario_map",
            "evaluation_map",
            "ablation_variant",
            "ablation_architecture",
            *CURRICULUM_COLUMNS,
            "learning_evidence_condition",
            "learning_evidence_policy_mode",
            "learning_evidence_training_regime",
            "learning_evidence_train_episodes",
            "learning_evidence_frozen_after_episode",
            "learning_evidence_checkpoint_source",
            "learning_evidence_budget_profile",
            "learning_evidence_budget_benchmark_strength",
            "reflex_scale",
            "reflex_anneal_final_scale",
            "competence_type",
            "is_primary_benchmark",
            "eval_reflex_scale",
            "budget_profile",
            "benchmark_strength",
            "architecture_version",
            "architecture_fingerprint",
            "operational_profile",
            "operational_profile_version",
            "train_noise_profile",
            "train_noise_profile_config",
            "eval_noise_profile",
            "eval_noise_profile_config",
            "noise_profile",
            "noise_profile_config",
            "checkpoint_source",
            "simulation_seed",
            "episode_seed",
            "scenario",
            "scenario_description",
            "scenario_objective",
            "scenario_focus",
            "episode",
            "success",
            "failure_count",
            "failures",
        ]
        extra = sorted(
            {
                key
                for row in rows
                for key in row
                if key not in preferred
            }
        )
        fieldnames = preferred + extra
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
