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


class SimulationSummaryMixin:
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

    def build_summary(
        self,
        training_history: List[EpisodeStats],
        evaluation_history: List[EpisodeStats],
    ) -> Dict[str, object]:
        """Package-internal API for entrypoints to build a run summary."""
        return self._build_summary(training_history, evaluation_history)
