"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from ..ablations import (
    BrainAblationConfig,
    compare_predator_type_ablation_performance,
    default_brain_config,
    resolve_ablation_configs,
    resolve_ablation_scenario_group,
)
from ..benchmark_types import SeedLevelResult, UncertaintyEstimate
from ..budget_profiles import BudgetProfile, resolve_budget
from ..checkpointing import (
    CheckpointSelectionConfig,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    mean_reward_from_behavior_payload,
    resolve_checkpoint_selection_config,
    resolve_checkpoint_load_dir,
)
from ..curriculum import (
    CURRICULUM_FOCUS_SCENARIOS,
    empty_curriculum_summary,
    validate_curriculum_profile,
)
from ..distillation import DistillationConfig
from ..export import compact_aggregate, compact_behavior_payload
from ..learning_evidence import resolve_learning_evidence_conditions
from ..memory import memory_leakage_audit
from ..metrics import (
    EpisodeStats,
    aggregate_episode_stats,
    competence_label_from_eval_reflex_scale,
    flatten_behavior_rows,
)
from ..noise import (
    NoiseConfig,
    RobustnessMatrixSpec,
    canonical_robustness_matrix,
    resolve_noise_profile,
)
from ..operational_profiles import OperationalProfile
from ..perception import observation_leakage_audit
from ..reward import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_PROFILES,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    reward_component_audit,
    reward_profile_audit,
    validate_gap_policy,
)
from ..scenarios import SCENARIO_NAMES, get_scenario
from ..simulation import EXPERIMENT_OF_RECORD_REGIME, SpiderSimulation
from ..statistics import bootstrap_confidence_interval, cohens_d
from ..training_regimes import TrainingRegimeSpec, resolve_training_regime

from ..comparison_utils import (
    aggregate_with_uncertainty,
    attach_behavior_seed_statistics,
    build_checkpoint_path,
    build_run_budget_summary,
    condition_compact_summary,
    condition_mean_reward,
    create_comparison_simulation,
    metric_seed_values_from_payload,
    noise_profile_metadata,
    paired_seed_delta_rows,
    seed_level_dicts,
)
from ..comparison_noise import condense_robustness_summary


__all__ = [name for name in globals() if not name.startswith("__")]
