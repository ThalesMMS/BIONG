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

"""Simulation runtime facade.

Multi-seed comparison workflows and comparison statistics live in
`spider_cortex_sim.comparison`; claim-test evaluation lives in
`spider_cortex_sim.claim_evaluation`.

``default_behavior_evaluation`` and ``ensure_behavior_evaluation`` are the
public homes for the default behavior-evaluation helpers previously kept in
the CLI entrypoint.
"""

EXPERIMENT_OF_RECORD_REGIME = "late_finetuning"

CAPABILITY_PROBE_SCENARIOS: tuple[str, ...] = SCENARIO_CAPABILITY_PROBE_SCENARIOS

def default_behavior_evaluation() -> dict[str, object]:
    """Create the default behavior-evaluation payload structure."""
    return {
        "suite": {},
        "legacy_scenarios": {},
        "summary": {
            "scenario_count": 0,
            "episode_count": 0,
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "competence_type": "mixed",
            "regressions": [],
        },
    }

def ensure_behavior_evaluation(summary: dict[str, object]) -> None:
    """Initialise the behavior_evaluation key in summary if absent."""
    if "behavior_evaluation" not in summary:
        summary["behavior_evaluation"] = default_behavior_evaluation()

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
