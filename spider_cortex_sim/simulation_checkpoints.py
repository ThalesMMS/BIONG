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


class SimulationCheckpointMixin:
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
