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


class SimulationEvaluationMixin:
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
