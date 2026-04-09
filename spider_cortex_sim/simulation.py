from __future__ import annotations

import csv
import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from .ablations import (
    BrainAblationConfig,
    PROPOSAL_SOURCE_NAMES,
    default_brain_config,
    resolve_ablation_configs,
)
from .agent import SpiderBrain
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
    aggregate_behavior_scores,
    aggregate_episode_stats,
    flatten_behavior_rows,
    summarize_behavior_suite,
)
from .noise import NoiseConfig, resolve_noise_profile
from .operational_profiles import OperationalProfile, resolve_operational_profile
from .perception import observation_leakage_audit
from .predator import PREDATOR_STATES
from .reward import REWARD_PROFILES, reward_component_audit, reward_profile_audit
from .scenarios import SCENARIO_NAMES, get_scenario
from .world import ACTIONS, REWARD_COMPONENT_NAMES, SpiderWorld


CURRICULUM_PROFILE_NAMES: tuple[str, ...] = ("none", "ecological_v1")
CURRICULUM_FOCUS_SCENARIOS: tuple[str, ...] = (
    "open_field_foraging",
    "corridor_gauntlet",
    "exposed_day_foraging",
    "food_deprivation",
)


@dataclass(frozen=True)
class CurriculumPhaseDefinition:
    name: str
    training_scenarios: tuple[str, ...]
    promotion_scenarios: tuple[str, ...]
    success_threshold: float
    max_episodes: int
    min_episodes: int


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
        Initialize the simulation and configure the world, brain, messaging bus, and budget defaults.
        
        Creates or resolves the provided brain ablation configuration, operational and noise profiles, constructs SpiderWorld and SpiderBrain using the resolved settings, and initializes bookkeeping fields (seed, max steps, default map template, budget summary, checkpoint and reflex/evaluation metadata).
        
        Parameters:
            brain_config (BrainAblationConfig | None): Optional ablation-aware brain configuration. If omitted, a default is created using `module_dropout`.
            module_dropout (float): Dropout probability used to create the default brain configuration when `brain_config` is not provided.
            operational_profile (str | OperationalProfile | None): Operational profile name or explicit `OperationalProfile` instance; `None` uses the default profile. Invalid profile names raise `ValueError` during resolution.
            noise_profile (str | NoiseConfig | None): Noise profile name or explicit `NoiseConfig` instance; `None` uses the default noise profile. Invalid names raise `ValueError` during resolution.
            map_template (str): Default map template name used when preparing episodes unless overridden per-scenario.
            budget_profile_name (str): Logical budget profile identifier stored in the simulation summary.
            benchmark_strength (str): Benchmark strength identifier stored in the simulation summary.
            budget_summary (Dict[str, object] | None): Optional budget override structure; when omitted a default resolved-summary is created and stored.
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
            "mode": "flat",
            "curriculum_profile": "none",
            "resolved_budget": {
                "total_training_episodes": int(
                    self.budget_summary.get("resolved", {}).get("episodes", 0)
                ),
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
        Resolve the curriculum phases for a given curriculum profile and total episode budget.
        
        If `curriculum_profile` is "none", returns an empty list. If an unsupported profile is provided,
        raises ValueError. For supported curricula, returns a list of CurriculumPhaseDefinition records
        (each containing phase name, training and promotion scenario tuples, success threshold, and
        per-phase `max_episodes`/`min_episodes`) whose episode budgets are derived from `total_episodes`.
        
        Parameters:
            curriculum_profile (str): Identifier of the curriculum profile ("none" or "ecological_v1").
            total_episodes (int): Total number of training episodes to distribute across phases.
        
        Returns:
            list[CurriculumPhaseDefinition]: Ordered phase definitions for the requested curriculum.
        """
        profile_name = str(curriculum_profile)
        if profile_name == "none":
            return []
        if profile_name != "ecological_v1":
            raise ValueError(
                "Invalid curriculum_profile. Use 'none' or 'ecological_v1'."
            )
        budgets = cls._resolve_curriculum_phase_budgets(total_episodes)
        phase_specs = (
            (
                "phase_1_night_rest_predator_edge",
                ("night_rest", "predator_edge"),
                1.0,
            ),
            (
                "phase_2_entrance_ambush_shelter_blockade",
                ("entrance_ambush", "shelter_blockade"),
                1.0,
            ),
            (
                "phase_3_open_field_exposed_day",
                ("open_field_foraging", "exposed_day_foraging"),
                0.5,
            ),
            (
                "phase_4_corridor_food_deprivation",
                ("corridor_gauntlet", "food_deprivation"),
                0.5,
            ),
        )
        phases: list[CurriculumPhaseDefinition] = []
        for budget, (name, scenarios, threshold) in zip(
            budgets, phase_specs, strict=True
        ):
            phases.append(
                CurriculumPhaseDefinition(
                    name=name,
                    training_scenarios=tuple(scenarios),
                    promotion_scenarios=tuple(scenarios),
                    success_threshold=float(threshold),
                    max_episodes=int(budget),
                    min_episodes=max(1, int(budget) // 2) if int(budget) > 0 else 0,
                )
            )
        return phases

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
        """Expose training-regime metadata in behavior CSV rows."""
        latest_phase: Dict[str, object] | None = None
        if isinstance(curriculum_summary, dict):
            phases = curriculum_summary.get("phases", [])
            if isinstance(phases, list) and phases:
                for phase in reversed(phases):
                    if not isinstance(phase, dict):
                        continue
                    if int(phase.get("episodes_executed", 0)) > 0:
                        latest_phase = phase
                        break
                if latest_phase is None:
                    phase = phases[-1]
                    if isinstance(phase, dict):
                        latest_phase = phase
        return {
            "training_regime": str(training_regime.get("mode", "flat")),
            "curriculum_profile": str(
                training_regime.get("curriculum_profile", "none")
            ),
            "curriculum_phase": (
                str(latest_phase.get("name", "")) if latest_phase is not None else ""
            ),
            "curriculum_phase_status": (
                str(latest_phase.get("status", ""))
                if latest_phase is not None
                else ""
            ),
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
        }
        for module_name in PROPOSAL_SOURCE_NAMES:
            metrics[f"module_contribution_{module_name}"] = float(
                stats.module_contribution_share.get(module_name, 0.0)
            )
        return metrics

    def _set_training_regime_metadata(
        self,
        *,
        curriculum_profile: str,
        episodes: int,
        curriculum_summary: Dict[str, object] | None = None,
    ) -> None:
        """Update cached regime metadata used by summaries and CSV exports."""
        phase_budgets: list[int] = []
        if curriculum_profile != "none":
            phase_budgets = self._resolve_curriculum_phase_budgets(episodes)
        self._latest_curriculum_summary = (
            deepcopy(curriculum_summary) if curriculum_summary is not None else None
        )
        self._latest_training_regime_summary = {
            "mode": "curriculum" if curriculum_profile != "none" else "flat",
            "curriculum_profile": str(curriculum_profile),
            "resolved_budget": {
                "total_training_episodes": int(episodes),
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
        Execute a single simulation episode and return aggregated episode statistics plus an optional per-tick trace.
        
        When `capture_trace` is True the trace contains one dict per tick with tick metadata, environment state, rewards, action/result flags, and bus messages; when `debug_trace` is also True trace items include expanded diagnostic fields (serialized observations, per-module reflex diagnostics, decision payloads, and predator internal state).
        
        Parameters:
            episode_index (int): Index used to derive the episode RNG seed and to tag trace items.
            training (bool): If True, enable learning updates during the episode.
            sample (bool): If True, allow stochastic action sampling; otherwise select the policy argmax.
            render (bool): If True, render the world each tick (text output).
            capture_trace (bool): If True, collect and return a list of per-tick trace dictionaries.
            scenario_name (str | None): Optional scenario identifier; when provided the scenario may override the map template and max steps.
            debug_trace (bool): When True and `capture_trace` is True, include expanded diagnostic fields in each trace item.
            policy_mode (str): Inference mode passed to the brain; valid values are `"normal"` or `"reflex_only"`.
        
        Returns:
            tuple[EpisodeStats, List[Dict[str, object]]]: Aggregated episode statistics and the per-tick trace list (an empty list when `capture_trace` is False).
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

            decision = self.brain.act(
                observation,
                self.bus,
                sample=sample,
                policy_mode=policy_mode,
            )
            metrics.record_decision(decision)
            predator_state_before = self.world.lizard.mode
            next_observation, reward, done, info = self.world.step(decision.action_idx)
            learn_stats: Dict[str, float] = {}
            if training:
                learn_stats = self.brain.learn(decision, reward, next_observation, done)
                self.bus.publish(sender="learning", topic="td_update", payload=learn_stats)

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
                            "selected_action": ACTIONS[decision.motor_action_idx],
                            "executed_action": info.get("executed_action", ACTIONS[decision.action_idx]),
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
        """Attach resolved noise-profile metadata to a returned payload."""
        enriched = dict(payload)
        enriched.update(cls._noise_profile_metadata(noise_profile))
        return enriched

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
        eval_reflex_scale: float | None = None,
    ) -> Dict[str, object]:
        """
        Create a checkpoint candidate by saving the current brain state and evaluating its behavior on the specified scenarios.
        
        Parameters:
            root_dir (Path): Directory where the checkpoint directory will be created.
            episode (int): Episode index used to name the checkpoint.
            scenario_names (Sequence[str]): Sequence of scenario names to evaluate the checkpoint against.
            metric (str): Primary metric name to attach to the candidate metadata.
            selection_scenario_episodes (int): Number of evaluation episodes to run per scenario when generating metrics.
            eval_reflex_scale (float | None): Optional reflex scale override used only while scoring this candidate.
        
        Returns:
            candidate (Dict[str, object]): Metadata for the checkpoint candidate containing:
                - `name` (str): Checkpoint directory name (e.g., "episode_00042").
                - `episode` (int): Episode index.
                - `path` (Path): Path to the saved checkpoint directory.
                - `metric` (str): Provided metric name.
                - `scenario_success_rate` (float): Aggregated scenario success rate from the evaluation summary.
                - `episode_success_rate` (float): Aggregated episode success rate from the evaluation summary.
                - `mean_reward` (float): Mean reward computed from the behavior payload.
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
        summary = payload.get("summary", {})
        candidate = {
            "name": checkpoint_name,
            "episode": int(episode),
            "path": checkpoint_path,
            "metric": str(metric),
            "scenario_success_rate": float(summary.get("scenario_success_rate", 0.0)),
            "episode_success_rate": float(summary.get("episode_success_rate", 0.0)),
            "mean_reward": self._mean_reward_from_behavior_payload(payload),
        }
        return candidate

    @staticmethod
    def _checkpoint_candidate_sort_key(
        candidate: Dict[str, object],
        *,
        primary_metric: str,
    ) -> tuple[float, float, float, int]:
        """
        Produce a sort key tuple for ordering checkpoint candidates by a prioritized metric.
        
        Parameters:
        	candidate (Dict[str, object]): Mapping containing candidate metadata; expected keys include metric names ('scenario_success_rate', 'episode_success_rate', 'mean_reward') and 'episode'.
        	primary_metric (str): Metric name to prioritize; must be one of 'scenario_success_rate', 'episode_success_rate', or 'mean_reward'.
        
        Returns:
        	tuple[float, float, float, int]: A 4-tuple (primary, secondary, tertiary, episode) where the first three elements are the metric values (converted to float, default 0.0) in order of priority and the final element is the episode number (converted to int, default 0).
        
        Raises:
        	ValueError: If `primary_metric` is not one of the accepted metric names.
        """
        metric_order = ["scenario_success_rate", "episode_success_rate", "mean_reward"]
        if primary_metric not in metric_order:
            raise ValueError(
                "Invalid checkpoint_metric. Use 'scenario_success_rate', "
                "'episode_success_rate' or 'mean_reward'."
            )
        ordered_metrics = [primary_metric] + [
            metric_name for metric_name in metric_order if metric_name != primary_metric
        ]
        return (
            float(candidate.get(ordered_metrics[0], 0.0)),
            float(candidate.get(ordered_metrics[1], 0.0)),
            float(candidate.get(ordered_metrics[2], 0.0)),
            int(candidate.get("episode", 0)),
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
        curriculum_profile: str = "none",
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> List[EpisodeStats]:
        """
        Run the configured training schedule either as flat training or following a curriculum.
        
        When `curriculum_profile` is "none" the function performs flat training for `episodes` episodes.
        When a curriculum profile is selected it executes phase-by-phase training, performs promotion
        checks using the phase's promotion scenarios, and may carry over unused phase budget to the
        next phase. The optional `checkpoint_callback`, if provided, is invoked with the 1-based count
        of completed training episodes after each training episode.
        
        Parameters:
            episodes (int): Total number of training episodes to execute (non-negative).
            episode_start (int): Base index added to per-episode indices reported to run_episode.
            reflex_anneal_final_scale (float | None): If provided, clamps to >= 0.0 and becomes the
                reflex scale applied at the end of training; if None, uses the brain's configured reflex scale.
            curriculum_profile (str): Curriculum identifier; supported values are "none" and "ecological_v1".
            checkpoint_callback (Callable[[int], None] | None): Optional callback called after each completed
                training episode with the completed episode count (1-based) to support checkpoint/candidate capture.
        
        Returns:
            training_history (List[EpisodeStats]): List of per-episode statistics collected in execution order.
        """
        training_history: List[EpisodeStats] = []
        total_episodes = max(0, int(episodes))
        final_training_reflex_scale = (
            float(self.brain.config.reflex_scale)
            if reflex_anneal_final_scale is None
            else max(0.0, float(reflex_anneal_final_scale))
        )
        profile_name = str(curriculum_profile)
        if profile_name not in CURRICULUM_PROFILE_NAMES:
            raise ValueError(
                "Invalid curriculum_profile. Use 'none' or 'ecological_v1'."
            )

        if profile_name == "none":
            self._set_training_regime_metadata(
                curriculum_profile="none",
                episodes=total_episodes,
            )
            for episode in range(total_episodes):
                self._set_training_episode_reflex_scale(
                    episode_index=episode,
                    total_episodes=total_episodes,
                    final_scale=final_training_reflex_scale,
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
                "training_scenarios": list(phase.training_scenarios),
                "promotion_scenarios": list(phase.promotion_scenarios),
                "success_threshold": float(phase.success_threshold),
                "max_episodes": int(phase.max_episodes),
                "min_episodes": int(phase.min_episodes),
                "allocated_episodes": int(allocated_episodes),
                "carryover_in": carryover_in,
                "episodes_executed": 0,
                "status": "pending",
                "promotion_checks": [],
                "final_metrics": {},
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

                microeval_payload, _, _ = self.evaluate_behavior_suite(
                    list(phase.promotion_scenarios),
                    episodes_per_scenario=1,
                    capture_trace=False,
                    debug_trace=False,
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
                phase_record["promotion_checks"].append(check)
                phase_record["final_metrics"] = dict(check)
                if check["scenario_success_rate"] >= phase.success_threshold:
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
        )
        self.brain.set_runtime_reflex_scale(final_training_reflex_scale)
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
        evaluation_reflex_scale: float | None = None,
        curriculum_profile: str = "none",
        preserve_training_metadata: bool = False,
    ) -> tuple[List[EpisodeStats], List[EpisodeStats], List[Dict[str, object]]]:
        """
        Execute a training phase of episodes followed by an evaluation phase and collect per-episode statistics and an optional evaluation trace.
        
        During training, the runtime reflex scale is annealed per episode toward `reflex_anneal_final_scale` when provided (clamped to >= 0.0); after training the brain's runtime reflex scale is set to that final scale. If `evaluation_reflex_scale` is provided, it is applied for the evaluation runs and restored afterward. Only the last evaluation run that produces a non-empty trace is returned when trace capture is enabled.
        
        Parameters:
            episode_start (int): Base index added to episode indices when running episodes.
            reflex_anneal_final_scale (float | None): Final reflex scale to anneal to during training; if `None`, the brain's configured reflex scale is used.
            evaluation_reflex_scale (float | None): If provided, temporarily override the brain's runtime reflex scale for the evaluation phase.
        
        Returns:
            tuple: 
                - training_history (List[EpisodeStats]): Per-episode statistics for all training episodes.
                - evaluation_history (List[EpisodeStats]): Per-episode statistics for all evaluation episodes.
                - evaluation_trace (List[Dict[str, object]]): Trace captured from the last non-empty evaluation run when capture is enabled, or an empty list.
        """
        training_history: List[EpisodeStats] = []
        if episodes > 0 or not preserve_training_metadata:
            training_history = self._execute_training_schedule(
                episodes=episodes,
                episode_start=episode_start,
                reflex_anneal_final_scale=reflex_anneal_final_scale,
                curriculum_profile=curriculum_profile,
            )
        final_training_reflex_scale = (
            float(self.brain.config.reflex_scale)
            if reflex_anneal_final_scale is None
            else max(0.0, float(reflex_anneal_final_scale))
        )

        evaluation_history: List[EpisodeStats] = []
        evaluation_trace: List[Dict[str, object]] = []
        previous_reflex_scale = float(self.brain.current_reflex_scale)
        if evaluation_reflex_scale is not None:
            self.brain.set_runtime_reflex_scale(evaluation_reflex_scale)
        try:
            for i in range(evaluation_episodes):
                stats, trace = self.run_episode(
                    episode_start + episodes + i,
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
    ) -> float:
        """
        Compute and apply the per-episode runtime reflex scale used during training.
        
        The runtime reflex scale is chosen by linearly interpolating from the brain's configured
        reflex scale to `final_scale` across `total_episodes` (if `total_episodes > 1`), then
        applied to the brain via `brain.set_runtime_reflex_scale`.
        
        Parameters:
        	episode_index (int): Zero-based index of the current training episode.
        	total_episodes (int): Total number of training episodes over which to anneal.
        	final_scale (float): Desired reflex scale at the final episode (values < 0 are clamped to 0.0).
        
        Returns:
        	scale (float): The reflex scale applied for this episode.
        """
        start_scale = float(self.brain.config.reflex_scale)
        target_scale = max(0.0, float(final_scale))
        if total_episodes <= 1:
            scale = start_scale
        else:
            progress = float(episode_index) / float(total_episodes - 1)
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

    def _consume_episodes_without_learning(
        self,
        *,
        episodes: int,
        episode_start: int = 0,
        policy_mode: str = "normal",
    ) -> List[EpisodeStats]:
        """
        Run exploratory episodes without updating weights, consuming budget after training is frozen.

        Parameters:
            episodes (int): Number of non-learning episodes to execute.
            episode_start (int): Base episode index used to derive deterministic per-episode seeds.
            policy_mode (str): Inference path passed to `run_episode()`.

        Returns:
            List[EpisodeStats]: Per-episode statistics for the consumed episodes.
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
        reflex_anneal_final_scale: float | None = None,
        curriculum_profile: str = "none",
    ) -> tuple[List[EpisodeStats], List[EpisodeStats], List[Dict[str, object]]]:
        """
        Selects checkpoint candidates during training, evaluates them to pick the best checkpoint, loads that checkpoint, and runs the final evaluation pass.
        
        Parameters:
            episodes (int): Number of training episodes used to generate checkpoint candidates.
            evaluation_episodes (int): Number of evaluation episodes to run after selecting and loading the best checkpoint.
            render_last_evaluation (bool): Render the final evaluation episode when True.
            capture_evaluation_trace (bool): Capture a trace for the final evaluation episodes when True.
            debug_trace (bool): Include detailed debug information in captured traces when True.
            checkpoint_metric (str): Primary metric name used to rank checkpoint candidates (e.g., "scenario_success_rate").
            checkpoint_interval (int): Frequency (in completed episodes) at which to create/evaluate checkpoint candidates; treated as at least 1.
            checkpoint_dir (str | Path | None): Optional destination directory to persist the selected "best" and the "last" checkpoints; if None, no persistence is performed.
            checkpoint_scenario_names (Sequence[str] | None): Sequence of scenario names to use when evaluating candidates; if None, all scenarios are used.
            selection_scenario_episodes (int): Number of evaluation episodes per scenario when assessing a checkpoint candidate; treated as at least 1.
            reflex_anneal_final_scale (float | None): Final reflex runtime scale to anneal to during training; if None, the brain's configured reflex_scale is used. Values are clamped to be >= 0.0.
        
        Returns:
            tuple[List[EpisodeStats], List[EpisodeStats], List[Dict[str, object]]]:
                - training_history: Episode statistics collected during the training phase while generating candidates.
                - evaluation_history: Episode statistics from the final evaluation pass after loading the selected checkpoint.
                - evaluation_trace: Captured trace from the final evaluation episodes (may be an empty list if tracing was disabled).
        """
        scenario_names = list(checkpoint_scenario_names or SCENARIO_NAMES)
        interval = max(1, int(checkpoint_interval))
        selection_runs = max(1, int(selection_scenario_episodes))
        training_history: List[EpisodeStats] = []
        self.checkpoint_source = "best"
        final_training_reflex_scale = (
            float(self.brain.config.reflex_scale)
            if reflex_anneal_final_scale is None
            else max(0.0, float(reflex_anneal_final_scale))
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
                        eval_reflex_scale=final_training_reflex_scale,
                    )
                )
            def maybe_capture_candidate(completed_episode: int) -> None:
                should_checkpoint = (
                    completed_episode % interval == 0 or completed_episode == episodes
                )
                if should_checkpoint:
                    candidates.append(
                        self._capture_checkpoint_candidate(
                            root_dir=checkpoint_root,
                            episode=completed_episode,
                            scenario_names=scenario_names,
                            metric=checkpoint_metric,
                            selection_scenario_episodes=selection_runs,
                            eval_reflex_scale=final_training_reflex_scale,
                        )
                    )

            training_history = self._execute_training_schedule(
                episodes=episodes,
                episode_start=0,
                reflex_anneal_final_scale=final_training_reflex_scale,
                curriculum_profile=curriculum_profile,
                checkpoint_callback=maybe_capture_candidate,
            )

            if not candidates:
                candidates.append(
                    self._capture_checkpoint_candidate(
                        root_dir=checkpoint_root,
                        episode=max(0, int(episodes)),
                        scenario_names=scenario_names,
                        metric=checkpoint_metric,
                        selection_scenario_episodes=selection_runs,
                        eval_reflex_scale=final_training_reflex_scale,
                    )
                )

            best_candidate = max(
                candidates,
                key=lambda item: self._checkpoint_candidate_sort_key(
                    item,
                    primary_metric=checkpoint_metric,
                ),
            )
            last_candidate = max(candidates, key=lambda item: int(item.get("episode", 0)))
            self.brain.load(best_candidate["path"])
            persisted = self._persist_checkpoint_pair(
                checkpoint_dir=checkpoint_dir,
                best_candidate=best_candidate,
                last_candidate=last_candidate,
            )
            generated = [
                {
                    "name": str(candidate["name"]),
                    "episode": int(candidate["episode"]),
                    "scenario_success_rate": float(candidate["scenario_success_rate"]),
                    "episode_success_rate": float(candidate["episode_success_rate"]),
                    "mean_reward": float(candidate["mean_reward"]),
                }
                for candidate in candidates
            ]
            selected_summary = {
                "name": str(best_candidate["name"]),
                "episode": int(best_candidate["episode"]),
                "scenario_success_rate": float(best_candidate["scenario_success_rate"]),
                "episode_success_rate": float(best_candidate["episode_success_rate"]),
                "mean_reward": float(best_candidate["mean_reward"]),
            }
            if "best" in persisted:
                selected_summary["path"] = persisted["best"]
            self._latest_checkpointing_summary = {
                "enabled": True,
                "selection": "best",
                "metric": str(checkpoint_metric),
                "checkpoint_interval": interval,
                "evaluation_source": "behavior_suite",
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
            curriculum_profile=curriculum_profile,
            preserve_training_metadata=True,
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
        reflex_anneal_final_scale: float | None = None,
        curriculum_profile: str = "none",
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Run training episodes, optionally perform checkpoint selection, and evaluate the trained agent.
        
        Parameters:
            episodes (int): Number of training episodes to execute.
            evaluation_episodes (int): Number of evaluation episodes to run after training.
            render_last_evaluation (bool): Render the environment on the final evaluation episode when True.
            capture_evaluation_trace (bool): Capture and return a per-tick trace for the final evaluation episode when True.
            debug_trace (bool): Include expanded diagnostic fields in captured traces when True.
            checkpoint_selection (str): Either "none" to skip candidate selection or "best" to select the best checkpoint.
            checkpoint_metric (str): Primary metric used to rank checkpoint candidates when `checkpoint_selection` is "best".
            checkpoint_interval (int | None): Episode interval for capturing checkpoint candidates; when None the resolved budget default is used.
            checkpoint_dir (str | Path | None): Directory in which to persist selected "best" and "last" checkpoints; persistence is skipped when None.
            checkpoint_scenario_names (Sequence[str] | None): Scenario names used to evaluate checkpoint candidates; defaults are used when None.
            selection_scenario_episodes (int): Episodes per scenario used when evaluating checkpoint candidates.
            reflex_anneal_final_scale (float | None): If provided, linearly anneal the runtime reflex scale from the configured start value to this final scale during training (clamped to >= 0.0). When None, no annealing occurs.
        
        Returns:
            summary (Dict[str, object]): Consolidated training and evaluation summary suitable for serialization.
            evaluation_trace (List[Dict[str, object]]): Captured trace for the final evaluation run when `capture_evaluation_trace` is True; otherwise an empty list.
        
        Raises:
            ValueError: If `checkpoint_selection` is not one of "none" or "best", or if `checkpoint_metric` is invalid when `checkpoint_selection` is "best".
        """
        if checkpoint_selection not in {"none", "best"}:
            raise ValueError(
                "Invalid checkpoint_selection. Use 'none' or 'best'."
            )
        if curriculum_profile not in CURRICULUM_PROFILE_NAMES:
            raise ValueError(
                "Invalid curriculum_profile. Use 'none' or 'ecological_v1'."
            )
        if checkpoint_selection == "best":
            # Intentionally call _checkpoint_candidate_sort_key() with an empty
            # candidate to validate checkpoint_metric early; invalid metrics
            # raise ValueError here before any training or checkpoint work runs.
            self._checkpoint_candidate_sort_key(
                {},
                primary_metric=checkpoint_metric,
            )
        self._set_runtime_budget(
            episodes=episodes,
            evaluation_episodes=evaluation_episodes,
            checkpoint_interval=checkpoint_interval,
        )
        self._latest_checkpointing_summary = None
        start_reflex_scale = float(self.brain.config.reflex_scale)
        final_training_reflex_scale = (
            start_reflex_scale
            if reflex_anneal_final_scale is None
            else max(0.0, float(reflex_anneal_final_scale))
        )
        self._latest_reflex_schedule_summary = {
            "enabled": episodes > 0 and abs(final_training_reflex_scale - start_reflex_scale) > 1e-8,
            "start_scale": start_reflex_scale,
            "final_scale": final_training_reflex_scale,
            "episodes": int(episodes),
            "mode": "linear",
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
                    reflex_anneal_final_scale=final_training_reflex_scale,
                    curriculum_profile=curriculum_profile,
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
                curriculum_profile=curriculum_profile,
            )
        summary = self._build_summary(training_history, evaluation_history)
        if evaluation_episodes > 0 and self.brain.config.is_modular:
            _, evaluation_without_reflex_support, _ = self._train_histories(
                episodes=0,
                evaluation_episodes=evaluation_episodes,
                render_last_evaluation=False,
                capture_evaluation_trace=False,
                debug_trace=False,
                episode_start=episodes,
                reflex_anneal_final_scale=final_training_reflex_scale,
                evaluation_reflex_scale=0.0,
                curriculum_profile=curriculum_profile,
                preserve_training_metadata=True,
            )
            self._latest_evaluation_without_reflex_support = {
                "eval_reflex_scale": 0.0,
                "summary": self._aggregate_group(evaluation_without_reflex_support),
                "delta_vs_default_eval": {
                    "mean_reward_delta": round(
                        float(
                            self._aggregate_group(evaluation_without_reflex_support)["mean_reward"]
                            - summary["evaluation"]["mean_reward"]
                        ),
                        6,
                    ),
                    "scenario_reflex_usage_delta": round(
                        float(
                            self._aggregate_group(evaluation_without_reflex_support)["mean_reflex_usage_rate"]
                            - summary["evaluation"]["mean_reflex_usage_rate"]
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
    ) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
        """
        Evaluate a set of behavior scenarios and produce aggregated metrics, an optional execution trace, and flattened CSV-ready rows.
        
        Parameters:
            names (List[str] | None): Sequence of scenario names to evaluate; evaluates all default scenarios when None.
            episodes_per_scenario (int): Number of episodes to run for each scenario.
            capture_trace (bool): If True, capture and return the public execution trace from the final run (the last episode of the last scenario); otherwise no trace is captured.
            debug_trace (bool): If True and a trace is captured, include debug-level details in each trace item.
            eval_reflex_scale (float | None): If provided, temporarily override the brain's runtime reflex scale for the evaluation runs; the previous scale is restored afterward.
            policy_mode (str): Inference path passed to run_episode() for the behavior-suite runs.
            extra_row_metadata (Dict[str, object] | None): Optional metadata merged into every flattened row after the standard simulation annotations.
        
        Returns:
            payload (Dict[str, object]): Behavior-suite payload containing `suite` (per-scenario behavioral summaries), `summary` (aggregated suite summary), and `legacy_scenarios` (legacy aggregated episode statistics per scenario).
            trace (List[Dict[str, object]]): Captured public trace from the final run when `capture_trace` is True; empty list otherwise.
            rows (List[Dict[str, object]]): Flattened per-episode behavior rows suitable for CSV export, annotated with evaluation/reflex metadata and merged `extra_row_metadata` when provided.
        """
        previous_reflex_scale = float(self.brain.current_reflex_scale)
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
        )
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
                )
            )
        return payload, trace, self._annotate_behavior_rows(
            rows,
            eval_reflex_scale=eval_reflex_scale,
            extra_metadata=extra_row_metadata,
        )

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
    ) -> Dict[str, object]:
        """
        Builds a structured behavior-suite payload combining per-episode behavioral scores with legacy aggregated episode metrics.
        
        Parameters:
            stats_histories (Dict[str, List[EpisodeStats]]): Mapping from scenario name to the list of legacy episode statistics for that scenario.
            behavior_histories (Dict[str, List[BehavioralEpisodeScore]]): Mapping from scenario name to the list of per-episode behavioral score objects for that scenario.
        
        Returns:
            Dict[str, object]: A payload containing:
                - "suite": dict mapping each scenario name to its aggregated behavioral scoring summary (includes description, objective, check results, and legacy metrics).
                - "summary": an overall summary of the suite produced by summarize_behavior_suite(suite).
                - "legacy_scenarios": dict mapping each scenario name to its legacy aggregated episode metrics (compactable via _compact_aggregate).
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
            "summary": summarize_behavior_suite(suite),
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
        
        The summary includes a configuration snapshot (seed, max_steps, brain architecture and config, world parameters, learning hyperparameters, operational and noise profile summaries, and a deep-copied budget), aggregated training metrics (full history and last-window up to 10 episodes), aggregated evaluation metrics, model parameter norms, and optional entries for reflex schedule, checkpointing, and evaluation without reflex support when available.
        
        Returns:
            summary (dict): Mapping with keys:
                - "config": configuration snapshot
                - "training": aggregated metrics for the full training history
                - "training_last_window": aggregated metrics for up to the last 10 training episodes
                - "evaluation": aggregated metrics for the evaluation history
                - "parameter_norms": model parameter norms from the brain
                - "checkpointing" (optional): latest checkpointing summary
                - "evaluation_without_reflex_support" (optional): evaluation summary with reflexes disabled
        """
        last_window = training_history[-min(10, len(training_history)) :]
        summary = {
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
                    "motor_lr": self.brain.motor_lr,
                    "module_dropout": self.brain.module_dropout,
                },
                "operational_profile": self.operational_profile.to_summary(),
                "noise_profile": self.noise_profile.to_summary(),
                "budget": deepcopy(self.budget_summary),
                "training_regime": deepcopy(self._latest_training_regime_summary),
            },
            "reward_audit": self._build_reward_audit(
                current_profile=self.world.reward_profile
            ),
            "training": self._aggregate_group(training_history),
            "training_last_window": self._aggregate_group(last_window),
            "evaluation": self._aggregate_group(evaluation_history),
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
        extra_metadata: Dict[str, object] | None = None,
    ) -> List[Dict[str, object]]:
        """
        Annotates each flattened behavior row with simulation and configuration metadata.
        
        Parameters:
            rows (List[Dict[str, object]]): Flattened behavior rows to annotate.
            eval_reflex_scale (float | None): Reflex runtime scale to record for these rows; when None the effective runtime reflex scale is recorded.
            extra_metadata (Dict[str, object] | None): Additional key/value pairs merged into each annotated row after the standard fields.
        
        Returns:
            List[Dict[str, object]]: Shallow copies of the input rows augmented with simulation/configuration fields such as:
                ablation_variant, ablation_architecture, budget_profile, benchmark_strength,
                architecture_version, architecture_fingerprint, operational_profile,
                operational_profile_version, noise_profile, noise_profile_config,
                checkpoint_source, reflex_scale, reflex_anneal_final_scale, eval_reflex_scale,
                plus any keys from `extra_metadata`.
        """
        annotated: List[Dict[str, object]] = []
        architecture_fingerprint = self.brain._architecture_fingerprint()
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
            item["noise_profile"] = self.noise_profile.name
            item["noise_profile_config"] = self._noise_profile_csv_value(self.noise_profile)
            item["checkpoint_source"] = self.checkpoint_source
            item["reflex_scale"] = float(self.brain.config.reflex_scale)
            item["reflex_anneal_final_scale"] = (
                float(self._latest_reflex_schedule_summary["final_scale"])
                if self._latest_reflex_schedule_summary is not None
                else float(self.brain.config.reflex_scale)
            )
            item["eval_reflex_scale"] = self._effective_reflex_scale(
                eval_reflex_scale
                if eval_reflex_scale is not None
                else self.brain.current_reflex_scale
            )
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
        Build a compact comparison of reward profiles and their metrics, and compute deltas versus the "austere" baseline when present.
        
        Parameters:
            comparison_payload (dict | None): The reward-audit comparison payload expected to contain a
                "reward_profiles" mapping of profile names to per-profile payloads (each a dict).
        
        Returns:
            dict | None: A summary dict with the following keys, or `None` if input is missing or invalid:
              - "minimal_profile": name of the austere baseline profile if present, else `None`.
              - "profiles": mapping of profile name -> rounded metrics (`scenario_success_rate`,
                `episode_success_rate`, `mean_reward`) with values rounded to 6 decimal places.
              - "deltas_vs_minimal": mapping of profile name -> deltas vs the minimal profile (same
                metric keys, rounded to 6 decimals). Empty if no austere baseline is available.
              - "notes": list of human-readable notes about how metrics were derived.
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
            "notes": [
                "scenario_success_rate and episode_success_rate mirror the existing comparison payload.",
                "mean_reward is derived from the summary when available or from the corresponding compact aggregate.",
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
            reference_condition (str): Name of the condition to use as the primary trained reference.
        
        Returns:
            Dict[str, object]: Summary containing:
                - reference_condition: the provided reference name.
                - primary_gate_metric: the metric used for the primary evidence gate ("scenario_success_rate").
                - supports_primary_evidence: `true` if `reference_condition`, `random_init`, and `reflex_only` are present and not skipped.
                - has_learning_evidence: `true` if `scenario_success_rate` for the reference exceeds both `random_init` and `reflex_only`.
                - trained_final, random_init, reflex_only, trained_without_reflex_support: compact summaries for each condition (zeroed defaults when missing).
                - trained_vs_random_init, trained_vs_reflex_only: per-metric deltas (`scenario_success_rate`, `episode_success_rate`, `mean_reward`) computed as reference minus comparator and rounded to 6 decimals.
                - notes: list of explanatory messages about gating, supporting metrics, and reflex-only availability.
        """
        trained_final = cls._condition_compact_summary(conditions.get(reference_condition))
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
                trained_final["scenario_success_rate"]
                - random_init["scenario_success_rate"],
                6,
            ),
            "episode_success_rate_delta": round(
                trained_final["episode_success_rate"]
                - random_init["episode_success_rate"],
                6,
            ),
            "mean_reward_delta": round(
                trained_final["mean_reward"] - random_init["mean_reward"],
                6,
            ),
        }
        trained_vs_reflex = {
            "scenario_success_rate_delta": round(
                trained_final["scenario_success_rate"]
                - reflex_only["scenario_success_rate"],
                6,
            ),
            "episode_success_rate_delta": round(
                trained_final["episode_success_rate"]
                - reflex_only["episode_success_rate"],
                6,
            ),
            "mean_reward_delta": round(
                trained_final["mean_reward"] - reflex_only["mean_reward"],
                6,
            ),
        }
        notes = [
            "Primary evidence uses only scenario_success_rate as the gate.",
            "episode_success_rate and mean_reward are included only as supporting documentation.",
            "trained_without_reflex_support measures residual reflex dependence and does not participate in the primary gate.",
        ]
        if not reflex_only_available:
            notes.append(
                "The reflex_only condition is not available for the current architecture."
            )
        has_learning_evidence = (
            primary_supported
            and trained_final["scenario_success_rate"] > random_init["scenario_success_rate"]
            and trained_final["scenario_success_rate"] > reflex_only["scenario_success_rate"]
        )
        return {
            "reference_condition": reference_condition,
            "primary_gate_metric": "scenario_success_rate",
            "supports_primary_evidence": bool(primary_supported),
            "has_learning_evidence": bool(has_learning_evidence),
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
    def compare_training_regimes(
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
        episodes_per_scenario: int | None = None,
        checkpoint_selection: str = "none",
        checkpoint_metric: str = "scenario_success_rate",
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
        curriculum_profile: str = "ecological_v1",
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Compare flat and curriculum training regimes by training each under the same resolved budget and seeds, then evaluating their behavior across a scenario suite.
        
        Trains a "flat" regime and a "curriculum" regime (using the provided curriculum_profile) per seed, evaluates both on the same scenarios and episodes-per-scenario, and aggregates per-scenario episode statistics, behavioral scores, annotated CSV rows, and computed deltas comparing curriculum vs flat.
        
        Returns:
            result_payload (Dict[str, object]): Aggregated comparison payload containing budget and seed metadata, per-regime compact behavior payloads under "regimes", computed deltas vs the flat reference under "deltas_vs_flat", focus summaries, and noise profile metadata.
            rows (List[Dict[str, object]]): A flattened, annotated list of per-episode/behavior rows suitable for CSV export.
        """
        if curriculum_profile == "none":
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
                "compare_training_regimes() requer pelo menos uma seed."
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
        regime_training_metadata: Dict[str, Dict[str, object]] = {}
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
                        curriculum_profile=curriculum_profile,
                        episodes=int(budget.episodes),
                        curriculum_summary=cls._empty_curriculum_summary(
                            curriculum_profile,
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
                        checkpoint_interval=budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=budget.selection_scenario_episodes,
                        curriculum_profile=(
                            curriculum_profile if regime == "curriculum" else "none"
                        ),
                    )
                regime_training_metadata[regime] = deepcopy(
                    sim._latest_training_regime_summary
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
            regime_payloads[regime]["training_regime"] = deepcopy(
                regime_training_metadata.get(regime, {})
            )
            regime_payloads[regime]["episode_allocation"] = {
                "total_training_episodes": int(budget.episodes),
                "evaluation_episodes": int(budget.eval_episodes),
                "episodes_per_scenario": int(run_count),
            }
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
            "curriculum_profile": curriculum_profile,
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
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Compare ablation variants of the brain configuration by running behavior suites across multiple seeds and scenarios and produce aggregated metrics and annotated CSV rows.
        
        Runs each ablation variant (including a default reference) across the requested seeds; for each variant it may perform training, executes the behavior suite (with and without reflex support), aggregates per-scenario statistics and behavioral scores across seeds, compacts per-variant behavior payloads, and computes deltas versus the reference variant. Also returns flattened, annotated rows suitable for CSV export.
        
        Parameters:
            operational_profile: Registered operational profile name or an OperationalProfile instance; when None the simulation uses the default operational profile.
            noise_profile: Registered noise profile name or a NoiseConfig instance to apply environment/motor noise; when None no explicit noise profile is applied.
            budget_profile: BudgetProfile name or instance to resolve episodes, evaluation runs, and seeds for the comparisons.
            seeds: Sequence of random seeds to run; metrics are aggregated across these seeds. If None, seeds are taken from the resolved budget.
            names: Sequence of scenario names to evaluate; when None all available scenarios are used.
            variant_names: Sequence of ablation variant names to include; when omitted the resolver determines available variants.
            episodes_per_scenario: Number of evaluation runs to perform per scenario per seed; when None this is taken from the resolved budget.
            checkpoint_selection: One of "none" or "best"; controls per-variant checkpointing behavior during optional training.
            checkpoint_metric: Primary metric used to select best checkpoints when checkpoint_selection == "best".
            checkpoint_interval: Checkpoint interval override passed into the resolved budget when provided.
            checkpoint_dir: Optional base directory to persist per-variant checkpoint artifacts.
        
        Returns:
            tuple:
                - payload (Dict[str, object]): Summary payload containing:
                    - "reference_variant": name of the reference config,
                    - "scenario_names": list of evaluated scenario names,
                    - "episodes_per_scenario": number of runs per scenario,
                    - "variants": mapping from variant name to a compact behavior-suite payload (includes per-scenario aggregated metrics, a "config" summary, and a "without_reflex_support" entry),
                    - "deltas_vs_reference": per-variant deltas of success rates versus the reference.
                - rows (List[Dict[str, object]]): Flattened, CSV-ready rows for all evaluated episodes across seeds and scenarios; each row is annotated with ablation metadata and evaluation/reflex metadata.
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
                            ),
                            eval_reflex_scale=0.0,
                        )
                    )
            if sim is None:
                continue
            compact_payload = cls._compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=combined_stats,
                    behavior_histories=combined_scores,
                )
            )
            compact_payload["config"] = config.to_summary()
            compact_payload.update(cls._noise_profile_metadata(resolved_noise_profile))
            compact_payload["without_reflex_support"] = cls._with_noise_profile_metadata(
                cls._compact_behavior_payload(
                    sim._build_behavior_payload(
                        stats_histories=combined_stats_without_reflex,
                        behavior_histories=combined_scores_without_reflex,
                    )
                ),
                resolved_noise_profile,
            )
            variants[config.name] = compact_payload

        reference_variant = reference_config.name
        return {
            "budget_profile": budget.profile,
            "benchmark_strength": budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
            "reference_variant": reference_variant,
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
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> tuple[Dict[str, object], List[Dict[str, object]]]:
        """
        Run a registry of learning-evidence conditions across seeds and scenarios and produce compact comparison results.
        
        For each resolved condition this method trains or initializes simulations according to the condition's specified budget (base, long, freeze_half, or initial), optionally persists checkpoints, executes the behavior suite with the condition's evaluation policy mode, and aggregates per-condition behavior-suite payloads and flattened CSV rows. Conditions that are incompatible with the base architecture are marked as skipped. The returned payload contains budget and noise metadata, per-condition compact behavior summaries under `"conditions"`, deltas versus the reference condition (`"trained_final"`), and a synthesized evidence summary.
        
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
        if condition_names is not None and not any(
            spec.name == "trained_final" for spec in condition_specs
        ):
            condition_specs = resolve_learning_evidence_conditions(
                ("trained_final", *tuple(condition_names))
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
                        checkpoint_interval=condition_budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=condition_budget.selection_scenario_episodes,
                    )
                    train_episodes = int(condition_budget.episodes)
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
                        checkpoint_interval=long_budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=long_budget.selection_scenario_episodes,
                    )
                    train_episodes = int(long_budget.episodes)
                    observed_checkpoint_source = str(sim.checkpoint_source)
                elif condition.train_budget == "freeze_half":
                    train_episodes = max(0, int(base_budget.episodes) // 2)
                    frozen_after_episode = train_episodes
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
                        checkpoint_interval=base_budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=base_budget.selection_scenario_episodes,
                    )
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
                if condition.eval_reflex_scale is not None:
                    sim.brain.set_runtime_reflex_scale(condition.eval_reflex_scale)
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
                            ),
                            eval_reflex_scale=condition.eval_reflex_scale,
                            extra_metadata=extra_row_metadata,
                        )
                    )

            if exemplar_sim is None:
                continue

            compact_payload = cls._compact_behavior_payload(
                exemplar_sim._build_behavior_payload(
                    stats_histories=combined_stats,
                    behavior_histories=combined_scores,
                )
            )
            compact_payload.update(
                {
                    "condition": condition.name,
                    "description": condition.description,
                    "policy_mode": condition.policy_mode,
                    "train_budget": condition.train_budget,
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

        reference_condition = "trained_final"
        return {
            "budget_profile": base_budget.profile,
            "benchmark_strength": base_budget.benchmark_strength,
            "long_budget_profile": long_budget.profile,
            "long_budget_benchmark_strength": long_budget.benchmark_strength,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_metric": checkpoint_metric,
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
            "training_regime",
            "curriculum_profile",
            "curriculum_phase",
            "curriculum_phase_status",
            "learning_evidence_condition",
            "learning_evidence_policy_mode",
            "learning_evidence_train_episodes",
            "learning_evidence_frozen_after_episode",
            "learning_evidence_checkpoint_source",
            "learning_evidence_budget_profile",
            "learning_evidence_budget_benchmark_strength",
            "reflex_scale",
            "reflex_anneal_final_scale",
            "eval_reflex_scale",
            "budget_profile",
            "benchmark_strength",
            "architecture_version",
            "architecture_fingerprint",
            "operational_profile",
            "operational_profile_version",
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
