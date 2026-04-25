"""Simulation runtime facade.

Multi-seed comparison workflows and comparison statistics live in
`spider_cortex_sim.comparison`; claim-test evaluation lives in
`spider_cortex_sim.claim_evaluation`.

``default_behavior_evaluation`` and ``ensure_behavior_evaluation`` are the
public homes for the default behavior-evaluation helpers previously kept in
the CLI entrypoint.
"""

from __future__ import annotations

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
from .distillation import DistillationDataset
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
from .observation_adapters import (
    adapt_observation_contracts,
    adapter_trace_summary,
    observation_vectors_from_adapters,
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


class SimulationEpisodeMixin:
    def collect_teacher_rollout(
        self,
        teacher_brain: SpiderBrain,
        *,
        episodes: int,
        teacher_metadata: Dict[str, object] | None = None,
        episode_start: int = 0,
        policy_mode: str = "normal",
        sample: bool = False,
        scenario_name: str | None = None,
    ) -> DistillationDataset:
        """
        Collect rollouts from a teacher policy and assemble them into a DistillationDataset for policy distillation.
        
        This runs the teacher brain in inference mode across a sequence of episodes (optionally using a named scenario), records per-step observation vectors and teacher outputs, and returns a dataset suitable for supervised distillation of student policies.
        
        Parameters:
        	teacher_brain (SpiderBrain): Teacher policy/brain providing `act_inference` and `VALENCE_ORDER`.
        	episodes (int): Number of episodes to collect.
        	teacher_metadata (Dict[str, object] | None): Optional metadata stored on the returned dataset.
        	episode_start (int): Base index added to sequential episode indices written into samples.
        	policy_mode (str): Decision routing mode to use; must be `"normal"` or `"reflex_only"`.
        	sample (bool): If true, the teacher will sample stochastic actions when available.
        	scenario_name (str | None): Optional scenario identifier to load per-episode environment setup.
        
        Returns:
        	DistillationDataset: A dataset populated with per-step samples containing episode and step indices, vectorized observations, teacher action selections, logits, per-module policy outputs, and optional valence scores.
        
        Raises:
        	ValueError: If `policy_mode` is not `"normal"` or `"reflex_only"`.
        """
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "Invalid policy_mode. Use 'normal' or 'reflex_only'."
            )
        dataset = DistillationDataset(teacher_metadata=teacher_metadata)
        total_episodes = max(0, int(episodes))
        scenario = get_scenario(scenario_name) if scenario_name is not None else None
        for episode_offset in range(total_episodes):
            episode_index = int(episode_start + episode_offset)
            episode_seed = self.seed + 997 * (episode_index + 1)
            episode_map_template = (
                scenario.map_template if scenario is not None else self.default_map_template
            )
            if self.world.map_template_name != episode_map_template:
                self.world.configure_map_template(episode_map_template)
            observation = self.world.reset(seed=episode_seed)
            teacher_brain.reset_hidden_states()
            episode_max_steps = self.max_steps
            if scenario is not None:
                scenario.setup(self.world)
                observation = self.world.observe()
                episode_max_steps = scenario.max_steps
            for step in range(episode_max_steps):
                observation_adapters = adapt_observation_contracts(
                    observation,
                    tick=step,
                )
                brain_observation = observation_vectors_from_adapters(
                    observation_adapters
                )
                decision = teacher_brain.act_inference(
                    brain_observation,
                    bus=None,
                    sample=sample,
                    policy_mode=policy_mode,
                )
                arbitration = decision.arbitration_decision
                dataset.add_sample(
                    episode=episode_index,
                    step=step,
                    observation=brain_observation,
                    teacher_policy=decision.policy,
                    teacher_total_logits=decision.total_logits,
                    teacher_action_center_policy=decision.action_center_policy,
                    teacher_action_center_logits=decision.action_center_logits,
                    teacher_action_intent_idx=decision.action_intent_idx,
                    teacher_valence_probs=(
                        [
                            float(arbitration.valence_scores.get(name, 0.0))
                            for name in teacher_brain.VALENCE_ORDER
                        ]
                        if arbitration is not None
                        else None
                    ),
                    teacher_valence_logits=(
                        [
                            float(arbitration.valence_logits.get(name, 0.0))
                            for name in teacher_brain.VALENCE_ORDER
                        ]
                        if arbitration is not None
                        else None
                    ),
                    teacher_module_policies={
                        result.name: result.probs
                        for result in decision.module_results
                        if result.active
                    },
                )
                next_observation, _, done, _ = self.world.step(decision.action_idx)
                observation = next_observation
                if done:
                    break
        return dataset

    @staticmethod
    def _episode_stats_behavior_metrics(stats: EpisodeStats) -> Dict[str, object]:
        """
        Produce a flat mapping of behavior-metric fields derived from an EpisodeStats object.
        
        The returned dictionary contains named metric entries (floats, ints, and dicts) such as module dominance/share, routing/router statistics, owner alignment/rank/suppression, motor slip and orientation metrics, per-proposal-source contribution shares (keys prefixed with `module_contribution_`), and per-terrain slip rates (keys prefixed with `terrain_slip_rate_`). These fields are intended for inclusion in scenario scorecards and downstream reporting.
        
        Returns:
            Dict[str, object]: A dictionary mapping metric names to primitive values suitable for reporting.
        """
        metrics: Dict[str, object] = {
            "dominant_module": stats.dominant_module,
            "dominant_module_share": float(stats.dominant_module_share),
            "effective_module_count": float(stats.effective_module_count),
            "gate_entropy": float(stats.gate_entropy),
            "dominance_rate": float(stats.dominance_rate),
            "effective_proposer_count": float(stats.effective_proposer_count),
            "module_agreement_rate": float(stats.module_agreement_rate),
            "module_disagreement_rate": float(stats.module_disagreement_rate),
            "route_active_modules": dict(stats.route_active_modules),
            "router_health": dict(stats.router_health),
            "owner_alignment": float(stats.owner_alignment),
            "owner_rank": int(stats.owner_rank),
            "owner_suppressed_rate": float(stats.owner_suppressed_rate),
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
        Execute one simulation episode and collect aggregated episode metrics and an optional per-tick trace.
        
        Parameters:
            episode_index (int): Index used to derive the episode RNG seed and to tag trace items.
            training (bool): If True, enables learning updates during the episode; only allowed with policy_mode="normal".
            sample (bool): If True, allow stochastic action sampling; otherwise use deterministic selection.
            render (bool): If True, render the world each tick.
            capture_trace (bool): If True, collect a list of per-tick dictionaries describing state, actions, rewards, and messages.
            scenario_name (str | None): Optional scenario identifier that may override the map template and max steps when provided.
            debug_trace (bool): If True and capture_trace is True, include expanded diagnostic fields in each trace item.
            policy_mode (str): Inference mode for the brain; valid values are "normal" or "reflex_only". "reflex_only" requires a modular brain with reflexes enabled.
        
        Returns:
            tuple[EpisodeStats, List[Dict[str, object]]]: EpisodeStats with aggregated episode metrics, and a list of per-tick trace dictionaries (empty when capture_trace is False).
        
        Raises:
            ValueError: If policy_mode is invalid or incompatible with the brain configuration, or if training=True is used with a non-"normal" policy_mode.
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
            observation_adapters = adapt_observation_contracts(
                observation,
                tick=step,
            )
            brain_observation = observation_vectors_from_adapters(
                observation_adapters
            )

            if training:
                decision = self.brain.act_train(
                    brain_observation,
                    self.bus,
                    sample=sample,
                    policy_mode=policy_mode,
                )
            elif sample:
                decision = self.brain.act_exploration(
                    brain_observation,
                    self.bus,
                    policy_mode=policy_mode,
                )
            else:
                decision = self.brain.act_inference(
                    brain_observation,
                    self.bus,
                    sample=False,
                    policy_mode=policy_mode,
                )
            metrics.record_decision(decision)
            predator_state_before = self.world.lizard.mode
            next_observation, reward, done, info = self.world.step(decision.action_idx)
            next_observation_adapters = adapt_observation_contracts(
                next_observation,
                tick=step + 1,
            )
            next_brain_observation = observation_vectors_from_adapters(
                next_observation_adapters
            )
            self._attach_motor_execution_info(decision, info)
            learn_stats: Dict[str, object] = {}
            if training:
                learn_stats = self.brain.learn(
                    decision,
                    reward,
                    next_brain_observation,
                    done,
                )
                if decision.arbitration_decision is not None:
                    decision.arbitration_decision = replace(
                        decision.arbitration_decision,
                        route_active_modules=list(
                            learn_stats.get("route_active_modules", [])
                        ),
                        route_credit_weights=dict(
                            learn_stats.get("route_credit_weights", {})
                        ),
                    )
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
                        "observation_contracts": adapter_trace_summary(
                            observation_adapters
                        ),
                        "next_observation_contracts": adapter_trace_summary(
                            next_observation_adapters
                        ),
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
