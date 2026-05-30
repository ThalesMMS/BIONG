from __future__ import annotations

from .simulation_episode_shared import *
from .simulation_episode_b_series_trace import append_b_series_trace_fields
from .simulation_episode_debug_trace import build_debug_trace_payload
from .simulation_episode_teacher_targets import _SimulationEpisodeTeacherTargetsMixin
from .simulation_episode_distillation_rollouts import _SimulationEpisodeDistillationRolloutsMixin
from .simulation_episode_metrics_trace import _SimulationEpisodeMetricsTraceMixin


class SimulationEpisodeMixin(_SimulationEpisodeTeacherTargetsMixin, _SimulationEpisodeDistillationRolloutsMixin, _SimulationEpisodeMetricsTraceMixin):
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
        normalized_scenario = str(scenario_name) if scenario_name is not None else None
        scenario = get_scenario(normalized_scenario) if normalized_scenario is not None else None
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
        self._reset_direct_policy_handoff_teacher_state()

        for step in range(episode_max_steps):
            self.bus.set_tick(step)
            self.brain.set_direct_policy_event_clock(step)
            current_state_snapshot = self.world.state_dict()
            self.bus.publish(
                sender="environment",
                topic="observation",
                payload={
                    "state": current_state_snapshot,
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
            if self.brain.config.is_b_series:
                brain_observation["meta"] = observation["meta"]

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
            if self.brain.config.direct_policy_phase_head:
                phase_target = derive_phase_target(
                    state=self.world.state_dict(),
                    observation_meta=observation["meta"],
                )
                decision.phase_target = phase_target
                decision.phase_target_idx = int(PHASE_TO_INDEX[phase_target])
                self.bus.publish(
                    sender=self.brain.TRUE_MONOLITHIC_POLICY_NAME,
                    topic="phase.prediction",
                    payload={
                        "phase_target": phase_target,
                        "phase_prediction": decision.phase_prediction,
                        "phase_prediction_confidence": round(
                            float(decision.phase_prediction_confidence),
                            6,
                        ),
                        "selected_action": ACTIONS[decision.action_idx],
                    },
                )
            decision.scenario_name = normalized_scenario
            if self.brain.config.direct_policy_affordance_head:
                (
                    decision.affordance_blocked_targets,
                    decision.affordance_role_targets,
                ) = self._direct_policy_affordance_targets(
                    current_state=current_state_snapshot,
                )
            if self.brain.config.direct_policy_geometry_head:
                decision.geometry_targets = self._direct_policy_geometry_targets(
                    current_state=current_state_snapshot,
                )
            if self.brain.config.direct_policy_shelter_column_head:
                decision.shelter_column_targets = (
                    self._direct_policy_shelter_column_targets(
                        current_state=current_state_snapshot,
                    )
                )
            if self.brain.config.direct_policy_shelter_position_head:
                decision.shelter_position_targets = (
                    self._direct_policy_shelter_position_targets(
                        current_state=current_state_snapshot,
                    )
                )
            if self.brain.config.direct_policy_transition_prediction_head:
                decision.transition_prediction_targets = (
                    self._direct_policy_transition_prediction_targets(
                        observation_meta=observation["meta"],
                    )
                )
            if self.brain.config.direct_policy_transition_rollout_prediction_head:
                decision.transition_rollout_prediction_targets = (
                    self._direct_policy_transition_rollout_prediction_targets(
                        observation_meta=observation["meta"],
                    )
                )
            if self.brain.config.direct_policy_handoff_teacher:
                food_direction_action = self.brain._food_direction_bias_action(
                    brain_observation
                )
                teacher_action_target_idx, teacher_action_target_stage = (
                    self._direct_policy_handoff_teacher_target(
                        current_state=current_state_snapshot,
                        food_direction_action=food_direction_action,
                        tick=step,
                    )
                )
                decision.teacher_action_target_idx = int(teacher_action_target_idx)
                decision.teacher_action_target_stage = teacher_action_target_stage
                decision.teacher_action_target_name = (
                    None
                    if teacher_action_target_idx < 0
                    else ACTIONS[int(teacher_action_target_idx)]
                )
                if self.brain.config.direct_policy_handoff_option_teacher:
                    teacher_option_target_idx, teacher_option_target_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state_snapshot,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    decision.teacher_option_target_idx = int(
                        teacher_option_target_idx
                    )
                    decision.teacher_option_target_stage = (
                        teacher_option_target_stage
                    )
                    decision.teacher_option_target_name = (
                        None
                        if teacher_option_target_idx < 0
                        else OPTION_NAMES[int(teacher_option_target_idx)]
                    )
            metrics.record_decision(decision)
            predator_state_before = self.world.lizard.mode
            next_observation, reward, done, info = self.world.step(decision.action_idx)
            next_state_snapshot = self.world.state_dict()
            next_observation_adapters = adapt_observation_contracts(
                next_observation,
                tick=step + 1,
            )
            next_brain_observation = observation_vectors_from_adapters(
                next_observation_adapters
            )
            if self.brain.config.is_b_series:
                next_brain_observation["meta"] = next_observation["meta"]
            self._attach_motor_execution_info(decision, info)
            self._record_direct_policy_events(
                step=step,
                decision=decision,
                observation=observation,
                next_observation=next_observation,
                current_state=current_state_snapshot,
                next_state=next_state_snapshot,
                info=info,
            )
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
                state_snapshot = self.world.state_dict()
                threat_snapshot = self.brain._bound_observation(
                    "threat_center",
                    next_brain_observation,
                )
                state_snapshot["predator_smell_strength"] = float(
                    threat_snapshot["predator_smell_strength"]
                )
                item: Dict[str, object] = {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "tick": step,
                    "training": training,
                    "policy_mode": policy_mode,
                    "scenario": normalized_scenario,
                    "action": info.get("action"),
                    "intended_action": info.get("intended_action"),
                    "executed_action": info.get("executed_action"),
                    "motor_noise_applied": bool(info.get("motor_noise_applied", False)),
                    "slip_reason": str(info.get("slip_reason", "none")),
                    "ate": bool(info.get("ate")),
                    "slept": bool(info.get("slept")),
                    "predator_contact": bool(info.get("predator_contact")),
                    "predator_escape": bool(info.get("predator_escape")),
                    "state": state_snapshot,
                    "reward": float(reward),
                    "reward_components": info["reward_components"],
                    "predator_transition": info.get("predator_transition"),
                    "distance_deltas": info.get("distance_deltas", {}),
                    "event_log": info["event_log"],
                    "done": bool(done),
                    "messages": self.bus.serialize_current_tick(),
                }
                if self.brain.config.is_b_series:
                    append_b_series_trace_fields(item, decision)
                if debug_trace:
                    item["observation"] = jsonify_observation(observation)
                    item["next_observation"] = jsonify_observation(next_observation)
                    item["debug"] = build_debug_trace_payload(
                        self,
                        decision=decision,
                        observation=observation,
                        next_observation=next_observation,
                        observation_adapters=observation_adapters,
                        next_observation_adapters=next_observation_adapters,
                        info=info,
                    )
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
            scenario=normalized_scenario,
            total_reward=float(total_reward),
            state=state,
        )
        return stats, trace
