from __future__ import annotations

from .simulation_episode_shared import *


class _SimulationEpisodeDistillationRolloutsMixin:
    @staticmethod
    def _direct_policy_transition_rollout_prediction_targets(
        *,
        observation_meta: Dict[str, object],
    ) -> np.ndarray:
        targets = np.zeros(
            len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4,
            dtype=float,
        )
        rollouts = observation_meta.get("local_transition_rollouts")
        if not isinstance(rollouts, dict):
            return targets
        offset = 0
        for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES:
            rollout = rollouts.get(action_name, {})
            if not isinstance(rollout, dict):
                rollout = {}
            targets[offset + 0] = float(
                np.clip(rollout.get("best_food_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 1] = float(
                np.clip(rollout.get("best_shelter_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 2] = float(
                np.clip(rollout.get("best_predator_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 3] = float(
                bool(rollout.get("food_reachable_within_two_steps", False))
            )
            offset += 4
        return targets

    def collect_direct_policy_probe_distillation_rollout(
        self,
        *,
        scenario_name: str,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        if scenario_name not in {
            "continuous_survival_post_rest_inside_v1",
            "continuous_survival_post_rest_entrance_v1",
        }:
            raise ValueError(
                "Direct-policy probe distillation only supports post-rest continuation scenarios."
            )
        scenario = get_scenario(scenario_name)
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe",
                "scenario_name": str(scenario_name),
                "episodes": int(max(0, episodes)),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            episode_index = int(episode_start + episode_offset)
            episode_seed = self.seed + 997 * (episode_index + 1)
            episode_action_stages: list[str] = []
            episode_option_stages: list[str] = []
            if self.world.map_template_name != scenario.map_template:
                self.world.configure_map_template(scenario.map_template)
            self.world.reset(seed=episode_seed)
            scenario.setup(self.world)
            observation = self.world.observe()
            self._reset_direct_policy_handoff_teacher_state()
            for step in range(int(scenario.max_steps)):
                observation_adapters = adapt_observation_contracts(
                    observation,
                    tick=step,
                )
                brain_observation = observation_vectors_from_adapters(
                    observation_adapters
                )
                current_state = self.world.state_dict()
                food_memory = current_state.get("food_memory")
                if isinstance(food_memory, dict) and isinstance(
                    food_memory.get("target"),
                    tuple,
                ):
                    current_state["food_memory"] = dict(food_memory)
                    current_state["food_memory"]["target"] = list(
                        food_memory["target"]
                    )
                food_direction_action = self.brain._food_direction_bias_action(
                    brain_observation
                )
                teacher_action_idx, teacher_action_stage = (
                    self._direct_policy_handoff_teacher_target(
                        current_state=current_state,
                        food_direction_action=food_direction_action,
                        tick=step,
                    )
                )
                if teacher_action_idx < 0:
                    break
                _, teacher_option_stage = (
                    self._direct_policy_handoff_option_teacher_target(
                        current_state=current_state,
                        food_direction_action=food_direction_action,
                        tick=step,
                    )
                )
                if teacher_action_stage is not None:
                    episode_action_stages.append(str(teacher_action_stage))
                if teacher_option_stage is not None:
                    episode_option_stages.append(str(teacher_option_stage))
                teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                teacher_policy[int(teacher_action_idx)] = 1.0
                teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                teacher_logits[int(teacher_action_idx)] = 6.0
                dataset.add_sample(
                    episode=episode_index,
                    step=step,
                    observation=brain_observation,
                    teacher_policy=teacher_policy,
                    teacher_total_logits=teacher_logits,
                    teacher_action_center_policy=teacher_policy,
                    teacher_action_center_logits=teacher_logits,
                    teacher_action_intent_idx=int(teacher_action_idx),
                    teacher_module_policies={},
                )
                next_observation, _, done, _ = self.world.step(int(teacher_action_idx))
                observation = next_observation
                if done:
                    break
            dataset.teacher_metadata.setdefault("stages", [])
            dataset.teacher_metadata["stages"].extend(episode_action_stages)
            dataset.teacher_metadata.setdefault("option_stages", [])
            dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        return dataset

    def collect_direct_policy_probe_sequence_distillation_rollout(
        self,
        *,
        scenario_name: str,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        scripted_sequences: dict[str, tuple[str, ...]] = {
            "continuous_survival_post_rest_inside_v1": (
                "MOVE_UP",
                "MOVE_UP",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
            ),
            "continuous_survival_post_rest_entrance_v1": (
                "MOVE_UP",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
            ),
            "continuous_survival_return_after_late_forage_v1": (
                "MOVE_LEFT",
                "MOVE_LEFT",
                "MOVE_LEFT",
                "MOVE_LEFT",
                "STAY",
                "STAY",
            ),
            "continuous_survival_re_rest_after_return_v1": (
                "MOVE_DOWN",
                "STAY",
                "STAY",
            ),
        }
        if scenario_name not in scripted_sequences:
            raise ValueError(
                "Direct-policy probe sequence distillation only supports scripted continuation scenarios."
            )
        scenario = get_scenario(scenario_name)
        sequence = scripted_sequences[str(scenario_name)]
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe_sequence",
                "scenario_name": str(scenario_name),
                "episodes": int(max(0, episodes)),
                "scripted_sequence": list(sequence),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            episode_index = int(episode_start + episode_offset)
            episode_seed = self.seed + 997 * (episode_index + 1)
            if self.world.map_template_name != scenario.map_template:
                self.world.configure_map_template(scenario.map_template)
            self.world.reset(seed=episode_seed)
            scenario.setup(self.world)
            observation = self.world.observe()
            for step, action_name in enumerate(sequence):
                action_idx = int(ACTION_TO_INDEX[action_name])
                observation_adapters = adapt_observation_contracts(
                    observation,
                    tick=step,
                )
                brain_observation = observation_vectors_from_adapters(
                    observation_adapters
                )
                teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                teacher_policy[action_idx] = 1.0
                teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                teacher_logits[action_idx] = 6.0
                dataset.add_sample(
                    episode=episode_index,
                    step=step,
                    observation=brain_observation,
                    teacher_policy=teacher_policy,
                    teacher_total_logits=teacher_logits,
                    teacher_action_center_policy=teacher_policy,
                    teacher_action_center_logits=teacher_logits,
                    teacher_action_intent_idx=action_idx,
                    teacher_module_policies={},
                )
                next_observation, _, done, _ = self.world.step(action_idx)
                observation = next_observation
                if done:
                    break
        return dataset

    def _direct_policy_probe_family_fallback_action(
        self,
        *,
        observation: Dict[str, np.ndarray],
        current_state: Dict[str, object],
        food_direction_action: str | None,
    ) -> tuple[int, str]:
        threat_escape_action = self.brain._threat_escape_bias_action(observation)
        if threat_escape_action is not None:
            return int(ACTION_TO_INDEX[threat_escape_action]), "fallback_escape"
        sleep_rest_action = self.brain._sleep_rest_bias_action(observation)
        if sleep_rest_action == "STAY":
            return int(ACTION_TO_INDEX["STAY"]), "fallback_rest"
        current_role = str(current_state.get("shelter_role", "outside"))
        hunger = float(current_state.get("hunger", 0.0))
        if food_direction_action is not None and (
            current_role in {"entrance", "outside"} or hunger >= 0.18
        ):
            return int(ACTION_TO_INDEX[food_direction_action]), "fallback_food"
        sleep_obs = self.brain._bound_observation("sleep_center", observation)
        if (
            sleep_obs["shelter_memory_age"] < 1.0
            and (
                sleep_obs["night"] > 0.0
                or float(current_state.get("fatigue", 0.0)) >= 0.12
                or float(current_state.get("sleep_debt", 0.0)) >= 0.08
                or current_role == "outside"
            )
        ):
            shelter_action = direction_action(
                sleep_obs["shelter_memory_dx"],
                sleep_obs["shelter_memory_dy"],
            )
            if shelter_action != "STAY":
                return int(ACTION_TO_INDEX[shelter_action]), "fallback_return"
        if current_role == "entrance":
            return int(ACTION_TO_INDEX["STAY"]), "fallback_hold"
        return int(ACTION_TO_INDEX["STAY"]), "fallback_idle"

    def collect_direct_policy_probe_family_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe_family",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self._reset_direct_policy_handoff_teacher_state()
                self._reset_direct_policy_probe_family_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(
                            food_memory.get("target"),
                            tuple,
                        )
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_family_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx, teacher_action_stage = (
                            self._direct_policy_probe_family_fallback_action(
                                observation=brain_observation,
                                current_state=current_state,
                                food_direction_action=food_direction_action,
                            )
                        )
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        return dataset

    def collect_direct_policy_probe_handoff_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe_handoff",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self._reset_direct_policy_handoff_teacher_state()
                self._reset_direct_policy_probe_family_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if isinstance(food_memory, dict) and isinstance(
                        food_memory.get("target"),
                        tuple,
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_handoff_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx, teacher_action_stage = (
                            self._direct_policy_probe_family_fallback_action(
                                observation=brain_observation,
                                current_state=current_state,
                                food_direction_action=food_direction_action,
                            )
                        )
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        return dataset

    def collect_direct_policy_probe_trajectory_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_trajectory",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_trajectory_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(
                            food_memory.get("target"),
                            tuple,
                        )
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    baseline_action_idx = int(teacher_decision.action_idx)
                    baseline_action_name = ACTIONS[baseline_action_idx]
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_trajectory_redirect_action(
                            current_state=current_state,
                            baseline_action_name=baseline_action_name,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = baseline_action_idx
                        teacher_action_stage = "live_policy"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_cycle_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_cycle",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_cycle_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    baseline_action_idx = int(teacher_decision.action_idx)
                    baseline_action_name = ACTIONS[baseline_action_idx]
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_cycle_redirect_action(
                            observation=brain_observation,
                            current_state=current_state,
                            baseline_action_name=baseline_action_name,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = baseline_action_idx
                        teacher_action_stage = "live_policy"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_trace_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
            "continuous_survival_return_after_late_forage_v1",
            "continuous_survival_re_rest_after_return_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_trace",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_trace_teacher_state()
                if scenario_name == "continuous_survival_return_after_late_forage_v1":
                    self._direct_policy_probe_trace_teacher_state["stage"] = "return_window"
                    self._direct_policy_probe_trace_teacher_state["return_tick"] = 0
                    self._direct_policy_probe_trace_teacher_state["sleep_events_before"] = int(
                        self.world.state.sleep_events
                    )
                elif scenario_name == "continuous_survival_re_rest_after_return_v1":
                    self._direct_policy_probe_trace_teacher_state["stage"] = "rerest_window"
                    self._direct_policy_probe_trace_teacher_state["sleep_events_before"] = int(
                        self.world.state.sleep_events
                    )
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_trace_redirect_action(
                            observation=brain_observation,
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_decision = self.brain.act(
                            brain_observation,
                            bus=None,
                            sample=False,
                            training=False,
                        )
                        teacher_action_idx = int(teacher_decision.action_idx)
                        teacher_action_stage = "live_policy"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_rollout_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_rollout",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_rollout_teacher_state()
                triggered = False
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage, active = (
                        self._direct_policy_probe_rollout_redirect_action(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = int(teacher_decision.action_idx)
                    if teacher_action_stage is None and active:
                        teacher_action_stage = "rollout_live"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if active:
                        triggered = True
                        episode_action_stages.append(str(teacher_action_stage))
                        if teacher_option_stage is not None:
                            episode_option_stages.append(str(teacher_option_stage))
                        teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                        teacher_policy[int(teacher_action_idx)] = 1.0
                        teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                        teacher_logits[int(teacher_action_idx)] = 6.0
                        dataset.add_sample(
                            episode=episode_index,
                            step=step,
                            observation=brain_observation,
                            teacher_policy=teacher_policy,
                            teacher_total_logits=teacher_logits,
                            teacher_action_center_policy=teacher_policy,
                            teacher_action_center_logits=teacher_logits,
                            teacher_action_intent_idx=int(teacher_action_idx),
                            teacher_module_policies={},
                        )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                if triggered:
                    dataset.teacher_metadata.setdefault("action_stages", [])
                    dataset.teacher_metadata["action_stages"].extend(
                        episode_action_stages
                    )
                    dataset.teacher_metadata.setdefault("option_stages", [])
                    dataset.teacher_metadata["option_stages"].extend(
                        episode_option_stages
                    )
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_frontier_teacher_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        checkpoint_dir = self._frontier_teacher_checkpoint_dir()
        teacher_checkpoint_error: str | None = None
        try:
            teacher_brain, teacher_checkpoint_metadata = load_teacher_checkpoint(
                checkpoint_dir
            )
        except (FileNotFoundError, ValueError) as exc:
            teacher_brain = self.brain
            teacher_checkpoint_metadata = {}
            teacher_checkpoint_error = str(exc)
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "checkpoint_direct_policy_post_rest_probe_frontier_teacher",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
                "teacher_checkpoint": str(checkpoint_dir),
                "teacher_ablation": str(
                    teacher_checkpoint_metadata.get("ablation_config", {}).get(
                        "name",
                        "",
                    )
                ),
                "teacher_checkpoint_error": teacher_checkpoint_error,
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                teacher_brain.reset_hidden_states()
                self._reset_direct_policy_probe_rollout_teacher_state()
                triggered = False
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = teacher_brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage, active = (
                        self._direct_policy_probe_rollout_redirect_action(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    teacher_decision = teacher_brain.act_inference(
                        brain_observation,
                        bus=None,
                        sample=False,
                        policy_mode="normal",
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = int(teacher_decision.action_idx)
                    if teacher_action_stage is None and active:
                        teacher_action_stage = "frontier_live"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if active:
                        triggered = True
                        episode_action_stages.append(str(teacher_action_stage))
                        if teacher_option_stage is not None:
                            episode_option_stages.append(str(teacher_option_stage))
                        if str(teacher_action_stage).startswith("rollout_"):
                            teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                            teacher_policy[int(teacher_action_idx)] = 1.0
                            teacher_total_logits = np.zeros(len(ACTIONS), dtype=float)
                            teacher_total_logits[int(teacher_action_idx)] = 6.0
                            teacher_action_center_policy = teacher_policy
                            teacher_action_center_logits = teacher_total_logits
                            teacher_action_intent_idx = int(teacher_action_idx)
                        else:
                            teacher_policy = teacher_decision.policy
                            teacher_total_logits = teacher_decision.total_logits
                            teacher_action_center_policy = (
                                teacher_decision.action_center_policy
                            )
                            teacher_action_center_logits = (
                                teacher_decision.action_center_logits
                            )
                            teacher_action_intent_idx = int(
                                teacher_decision.action_intent_idx
                            )
                        dataset.add_sample(
                            episode=episode_index,
                            step=step,
                            observation=brain_observation,
                            teacher_policy=teacher_policy,
                            teacher_total_logits=teacher_total_logits,
                            teacher_action_center_policy=teacher_action_center_policy,
                            teacher_action_center_logits=teacher_action_center_logits,
                            teacher_action_intent_idx=teacher_action_intent_idx,
                            teacher_module_policies={},
                        )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                if triggered:
                    dataset.teacher_metadata.setdefault("action_stages", [])
                    dataset.teacher_metadata["action_stages"].extend(
                        episode_action_stages
                    )
                    dataset.teacher_metadata.setdefault("option_stages", [])
                    dataset.teacher_metadata["option_stages"].extend(
                        episode_option_stages
                    )
        teacher_brain.reset_hidden_states()
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_replayable_teacher_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = ("continuous_survival_canonical",)
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "stateful_direct_policy_post_rest_probe_replayable_teacher",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_replayable_teacher_state()
                triggered = False
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage, active = (
                        self._direct_policy_probe_replayable_teacher_action(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                        )
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = int(teacher_decision.action_idx)
                    if teacher_action_stage is None and active:
                        teacher_action_stage = "probe_live"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if active:
                        triggered = True
                        episode_action_stages.append(str(teacher_action_stage))
                        if teacher_option_stage is not None:
                            episode_option_stages.append(str(teacher_option_stage))
                        teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                        teacher_policy[int(teacher_action_idx)] = 1.0
                        teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                        teacher_logits[int(teacher_action_idx)] = 6.0
                        dataset.add_sample(
                            episode=episode_index,
                            step=step,
                            observation=brain_observation,
                            teacher_policy=teacher_policy,
                            teacher_total_logits=teacher_logits,
                            teacher_action_center_policy=teacher_policy,
                            teacher_action_center_logits=teacher_logits,
                            teacher_action_intent_idx=int(teacher_action_idx),
                            teacher_module_policies={},
                        )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                if triggered:
                    dataset.teacher_metadata.setdefault("action_stages", [])
                    dataset.teacher_metadata["action_stages"].extend(
                        episode_action_stages
                    )
                    dataset.teacher_metadata.setdefault("option_stages", [])
                    dataset.teacher_metadata["option_stages"].extend(
                        episode_option_stages
                    )
                    dataset.teacher_metadata.setdefault(
                        "first_up_redirects",
                        0,
                    )
                    dataset.teacher_metadata["first_up_redirects"] += int(
                        self._direct_policy_probe_replayable_teacher_state.get(
                            "first_up_redirects",
                            0,
                        )
                    )
                    dataset.teacher_metadata.setdefault(
                        "down_redirects",
                        0,
                    )
                    dataset.teacher_metadata["down_redirects"] += int(
                        self._direct_policy_probe_replayable_teacher_state.get(
                            "down_redirects",
                            0,
                        )
                    )
        self.brain.reset_hidden_states()
        return dataset

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
