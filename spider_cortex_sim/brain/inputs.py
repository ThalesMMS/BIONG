from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from ..direct_policy_affordances import (
    AFFORDANCE_SHELTER_ROLE_TO_INDEX,
    AFFORDANCE_SHELTER_ROLE_NAMES,
    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
    DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM,
    DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM,
    DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM,
)
from ..modules import ModuleResult
from ..nn import softmax
from ..noise import _compute_execution_difficulty_core
from ..world import ACTIONS


class BrainInputMixin:
    def reset_hidden_states(self) -> None:
        """
        Reset recurrent hidden state for all modules owned by this brain.
        
        If a module bank is present, delegates to its reset_hidden_states(); otherwise this is a no-op.
        """
        if self.module_bank is not None:
            self.module_bank.reset_hidden_states()
        if self.true_monolithic_policy is not None and hasattr(
            self.true_monolithic_policy,
            "reset_hidden_state",
        ):
            self.true_monolithic_policy.reset_hidden_state()
            if hasattr(self.true_monolithic_policy, "reset_event_memory"):
                self.true_monolithic_policy.reset_event_memory()
            self._direct_policy_hidden_reset_pending = True

    def snapshot_direct_policy_hidden_state(self) -> np.ndarray | None:
        if self.true_monolithic_policy is None or not hasattr(
            self.true_monolithic_policy,
            "get_hidden_state",
        ):
            return None
        return np.asarray(self.true_monolithic_policy.get_hidden_state(), dtype=float).copy()

    def restore_direct_policy_hidden_state(self, hidden_state: np.ndarray | None) -> None:
        if hidden_state is None:
            return
        if self.true_monolithic_policy is None or not hasattr(
            self.true_monolithic_policy,
            "set_hidden_state",
        ):
            raise ValueError("Direct policy hidden-state restore requested for non-recurrent policy.")
        self.true_monolithic_policy.set_hidden_state(hidden_state)

    def snapshot_direct_policy_runtime_state(self) -> dict[str, object] | None:
        if self.true_monolithic_policy is None:
            return None
        if hasattr(self.true_monolithic_policy, "get_runtime_state"):
            runtime_state = self.true_monolithic_policy.get_runtime_state()
            return dict(runtime_state) if isinstance(runtime_state, dict) else None
        hidden_state = self.snapshot_direct_policy_hidden_state()
        if hidden_state is None:
            return None
        return {"hidden_state": hidden_state}

    def restore_direct_policy_runtime_state(
        self,
        runtime_state: dict[str, object] | None,
    ) -> None:
        if runtime_state is None or self.true_monolithic_policy is None:
            return
        if hasattr(self.true_monolithic_policy, "set_runtime_state"):
            self.true_monolithic_policy.set_runtime_state(dict(runtime_state))
            return
        hidden_state = runtime_state.get("hidden_state")
        if hidden_state is not None:
            self.restore_direct_policy_hidden_state(
                np.asarray(hidden_state, dtype=float)
            )

    def set_direct_policy_event_clock(self, tick: int) -> None:
        if self.true_monolithic_policy is not None and hasattr(
            self.true_monolithic_policy,
            "set_event_clock",
        ):
            self.true_monolithic_policy.set_event_clock(int(tick))

    def record_direct_policy_event(
        self,
        event_type: str,
        *,
        features: np.ndarray,
        tick: int,
    ) -> None:
        if self.true_monolithic_policy is not None and hasattr(
            self.true_monolithic_policy,
            "record_event",
        ):
            self.true_monolithic_policy.record_event(
                event_type,
                features=np.asarray(features, dtype=float),
                tick=int(tick),
            )

    def _build_direct_policy_local_affordance_input(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        affordance_input = np.zeros(
            DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM,
            dtype=float,
        )
        meta = observation.get("meta")
        if not isinstance(meta, dict):
            return affordance_input
        current_role = str(meta.get("shelter_role", "outside"))
        current_role_idx = AFFORDANCE_SHELTER_ROLE_TO_INDEX.get(current_role)
        if current_role_idx is not None:
            affordance_input[current_role_idx] = 1.0
        current_role_level = float(meta.get("shelter_role_level", 0.0))
        local_affordances = meta.get("local_affordances")
        if not isinstance(local_affordances, dict):
            return affordance_input
        offset = len(AFFORDANCE_SHELTER_ROLE_NAMES)
        for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES:
            affordance = local_affordances.get(action_name, {})
            if not isinstance(affordance, dict):
                affordance = {}
            next_role = str(affordance.get("next_role", current_role))
            next_role_level = float(
                affordance.get("next_role_level", current_role_level)
            )
            affordance_input[offset + 0] = float(bool(affordance.get("blocked", False)))
            affordance_input[offset + 1] = float(
                np.clip(next_role_level - current_role_level, -1.0, 1.0)
            )
            affordance_input[offset + 2] = float(next_role == "entrance")
            affordance_input[offset + 3] = float(next_role == "outside")
            offset += 4
        return affordance_input

    def _build_direct_policy_local_spatial_input(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        spatial_input = np.zeros(
            DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM,
            dtype=float,
        )
        meta = observation.get("meta")
        if not isinstance(meta, dict):
            return spatial_input
        local_patch = meta.get("local_spatial_patch")
        if not isinstance(local_patch, dict):
            return spatial_input
        offset = 0
        for key in ("blocked", "shelter_role_level", "food"):
            values = local_patch.get(key, ())
            if not isinstance(values, (list, tuple)):
                values = ()
            count = min(len(values), 9)
            if count > 0:
                spatial_input[offset : offset + count] = np.asarray(
                    values[:count],
                    dtype=float,
                )
            offset += 9
        return spatial_input

    def _build_direct_policy_local_transition_input(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        transition_input = np.zeros(
            DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM,
            dtype=float,
        )
        meta = observation.get("meta")
        if not isinstance(meta, dict):
            return transition_input
        transitions = meta.get("local_transition_consequences")
        if not isinstance(transitions, dict):
            return transition_input
        offset = 0
        for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES:
            consequence = transitions.get(action_name, {})
            if not isinstance(consequence, dict):
                consequence = {}
            transition_input[offset + 0] = float(
                np.clip(consequence.get("food_dist_delta", 0.0), -1.0, 1.0)
            )
            transition_input[offset + 1] = float(
                np.clip(consequence.get("shelter_dist_delta", 0.0), -1.0, 1.0)
            )
            transition_input[offset + 2] = float(
                np.clip(consequence.get("predator_dist_delta", 0.0), -1.0, 1.0)
            )
            transition_input[offset + 3] = float(
                bool(consequence.get("next_cell_has_food", False))
            )
            offset += 4
        return transition_input

    def _build_direct_policy_local_transition_rollout_input(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        rollout_input = np.zeros(
            DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM,
            dtype=float,
        )
        meta = observation.get("meta")
        if not isinstance(meta, dict):
            return rollout_input
        rollouts = meta.get("local_transition_rollouts")
        if not isinstance(rollouts, dict):
            return rollout_input
        offset = 0
        for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES:
            rollout = rollouts.get(action_name, {})
            if not isinstance(rollout, dict):
                rollout = {}
            rollout_input[offset + 0] = float(
                np.clip(rollout.get("best_food_dist_delta", 0.0), -1.0, 1.0)
            )
            rollout_input[offset + 1] = float(
                np.clip(rollout.get("best_shelter_dist_delta", 0.0), -1.0, 1.0)
            )
            rollout_input[offset + 2] = float(
                np.clip(rollout.get("best_predator_dist_delta", 0.0), -1.0, 1.0)
            )
            rollout_input[offset + 3] = float(
                bool(rollout.get("food_reachable_within_two_steps", False))
            )
            offset += 4
        return rollout_input

    def _build_direct_policy_local_geodesic_input(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        geodesic_input = np.zeros(
            DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
            dtype=float,
        )
        meta = observation.get("meta")
        if not isinstance(meta, dict):
            return geodesic_input
        geodesics = meta.get("local_geodesic_consequences")
        if not isinstance(geodesics, dict):
            return geodesic_input
        offset = 0
        for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES:
            consequence = geodesics.get(action_name, {})
            if not isinstance(consequence, dict):
                consequence = {}
            geodesic_input[offset + 0] = float(
                np.clip(consequence.get("exit_geodesic_delta", 0.0), -1.0, 1.0)
            )
            geodesic_input[offset + 1] = float(
                np.clip(consequence.get("deep_geodesic_delta", 0.0), -1.0, 1.0)
            )
            geodesic_input[offset + 2] = float(
                bool(consequence.get("next_on_exit_target", False))
            )
            geodesic_input[offset + 3] = float(
                bool(consequence.get("next_on_deep_target", False))
            )
            offset += 4
        return geodesic_input

    def _build_monolithic_observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build a flat numeric observation vector for the monolithic proposal network.
        
        Converts each module interface's observation array to float, replaces NaN with 0.0,
        +inf with 1.0 and -inf with -1.0, then concatenates them in the order of MODULE_INTERFACES.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Mapping of observation keys to arrays; keys must include
                each spec.observation_key from MODULE_INTERFACES.
        
        Returns:
            np.ndarray: 1-D concatenated float vector containing the sanitized observations for all interfaces.
        """
        monolithic_observation = np.concatenate(
            [
                np.nan_to_num(
                    np.asarray(observation[spec.observation_key], dtype=float),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                for spec in MODULE_INTERFACES
            ],
            axis=0,
        )
        if bool(getattr(self.config, "direct_policy_local_affordance_inputs", False)):
            monolithic_observation = np.concatenate(
                [
                    monolithic_observation,
                    self._build_direct_policy_local_affordance_input(observation),
                ],
                axis=0,
            )
        if bool(getattr(self.config, "direct_policy_local_spatial_inputs", False)):
            monolithic_observation = np.concatenate(
                [
                    monolithic_observation,
                    self._build_direct_policy_local_spatial_input(observation),
                ],
                axis=0,
            )
        if bool(getattr(self.config, "direct_policy_local_transition_inputs", False)):
            monolithic_observation = np.concatenate(
                [
                    monolithic_observation,
                    self._build_direct_policy_local_transition_input(observation),
                ],
                axis=0,
            )
        if bool(
            getattr(self.config, "direct_policy_local_transition_rollout_inputs", False)
        ):
            monolithic_observation = np.concatenate(
                [
                    monolithic_observation,
                    self._build_direct_policy_local_transition_rollout_input(
                        observation
                    ),
                ],
                axis=0,
            )
        if bool(getattr(self.config, "direct_policy_local_geodesic_inputs", False)):
            monolithic_observation = np.concatenate(
                [
                    monolithic_observation,
                    self._build_direct_policy_local_geodesic_input(observation),
                ],
                axis=0,
            )
        return monolithic_observation

    def _build_action_input(self, module_results: List[ModuleResult], observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Construct the flattened input vector for the action center by concatenating proposal logits and bound action context.

        Parameters:
            module_results (List[ModuleResult]): Per-module proposal results whose gated logits, when present, are concatenated in proposal order.
            observation (Dict[str, np.ndarray]): Full observation mapping; the action context is taken from the key defined by ACTION_CONTEXT_INTERFACE.observation_key and bound/flattened via ACTION_CONTEXT_INTERFACE.

        Returns:
            np.ndarray: 1-D array formed by concatenating all action-path module logits followed by the action context vector.
        """
        logits_flat = np.concatenate(
            [
                gated_logits
                if (gated_logits := getattr(result, "gated_logits", None)) is not None
                else result.logits
                for result in module_results
            ],
            axis=0,
        )
        action_context_mapping = self._bound_action_context(observation)
        action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
        return np.concatenate([logits_flat, action_context], axis=0)

    def _build_motor_input(self, action_intent: np.ndarray, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build the flattened input vector for the motor cortex by combining a sanitized action intent and the bound motor context.
        
        The intent vector is converted to a finite float array (NaN -> 0.0, +inf -> 1.0, -inf -> -1.0) and must match shape (action_dim,), otherwise a ValueError is raised. The motor context is obtained from the observation via MOTOR_CONTEXT_INTERFACE and concatenated to the intent.
        
        Parameters:
            action_intent (np.ndarray): One-hot or real-valued intent vector for the chosen action.
            observation (Dict[str, np.ndarray]): Raw observation dict used to bind motor context.
        
        Returns:
            np.ndarray: Concatenation of the sanitized intent and the motor-context vector.
        """
        intent = np.nan_to_num(np.asarray(action_intent, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        if intent.shape != (self.action_dim,):
            raise ValueError(
                f"action_intent expected shape {(self.action_dim,)}, received {intent.shape}."
            )
        motor_context_values = np.nan_to_num(
            np.asarray(
                observation[MOTOR_CONTEXT_INTERFACE.observation_key],
                dtype=float,
            ),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        motor_context_mapping = MOTOR_CONTEXT_INTERFACE.bind_values(
            motor_context_values
        )
        motor_context = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(motor_context_mapping)
        return np.concatenate([intent, motor_context], axis=0)

    def _proposal_results(
        self,
        observation: Dict[str, np.ndarray],
        *,
        store_cache: bool,
        training: bool,
    ) -> List[ModuleResult]:
        """
        Produce per-module proposal outputs for the current architecture.
        
        For the modular configuration this delegates to the module bank and returns its
        per-module ModuleResult list. For monolithic and true-monolithic configurations
        this constructs a single ModuleResult from the concatenated observation vector.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Raw observation mapping keyed by module/interface names.
            store_cache (bool): Allow proposal components to cache intermediate state for later learning.
            training (bool): Run proposal components in training mode (e.g., with dropout enabled).
        
        Returns:
            List[ModuleResult]: Per-proposal-module results including logits, probabilities and reflex/diagnostic fields.
        
        Raises:
            RuntimeError: If the configured proposal backend is not available.
        """
        if self.config.is_modular:
            if self.module_bank is None:
                raise RuntimeError("Module bank unavailable for modular architecture.")
            return self.module_bank.forward(
                observation,
                store_cache=store_cache,
                training=training,
            )

        network = self.true_monolithic_policy if self.config.is_true_monolithic else self.monolithic_policy
        network_name = (
            self.TRUE_MONOLITHIC_POLICY_NAME
            if self.config.is_true_monolithic
            else self.MONOLITHIC_POLICY_NAME
        )
        if network is None:
            raise RuntimeError("Monolithic network unavailable for the configured architecture.")
        monolithic_observation = self._build_monolithic_observation(observation)
        if self.config.is_true_monolithic:
            direct_forward = network.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            logits = np.asarray(direct_forward[0], dtype=float)
        else:
            logits = network.forward(
                monolithic_observation,
                store_cache=store_cache,
                training=training,
            )
        return [
            ModuleResult(
                interface=None,
                name=network_name,
                observation_key=network_name,
                observation=monolithic_observation.copy(),
                logits=logits,
                probs=softmax(logits),
                active=True,
                reflex=None,
                neural_logits=logits.copy(),
                reflex_delta_logits=np.zeros_like(logits),
                post_reflex_logits=logits.copy(),
            )
        ]

    def _bound_observation(
        self,
        interface_name: str,
        observation: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Bind the named interface's raw observation and sanitize its values into bounded scalar evidence.
        
        Parameters:
            interface_name (str): Interface identifier present in MODULE_INTERFACE_BY_NAME; the function looks up that interface and uses its observation_key to fetch data from `observation`.
            observation (Dict[str, np.ndarray]): Full observation mapping from observation keys to arrays.
        
        Returns:
            Dict[str, float]: The interface's bound observation fields as finite float values; NaN values are converted to 0.0, +inf to 1.0, and -inf to -1.0 before binding.
        
        Raises:
            KeyError: If no interface with the given name exists.
        """
        interface = MODULE_INTERFACE_BY_NAME.get(interface_name)
        if interface is None:
            raise KeyError(f"Unknown interface: {interface_name}")
        sanitized_obs = np.nan_to_num(
            np.asarray(observation[interface.observation_key], dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return interface.bind_values(sanitized_obs)

    def _bound_action_context(self, observation: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Produce a mapping of action-context field names to finite scalar values.
        
        The method reads the action-context observation, replaces NaN with 0.0, +inf with 1.0, and -inf with -1.0, and returns the interface-bound mapping of scalar fields.

        Parameters:
            observation (Dict[str, np.ndarray]): Observation payload containing
                `ACTION_CONTEXT_INTERFACE.observation_key`. The corresponding
                array is coerced to float, sanitized with NaN -> 0.0,
                +inf -> 1.0, and -inf -> -1.0, then bound with
                `ACTION_CONTEXT_INTERFACE.bind_values`.
        
        Returns:
            Dict[str, float]: Mapping from action-context field names to their sanitized float values.
        """
        sanitized_obs = np.nan_to_num(
            np.asarray(observation[ACTION_CONTEXT_INTERFACE.observation_key], dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return ACTION_CONTEXT_INTERFACE.bind_values(sanitized_obs)

    def _motor_execution_diagnostics(
        self,
        observation: Dict[str, np.ndarray],
        action_idx: int,
    ) -> Dict[str, float]:
        """
        Estimate execution difficulty and related components for a selected action using motor-context inputs.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Perception/state mapping that must include the motor context at
                MOTOR_CONTEXT_INTERFACE.observation_key; values will be read and interpreted as motor-context fields.
            action_idx (int): Index into ACTIONS selecting the proposed action.
        
        Returns:
            Dict[str, float]: A mapping with:
                - "orientation_alignment": alignment between current heading and the action's movement direction.
                - "terrain_difficulty": terrain difficulty value clipped to the range [0.0, 1.0].
                - "momentum": momentum value clipped to the range [0.0, 1.0].
                - "execution_difficulty": aggregate difficulty score combining orientation, terrain, fatigue, and momentum.
        """
        sanitized_obs = np.nan_to_num(
            np.asarray(observation[MOTOR_CONTEXT_INTERFACE.observation_key], dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        motor_context = MOTOR_CONTEXT_INTERFACE.bind_values(sanitized_obs)
        action_name = ACTIONS[int(action_idx)]
        move_dx, move_dy = ACTION_DELTAS.get(action_name, (0, 0))
        heading_dx = float(motor_context.get("heading_dx", 0.0))
        heading_dy = float(motor_context.get("heading_dy", 0.0))
        terrain_difficulty = float(
            np.clip(float(motor_context.get("terrain_difficulty", 0.0)), 0.0, 1.0)
        )
        fatigue = float(np.clip(float(motor_context.get("fatigue", 0.0)), 0.0, 1.0))
        momentum = float(np.clip(float(motor_context.get("momentum", 0.0)), 0.0, 1.0))
        execution_difficulty, components = _compute_execution_difficulty_core(
            (heading_dx, heading_dy),
            (float(move_dx), float(move_dy)),
            terrain_difficulty=terrain_difficulty,
            fatigue=fatigue,
            momentum=momentum,
        )
        return {
            "orientation_alignment": float(components["orientation_alignment"]),
            "terrain_difficulty": float(components["terrain_difficulty"]),
            "momentum": float(components["momentum"]),
            "execution_difficulty": execution_difficulty,
        }
