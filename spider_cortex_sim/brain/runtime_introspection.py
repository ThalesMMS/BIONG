from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart11Mixin:
    def estimate_value(self, observation: Dict[str, np.ndarray]) -> float:
        """
        Estimate the action-center state value for a single observation.
        
        Builds proposals, applies reflex and arbitration gating as used at inference time, and returns the scalar value produced by the action-center. If the brain contains recurrent modules, their hidden states are snapshot and restored so this call does not mutate recurrent state.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Observation arrays keyed by interface names used to produce proposals and construct the action-center input.
        
        Returns:
            float: Scalar value estimate for the provided observation.
        """
        if self.config.is_b_series:
            if self.b_series_policy is None:
                raise RuntimeError(
                    "B-series policy unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            return float(
                self.b_series_policy.value_only(monolithic_observation)
            )
        if self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            if hasattr(self.true_monolithic_policy, "set_runtime_observation_meta"):
                self.true_monolithic_policy.set_runtime_observation_meta(
                    observation.get("meta", {})
                )
            runtime_state_snapshot = self.snapshot_direct_policy_runtime_state()
            try:
                direct_forward = self.true_monolithic_policy.forward(
                    monolithic_observation,
                    store_cache=False,
                )
                if len(direct_forward) >= 3:
                    _, value, *_ = direct_forward
                else:
                    _, value = direct_forward
                return float(value)
            finally:
                self.restore_direct_policy_runtime_state(runtime_state_snapshot)
        hidden_state_snapshot: Dict[str, np.ndarray] | None = None
        if self.module_bank is not None and self.module_bank.has_recurrent_modules:
            hidden_state_snapshot = self.module_bank.snapshot_hidden_states()
        try:
            module_results = self._proposal_results(
                observation,
                store_cache=False,
                training=False,
            )
            apply_reflex_path(
                module_results,
                ablation_config=self.config,
                operational_profile=self.operational_profile,
                interface_registry=self._interface_registry(),
                current_reflex_scale=self.current_reflex_scale,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            arbitration = self._compute_arbitration(
                module_results,
                observation,
                training=False,
                store_cache=False,
            )
            apply_priority_gating(
                module_results,
                arbitration,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            action_input = self._build_action_input(module_results, observation)
            if self.action_center is None:
                raise RuntimeError("Action center unavailable for the configured architecture.")
            _, value = self.action_center.forward(action_input, store_cache=False)
            return float(value)
        finally:
            if hidden_state_snapshot is not None and self.module_bank is not None:
                self.module_bank.restore_hidden_states(hidden_state_snapshot)

    def _proposal_stage_names(self) -> List[str]:
        """
        Return the ordered proposal sources that feed the action-center input.
        """
        if self.module_bank is not None:
            return [spec.name for spec in self.module_bank.enabled_specs]
        if self.true_monolithic_policy is not None:
            return [self.TRUE_MONOLITHIC_POLICY_NAME]
        if getattr(self, "b_series_policy", None) is not None:
            return [B_SERIES_POLICY_NAME]
        return [self.MONOLITHIC_POLICY_NAME]

    def _architecture_signature(self) -> dict[str, object]:
        """
        Compute the runtime architecture signature for the active proposal backend.
        
        Returns:
            signature (dict): A mapping describing architecture identifiers and configuration used for compatibility/fingerprinting (includes proposal backend name and order, whether learned arbitration is enabled, and arbitration network input/hidden dims and regularization weight).
        """
        arbitration_input_dim = (
            self.arbitration_network.input_dim
            if self.arbitration_network is not None
            else 0
        )
        arbitration_hidden_dim = (
            self.arbitration_network.hidden_dim
            if self.arbitration_network is not None
            else 0
        )
        return architecture_signature(
            proposal_backend=self.config.architecture,
            proposal_order=self._proposal_stage_names(),
            module_variants=getattr(self.config, "module_variants", None),
            learned_arbitration=(
                self.config.use_learned_arbitration and self.arbitration_network is not None
            ),
            arbitration_input_dim=arbitration_input_dim,
            arbitration_hidden_dim=arbitration_hidden_dim,
            arbitration_regularization_weight=self.arbitration_regularization_weight,
            capacity_profile_name=self.config.capacity_profile_name,
            module_hidden_dims=self.config.module_hidden_dims,
            action_center_hidden_dim=self.config.action_center_hidden_dim,
            motor_hidden_dim=self.config.motor_hidden_dim,
            integration_hidden_dim=self.config.integration_hidden_dim,
            monolithic_hidden_dim=self.config.monolithic_hidden_dim,
            b_level=getattr(self.config, "b_level", 0),
            b_mode=getattr(self.config, "b_mode", "current_bridge"),
            direct_policy_hidden_dims=self.config.direct_policy_hidden_dims or None,
            direct_policy_recurrent=bool(self.config.direct_policy_recurrent),
            direct_policy_phase_head=bool(self.config.direct_policy_phase_head),
            direct_policy_event_attention=bool(
                self.config.direct_policy_event_attention
            ),
            direct_policy_event_buffer_size=int(
                self.config.direct_policy_event_buffer_size
            ),
            direct_policy_option_head=bool(self.config.direct_policy_option_head),
            direct_policy_owned_option_controller=bool(
                self.config.direct_policy_owned_option_controller
            ),
            direct_policy_option_ttl=int(self.config.direct_policy_option_ttl),
            direct_policy_affordance_head=bool(
                self.config.direct_policy_affordance_head
            ),
            direct_policy_affordance_feedback=bool(
                self.config.direct_policy_affordance_feedback
            ),
            direct_policy_geometry_head=bool(
                self.config.direct_policy_geometry_head
            ),
            direct_policy_shelter_column_head=bool(
                self.config.direct_policy_shelter_column_head
            ),
            direct_policy_shelter_position_head=bool(
                self.config.direct_policy_shelter_position_head
            ),
            direct_policy_local_affordance_inputs=bool(
                getattr(self.config, "direct_policy_local_affordance_inputs", False)
            ),
            direct_policy_local_spatial_inputs=bool(
                getattr(self.config, "direct_policy_local_spatial_inputs", False)
            ),
            direct_policy_local_transition_inputs=bool(
                getattr(self.config, "direct_policy_local_transition_inputs", False)
            ),
            direct_policy_local_transition_rollout_inputs=bool(
                getattr(
                    self.config,
                    "direct_policy_local_transition_rollout_inputs",
                    False,
                )
            ),
            direct_policy_transition_prediction_head=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_prediction_head",
                    False,
                )
            ),
            direct_policy_transition_prediction_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_prediction_feedback",
                    False,
                )
            ),
            direct_policy_transition_rollout_prediction_head=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_rollout_prediction_head",
                    False,
                )
            ),
            direct_policy_transition_rollout_prediction_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_rollout_prediction_feedback",
                    False,
                )
            ),
            direct_policy_handoff_teacher=bool(
                self.config.direct_policy_handoff_teacher
            ),
            direct_policy_handoff_option_teacher=bool(
                getattr(self.config, "direct_policy_handoff_option_teacher", False)
            ),
            direct_policy_post_rest_action_teacher=bool(
                getattr(self.config, "direct_policy_post_rest_action_teacher", False)
            ),
            direct_policy_post_rest_release_sequence_teacher=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_teacher",
                    False,
                )
            ),
            direct_policy_post_rest_release_sequence_replay_boost=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_replay_boost",
                    False,
                )
            ),
            direct_policy_post_rest_release_sequence_distill=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_distill",
                    False,
                )
            ),
            direct_policy_post_rest_probe_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_sequence_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_sequence_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_family_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_family_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_handoff_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_handoff_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_trajectory_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_trajectory_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_cycle_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_cycle_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_trace_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_trace_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_rollout_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_rollout_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_frontier_teacher_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_frontier_teacher_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_replayable_teacher_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_replayable_teacher_distillation",
                    False,
                )
            ),
            direct_policy_continuation_replay_passes=int(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_passes",
                    0,
                )
            ),
            direct_policy_continuation_replay_lr_scale=float(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_lr_scale",
                    0.0,
                )
            ),
            direct_policy_continuation_margin_weight=float(
                getattr(
                    self.config,
                    "direct_policy_continuation_margin_weight",
                    0.0,
                )
            ),
            direct_policy_phase_option_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_phase_option_feedback",
                    False,
                )
            ),
            direct_policy_option_transition_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_option_transition_feedback",
                    False,
                )
            ),
            direct_policy_option_termination_cooldown=bool(
                getattr(
                    self.config,
                    "direct_policy_option_termination_cooldown",
                    False,
                )
            ),
            direct_policy_option_action_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_head",
                    False,
                )
            ),
            direct_policy_option_decoder_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_decoder_state",
                    False,
                )
            ),
            direct_policy_option_recurrent_dynamics=bool(
                getattr(
                    self.config,
                    "direct_policy_option_recurrent_dynamics",
                    False,
                )
            ),
            direct_policy_option_sequence_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_sequence_head",
                    False,
                )
            ),
            direct_policy_option_decoder_recurrent_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_decoder_recurrent_state",
                    False,
                )
            ),
            direct_policy_option_action_transition_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_transition_state",
                    False,
                )
            ),
            direct_policy_option_action_controller_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_controller_state",
                    False,
                )
            ),
            direct_policy_option_action_token_decoder=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_token_decoder",
                    False,
                )
            ),
            direct_policy_option_action_recurrent_core=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_recurrent_core",
                    False,
                )
            ),
            direct_policy_option_action_separate_recurrent_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_recurrent_head",
                    False,
                )
            ),
            direct_policy_option_action_separate_policy_path=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_policy_path",
                    False,
                )
            ),
            direct_policy_option_action_separate_backbone=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_backbone",
                    False,
                )
            ),
            direct_policy_executive_physiology_option_gating=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_physiology_option_gating",
                    False,
                )
            ),
            direct_policy_executive_affordance_action_gating=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_affordance_action_gating",
                    False,
                )
            ),
            direct_policy_executive_option_action_masking=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_option_action_masking",
                    False,
                )
            ),
            direct_policy_executive_event_release_latching=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_event_release_latching",
                    False,
                )
            ),
            direct_policy_executive_event_release_action_commitment=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_event_release_action_commitment",
                    False,
                )
            ),
            direct_policy_executive_release_phase_state=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_phase_state",
                    False,
                )
            ),
            direct_policy_executive_release_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_progression",
                    False,
                )
            ),
            direct_policy_executive_release_exit_contract=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_exit_contract",
                    False,
                )
            ),
            direct_policy_executive_release_substate_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_substate_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_continuation=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_continuation",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_guidance=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_guidance",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_commitment=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_commitment",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_heading_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_heading_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_smell_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_smell_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_corridor_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_corridor_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_corridor_affordance_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_corridor_affordance_progression",
                    False,
                )
            ),
            direct_policy_executive_post_food_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_return",
                    False,
                )
            ),
            direct_policy_executive_post_food_vector_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_vector_return",
                    False,
                )
            ),
            direct_policy_executive_post_food_path_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_path_return",
                    False,
                )
            ),
            capacity_profile=self.config.capacity_profile.to_summary(),
        )

    def _interface_registry(self) -> dict[str, object]:
        """
        Retrieve the runtime-governed interface registry used by this brain.
        
        Returns:
            dict[str, object]: Mapping of interface names to their interface objects as provided by the active runtime registry.
        """
        return interface_registry()

    def _architecture_fingerprint(self) -> str:
        """
        Get the stable fingerprint of the brain's runtime architecture signature.
        
        Returns:
            fingerprint (str): String representation of the architecture signature's `fingerprint` field.
        """
        return str(self._architecture_signature()["fingerprint"])

    def parameter_norms(self) -> Dict[str, float]:
        """
        Compute the L2 norm of parameters for each trainable network component.
        
        Returns:
            Mapping from component name to its L2 parameter norm for each active
            trainable network in the configured topology.
        """
        norms: Dict[str, float] = {}
        if self.module_bank is not None:
            norms.update(self.module_bank.parameter_norms())
        if self.monolithic_policy is not None:
            norms[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.parameter_norm()
        if self.true_monolithic_policy is not None:
            norms[self.TRUE_MONOLITHIC_POLICY_NAME] = (
                self.true_monolithic_policy.parameter_norm()
            )
        if getattr(self, "b_series_policy", None) is not None:
            norms[B_SERIES_POLICY_NAME] = self.b_series_policy.parameter_norm()
        if self.arbitration_network is not None:
            norms[self.ARBITRATION_NETWORK_NAME] = self.arbitration_network.parameter_norm()
        if self.action_center is not None:
            norms["action_center"] = self.action_center.parameter_norm()
        if self.motor_cortex is not None:
            norms["motor_cortex"] = self.motor_cortex.parameter_norm()
        return norms

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters for each trainable network component.

        Returns:
            Mapping from component name to trainable parameter count for each
            active trainable network in the configured topology.
        """
        counts: Dict[str, int] = {}
        if self.module_bank is not None:
            counts.update(self.module_bank.parameter_counts())
        if self.monolithic_policy is not None:
            counts[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.count_parameters()
        if self.true_monolithic_policy is not None:
            counts[self.TRUE_MONOLITHIC_POLICY_NAME] = (
                self.true_monolithic_policy.count_parameters()
            )
        if getattr(self, "b_series_policy", None) is not None:
            counts[B_SERIES_POLICY_NAME] = self.b_series_policy.count_parameters()
        if self.arbitration_network is not None:
            counts[self.ARBITRATION_NETWORK_NAME] = self.arbitration_network.count_parameters()
        if self.action_center is not None:
            counts["action_center"] = self.action_center.count_parameters()
        if self.motor_cortex is not None:
            counts["motor_cortex"] = self.motor_cortex.count_parameters()
        return counts

    def _module_names(self) -> List[str]:
        """
        List module names present in the brain for inspection.
        
        When the brain is modular, returns the module spec names in their
        configured order followed by the active downstream controllers.
        Monolithic variants return only the networks present in that topology.
        
        Returns:
            names (List[str]): Ordered list of module names and the two controller component names.
        """
        if self.module_bank is not None:
            return [
                spec.name for spec in self.module_bank.enabled_specs
            ] + [self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
        if self.true_monolithic_policy is not None:
            return [self.TRUE_MONOLITHIC_POLICY_NAME]
        if getattr(self, "b_series_policy", None) is not None:
            return [B_SERIES_POLICY_NAME]
        return [self.MONOLITHIC_POLICY_NAME, self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
