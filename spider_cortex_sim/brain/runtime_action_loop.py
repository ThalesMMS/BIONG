from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart10Mixin:
    def act(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
        policy_mode: str = "normal",
        training: bool | None = None,
    ) -> BrainStep:
        """
        Choose and execute an action for the provided observation.
        
        Builds per-module proposals, optionally applies reflexes, computes arbitration and priority gating, runs action-center and motor-cortex corrections (unless `policy_mode == "reflex_only"`), and returns a populated BrainStep describing module results, logits/policies, selected intent/action, overrides, controller inputs, and the arbitration decision.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Mapping of interface observation arrays consumed by proposal modules and context interfaces.
            bus (MessageBus | None): Optional message bus for publishing per-module proposal diagnostics and final selection/execution diagnostics; pass None to disable publishing.
            sample (bool): If True, sample the executed action from the final policy distribution; if False, select the greedy argmax action.
            policy_mode (str): Execution mode, either "normal" to apply learned action-center and motor-cortex corrections, or "reflex_only" to select directly from post-reflex modular proposals. "reflex_only" requires a modular architecture with reflexes enabled.
            training (bool | None): If provided, forces training mode on/off for internal network cache and learned-arbitration behavior; if None, training mode is inferred from `sample` or an internal override.
        
        Returns:
            BrainStep: Decision container populated with per-module ModuleResult entries, action-center and motor-cortex logits/policies, combined logits with and without reflexes, the final policy and value estimate, selected intent and action indices, override flags, controller input vectors, the active `policy_mode`, and the computed `arbitration_decision`.
        
        Raises:
            ValueError: If `policy_mode` is not "normal" or "reflex_only", or if `policy_mode == "reflex_only"` is requested but the brain is not modular or reflexes are disabled.
        """
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "Invalid policy_mode. Use 'normal' or 'reflex_only'."
            )
        if policy_mode == "reflex_only" and not self.config.is_modular:
            raise ValueError(
                "policy_mode='reflex_only' requires the modular architecture."
            )
        if policy_mode == "reflex_only" and not self.config.enable_reflexes:
            raise ValueError(
                "policy_mode='reflex_only' requires reflexes to be enabled."
            )

        runtime_training = getattr(self, "_act_training_override", None)
        if training is None:
            training_mode = bool(sample if runtime_training is None else runtime_training)
        else:
            training_mode = bool(training)
        store_cache = training_mode and policy_mode == "normal"
        proposal_sum = np.zeros(self.action_dim, dtype=float)
        action_center_input = np.zeros(0, dtype=float)
        motor_input = np.zeros(0, dtype=float)
        action_center_correction_logits = np.zeros(self.action_dim, dtype=float)
        motor_correction_logits = np.zeros(self.action_dim, dtype=float)
        value = 0.0
        arbitration = None
        direct_policy_trace_payload: Dict[str, object] = {}
        phase_logits = np.zeros(0, dtype=float)
        phase_prediction: str | None = None
        phase_prediction_confidence = 0.0
        event_attention_top_type: str | None = None
        event_attention_top_age = -1
        event_attention_entropy = 0.0
        selected_option: str | None = None
        option_age = -1
        option_termination_reason = "none"
        option_logits = np.zeros(0, dtype=float)
        option_leaf_logits = np.zeros(0, dtype=float)
        option_owned_action: str | None = None
        safety_mask_applied = False
        safety_masked_actions: list[str] = []
        external_override_count = 0
        affordance_blocked_logits = np.zeros(0, dtype=float)
        affordance_role_logits = np.zeros(0, dtype=float)
        geometry_logits = np.zeros(0, dtype=float)
        shelter_column_logits = np.zeros(0, dtype=float)
        shelter_position_logits = np.zeros(0, dtype=float)
        transition_prediction_logits = np.zeros(0, dtype=float)
        transition_rollout_prediction_logits = np.zeros(0, dtype=float)
        semantic_logits = np.zeros(0, dtype=float)
        semantic_policy = np.zeros(0, dtype=float)
        if self.config.is_b_series:
            if self.b_series_policy is None:
                raise RuntimeError(
                    "B-series policy unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            semantic_logits, value = self.b_series_policy.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            semantic_policy = softmax(semantic_logits)
            if sample:
                learned_semantic_action_idx = int(
                    self.rng.choice(len(B_SEMANTIC_ACTIONS), p=semantic_policy)
                )
            else:
                learned_semantic_action_idx = int(np.argmax(semantic_policy))
            learned_semantic_action = B_SEMANTIC_ACTIONS[learned_semantic_action_idx]
            semantic_action = learned_semantic_action
            semantic_action_source = "network_policy"
            semantic_action_reason = "network_argmax_or_sample"
            semantic_override_count = 0
            b_temporal_threat_trace: dict[str, object] = {}
            b_level = int(getattr(self.config, "b_level", 0))
            b_effective_level = f"B{b_level}"
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
                b_effective_level,
            ) = self._select_b_series_semantic_action(
                observation=observation,
                learned_semantic_action=learned_semantic_action,
                semantic_action=semantic_action,
                semantic_action_source=semantic_action_source,
                semantic_action_reason=semantic_action_reason,
                semantic_override_count=semantic_override_count,
                b_temporal_threat_trace=b_temporal_threat_trace,
                b_level=b_level,
                b_effective_level=b_effective_level,
            )
            semantic_action_idx = int(B_SEMANTIC_ACTION_TO_INDEX[semantic_action])
            bridge_observation = observation
            if (
                semantic_action == "MOVE_TO_SHELTER"
                and (
                    float(
                        b_temporal_threat_trace.get("b_predator_trace_pressure", 0.0)
                        or 0.0
                    )
                    >= 0.50
                    or float(
                        b_temporal_threat_trace.get(
                            "b_predator_memory_pressure",
                            0.0,
                        )
                        or 0.0
                    )
                    >= 0.85
                )
            ):
                meta = observation.get("meta")
                if isinstance(meta, dict):
                    bridge_meta = dict(meta)
                    bridge_meta["predator_smell_strength"] = max(
                        self._b_series_float(bridge_meta, "predator_smell_strength"),
                        float(
                            b_temporal_threat_trace[
                                "b_temporal_threat_pressure"
                            ]
                        ),
                    )
                    bridge_observation = dict(observation)
                    bridge_observation["meta"] = bridge_meta
            bridge_decision = bridge_b_semantic_action(
                semantic_action,
                bridge_observation,
                rng=self.rng,
                sample=bool(sample),
            )
            bridge_meta = bridge_observation.get("meta")
            if (
                b_level in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
                and semantic_action == "MOVE_TO_FOOD"
                and isinstance(bridge_meta, dict)
                and str(bridge_meta.get("map_template", "")) == "corridor_escape"
                and str(
                    b_temporal_threat_trace.get("b6_decision", "")
                ).startswith("corridor_commitment")
            ):
                transitions = bridge_meta.get("local_transition_consequences")
                transitions = transitions if isinstance(transitions, dict) else {}
                affordances = bridge_meta.get("local_affordances")
                affordances = affordances if isinstance(affordances, dict) else {}
                up_transition = transitions.get("MOVE_UP")
                up_transition = up_transition if isinstance(up_transition, dict) else {}
                right_transition = transitions.get("MOVE_RIGHT")
                right_transition = (
                    right_transition if isinstance(right_transition, dict) else {}
                )
                up_affordance = affordances.get("MOVE_UP")
                up_affordance = up_affordance if isinstance(up_affordance, dict) else {}
                right_affordance = affordances.get("MOVE_RIGHT")
                right_affordance = (
                    right_affordance if isinstance(right_affordance, dict) else {}
                )
                up_delta = self._b_series_float(up_transition, "food_dist_delta")
                right_delta = self._b_series_float(right_transition, "food_dist_delta")
                try:
                    food_dist = float(bridge_meta.get("food_dist", 0.0))
                except (TypeError, ValueError):
                    food_dist = 0.0
                if not np.isfinite(food_dist):
                    food_dist = 0.0
                if (
                    not bool(right_affordance.get("blocked", False))
                    and (
                        bool(right_transition.get("next_cell_has_food", False))
                        or (food_dist > 7.0 and right_delta >= 0.0)
                    )
                ):
                    bridge_decision = replace(
                        bridge_decision,
                        primitive_action="MOVE_RIGHT",
                        reason="b6_corridor_horizontal_commitment",
                        food_delta_used=float(right_delta),
                    )
                elif (
                    not bool(up_affordance.get("blocked", False))
                    and food_dist <= 7.0
                    and up_delta > 0.0
                ):
                    bridge_decision = replace(
                        bridge_decision,
                        primitive_action="MOVE_UP",
                        reason="b6_corridor_vertical_commitment",
                        food_delta_used=float(up_delta),
                    )
            action_idx = int(bridge_decision.primitive_action_idx)
            motor_action_idx = action_idx
            action_intent_idx = action_idx
            action_intent_without_reflex_idx = action_idx
            action_without_reflex_idx = action_idx
            primitive_logits = np.zeros(self.action_dim, dtype=float)
            primitive_logits[action_idx] = 6.0
            total_logits_without_reflex = primitive_logits.copy()
            total_logits = primitive_logits.copy()
            action_center_logits = primitive_logits.copy()
            action_center_policy = softmax(action_center_logits)
            policy = softmax(total_logits)
            proposal_sum = total_logits.copy()
            module_results = [
                ModuleResult(
                    interface=None,
                    name=B_SERIES_POLICY_NAME,
                    observation_key=B_SERIES_POLICY_NAME,
                    observation=monolithic_observation.copy(),
                    logits=primitive_logits.copy(),
                    probs=policy.copy(),
                    active=True,
                    reflex=None,
                    neural_logits=primitive_logits.copy(),
                    reflex_delta_logits=np.zeros_like(primitive_logits),
                    post_reflex_logits=primitive_logits.copy(),
                )
            ]
            module_results[0].valence_role = "semantic_bridge"
            module_results[0].gate_weight = 1.0
            module_results[0].gated_logits = primitive_logits.copy()
            module_results[0].contribution_share = 1.0
            module_results[0].intent_before_gating = bridge_decision.semantic_action
            module_results[0].intent_after_gating = bridge_decision.primitive_action
            arbitration = self._true_monolithic_arbitration_decision(
                module_name=B_SERIES_POLICY_NAME,
                action_idx=action_idx,
            )
            b_transfer_report = getattr(self, "b_series_transfer_report", None) or {}
            b_parent_level_raw = getattr(self.config, "b_parent_level", None)
            b_parent_level = (
                None if b_parent_level_raw is None else int(b_parent_level_raw)
            )
            b_transfer_coverage = b_transfer_report.get("coverage")
            try:
                b_transfer_coverage_value = (
                    None
                    if b_transfer_coverage is None
                    else round(float(b_transfer_coverage), 6)
                )
            except (TypeError, ValueError):
                b_transfer_coverage_value = None
            direct_policy_trace_payload = {
                "b_level": int(self.config.b_level),
                "b_effective_level": b_effective_level,
                "b_mode": str(self.config.b_mode),
                "b_parent_level": b_parent_level,
                "b_transfer_source_checkpoint": b_transfer_report.get(
                    "source_checkpoint"
                ),
                "b_transfer_coverage": b_transfer_coverage_value,
                **b_temporal_threat_trace,
                "semantic_action": semantic_action,
                "semantic_action_idx": int(semantic_action_idx),
                "learned_semantic_action": learned_semantic_action,
                "learned_semantic_action_idx": int(learned_semantic_action_idx),
                "semantic_action_source": semantic_action_source,
                "semantic_action_reason": semantic_action_reason,
                "semantic_override_count": int(semantic_override_count),
                "semantic_logits": np.asarray(semantic_logits, dtype=float)
                .round(6)
                .tolist(),
                "semantic_policy": np.asarray(semantic_policy, dtype=float)
                .round(6)
                .tolist(),
                "bridge_primitive_action": bridge_decision.primitive_action,
                "bridge_reason": bridge_decision.reason,
                "blocked_mask": dict(bridge_decision.blocked_mask),
                "food_delta_used": round(float(bridge_decision.food_delta_used), 6),
                "shelter_delta_used": round(
                    float(bridge_decision.shelter_delta_used),
                    6,
                ),
                "external_override_count": int(
                    bridge_decision.external_override_count
                ),
            }
            external_override_count = int(bridge_decision.external_override_count)
            motor_override = False
            final_reflex_override = False
        elif self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
            )
            monolithic_observation = self._build_monolithic_observation(observation)
            if hasattr(self.true_monolithic_policy, "set_runtime_observation_meta"):
                self.true_monolithic_policy.set_runtime_observation_meta(
                    observation.get("meta", {})
                )
            hidden_before = self.snapshot_direct_policy_hidden_state()
            direct_forward = self.true_monolithic_policy.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            if len(direct_forward) == 4:
                policy_logits, value, option_logits_raw, phase_logits_raw = direct_forward
                option_logits = np.asarray(option_logits_raw, dtype=float).copy()
            elif len(direct_forward) == 3:
                policy_logits, value, aux_logits_raw = direct_forward
                phase_logits_raw = (
                    aux_logits_raw if self.config.direct_policy_phase_head else None
                )
                if self.config.direct_policy_option_head:
                    option_logits = np.asarray(aux_logits_raw, dtype=float).copy()
            else:
                policy_logits, value = direct_forward
                phase_logits_raw = None
            if phase_logits_raw is not None:
                phase_logits = np.asarray(phase_logits_raw, dtype=float).copy()
                phase_probs = softmax(phase_logits)
                phase_prediction_idx = int(np.argmax(phase_probs))
                phase_prediction = PHASE_LABELS[phase_prediction_idx]
                phase_prediction_confidence = float(phase_probs[phase_prediction_idx])
            hidden_after = self.snapshot_direct_policy_hidden_state()
            if hidden_after is not None:
                hidden_before_array = (
                    hidden_before
                    if hidden_before is not None
                    else np.zeros_like(hidden_after, dtype=float)
                )
                direct_policy_trace_payload = {
                    "recurrent_hidden_norm": round(float(np.linalg.norm(hidden_after)), 6),
                    "recurrent_hidden_delta_norm": round(
                        float(np.linalg.norm(hidden_after - hidden_before_array)),
                        6,
                    ),
                    "hidden_reset_event": bool(self._direct_policy_hidden_reset_pending),
                    "architecture_metadata": {
                        "direct_policy_recurrent": bool(self.config.direct_policy_recurrent),
                        "direct_policy_hidden_dims": list(self.config.direct_policy_hidden_dims),
                        "direct_policy_phase_head": bool(self.config.direct_policy_phase_head),
                        "direct_policy_event_attention": bool(
                            self.config.direct_policy_event_attention
                        ),
                        "direct_policy_event_buffer_size": int(
                            self.config.direct_policy_event_buffer_size
                        ),
                        "direct_policy_option_head": bool(
                            self.config.direct_policy_option_head
                        ),
                        "direct_policy_owned_option_controller": bool(
                            self.config.direct_policy_owned_option_controller
                        ),
                        "direct_policy_option_ttl": int(
                            self.config.direct_policy_option_ttl
                        ),
                        "direct_policy_affordance_head": bool(
                            self.config.direct_policy_affordance_head
                        ),
                        "direct_policy_affordance_feedback": bool(
                            self.config.direct_policy_affordance_feedback
                        ),
                        "direct_policy_geometry_head": bool(
                            self.config.direct_policy_geometry_head
                        ),
                        "direct_policy_shelter_column_head": bool(
                            self.config.direct_policy_shelter_column_head
                        ),
                        "direct_policy_shelter_position_head": bool(
                            self.config.direct_policy_shelter_position_head
                        ),
                        "direct_policy_local_affordance_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_affordance_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_spatial_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_spatial_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_transition_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_transition_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_transition_rollout_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_transition_rollout_inputs",
                                False,
                            )
                        ),
                        "direct_policy_transition_prediction_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_prediction_head",
                                False,
                            )
                        ),
                        "direct_policy_transition_prediction_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_prediction_feedback",
                                False,
                            )
                        ),
                        "direct_policy_transition_rollout_prediction_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_rollout_prediction_head",
                                False,
                            )
                        ),
                        "direct_policy_transition_rollout_prediction_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_rollout_prediction_feedback",
                                False,
                            )
                        ),
                        "direct_policy_handoff_teacher": bool(
                            self.config.direct_policy_handoff_teacher
                        ),
                        "direct_policy_handoff_option_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_handoff_option_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_action_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_action_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_replay_boost": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_replay_boost",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_distill": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_distill",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_sequence_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_sequence_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_family_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_family_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_handoff_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_handoff_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_trajectory_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_trajectory_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_cycle_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_cycle_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_trace_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_trace_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_rollout_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_rollout_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_frontier_teacher_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_frontier_teacher_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_replayable_teacher_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_replayable_teacher_distillation",
                                False,
                            )
                        ),
                        "direct_policy_continuation_replay_passes": int(
                            getattr(
                                self.config,
                                "direct_policy_continuation_replay_passes",
                                0,
                            )
                        ),
                        "direct_policy_continuation_replay_lr_scale": float(
                            getattr(
                                self.config,
                                "direct_policy_continuation_replay_lr_scale",
                                0.0,
                            )
                        ),
                        "direct_policy_continuation_margin_weight": float(
                            getattr(
                                self.config,
                                "direct_policy_continuation_margin_weight",
                                0.0,
                            )
                        ),
                        "direct_policy_phase_option_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_phase_option_feedback",
                                False,
                            )
                        ),
                        "direct_policy_option_transition_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_transition_feedback",
                                False,
                            )
                        ),
                        "direct_policy_option_termination_cooldown": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_termination_cooldown",
                                False,
                            )
                        ),
                        "direct_policy_option_action_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_head",
                                False,
                            )
                        ),
                        "direct_policy_option_decoder_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_decoder_state",
                                False,
                            )
                        ),
                        "direct_policy_option_recurrent_dynamics": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_recurrent_dynamics",
                                False,
                            )
                        ),
                        "direct_policy_option_sequence_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_sequence_head",
                                False,
                            )
                        ),
                        "direct_policy_option_decoder_recurrent_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_decoder_recurrent_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_transition_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_transition_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_controller_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_controller_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_token_decoder": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_token_decoder",
                                False,
                            )
                        ),
                        "direct_policy_option_action_recurrent_core": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_recurrent_core",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_recurrent_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_recurrent_head",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_policy_path": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_policy_path",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_backbone": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_backbone",
                                False,
                            )
                        ),
                        "direct_policy_executive_physiology_option_gating": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_physiology_option_gating",
                                False,
                            )
                        ),
                        "direct_policy_executive_affordance_action_gating": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_affordance_action_gating",
                                False,
                            )
                        ),
                        "direct_policy_executive_option_action_masking": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_option_action_masking",
                                False,
                            )
                        ),
                        "direct_policy_executive_event_release_latching": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_event_release_latching",
                                False,
                            )
                        ),
                        "direct_policy_executive_event_release_action_commitment": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_event_release_action_commitment",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_phase_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_phase_state",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_exit_contract": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_exit_contract",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_substate_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_substate_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_continuation": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_continuation",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_guidance": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_guidance",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_commitment": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_commitment",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_heading_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_heading_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_smell_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_smell_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_corridor_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_corridor_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_corridor_affordance_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_corridor_affordance_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_return",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_vector_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_vector_return",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_path_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_path_return",
                                False,
                            )
                        ),
                    },
                }
            if phase_prediction is not None:
                direct_policy_trace_payload.update(
                    {
                        "phase_prediction": phase_prediction,
                        "phase_prediction_confidence": round(
                            float(phase_prediction_confidence),
                            6,
                        ),
                    }
                )
            attention_summary = getattr(
                self.true_monolithic_policy,
                "last_attention_summary",
                None,
            )
            if isinstance(attention_summary, dict):
                event_attention_top_type = attention_summary.get(
                    "event_attention_top_type"
                )
                event_attention_top_age = int(
                    attention_summary.get("event_attention_top_age", -1)
                )
                event_attention_entropy = float(
                    attention_summary.get("event_attention_entropy", 0.0)
                )
                if event_attention_top_type is not None:
                    direct_policy_trace_payload.update(
                        {
                            "event_attention_top_type": str(
                                event_attention_top_type
                            ),
                            "event_attention_top_age": int(
                                event_attention_top_age
                            ),
                            "event_attention_entropy": round(
                                float(event_attention_entropy),
                                6,
                            ),
                        }
                    )
            option_summary = getattr(self.true_monolithic_policy, "last_option_summary", None)
            if isinstance(option_summary, dict):
                selected_option_raw = option_summary.get("selected_option")
                if selected_option_raw is not None:
                    selected_option = str(selected_option_raw)
                    option_age = int(option_summary.get("option_age", -1))
                    option_termination_reason = str(
                        option_summary.get("option_termination_reason", "none")
                    )
                    option_logits = np.asarray(
                        option_summary.get("option_logits", []),
                        dtype=float,
                    ).copy()
                    direct_policy_trace_payload.update(
                        {
                            "selected_option": selected_option,
                            "option_age": int(option_age),
                            "option_termination_reason": option_termination_reason,
                            "option_logits": option_logits.round(6).tolist(),
                        }
                    )
                    option_leaf_logits = np.asarray(
                        option_summary.get("option_leaf_logits", []),
                        dtype=float,
                    ).copy()
                    option_owned_action_raw = option_summary.get("option_owned_action")
                    option_owned_action = (
                        None
                        if option_owned_action_raw is None
                        else str(option_owned_action_raw)
                    )
                    safety_mask_applied = bool(
                        option_summary.get("safety_mask_applied", False)
                    )
                    safety_masked_actions = [
                        str(action)
                        for action in option_summary.get(
                            "safety_masked_actions",
                            [],
                        )
                    ]
                    external_override_count = int(
                        option_summary.get("external_override_count", 0)
                    )
                    if option_leaf_logits.size > 0:
                        direct_policy_trace_payload["option_leaf_logits"] = (
                            option_leaf_logits.round(6).tolist()
                        )
                    if option_owned_action is not None:
                        direct_policy_trace_payload["option_owned_action"] = (
                            option_owned_action
                        )
                    direct_policy_trace_payload["safety_mask_applied"] = bool(
                        safety_mask_applied
                    )
                    direct_policy_trace_payload["safety_masked_actions"] = list(
                        safety_masked_actions
                    )
                    direct_policy_trace_payload["external_override_count"] = int(
                        external_override_count
                    )
            affordance_summary = getattr(
                self.true_monolithic_policy,
                "last_affordance_summary",
                None,
            )
            if isinstance(affordance_summary, dict):
                affordance_blocked_logits = np.asarray(
                    affordance_summary.get("blocked_logits", []),
                    dtype=float,
                ).copy()
                affordance_role_logits = np.asarray(
                    affordance_summary.get("role_logits", []),
                    dtype=float,
                ).copy()
                if affordance_blocked_logits.size > 0:
                    direct_policy_trace_payload["affordance_blocked_logits"] = (
                        affordance_blocked_logits.round(6).tolist()
                    )
                if affordance_role_logits.size > 0:
                    direct_policy_trace_payload["affordance_role_logits"] = (
                        affordance_role_logits.round(6).tolist()
                    )
                geometry_logits = np.asarray(
                    affordance_summary.get("geometry_logits", []),
                    dtype=float,
                ).copy()
                if geometry_logits.size > 0:
                    direct_policy_trace_payload["geometry_logits"] = (
                        geometry_logits.round(6).tolist()
                    )
                shelter_column_logits = np.asarray(
                    affordance_summary.get("shelter_column_logits", []),
                    dtype=float,
                ).copy()
                if shelter_column_logits.size > 0:
                    direct_policy_trace_payload["shelter_column_logits"] = (
                        shelter_column_logits.round(6).tolist()
                    )
                shelter_position_logits = np.asarray(
                    affordance_summary.get("shelter_position_logits", []),
                    dtype=float,
                ).copy()
                if shelter_position_logits.size > 0:
                    direct_policy_trace_payload["shelter_position_logits"] = (
                        shelter_position_logits.round(6).tolist()
                    )
                transition_prediction_logits = np.asarray(
                    affordance_summary.get("transition_prediction_logits", []),
                    dtype=float,
                ).copy()
                if transition_prediction_logits.size > 0:
                    direct_policy_trace_payload["transition_prediction_logits"] = (
                        transition_prediction_logits.round(6).tolist()
                    )
                transition_rollout_prediction_logits = np.asarray(
                    affordance_summary.get(
                        "transition_rollout_prediction_logits",
                        [],
                    ),
                    dtype=float,
                ).copy()
                if transition_rollout_prediction_logits.size > 0:
                    direct_policy_trace_payload[
                        "transition_rollout_prediction_logits"
                    ] = transition_rollout_prediction_logits.round(6).tolist()
            direct_result = ModuleResult(
                interface=None,
                name=self.TRUE_MONOLITHIC_POLICY_NAME,
                observation_key=self.TRUE_MONOLITHIC_POLICY_NAME,
                observation=monolithic_observation.copy(),
                logits=policy_logits.copy(),
                probs=softmax(policy_logits),
                active=True,
                reflex=None,
                neural_logits=policy_logits.copy(),
                reflex_delta_logits=np.zeros_like(policy_logits),
                post_reflex_logits=policy_logits.copy(),
            )
            module_results = [direct_result]
            direct_result.valence_role = "integrated_policy"
            direct_result.gate_weight = 1.0
            direct_result.gated_logits = direct_result.logits.copy()
            direct_result.contribution_share = 1.0
            direct_result.intent_before_gating = ACTIONS[int(np.argmax(direct_result.logits))]
            direct_result.intent_after_gating = direct_result.intent_before_gating
            total_logits_without_reflex = direct_result.logits.copy()
            total_logits = direct_result.logits.copy()
            action_center_logits = total_logits.copy()
            action_center_policy = softmax(action_center_logits)
            proposal_sum = total_logits.copy()
            policy = softmax(total_logits)
            action_intent_idx = int(np.argmax(action_center_policy))
            action_intent_without_reflex_idx = action_intent_idx
            action_without_reflex_idx = action_intent_idx
            motor_action_idx = action_intent_idx
            arbitration = self._true_monolithic_arbitration_decision(
                module_name=self.TRUE_MONOLITHIC_POLICY_NAME,
                action_idx=action_intent_idx,
            )
            if (
                self.config.enable_food_direction_bias
                and not self.config.direct_policy_owned_option_controller
                and not training_mode
                and not sample
            ):
                bias_action = self._threat_escape_bias_action(observation)
                if bias_action is None:
                    bias_action = self._sleep_rest_bias_action(observation)
                if (
                    bias_action is None
                    and self._true_monolithic_allows_food_direction_bias()
                ):
                    bias_action = self._food_direction_bias_action(observation)
                if bias_action is not None:
                    total_logits = total_logits.copy()
                    threat_bias_action = self._threat_escape_bias_action(observation)
                    sleep_bias_action = self._sleep_rest_bias_action(observation)
                    if threat_bias_action is not None and bias_action == threat_bias_action:
                        bias_bonus = TRUE_MONOLITHIC_THREAT_ESCAPE_BIAS_LOGIT
                    elif sleep_bias_action is not None and bias_action == sleep_bias_action:
                        bias_bonus = TRUE_MONOLITHIC_SLEEP_REST_BIAS_LOGIT
                    else:
                        bias_bonus = TRUE_MONOLITHIC_DIRECTION_BIAS_LOGIT
                    total_logits[ACTION_TO_INDEX[bias_action]] += bias_bonus
                    policy = softmax(total_logits)
                    action_intent_idx = int(np.argmax(policy))
                    action_intent_without_reflex_idx = action_intent_idx
                    action_without_reflex_idx = action_intent_idx
                    motor_action_idx = action_intent_idx
                    action_center_logits = total_logits.copy()
                    action_center_policy = policy.copy()
                    proposal_sum = total_logits.copy()
                    arbitration = replace(
                        arbitration,
                        food_bias_applied=True,
                        food_bias_action=bias_action,
                        intent_before_gating_idx=int(np.argmax(total_logits_without_reflex)),
                        intent_after_gating_idx=action_intent_idx,
                    )
                    if hasattr(self.true_monolithic_policy, "record_external_override"):
                        self.true_monolithic_policy.record_external_override("final_bias")
                        external_override_count = int(
                            getattr(
                                self.true_monolithic_policy,
                                "external_override_count",
                                external_override_count,
                            )
                        )
                        direct_policy_trace_payload["external_override_count"] = int(
                            external_override_count
                        )
            if sample:
                action_idx = int(self.rng.choice(self.action_dim, p=policy))
            else:
                action_idx = motor_action_idx
            if hasattr(self.true_monolithic_policy, "record_executed_action"):
                self.true_monolithic_policy.record_executed_action(action_idx)
            motor_override = False
            final_reflex_override = False
        else:
            module_results = self._proposal_results(
                observation,
                store_cache=store_cache,
                training=training_mode,
            )
            arbitration_without_reflex = self._compute_arbitration(
                module_results,
                observation,
                training=False,
                store_cache=False,
            )
            gated_logits_without_reflex = [
                arbitration_gate_weight_for(arbitration_without_reflex, result.name) * result.logits
                for result in module_results
            ]
            proposal_sum_without_reflex = np.sum(
                np.stack(gated_logits_without_reflex, axis=0),
                axis=0,
            )
            if policy_mode == "reflex_only":
                action_center_logits_without_reflex = proposal_sum_without_reflex.copy()
                action_intent_without_reflex_idx = int(
                    np.argmax(action_center_logits_without_reflex)
                )
                total_logits_without_reflex = proposal_sum_without_reflex.copy()
            else:
                action_context_mapping = self._bound_action_context(observation)
                action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
                action_input_without_reflex = np.concatenate(
                    [np.concatenate(gated_logits_without_reflex, axis=0), action_context],
                    axis=0,
                )
                if self.action_center is None:
                    raise RuntimeError("Action center unavailable for the configured architecture.")
                action_center_correction_without_reflex, _ = self.action_center.forward(
                    action_input_without_reflex,
                    store_cache=False,
                )
                action_center_logits_without_reflex = (
                    proposal_sum_without_reflex + action_center_correction_without_reflex
                )
                action_intent_without_reflex_idx = int(
                    np.argmax(action_center_logits_without_reflex)
                )
                motor_input_without_reflex = self._build_motor_input(
                    one_hot(action_intent_without_reflex_idx, self.action_dim),
                    observation,
                )
                if self.motor_cortex is None:
                    raise RuntimeError("Motor cortex unavailable for the configured architecture.")
                motor_correction_without_reflex = self.motor_cortex.forward(
                    motor_input_without_reflex,
                    store_cache=False,
                )
                total_logits_without_reflex = (
                    action_center_logits_without_reflex + motor_correction_without_reflex
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
                training=training_mode,
                store_cache=store_cache,
            )
            apply_priority_gating(
                module_results,
                arbitration,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )

        if bus is not None:
            for result in module_results:
                bus.publish(
                    sender=result.name,
                    topic="action.proposal",
                    payload={
                        "active": bool(result.active),
                        "action_logits": result.logits.round(6).tolist(),
                        "action_probs": result.probs.round(6).tolist(),
                        "neural_logits": result.neural_logits.round(6).tolist() if result.neural_logits is not None else None,
                        "reflex_delta_logits": result.reflex_delta_logits.round(6).tolist() if result.reflex_delta_logits is not None else None,
                        "post_reflex_logits": result.post_reflex_logits.round(6).tolist() if result.post_reflex_logits is not None else None,
                        "reflex_applied": bool(result.reflex_applied),
                        "effective_reflex_scale": round(float(result.effective_reflex_scale), 6),
                        "module_reflex_override": bool(result.module_reflex_override),
                        "module_reflex_dominance": round(float(result.module_reflex_dominance), 6),
                        "reflex": result.reflex.to_payload() if result.reflex is not None else None,
                        "valence_role": result.valence_role,
                        "gate_weight": round(float(result.gate_weight), 6),
                        "contribution_share": round(float(result.contribution_share), 6),
                        "gated_logits": result.gated_logits.round(6).tolist() if result.gated_logits is not None else None,
                        "intent_before_gating": result.intent_before_gating,
                        "intent_after_gating": result.intent_after_gating,
                        **(
                            direct_policy_trace_payload
                            if result.name
                            in {self.TRUE_MONOLITHIC_POLICY_NAME, B_SERIES_POLICY_NAME}
                            else {}
                        ),
                    },
                )

        if not (self.config.is_true_monolithic or self.config.is_b_series):
            proposal_sum = np.sum(
                np.stack(
                    [
                        result.gated_logits if result.gated_logits is not None else result.logits
                        for result in module_results
                    ],
                    axis=0,
                ),
                axis=0,
            )
            action_center_input = self._build_action_input(module_results, observation)
            if policy_mode == "reflex_only":
                action_center_logits = proposal_sum.copy()
                action_center_policy = softmax(action_center_logits)
                action_intent_idx = int(np.argmax(action_center_policy))
                motor_input = self._build_motor_input(
                    one_hot(action_intent_idx, self.action_dim),
                    observation,
                )
                total_logits = action_center_logits.copy()
                policy = softmax(total_logits)
                action_without_reflex_idx = int(np.argmax(total_logits_without_reflex))
                motor_action_idx = int(np.argmax(total_logits))
                if sample:
                    action_idx = int(self.rng.choice(self.action_dim, p=policy))
                else:
                    action_idx = motor_action_idx
                motor_override = False
                final_reflex_override = action_without_reflex_idx != motor_action_idx
            else:
                if self.action_center is None or self.motor_cortex is None:
                    raise RuntimeError(
                        "Action/motor pipeline unavailable for the configured architecture."
                    )
                action_center_correction_logits, value = self.action_center.forward(
                    action_center_input,
                    store_cache=store_cache,
                )
                action_center_logits = proposal_sum + action_center_correction_logits
                action_center_policy = softmax(action_center_logits)
                action_intent_idx = int(np.argmax(action_center_policy))
                motor_input = self._build_motor_input(
                    one_hot(action_intent_idx, self.action_dim),
                    observation,
                )
                motor_correction_logits = self.motor_cortex.forward(
                    motor_input,
                    store_cache=store_cache,
                )
                total_logits = action_center_logits + motor_correction_logits
                motor_action_idx_before_food_bias = int(np.argmax(total_logits))
                if (
                    self.config.enable_food_direction_bias
                    and not training_mode
                    and not sample
                    and arbitration is not None
                    and arbitration.winning_valence == "hunger"
                ):
                    food_bias_action = self._food_direction_bias_action(observation)
                    if food_bias_action is not None:
                        total_logits = total_logits.copy()
                        total_logits[ACTION_TO_INDEX[food_bias_action]] += (
                            MODULAR_DIRECTION_BIAS_LOGIT
                        )
                        arbitration = replace(
                            arbitration,
                            food_bias_applied=True,
                            food_bias_action=food_bias_action,
                        )
                policy = softmax(total_logits)
                action_without_reflex_idx = int(np.argmax(total_logits_without_reflex))
                motor_action_idx = int(np.argmax(total_logits))
                if sample:
                    action_idx = int(self.rng.choice(self.action_dim, p=policy))
                else:
                    action_idx = motor_action_idx
                motor_override = action_intent_idx != motor_action_idx
                final_reflex_override = (
                    action_intent_without_reflex_idx != action_intent_idx
                    or action_without_reflex_idx != motor_action_idx_before_food_bias
                )

        execution_diagnostics = self._motor_execution_diagnostics(
            observation,
            action_idx,
        )
        orientation_alignment = float(execution_diagnostics["orientation_alignment"])
        terrain_difficulty = float(execution_diagnostics["terrain_difficulty"])
        momentum = float(execution_diagnostics["momentum"])
        execution_difficulty = float(execution_diagnostics["execution_difficulty"])

        if bus is not None:
            if self.config.is_b_series:
                bus.publish(
                    sender=B_SERIES_POLICY_NAME,
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "direct_policy_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "value_estimate": round(float(value), 6),
                        **direct_policy_trace_payload,
                    },
                )
                if bool(direct_policy_trace_payload.get("b6_emit_action_center_payload", False)):
                    b6_action_center_payload = direct_policy_trace_payload.get(
                        "b6_action_center_payload",
                        {},
                    )
                    b6_action_center_payload = (
                        b6_action_center_payload
                        if isinstance(b6_action_center_payload, dict)
                        else {}
                    )
                    bus.publish(
                        sender="action_center",
                        topic="action.selection",
                        payload={
                            "policy_mode": policy_mode,
                            "direct_policy_logits": total_logits.round(6).tolist(),
                            "policy": policy.round(6).tolist(),
                            "selected_intent": ACTIONS[motor_action_idx],
                            "selected_action": ACTIONS[motor_action_idx],
                            "executed_action": ACTIONS[action_idx],
                            "value_estimate": round(float(value), 6),
                            **b6_action_center_payload,
                            "b6_controller_family": direct_policy_trace_payload.get(
                                "b6_controller_family"
                            ),
                            "b6_controller_profile": direct_policy_trace_payload.get(
                                "b6_controller_profile"
                            ),
                            "b6_decision": direct_policy_trace_payload.get(
                                "b6_decision"
                            ),
                        },
                    )
            elif self.config.is_true_monolithic:
                bus.publish(
                    sender=self.TRUE_MONOLITHIC_POLICY_NAME,
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "direct_policy_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "value_estimate": round(float(value), 6),
                        **direct_policy_trace_payload,
                    },
                )
                self._direct_policy_hidden_reset_pending = False
            else:
                arbitration_payload = arbitration.to_payload() if arbitration is not None else {}
                bus.publish(
                    sender="action_center",
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "proposal_sum_logits": proposal_sum.round(6).tolist(),
                        "action_center_correction_logits": action_center_correction_logits.round(6).tolist(),
                        "action_center_logits": action_center_logits.round(6).tolist(),
                        "action_center_policy": action_center_policy.round(6).tolist(),
                        "selected_intent": ACTIONS[action_intent_idx],
                        "selected_intent_without_reflex": ACTIONS[action_intent_without_reflex_idx],
                        "value_estimate": round(float(value), 6),
                        **arbitration_payload,
                    },
                )
                bus.publish(
                    sender="motor_cortex",
                    topic="action.execution",
                    payload={
                        "policy_mode": policy_mode,
                        "motor_correction_logits": motor_correction_logits.round(6).tolist(),
                        "total_logits_without_reflex": total_logits_without_reflex.round(6).tolist(),
                        "total_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_intent": ACTIONS[action_intent_idx],
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "selected_action_without_reflex": ACTIONS[action_without_reflex_idx],
                        "motor_override": bool(motor_override),
                        "final_reflex_override": bool(final_reflex_override),
                        "orientation_alignment": round(float(orientation_alignment), 6),
                        "terrain_difficulty": round(float(terrain_difficulty), 6),
                        "momentum": round(float(momentum), 6),
                        "execution_difficulty": round(float(execution_difficulty), 6),
                        "execution_slip_occurred": False,
                        "slip_reason": "none",
                    },
                )

        step_observation: Dict[str, np.ndarray] = {}
        if policy_mode == "normal" and (
            self.config.is_true_monolithic or self.config.is_b_series
        ):
            brain_observation_keys = set(observation.keys())
            step_observation = {
                key: np.asarray(observation[key], dtype=float).copy()
                for key in brain_observation_keys
                if key in observation and key != "meta"
            }
        elif (
            policy_mode == "normal"
            and self.config.is_modular
            and self.config.uses_counterfactual_credit
        ):
            brain_observation_keys = {
                spec.observation_key for spec in MODULE_INTERFACES
            }
            brain_observation_keys.update(
                {
                    ACTION_CONTEXT_INTERFACE.observation_key,
                    MOTOR_CONTEXT_INTERFACE.observation_key,
                }
            )
            step_observation = {
                key: np.asarray(observation[key], dtype=float).copy()
                for key in brain_observation_keys
                if key in observation
            }

        return BrainStep(
            module_results=module_results,
            action_center_logits=action_center_logits,
            action_center_policy=action_center_policy,
            motor_correction_logits=motor_correction_logits,
            observation=step_observation,
            total_logits_without_reflex=total_logits_without_reflex,
            total_logits=total_logits,
            policy=policy,
            value=float(value),
            action_intent_idx=action_intent_idx,
            motor_action_idx=motor_action_idx,
            action_idx=action_idx,
            orientation_alignment=orientation_alignment,
            terrain_difficulty=terrain_difficulty,
            momentum=momentum,
            execution_difficulty=execution_difficulty,
            execution_slip_occurred=False,
            motor_slip_occurred=False,
            motor_noise_applied=False,
            slip_reason="none",
            motor_override=bool(motor_override),
            final_reflex_override=bool(final_reflex_override),
            action_center_input=action_center_input,
            motor_input=motor_input,
            policy_mode=policy_mode,
            arbitration_decision=arbitration,
            phase_logits=phase_logits,
            phase_prediction=phase_prediction,
            phase_prediction_confidence=float(phase_prediction_confidence),
            event_attention_top_type=event_attention_top_type,
            event_attention_top_age=int(event_attention_top_age),
            event_attention_entropy=float(event_attention_entropy),
            selected_option=selected_option,
            option_age=int(option_age),
            option_termination_reason=option_termination_reason,
            option_logits=np.asarray(option_logits, dtype=float).copy(),
            option_leaf_logits=np.asarray(option_leaf_logits, dtype=float).copy(),
            option_owned_action=option_owned_action,
            safety_mask_applied=bool(safety_mask_applied),
            safety_masked_actions=tuple(safety_masked_actions),
            external_override_count=int(external_override_count),
            affordance_blocked_logits=np.asarray(
                affordance_blocked_logits,
                dtype=float,
            ).copy(),
            affordance_role_logits=np.asarray(
                affordance_role_logits,
                dtype=float,
            ).copy(),
            geometry_logits=np.asarray(geometry_logits, dtype=float).copy(),
            shelter_column_logits=np.asarray(
                shelter_column_logits,
                dtype=float,
            ).copy(),
            shelter_position_logits=np.asarray(
                shelter_position_logits,
                dtype=float,
            ).copy(),
            transition_prediction_logits=np.asarray(
                transition_prediction_logits,
                dtype=float,
            ).copy(),
            transition_rollout_prediction_logits=np.asarray(
                transition_rollout_prediction_logits,
                dtype=float,
            ).copy(),
            **self._b_series_step_payload(
                direct_policy_trace_payload,
                semantic_logits,
                semantic_policy,
            ),
        )
