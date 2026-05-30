from __future__ import annotations


def append_b_series_trace_fields(item: dict[str, object], decision) -> None:
    item["b_level"] = int(decision.b_level)
    item["b_effective_level"] = decision.b_effective_level
    item["b_mode"] = decision.b_mode
    item["b_parent_level"] = decision.b_parent_level
    item["b_transfer_source_checkpoint"] = (
        decision.b_transfer_source_checkpoint
    )
    item["b_transfer_coverage"] = decision.b_transfer_coverage
    item["b_current_threat_pressure"] = (
        decision.b_current_threat_pressure
    )
    item["b_temporal_threat_pressure"] = (
        decision.b_temporal_threat_pressure
    )
    item["b_predator_memory_pressure"] = (
        decision.b_predator_memory_pressure
    )
    item["b_predator_trace_pressure"] = (
        decision.b_predator_trace_pressure
    )
    item["b3_contact_cooldown"] = decision.b3_contact_cooldown
    item["b3_post_food_cooldown"] = (
        decision.b3_post_food_cooldown
    )
    item["b3_hunger_drop"] = decision.b3_hunger_drop
    item["b3_controller_profile"] = decision.b3_controller_profile
    item["b4_controller_profile"] = decision.b4_controller_profile
    item["b4_recovery_pressure"] = decision.b4_recovery_pressure
    item["b4_sleep_hold"] = decision.b4_sleep_hold
    item["b4_exit_blocked"] = decision.b4_exit_blocked
    item["b4_hunger_release"] = decision.b4_hunger_release
    item["b4_genetic_generation"] = decision.b4_genetic_generation
    item["b4_genetic_candidate"] = decision.b4_genetic_candidate
    item["b5_controller_profile"] = decision.b5_controller_profile
    item["b5_hunger_urgency"] = decision.b5_hunger_urgency
    item["b5_sleep_pressure"] = decision.b5_sleep_pressure
    item["b5_recovery_debt"] = decision.b5_recovery_debt
    item["b5_threat_gate"] = decision.b5_threat_gate
    item["b5_sleep_bout_lock"] = decision.b5_sleep_bout_lock
    item["b5_forage_commitment_lock"] = (
        decision.b5_forage_commitment_lock
    )
    item["b5_homeostatic_decision"] = (
        decision.b5_homeostatic_decision
    )
    item["b5_genetic_generation"] = decision.b5_genetic_generation
    item["b5_genetic_candidate"] = decision.b5_genetic_candidate
    item["b6_controller_family"] = decision.b6_controller_family
    item["b6_controller_profile"] = decision.b6_controller_profile
    item["b6_risk_pressure"] = decision.b6_risk_pressure
    item["b6_threat_priority"] = decision.b6_threat_priority
    item["b6_forage_suppressed"] = decision.b6_forage_suppressed
    item["b6_corridor_commitment"] = (
        decision.b6_corridor_commitment
    )
    item["b6_corridor_progress_memory"] = (
        decision.b6_corridor_progress_memory
    )
    item["b6_recurrent_state"] = decision.b6_recurrent_state
    item["b6_return_lock"] = decision.b6_return_lock
    item["b6_decision"] = decision.b6_decision
    item["b6_genetic_generation"] = decision.b6_genetic_generation
    item["b6_genetic_candidate"] = decision.b6_genetic_candidate
    item["b7_controller_profile"] = decision.b7_controller_profile
    item["b7_affordance_state"] = decision.b7_affordance_state
    item["b7_energy_budget"] = decision.b7_energy_budget
    item["b7_budget_margin"] = decision.b7_budget_margin
    item["b7_food_steps_estimate"] = (
        decision.b7_food_steps_estimate
    )
    item["b7_return_steps_estimate"] = (
        decision.b7_return_steps_estimate
    )
    item["b7_corridor_viability"] = (
        decision.b7_corridor_viability
    )
    item["b7_abort_return"] = decision.b7_abort_return
    item["b7_commitment_lock"] = decision.b7_commitment_lock
    item["b7_decision"] = decision.b7_decision
    item["b7_genetic_generation"] = decision.b7_genetic_generation
    item["b7_genetic_candidate"] = decision.b7_genetic_candidate
    item["b8_controller_profile"] = decision.b8_controller_profile
    item["b8_spatial_map_state"] = decision.b8_spatial_map_state
    item["b8_local_affordance_score"] = (
        decision.b8_local_affordance_score
    )
    item["b8_return_vector_strength"] = (
        decision.b8_return_vector_strength
    )
    item["b8_corridor_dead_end_risk"] = (
        decision.b8_corridor_dead_end_risk
    )
    item["b8_abort_executed"] = decision.b8_abort_executed
    item["b8_place_memory"] = decision.b8_place_memory
    item["b8_decision"] = decision.b8_decision
    item["b8_genetic_generation"] = decision.b8_genetic_generation
    item["b8_genetic_candidate"] = decision.b8_genetic_candidate
    item["b9_controller_profile"] = decision.b9_controller_profile
    item["b9_route_state"] = decision.b9_route_state
    item["b9_route_confidence"] = decision.b9_route_confidence
    item["b9_waypoint_lock"] = decision.b9_waypoint_lock
    item["b9_path_integrator"] = decision.b9_path_integrator
    item["b9_replan_signal"] = decision.b9_replan_signal
    item["b9_decision"] = decision.b9_decision
    item["b9_genetic_generation"] = decision.b9_genetic_generation
    item["b9_genetic_candidate"] = decision.b9_genetic_candidate
    item["b10_controller_profile"] = decision.b10_controller_profile
    item["b10_replay_state"] = decision.b10_replay_state
    item["b10_prospective_value"] = decision.b10_prospective_value
    item["b10_rollout_depth"] = decision.b10_rollout_depth
    item["b10_replay_memory"] = decision.b10_replay_memory
    item["b10_plan_commitment"] = decision.b10_plan_commitment
    item["b10_abort_signal"] = decision.b10_abort_signal
    item["b10_decision"] = decision.b10_decision
    item["b10_genetic_generation"] = decision.b10_genetic_generation
    item["b10_genetic_candidate"] = decision.b10_genetic_candidate
    item["b11_controller_profile"] = decision.b11_controller_profile
    item["b11_confidence_state"] = decision.b11_confidence_state
    item["b11_plan_confidence"] = decision.b11_plan_confidence
    item["b11_uncertainty"] = decision.b11_uncertainty
    item["b11_neuromod_signal"] = decision.b11_neuromod_signal
    item["b11_confidence_lock"] = decision.b11_confidence_lock
    item["b11_decision"] = decision.b11_decision
    item["b11_genetic_generation"] = decision.b11_genetic_generation
    item["b11_genetic_candidate"] = decision.b11_genetic_candidate
    item["b12_controller_profile"] = decision.b12_controller_profile
    item["b12_attention_state"] = decision.b12_attention_state
    item["b12_prediction_error"] = decision.b12_prediction_error
    item["b12_attention_gain"] = decision.b12_attention_gain
    item["b12_expected_progress"] = decision.b12_expected_progress
    item["b12_search_lock"] = decision.b12_search_lock
    item["b12_decision"] = decision.b12_decision
    item["b12_genetic_generation"] = decision.b12_genetic_generation
    item["b12_genetic_candidate"] = decision.b12_genetic_candidate
    item["b13_controller_profile"] = decision.b13_controller_profile
    item["b13_search_state"] = decision.b13_search_state
    item["b13_local_route_score"] = decision.b13_local_route_score
    item["b13_affordance_samples"] = decision.b13_affordance_samples
    item["b13_search_memory"] = decision.b13_search_memory
    item["b13_dead_end_score"] = decision.b13_dead_end_score
    item["b13_search_lock"] = decision.b13_search_lock
    item["b13_decision"] = decision.b13_decision
    item["b13_genetic_generation"] = decision.b13_genetic_generation
    item["b13_genetic_candidate"] = decision.b13_genetic_candidate
    item["b14_controller_profile"] = decision.b14_controller_profile
    item["b14_uncertainty_state"] = decision.b14_uncertainty_state
    item["b14_affordance_confidence"] = decision.b14_affordance_confidence
    item["b14_uncertainty"] = decision.b14_uncertainty
    item["b14_risk_adjusted_score"] = decision.b14_risk_adjusted_score
    item["b14_commitment_lock"] = decision.b14_commitment_lock
    item["b14_decision"] = decision.b14_decision
    item["b14_genetic_generation"] = decision.b14_genetic_generation
    item["b14_genetic_candidate"] = decision.b14_genetic_candidate
    item["b15_controller_profile"] = decision.b15_controller_profile
    item["b15_option_state"] = decision.b15_option_state
    item["b15_option_value"] = decision.b15_option_value
    item["b15_termination_pressure"] = decision.b15_termination_pressure
    item["b15_persistence_score"] = decision.b15_persistence_score
    item["b15_option_lock"] = decision.b15_option_lock
    item["b15_decision"] = decision.b15_decision
    item["b15_genetic_generation"] = decision.b15_genetic_generation
    item["b15_genetic_candidate"] = decision.b15_genetic_candidate
    item["b16_controller_profile"] = decision.b16_controller_profile
    item["b16_ensemble_state"] = decision.b16_ensemble_state
    item["b16_continue_vote"] = decision.b16_continue_vote
    item["b16_return_vote"] = decision.b16_return_vote
    item["b16_option_votes"] = decision.b16_option_votes
    item["b16_consensus_score"] = decision.b16_consensus_score
    item["b16_conflict_score"] = decision.b16_conflict_score
    item["b16_ensemble_lock"] = decision.b16_ensemble_lock
    item["b16_decision"] = decision.b16_decision
    item["b16_genetic_generation"] = decision.b16_genetic_generation
    item["b16_genetic_candidate"] = decision.b16_genetic_candidate
    item["b17_controller_profile"] = decision.b17_controller_profile
    item["b17_modulator_state"] = decision.b17_modulator_state
    item["b17_arousal_signal"] = decision.b17_arousal_signal
    item["b17_homeostatic_gain"] = decision.b17_homeostatic_gain
    item["b17_option_gain"] = decision.b17_option_gain
    item["b17_conflict_release"] = decision.b17_conflict_release
    item["b17_modulation_lock"] = decision.b17_modulation_lock
    item["b17_decision"] = decision.b17_decision
    item["b17_genetic_generation"] = decision.b17_genetic_generation
    item["b17_genetic_candidate"] = decision.b17_genetic_candidate
    item["b18_controller_profile"] = decision.b18_controller_profile
    item["b18_trace_state"] = decision.b18_trace_state
    item["b18_eligibility_trace"] = decision.b18_eligibility_trace
    item["b18_reward_prediction_proxy"] = (
        decision.b18_reward_prediction_proxy
    )
    item["b18_stability_bias"] = decision.b18_stability_bias
    item["b18_switch_pressure"] = decision.b18_switch_pressure
    item["b18_trace_lock"] = decision.b18_trace_lock
    item["b18_decision"] = decision.b18_decision
    item["b18_genetic_generation"] = decision.b18_genetic_generation
    item["b18_genetic_candidate"] = decision.b18_genetic_candidate
    item["b19_controller_profile"] = decision.b19_controller_profile
    item["b19_memory_state"] = decision.b19_memory_state
    item["b19_episode_memory"] = decision.b19_episode_memory
    item["b19_consolidation_score"] = decision.b19_consolidation_score
    item["b19_stability_vote"] = decision.b19_stability_vote
    item["b19_switch_suppression"] = decision.b19_switch_suppression
    item["b19_memory_lock"] = decision.b19_memory_lock
    item["b19_decision"] = decision.b19_decision
    item["b19_genetic_generation"] = decision.b19_genetic_generation
    item["b19_genetic_candidate"] = decision.b19_genetic_candidate
    item["b20_controller_profile"] = decision.b20_controller_profile
    item["b20_buffer_state"] = decision.b20_buffer_state
    item["b20_working_buffer"] = decision.b20_working_buffer
    item["b20_context_binding"] = decision.b20_context_binding
    item["b20_gate_vote"] = decision.b20_gate_vote
    item["b20_release_vote"] = decision.b20_release_vote
    item["b20_buffer_lock"] = decision.b20_buffer_lock
    item["b20_decision"] = decision.b20_decision
    item["b20_genetic_generation"] = decision.b20_genetic_generation
    item["b20_genetic_candidate"] = decision.b20_genetic_candidate
    item["b21_controller_profile"] = decision.b21_controller_profile
    item["b21_replay_state"] = decision.b21_replay_state
    item["b21_sequence_memory"] = decision.b21_sequence_memory
    item["b21_replay_score"] = decision.b21_replay_score
    item["b21_route_commitment"] = decision.b21_route_commitment
    item["b21_abort_prediction"] = decision.b21_abort_prediction
    item["b21_replay_lock"] = decision.b21_replay_lock
    item["b21_decision"] = decision.b21_decision
    item["b21_genetic_generation"] = decision.b21_genetic_generation
    item["b21_genetic_candidate"] = decision.b21_genetic_candidate
    item["b22_controller_profile"] = decision.b22_controller_profile
    item["b22_sim_state"] = decision.b22_sim_state
    item["b22_prospective_sim"] = decision.b22_prospective_sim
    item["b22_forward_model_score"] = (
        decision.b22_forward_model_score
    )
    item["b22_viability_projection"] = (
        decision.b22_viability_projection
    )
    item["b22_abort_projection"] = decision.b22_abort_projection
    item["b22_sim_lock"] = decision.b22_sim_lock
    item["b22_decision"] = decision.b22_decision
    item["b22_genetic_generation"] = decision.b22_genetic_generation
    item["b22_genetic_candidate"] = decision.b22_genetic_candidate
    item["b23_controller_profile"] = decision.b23_controller_profile
    item["b23_conflict_state"] = decision.b23_conflict_state
    item["b23_prediction_error"] = decision.b23_prediction_error
    item["b23_conflict_memory"] = decision.b23_conflict_memory
    item["b23_stability_vote"] = decision.b23_stability_vote
    item["b23_abort_bias"] = decision.b23_abort_bias
    item["b23_monitor_lock"] = decision.b23_monitor_lock
    item["b23_decision"] = decision.b23_decision
    item["b23_genetic_generation"] = decision.b23_genetic_generation
    item["b23_genetic_candidate"] = decision.b23_genetic_candidate
    item["b24_controller_profile"] = decision.b24_controller_profile
    item["b24_precision_state"] = decision.b24_precision_state
    item["b24_precision_memory"] = decision.b24_precision_memory
    item["b24_precision_vote"] = decision.b24_precision_vote
    item["b24_uncertainty_pressure"] = (
        decision.b24_uncertainty_pressure
    )
    item["b24_abort_precision"] = decision.b24_abort_precision
    item["b24_precision_lock"] = decision.b24_precision_lock
    item["b24_decision"] = decision.b24_decision
    item["b24_genetic_generation"] = decision.b24_genetic_generation
    item["b24_genetic_candidate"] = decision.b24_genetic_candidate
    item["b25_controller_profile"] = decision.b25_controller_profile
    item["b25_metacognitive_state"] = decision.b25_metacognitive_state
    item["b25_confidence_memory"] = decision.b25_confidence_memory
    item["b25_confidence_vote"] = decision.b25_confidence_vote
    item["b25_doubt_pressure"] = decision.b25_doubt_pressure
    item["b25_control_gain"] = decision.b25_control_gain
    item["b25_meta_lock"] = decision.b25_meta_lock
    item["b25_decision"] = decision.b25_decision
    item["b25_genetic_generation"] = decision.b25_genetic_generation
    item["b25_genetic_candidate"] = decision.b25_genetic_candidate
    item["b26_controller_profile"] = decision.b26_controller_profile
    item["b26_allostatic_state"] = decision.b26_allostatic_state
    item["b26_prediction_error"] = decision.b26_prediction_error
    item["b26_setpoint_pressure"] = decision.b26_setpoint_pressure
    item["b26_control_vote"] = decision.b26_control_vote
    item["b26_stability_lock"] = decision.b26_stability_lock
    item["b26_decision"] = decision.b26_decision
    item["b26_genetic_generation"] = decision.b26_genetic_generation
    item["b26_genetic_candidate"] = decision.b26_genetic_candidate
    item["b27_controller_profile"] = decision.b27_controller_profile
    item["b27_arousal_state"] = decision.b27_arousal_state
    item["b27_arousal_level"] = decision.b27_arousal_level
    item["b27_gain_modulation"] = decision.b27_gain_modulation
    item["b27_stress_pressure"] = decision.b27_stress_pressure
    item["b27_arousal_lock"] = decision.b27_arousal_lock
    item["b27_decision"] = decision.b27_decision
    item["b27_genetic_generation"] = decision.b27_genetic_generation
    item["b27_genetic_candidate"] = decision.b27_genetic_candidate
    item["b28_controller_profile"] = decision.b28_controller_profile
    item["b28_attention_state"] = decision.b28_attention_state
    item["b28_interoceptive_focus"] = decision.b28_interoceptive_focus
    item["b28_attention_gain"] = decision.b28_attention_gain
    item["b28_distractor_pressure"] = decision.b28_distractor_pressure
    item["b28_attention_lock"] = decision.b28_attention_lock
    item["b28_decision"] = decision.b28_decision
    item["b28_genetic_generation"] = decision.b28_genetic_generation
    item["b28_genetic_candidate"] = decision.b28_genetic_candidate
    item["b29_controller_profile"] = decision.b29_controller_profile
    item["b29_salience_state"] = decision.b29_salience_state
    item["b29_threat_salience"] = decision.b29_threat_salience
    item["b29_homeostatic_salience"] = (
        decision.b29_homeostatic_salience
    )
    item["b29_corridor_salience"] = decision.b29_corridor_salience
    item["b29_winner_channel"] = decision.b29_winner_channel
    item["b29_salience_lock"] = decision.b29_salience_lock
    item["b29_decision"] = decision.b29_decision
    item["b29_genetic_generation"] = decision.b29_genetic_generation
    item["b29_genetic_candidate"] = decision.b29_genetic_candidate
    item["b30_controller_profile"] = decision.b30_controller_profile
    item["b30_gate_state"] = decision.b30_gate_state
    item["b30_go_signal"] = decision.b30_go_signal
    item["b30_no_go_signal"] = decision.b30_no_go_signal
    item["b30_action_gate"] = decision.b30_action_gate
    item["b30_gate_lock"] = decision.b30_gate_lock
    item["b30_decision"] = decision.b30_decision
    item["b30_genetic_generation"] = decision.b30_genetic_generation
    item["b30_genetic_candidate"] = decision.b30_genetic_candidate
    item["b31_controller_profile"] = decision.b31_controller_profile
    item["b31_dopamine_state"] = decision.b31_dopamine_state
    item["b31_reward_prediction_error"] = (
        decision.b31_reward_prediction_error
    )
    item["b31_tonic_dopamine"] = decision.b31_tonic_dopamine
    item["b31_phasic_dopamine"] = decision.b31_phasic_dopamine
    item["b31_gate_bias"] = decision.b31_gate_bias
    item["b31_dopamine_lock"] = decision.b31_dopamine_lock
    item["b31_decision"] = decision.b31_decision
    item["b31_genetic_generation"] = decision.b31_genetic_generation
    item["b31_genetic_candidate"] = decision.b31_genetic_candidate
    item["b32_controller_profile"] = decision.b32_controller_profile
    item["b32_critic_value"] = decision.b32_critic_value
    item["b32_actor_advantage"] = decision.b32_actor_advantage
    item["b32_value_error"] = decision.b32_value_error
    item["b32_policy_bias"] = decision.b32_policy_bias
    item["b32_value_lock"] = decision.b32_value_lock
    item["b32_decision"] = decision.b32_decision
    item["b32_genetic_generation"] = decision.b32_genetic_generation
    item["b32_genetic_candidate"] = decision.b32_genetic_candidate
    item["b33_controller_profile"] = decision.b33_controller_profile
    item["b33_td_error"] = decision.b33_td_error
    item["b33_bootstrap_value"] = decision.b33_bootstrap_value
    item["b33_reward_trace"] = decision.b33_reward_trace
    item["b33_actor_update"] = decision.b33_actor_update
    item["b33_td_lock"] = decision.b33_td_lock
    item["b33_decision"] = decision.b33_decision
    item["b33_genetic_generation"] = decision.b33_genetic_generation
    item["b33_genetic_candidate"] = decision.b33_genetic_candidate
    item["b34_controller_profile"] = decision.b34_controller_profile
    item["b34_eligibility_trace"] = decision.b34_eligibility_trace
    item["b34_credit_assignment"] = decision.b34_credit_assignment
    item["b34_synaptic_tag"] = decision.b34_synaptic_tag
    item["b34_decay_memory"] = decision.b34_decay_memory
    item["b34_credit_lock"] = decision.b34_credit_lock
    item["b34_decision"] = decision.b34_decision
    item["b34_genetic_generation"] = decision.b34_genetic_generation
    item["b34_genetic_candidate"] = decision.b34_genetic_candidate
    item["b35_controller_profile"] = decision.b35_controller_profile
    item["b35_forward_value"] = decision.b35_forward_value
    item["b35_transition_error"] = decision.b35_transition_error
    item["b35_model_confidence"] = decision.b35_model_confidence
    item["b35_prediction_memory"] = decision.b35_prediction_memory
    item["b35_model_lock"] = decision.b35_model_lock
    item["b35_decision"] = decision.b35_decision
    item["b35_genetic_generation"] = decision.b35_genetic_generation
    item["b35_genetic_candidate"] = decision.b35_genetic_candidate
    item["b36_controller_profile"] = decision.b36_controller_profile
    item["b36_latent_state"] = decision.b36_latent_state
    item["b36_belief_error"] = decision.b36_belief_error
    item["b36_state_confidence"] = decision.b36_state_confidence
    item["b36_context_memory"] = decision.b36_context_memory
    item["b36_belief_lock"] = decision.b36_belief_lock
    item["b36_decision"] = decision.b36_decision
    item["b36_genetic_generation"] = decision.b36_genetic_generation
    item["b36_genetic_candidate"] = decision.b36_genetic_candidate
    item["b37_controller_profile"] = decision.b37_controller_profile
    item["b37_external_state_factor"] = (
        decision.b37_external_state_factor
    )
    item["b37_internal_state_factor"] = (
        decision.b37_internal_state_factor
    )
    item["b37_factor_alignment"] = decision.b37_factor_alignment
    item["b37_factor_confidence"] = decision.b37_factor_confidence
    item["b37_factor_lock"] = decision.b37_factor_lock
    item["b37_decision"] = decision.b37_decision
    item["b37_genetic_generation"] = decision.b37_genetic_generation
    item["b37_genetic_candidate"] = decision.b37_genetic_candidate
    item["b38_controller_profile"] = decision.b38_controller_profile
    item["b38_external_attention"] = decision.b38_external_attention
    item["b38_internal_attention"] = decision.b38_internal_attention
    item["b38_attention_balance"] = decision.b38_attention_balance
    item["b38_attention_gain"] = decision.b38_attention_gain
    item["b38_attention_lock"] = decision.b38_attention_lock
    item["b38_decision"] = decision.b38_decision
    item["b38_genetic_generation"] = decision.b38_genetic_generation
    item["b38_genetic_candidate"] = decision.b38_genetic_candidate
    item["b39_controller_profile"] = decision.b39_controller_profile
    item["b39_binding_strength"] = decision.b39_binding_strength
    item["b39_cross_factor_coherence"] = (
        decision.b39_cross_factor_coherence
    )
    item["b39_bound_context"] = decision.b39_bound_context
    item["b39_binding_gain"] = decision.b39_binding_gain
    item["b39_binding_lock"] = decision.b39_binding_lock
    item["b39_decision"] = decision.b39_decision
    item["b39_genetic_generation"] = decision.b39_genetic_generation
    item["b39_genetic_candidate"] = decision.b39_genetic_candidate
    item["b40_controller_profile"] = decision.b40_controller_profile
    item["b40_workspace_activation"] = decision.b40_workspace_activation
    item["b40_broadcast_gain"] = decision.b40_broadcast_gain
    item["b40_context_availability"] = (
        decision.b40_context_availability
    )
    item["b40_workspace_stability"] = decision.b40_workspace_stability
    item["b40_workspace_lock"] = decision.b40_workspace_lock
    item["b40_decision"] = decision.b40_decision
    item["b40_genetic_generation"] = decision.b40_genetic_generation
    item["b40_genetic_candidate"] = decision.b40_genetic_candidate
    item["b41_controller_profile"] = decision.b41_controller_profile
    item["b41_executive_selection"] = decision.b41_executive_selection
    item["b41_inhibitory_pressure"] = decision.b41_inhibitory_pressure
    item["b41_goal_context"] = decision.b41_goal_context
    item["b41_executive_stability"] = decision.b41_executive_stability
    item["b41_executive_lock"] = decision.b41_executive_lock
    item["b41_decision"] = decision.b41_decision
    item["b41_genetic_generation"] = decision.b41_genetic_generation
    item["b41_genetic_candidate"] = decision.b41_genetic_candidate
    item["b42_controller_profile"] = decision.b42_controller_profile
    item["b42_error_signal"] = decision.b42_error_signal
    item["b42_conflict_signal"] = decision.b42_conflict_signal
    item["b42_performance_context"] = decision.b42_performance_context
    item["b42_monitor_stability"] = decision.b42_monitor_stability
    item["b42_monitor_lock"] = decision.b42_monitor_lock
    item["b42_decision"] = decision.b42_decision
    item["b42_genetic_generation"] = decision.b42_genetic_generation
    item["b42_genetic_candidate"] = decision.b42_genetic_candidate
    item["b43_controller_profile"] = decision.b43_controller_profile
    item["b43_precision_signal"] = decision.b43_precision_signal
    item["b43_adaptive_threshold"] = decision.b43_adaptive_threshold
    item["b43_arousal_context"] = decision.b43_arousal_context
    item["b43_control_stability"] = decision.b43_control_stability
    item["b43_precision_lock"] = decision.b43_precision_lock
    item["b43_decision"] = decision.b43_decision
    item["b43_genetic_generation"] = decision.b43_genetic_generation
    item["b43_genetic_candidate"] = decision.b43_genetic_candidate
    item["b44_controller_profile"] = decision.b44_controller_profile
    item["b44_relay_gate"] = decision.b44_relay_gate
    item["b44_sensory_precision"] = decision.b44_sensory_precision
    item["b44_context_relay"] = decision.b44_context_relay
    item["b44_gate_stability"] = decision.b44_gate_stability
    item["b44_relay_lock"] = decision.b44_relay_lock
    item["b44_decision"] = decision.b44_decision
    item["b44_genetic_generation"] = decision.b44_genetic_generation
    item["b44_genetic_candidate"] = decision.b44_genetic_candidate
    item["b45_controller_profile"] = decision.b45_controller_profile
    item["b45_inhibitory_gate"] = decision.b45_inhibitory_gate
    item["b45_sensory_filter"] = decision.b45_sensory_filter
    item["b45_context_suppression"] = decision.b45_context_suppression
    item["b45_loop_stability"] = decision.b45_loop_stability
    item["b45_inhibition_lock"] = decision.b45_inhibition_lock
    item["b45_decision"] = decision.b45_decision
    item["b45_genetic_generation"] = decision.b45_genetic_generation
    item["b45_genetic_candidate"] = decision.b45_genetic_candidate
    item["b46_controller_profile"] = decision.b46_controller_profile
    item["b46_feedback_gain"] = decision.b46_feedback_gain
    item["b46_topdown_context"] = decision.b46_topdown_context
    item["b46_prediction_match"] = decision.b46_prediction_match
    item["b46_feedback_stability"] = decision.b46_feedback_stability
    item["b46_feedback_lock"] = decision.b46_feedback_lock
    item["b46_decision"] = decision.b46_decision
    item["b46_genetic_generation"] = decision.b46_genetic_generation
    item["b46_genetic_candidate"] = decision.b46_genetic_candidate
    item["b47_controller_profile"] = decision.b47_controller_profile
    item["b47_phase_alignment"] = decision.b47_phase_alignment
    item["b47_synchrony_gain"] = decision.b47_synchrony_gain
    item["b47_cross_loop_coherence"] = (
        decision.b47_cross_loop_coherence
    )
    item["b47_phase_lock"] = decision.b47_phase_lock
    item["b47_decision"] = decision.b47_decision
    item["b47_genetic_generation"] = decision.b47_genetic_generation
    item["b47_genetic_candidate"] = decision.b47_genetic_candidate
    item["b48_controller_profile"] = decision.b48_controller_profile
    item["b48_timing_error"] = decision.b48_timing_error
    item["b48_predictive_timing"] = decision.b48_predictive_timing
    item["b48_corrective_gain"] = decision.b48_corrective_gain
    item["b48_calibration_lock"] = decision.b48_calibration_lock
    item["b48_decision"] = decision.b48_decision
    item["b48_genetic_generation"] = decision.b48_genetic_generation
    item["b48_genetic_candidate"] = decision.b48_genetic_candidate
    item["b49_controller_profile"] = decision.b49_controller_profile
    item["b49_go_signal"] = decision.b49_go_signal
    item["b49_no_go_signal"] = decision.b49_no_go_signal
    item["b49_action_gate_balance"] = decision.b49_action_gate_balance
    item["b49_selection_lock"] = decision.b49_selection_lock
    item["b49_decision"] = decision.b49_decision
    item["b49_genetic_generation"] = decision.b49_genetic_generation
    item["b49_genetic_candidate"] = decision.b49_genetic_candidate
    item["b50_controller_profile"] = decision.b50_controller_profile
    item["b50_habit_strength"] = decision.b50_habit_strength
    item["b50_chunk_value"] = decision.b50_chunk_value
    item["b50_habit_stability"] = decision.b50_habit_stability
    item["b50_chunk_lock"] = decision.b50_chunk_lock
    item["b50_decision"] = decision.b50_decision
    item["b50_genetic_generation"] = decision.b50_genetic_generation
    item["b50_genetic_candidate"] = decision.b50_genetic_candidate
    item["b51_controller_profile"] = decision.b51_controller_profile
    item["b51_prediction_error"] = decision.b51_prediction_error
    item["b51_dopamine_gain"] = decision.b51_dopamine_gain
    item["b51_habit_modulation"] = decision.b51_habit_modulation
    item["b51_modulation_lock"] = decision.b51_modulation_lock
    item["b51_decision"] = decision.b51_decision
    item["b51_genetic_generation"] = decision.b51_genetic_generation
    item["b51_genetic_candidate"] = decision.b51_genetic_candidate
    item["b52_controller_profile"] = decision.b52_controller_profile
    item["b52_acetylcholine_level"] = decision.b52_acetylcholine_level
    item["b52_precision_gain"] = decision.b52_precision_gain
    item["b52_uncertainty_signal"] = decision.b52_uncertainty_signal
    item["b52_attention_lock"] = decision.b52_attention_lock
    item["b52_decision"] = decision.b52_decision
    item["b52_genetic_generation"] = decision.b52_genetic_generation
    item["b52_genetic_candidate"] = decision.b52_genetic_candidate
    item["b53_controller_profile"] = decision.b53_controller_profile
    item["b53_norepinephrine_level"] = decision.b53_norepinephrine_level
    item["b53_arousal_gain"] = decision.b53_arousal_gain
    item["b53_surprise_signal"] = decision.b53_surprise_signal
    item["b53_gain_lock"] = decision.b53_gain_lock
    item["b53_decision"] = decision.b53_decision
    item["b53_genetic_generation"] = decision.b53_genetic_generation
    item["b53_genetic_candidate"] = decision.b53_genetic_candidate
    item["b54_controller_profile"] = decision.b54_controller_profile
    item["b54_serotonin_level"] = decision.b54_serotonin_level
    item["b54_patience_signal"] = decision.b54_patience_signal
    item["b54_impulse_suppression"] = decision.b54_impulse_suppression
    item["b54_patience_lock"] = decision.b54_patience_lock
    item["b54_decision"] = decision.b54_decision
    item["b54_genetic_generation"] = decision.b54_genetic_generation
    item["b54_genetic_candidate"] = decision.b54_genetic_candidate
    item["b55_controller_profile"] = decision.b55_controller_profile
    item["b55_hypothalamic_drive"] = decision.b55_hypothalamic_drive
    item["b55_satiety_signal"] = decision.b55_satiety_signal
    item["b55_recovery_bias"] = decision.b55_recovery_bias
    item["b55_drive_balance"] = decision.b55_drive_balance
    item["b55_drive_lock"] = decision.b55_drive_lock
    item["b55_decision"] = decision.b55_decision
    item["b55_genetic_generation"] = decision.b55_genetic_generation
    item["b55_genetic_candidate"] = decision.b55_genetic_candidate
    item["b56_controller_profile"] = decision.b56_controller_profile
    item["b56_cortisol_level"] = decision.b56_cortisol_level
    item["b56_stress_load"] = decision.b56_stress_load
    item["b56_recovery_signal"] = decision.b56_recovery_signal
    item["b56_endocrine_balance"] = decision.b56_endocrine_balance
    item["b56_stress_lock"] = decision.b56_stress_lock
    item["b56_decision"] = decision.b56_decision
    item["b56_genetic_generation"] = decision.b56_genetic_generation
    item["b56_genetic_candidate"] = decision.b56_genetic_candidate
    item["b57_controller_profile"] = decision.b57_controller_profile
    item["b57_interoceptive_awareness"] = (
        decision.b57_interoceptive_awareness
    )
    item["b57_visceral_salience"] = decision.b57_visceral_salience
    item["b57_body_state_confidence"] = (
        decision.b57_body_state_confidence
    )
    item["b57_awareness_balance"] = decision.b57_awareness_balance
    item["b57_awareness_lock"] = decision.b57_awareness_lock
    item["b57_decision"] = decision.b57_decision
    item["b57_genetic_generation"] = decision.b57_genetic_generation
    item["b57_genetic_candidate"] = decision.b57_genetic_candidate
    item["b58_controller_profile"] = decision.b58_controller_profile
    item["b58_conflict_signal"] = decision.b58_conflict_signal
    item["b58_error_likelihood"] = decision.b58_error_likelihood
    item["b58_control_allocation"] = decision.b58_control_allocation
    item["b58_resolution_balance"] = decision.b58_resolution_balance
    item["b58_conflict_lock"] = decision.b58_conflict_lock
    item["b58_decision"] = decision.b58_decision
    item["b58_genetic_generation"] = decision.b58_genetic_generation
    item["b58_genetic_candidate"] = decision.b58_genetic_candidate
    item["b59_controller_profile"] = decision.b59_controller_profile
    item["b59_goal_context"] = decision.b59_goal_context
    item["b59_working_set_stability"] = (
        decision.b59_working_set_stability
    )
    item["b59_task_set_confidence"] = decision.b59_task_set_confidence
    item["b59_executive_balance"] = decision.b59_executive_balance
    item["b59_executive_lock"] = decision.b59_executive_lock
    item["b59_decision"] = decision.b59_decision
    item["b59_genetic_generation"] = decision.b59_genetic_generation
    item["b59_genetic_candidate"] = decision.b59_genetic_candidate
    item["b60_controller_profile"] = decision.b60_controller_profile
    item["b60_outcome_value"] = decision.b60_outcome_value
    item["b60_reversal_signal"] = decision.b60_reversal_signal
    item["b60_goal_value_confidence"] = (
        decision.b60_goal_value_confidence
    )
    item["b60_value_balance"] = decision.b60_value_balance
    item["b60_value_lock"] = decision.b60_value_lock
    item["b60_decision"] = decision.b60_decision
    item["b60_genetic_generation"] = decision.b60_genetic_generation
    item["b60_genetic_candidate"] = decision.b60_genetic_candidate
    item["b61_controller_profile"] = decision.b61_controller_profile
    item["b61_safety_value"] = decision.b61_safety_value
    item["b61_threat_value"] = decision.b61_threat_value
    item["b61_safety_confidence"] = decision.b61_safety_confidence
    item["b61_affective_balance"] = decision.b61_affective_balance
    item["b61_safety_lock"] = decision.b61_safety_lock
    item["b61_decision"] = decision.b61_decision
    item["b61_genetic_generation"] = decision.b61_genetic_generation
    item["b61_genetic_candidate"] = decision.b61_genetic_candidate
    item["b62_controller_profile"] = decision.b62_controller_profile
    item["b62_defensive_mode"] = decision.b62_defensive_mode
    item["b62_freeze_pressure"] = decision.b62_freeze_pressure
    item["b62_flee_pressure"] = decision.b62_flee_pressure
    item["b62_shelter_bias"] = decision.b62_shelter_bias
    item["b62_defense_balance"] = decision.b62_defense_balance
    item["b62_defense_lock"] = decision.b62_defense_lock
    item["b62_decision"] = decision.b62_decision
    item["b62_genetic_generation"] = decision.b62_genetic_generation
    item["b62_genetic_candidate"] = decision.b62_genetic_candidate
    item["semantic_action"] = decision.semantic_action
    item["learned_semantic_action"] = decision.learned_semantic_action
    item["semantic_action_source"] = decision.semantic_action_source
    item["semantic_action_reason"] = decision.semantic_action_reason
    item["semantic_override_count"] = int(
        decision.semantic_override_count
    )
    item["semantic_logits"] = decision.semantic_logits.round(6).tolist()
    item["bridge_primitive_action"] = decision.bridge_primitive_action
    item["bridge_reason"] = decision.bridge_reason
    item["blocked_mask"] = dict(decision.blocked_mask)
    item["food_delta_used"] = round(float(decision.food_delta_used), 6)
    item["shelter_delta_used"] = round(
        float(decision.shelter_delta_used),
        6,
    )
    item["external_override_count"] = int(
        decision.external_override_count
    )
