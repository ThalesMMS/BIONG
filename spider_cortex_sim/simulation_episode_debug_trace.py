from __future__ import annotations

from .simulation_episode_shared import *


def build_debug_trace_payload(
    self,
    *,
    decision,
    observation,
    next_observation,
    observation_adapters,
    next_observation_adapters,
    info,
) -> dict[str, object]:
    arbitration_payload = (
        decision.arbitration_decision.to_payload()
        if decision.arbitration_decision is not None
        else {}
    )
    return {
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
        "phase": {
            "target": decision.phase_target,
            "prediction": decision.phase_prediction,
            "prediction_confidence": round(
                float(decision.phase_prediction_confidence),
                6,
            ),
            "logits": decision.phase_logits.round(6).tolist(),
            "selected_action": ACTIONS[decision.action_idx],
        },
        "event_attention": {
            "top_type": decision.event_attention_top_type,
            "top_age": int(decision.event_attention_top_age),
            "entropy": round(
                float(decision.event_attention_entropy),
                6,
            ),
        },
        "option": {
            "selected_option": decision.selected_option,
            "age": int(decision.option_age),
            "termination_reason": decision.option_termination_reason,
            "logits": decision.option_logits.round(6).tolist(),
            "outside_or_corridor": bool(
                (not bool(next_observation["meta"].get("on_shelter", False)))
                or str(next_observation["meta"].get("shelter_role", "")) == "outside"
            ),
        },
        "affordance": {
            "blocked_logits": decision.affordance_blocked_logits.round(6).tolist(),
            "blocked_targets": decision.affordance_blocked_targets.round(6).tolist(),
            "role_logits": decision.affordance_role_logits.round(6).tolist(),
            "role_targets": decision.affordance_role_targets.astype(int).tolist(),
            "geometry_logits": decision.geometry_logits.round(6).tolist(),
            "geometry_targets": decision.geometry_targets.round(6).tolist(),
            "shelter_column_logits": decision.shelter_column_logits.round(6).tolist(),
            "shelter_column_targets": decision.shelter_column_targets.astype(int).tolist(),
            "shelter_position_logits": decision.shelter_position_logits.round(6).tolist(),
            "shelter_position_targets": decision.shelter_position_targets.astype(int).tolist(),
            "transition_prediction_logits": decision.transition_prediction_logits.round(6).tolist(),
            "transition_prediction_targets": decision.transition_prediction_targets.round(6).tolist(),
            "transition_rollout_prediction_logits": decision.transition_rollout_prediction_logits.round(6).tolist(),
            "transition_rollout_prediction_targets": decision.transition_rollout_prediction_targets.round(6).tolist(),
        },
        "teacher": {
            "action_target": decision.teacher_action_target_name,
            "action_target_idx": int(decision.teacher_action_target_idx),
            "stage": decision.teacher_action_target_stage,
            "option_target": decision.teacher_option_target_name,
            "option_target_idx": int(decision.teacher_option_target_idx),
            "option_stage": decision.teacher_option_target_stage,
        },
        **(
            {
                "b_series": {
                    "b_level": int(decision.b_level),
                    "b_effective_level": decision.b_effective_level,
                    "b_mode": decision.b_mode,
                    "b_parent_level": decision.b_parent_level,
                    "b_transfer_source_checkpoint": (
                        decision.b_transfer_source_checkpoint
                    ),
                    "b_transfer_coverage": (
                        decision.b_transfer_coverage
                    ),
                    "b_current_threat_pressure": (
                        decision.b_current_threat_pressure
                    ),
                    "b_temporal_threat_pressure": (
                        decision.b_temporal_threat_pressure
                    ),
                    "b_predator_memory_pressure": (
                        decision.b_predator_memory_pressure
                    ),
                    "b_predator_trace_pressure": (
                        decision.b_predator_trace_pressure
                    ),
                    "b3_contact_cooldown": (
                        decision.b3_contact_cooldown
                    ),
                    "b3_post_food_cooldown": (
                        decision.b3_post_food_cooldown
                    ),
                    "b3_hunger_drop": decision.b3_hunger_drop,
                    "b3_controller_profile": (
                        decision.b3_controller_profile
                    ),
                    "b4_controller_profile": (
                        decision.b4_controller_profile
                    ),
                    "b4_recovery_pressure": (
                        decision.b4_recovery_pressure
                    ),
                    "b4_sleep_hold": decision.b4_sleep_hold,
                    "b4_exit_blocked": decision.b4_exit_blocked,
                    "b4_hunger_release": (
                        decision.b4_hunger_release
                    ),
                    "b4_genetic_generation": (
                        decision.b4_genetic_generation
                    ),
                    "b4_genetic_candidate": (
                        decision.b4_genetic_candidate
                    ),
                    "b5_controller_profile": (
                        decision.b5_controller_profile
                    ),
                    "b5_hunger_urgency": decision.b5_hunger_urgency,
                    "b5_sleep_pressure": decision.b5_sleep_pressure,
                    "b5_recovery_debt": decision.b5_recovery_debt,
                    "b5_threat_gate": decision.b5_threat_gate,
                    "b5_sleep_bout_lock": (
                        decision.b5_sleep_bout_lock
                    ),
                    "b5_forage_commitment_lock": (
                        decision.b5_forage_commitment_lock
                    ),
                    "b5_homeostatic_decision": (
                        decision.b5_homeostatic_decision
                    ),
                    "b5_genetic_generation": (
                        decision.b5_genetic_generation
                    ),
                    "b5_genetic_candidate": (
                        decision.b5_genetic_candidate
                    ),
                    "b6_controller_family": (
                        decision.b6_controller_family
                    ),
                    "b6_controller_profile": (
                        decision.b6_controller_profile
                    ),
                    "b6_risk_pressure": (
                        decision.b6_risk_pressure
                    ),
                    "b6_threat_priority": (
                        decision.b6_threat_priority
                    ),
                    "b6_forage_suppressed": (
                        decision.b6_forage_suppressed
                    ),
                    "b6_corridor_commitment": (
                        decision.b6_corridor_commitment
                    ),
                    "b6_corridor_progress_memory": (
                        decision.b6_corridor_progress_memory
                    ),
                    "b6_recurrent_state": (
                        decision.b6_recurrent_state
                    ),
                    "b6_return_lock": decision.b6_return_lock,
                    "b6_decision": decision.b6_decision,
                    "b6_genetic_generation": (
                        decision.b6_genetic_generation
                    ),
                    "b6_genetic_candidate": (
                        decision.b6_genetic_candidate
                    ),
                    "b7_controller_profile": (
                        decision.b7_controller_profile
                    ),
                    "b7_affordance_state": (
                        decision.b7_affordance_state
                    ),
                    "b7_energy_budget": decision.b7_energy_budget,
                    "b7_budget_margin": decision.b7_budget_margin,
                    "b7_food_steps_estimate": (
                        decision.b7_food_steps_estimate
                    ),
                    "b7_return_steps_estimate": (
                        decision.b7_return_steps_estimate
                    ),
                    "b7_corridor_viability": (
                        decision.b7_corridor_viability
                    ),
                    "b7_abort_return": decision.b7_abort_return,
                    "b7_commitment_lock": (
                        decision.b7_commitment_lock
                    ),
                    "b7_decision": decision.b7_decision,
                    "b7_genetic_generation": (
                        decision.b7_genetic_generation
                    ),
                    "b7_genetic_candidate": (
                        decision.b7_genetic_candidate
                    ),
                    "b8_controller_profile": (
                        decision.b8_controller_profile
                    ),
                    "b8_spatial_map_state": (
                        decision.b8_spatial_map_state
                    ),
                    "b8_local_affordance_score": (
                        decision.b8_local_affordance_score
                    ),
                    "b8_return_vector_strength": (
                        decision.b8_return_vector_strength
                    ),
                    "b8_corridor_dead_end_risk": (
                        decision.b8_corridor_dead_end_risk
                    ),
                    "b8_abort_executed": decision.b8_abort_executed,
                    "b8_place_memory": decision.b8_place_memory,
                    "b8_decision": decision.b8_decision,
                    "b8_genetic_generation": (
                        decision.b8_genetic_generation
                    ),
                    "b8_genetic_candidate": (
                        decision.b8_genetic_candidate
                    ),
                    "b9_controller_profile": (
                        decision.b9_controller_profile
                    ),
                    "b9_route_state": decision.b9_route_state,
                    "b9_route_confidence": (
                        decision.b9_route_confidence
                    ),
                    "b9_waypoint_lock": decision.b9_waypoint_lock,
                    "b9_path_integrator": (
                        decision.b9_path_integrator
                    ),
                    "b9_replan_signal": decision.b9_replan_signal,
                    "b9_decision": decision.b9_decision,
                    "b9_genetic_generation": (
                        decision.b9_genetic_generation
                    ),
                    "b9_genetic_candidate": (
                        decision.b9_genetic_candidate
                    ),
                    "b10_controller_profile": (
                        decision.b10_controller_profile
                    ),
                    "b10_replay_state": decision.b10_replay_state,
                    "b10_prospective_value": (
                        decision.b10_prospective_value
                    ),
                    "b10_rollout_depth": decision.b10_rollout_depth,
                    "b10_replay_memory": decision.b10_replay_memory,
                    "b10_plan_commitment": (
                        decision.b10_plan_commitment
                    ),
                    "b10_abort_signal": decision.b10_abort_signal,
                    "b10_decision": decision.b10_decision,
                    "b10_genetic_generation": (
                        decision.b10_genetic_generation
                    ),
                    "b10_genetic_candidate": (
                        decision.b10_genetic_candidate
                    ),
                    "b11_controller_profile": (
                        decision.b11_controller_profile
                    ),
                    "b11_confidence_state": (
                        decision.b11_confidence_state
                    ),
                    "b11_plan_confidence": (
                        decision.b11_plan_confidence
                    ),
                    "b11_uncertainty": decision.b11_uncertainty,
                    "b11_neuromod_signal": (
                        decision.b11_neuromod_signal
                    ),
                    "b11_confidence_lock": (
                        decision.b11_confidence_lock
                    ),
                    "b11_decision": decision.b11_decision,
                    "b11_genetic_generation": (
                        decision.b11_genetic_generation
                    ),
                    "b11_genetic_candidate": (
                        decision.b11_genetic_candidate
                    ),
                    "b12_controller_profile": (
                        decision.b12_controller_profile
                    ),
                    "b12_attention_state": (
                        decision.b12_attention_state
                    ),
                    "b12_prediction_error": (
                        decision.b12_prediction_error
                    ),
                    "b12_attention_gain": (
                        decision.b12_attention_gain
                    ),
                    "b12_expected_progress": (
                        decision.b12_expected_progress
                    ),
                    "b12_search_lock": decision.b12_search_lock,
                    "b12_decision": decision.b12_decision,
                    "b12_genetic_generation": (
                        decision.b12_genetic_generation
                    ),
                    "b12_genetic_candidate": (
                        decision.b12_genetic_candidate
                    ),
                    "b13_controller_profile": (
                        decision.b13_controller_profile
                    ),
                    "b13_search_state": decision.b13_search_state,
                    "b13_local_route_score": (
                        decision.b13_local_route_score
                    ),
                    "b13_affordance_samples": (
                        decision.b13_affordance_samples
                    ),
                    "b13_search_memory": decision.b13_search_memory,
                    "b13_dead_end_score": decision.b13_dead_end_score,
                    "b13_search_lock": decision.b13_search_lock,
                    "b13_decision": decision.b13_decision,
                    "b13_genetic_generation": (
                        decision.b13_genetic_generation
                    ),
                    "b13_genetic_candidate": (
                        decision.b13_genetic_candidate
                    ),
                    "b14_controller_profile": (
                        decision.b14_controller_profile
                    ),
                    "b14_uncertainty_state": (
                        decision.b14_uncertainty_state
                    ),
                    "b14_affordance_confidence": (
                        decision.b14_affordance_confidence
                    ),
                    "b14_uncertainty": decision.b14_uncertainty,
                    "b14_risk_adjusted_score": (
                        decision.b14_risk_adjusted_score
                    ),
                    "b14_commitment_lock": (
                        decision.b14_commitment_lock
                    ),
                    "b14_decision": decision.b14_decision,
                    "b14_genetic_generation": (
                        decision.b14_genetic_generation
                    ),
                    "b14_genetic_candidate": (
                        decision.b14_genetic_candidate
                    ),
                    "b15_controller_profile": (
                        decision.b15_controller_profile
                    ),
                    "b15_option_state": decision.b15_option_state,
                    "b15_option_value": decision.b15_option_value,
                    "b15_termination_pressure": (
                        decision.b15_termination_pressure
                    ),
                    "b15_persistence_score": (
                        decision.b15_persistence_score
                    ),
                    "b15_option_lock": decision.b15_option_lock,
                    "b15_decision": decision.b15_decision,
                    "b15_genetic_generation": (
                        decision.b15_genetic_generation
                    ),
                    "b15_genetic_candidate": (
                        decision.b15_genetic_candidate
                    ),
                    "b16_controller_profile": (
                        decision.b16_controller_profile
                    ),
                    "b16_ensemble_state": (
                        decision.b16_ensemble_state
                    ),
                    "b16_continue_vote": decision.b16_continue_vote,
                    "b16_return_vote": decision.b16_return_vote,
                    "b16_option_votes": decision.b16_option_votes,
                    "b16_consensus_score": (
                        decision.b16_consensus_score
                    ),
                    "b16_conflict_score": (
                        decision.b16_conflict_score
                    ),
                    "b16_ensemble_lock": decision.b16_ensemble_lock,
                    "b16_decision": decision.b16_decision,
                    "b16_genetic_generation": (
                        decision.b16_genetic_generation
                    ),
                    "b16_genetic_candidate": (
                        decision.b16_genetic_candidate
                    ),
                    "b17_controller_profile": (
                        decision.b17_controller_profile
                    ),
                    "b17_modulator_state": (
                        decision.b17_modulator_state
                    ),
                    "b17_arousal_signal": decision.b17_arousal_signal,
                    "b17_homeostatic_gain": (
                        decision.b17_homeostatic_gain
                    ),
                    "b17_option_gain": decision.b17_option_gain,
                    "b17_conflict_release": (
                        decision.b17_conflict_release
                    ),
                    "b17_modulation_lock": (
                        decision.b17_modulation_lock
                    ),
                    "b17_decision": decision.b17_decision,
                    "b17_genetic_generation": (
                        decision.b17_genetic_generation
                    ),
                    "b17_genetic_candidate": (
                        decision.b17_genetic_candidate
                    ),
                    "b18_controller_profile": (
                        decision.b18_controller_profile
                    ),
                    "b18_trace_state": decision.b18_trace_state,
                    "b18_eligibility_trace": (
                        decision.b18_eligibility_trace
                    ),
                    "b18_reward_prediction_proxy": (
                        decision.b18_reward_prediction_proxy
                    ),
                    "b18_stability_bias": (
                        decision.b18_stability_bias
                    ),
                    "b18_switch_pressure": (
                        decision.b18_switch_pressure
                    ),
                    "b18_trace_lock": decision.b18_trace_lock,
                    "b18_decision": decision.b18_decision,
                    "b18_genetic_generation": (
                        decision.b18_genetic_generation
                    ),
                    "b18_genetic_candidate": (
                        decision.b18_genetic_candidate
                    ),
                    "b19_controller_profile": (
                        decision.b19_controller_profile
                    ),
                    "b19_memory_state": decision.b19_memory_state,
                    "b19_episode_memory": (
                        decision.b19_episode_memory
                    ),
                    "b19_consolidation_score": (
                        decision.b19_consolidation_score
                    ),
                    "b19_stability_vote": (
                        decision.b19_stability_vote
                    ),
                    "b19_switch_suppression": (
                        decision.b19_switch_suppression
                    ),
                    "b19_memory_lock": decision.b19_memory_lock,
                    "b19_decision": decision.b19_decision,
                    "b19_genetic_generation": (
                        decision.b19_genetic_generation
                    ),
                    "b19_genetic_candidate": (
                        decision.b19_genetic_candidate
                    ),
                    "b20_controller_profile": (
                        decision.b20_controller_profile
                    ),
                    "b20_buffer_state": decision.b20_buffer_state,
                    "b20_working_buffer": (
                        decision.b20_working_buffer
                    ),
                    "b20_context_binding": (
                        decision.b20_context_binding
                    ),
                    "b20_gate_vote": decision.b20_gate_vote,
                    "b20_release_vote": decision.b20_release_vote,
                    "b20_buffer_lock": decision.b20_buffer_lock,
                    "b20_decision": decision.b20_decision,
                    "b20_genetic_generation": (
                        decision.b20_genetic_generation
                    ),
                    "b20_genetic_candidate": (
                        decision.b20_genetic_candidate
                    ),
                    "b21_controller_profile": (
                        decision.b21_controller_profile
                    ),
                    "b21_replay_state": decision.b21_replay_state,
                    "b21_sequence_memory": (
                        decision.b21_sequence_memory
                    ),
                    "b21_replay_score": decision.b21_replay_score,
                    "b21_route_commitment": (
                        decision.b21_route_commitment
                    ),
                    "b21_abort_prediction": (
                        decision.b21_abort_prediction
                    ),
                    "b21_replay_lock": decision.b21_replay_lock,
                    "b21_decision": decision.b21_decision,
                    "b21_genetic_generation": (
                        decision.b21_genetic_generation
                    ),
                    "b21_genetic_candidate": (
                        decision.b21_genetic_candidate
                    ),
                    "b22_controller_profile": (
                        decision.b22_controller_profile
                    ),
                    "b22_sim_state": decision.b22_sim_state,
                    "b22_prospective_sim": (
                        decision.b22_prospective_sim
                    ),
                    "b22_forward_model_score": (
                        decision.b22_forward_model_score
                    ),
                    "b22_viability_projection": (
                        decision.b22_viability_projection
                    ),
                    "b22_abort_projection": (
                        decision.b22_abort_projection
                    ),
                    "b22_sim_lock": decision.b22_sim_lock,
                    "b22_decision": decision.b22_decision,
                    "b22_genetic_generation": (
                        decision.b22_genetic_generation
                    ),
                    "b22_genetic_candidate": (
                        decision.b22_genetic_candidate
                    ),
                    "b23_controller_profile": (
                        decision.b23_controller_profile
                    ),
                    "b23_conflict_state": (
                        decision.b23_conflict_state
                    ),
                    "b23_prediction_error": (
                        decision.b23_prediction_error
                    ),
                    "b23_conflict_memory": (
                        decision.b23_conflict_memory
                    ),
                    "b23_stability_vote": decision.b23_stability_vote,
                    "b23_abort_bias": decision.b23_abort_bias,
                    "b23_monitor_lock": decision.b23_monitor_lock,
                    "b23_decision": decision.b23_decision,
                    "b23_genetic_generation": (
                        decision.b23_genetic_generation
                    ),
                    "b23_genetic_candidate": (
                        decision.b23_genetic_candidate
                    ),
                    "b24_controller_profile": (
                        decision.b24_controller_profile
                    ),
                    "b24_precision_state": (
                        decision.b24_precision_state
                    ),
                    "b24_precision_memory": (
                        decision.b24_precision_memory
                    ),
                    "b24_precision_vote": decision.b24_precision_vote,
                    "b24_uncertainty_pressure": (
                        decision.b24_uncertainty_pressure
                    ),
                    "b24_abort_precision": (
                        decision.b24_abort_precision
                    ),
                    "b24_precision_lock": decision.b24_precision_lock,
                    "b24_decision": decision.b24_decision,
                    "b24_genetic_generation": (
                        decision.b24_genetic_generation
                    ),
                    "b24_genetic_candidate": (
                        decision.b24_genetic_candidate
                    ),
                    "b25_controller_profile": (
                        decision.b25_controller_profile
                    ),
                    "b25_metacognitive_state": (
                        decision.b25_metacognitive_state
                    ),
                    "b25_confidence_memory": (
                        decision.b25_confidence_memory
                    ),
                    "b25_confidence_vote": (
                        decision.b25_confidence_vote
                    ),
                    "b25_doubt_pressure": (
                        decision.b25_doubt_pressure
                    ),
                    "b25_control_gain": decision.b25_control_gain,
                    "b25_meta_lock": decision.b25_meta_lock,
                    "b25_decision": decision.b25_decision,
                    "b25_genetic_generation": (
                        decision.b25_genetic_generation
                    ),
                    "b25_genetic_candidate": (
                        decision.b25_genetic_candidate
                    ),
                    "b26_controller_profile": (
                        decision.b26_controller_profile
                    ),
                    "b26_allostatic_state": (
                        decision.b26_allostatic_state
                    ),
                    "b26_prediction_error": (
                        decision.b26_prediction_error
                    ),
                    "b26_setpoint_pressure": (
                        decision.b26_setpoint_pressure
                    ),
                    "b26_control_vote": decision.b26_control_vote,
                    "b26_stability_lock": (
                        decision.b26_stability_lock
                    ),
                    "b26_decision": decision.b26_decision,
                    "b26_genetic_generation": (
                        decision.b26_genetic_generation
                    ),
                    "b26_genetic_candidate": (
                        decision.b26_genetic_candidate
                    ),
                    "b27_controller_profile": (
                        decision.b27_controller_profile
                    ),
                    "b27_arousal_state": decision.b27_arousal_state,
                    "b27_arousal_level": decision.b27_arousal_level,
                    "b27_gain_modulation": (
                        decision.b27_gain_modulation
                    ),
                    "b27_stress_pressure": (
                        decision.b27_stress_pressure
                    ),
                    "b27_arousal_lock": decision.b27_arousal_lock,
                    "b27_decision": decision.b27_decision,
                    "b27_genetic_generation": (
                        decision.b27_genetic_generation
                    ),
                    "b27_genetic_candidate": (
                        decision.b27_genetic_candidate
                    ),
                    "b28_controller_profile": (
                        decision.b28_controller_profile
                    ),
                    "b28_attention_state": (
                        decision.b28_attention_state
                    ),
                    "b28_interoceptive_focus": (
                        decision.b28_interoceptive_focus
                    ),
                    "b28_attention_gain": decision.b28_attention_gain,
                    "b28_distractor_pressure": (
                        decision.b28_distractor_pressure
                    ),
                    "b28_attention_lock": decision.b28_attention_lock,
                    "b28_decision": decision.b28_decision,
                    "b28_genetic_generation": (
                        decision.b28_genetic_generation
                    ),
                    "b28_genetic_candidate": (
                        decision.b28_genetic_candidate
                    ),
                    "b29_controller_profile": (
                        decision.b29_controller_profile
                    ),
                    "b29_salience_state": (
                        decision.b29_salience_state
                    ),
                    "b29_threat_salience": (
                        decision.b29_threat_salience
                    ),
                    "b29_homeostatic_salience": (
                        decision.b29_homeostatic_salience
                    ),
                    "b29_corridor_salience": (
                        decision.b29_corridor_salience
                    ),
                    "b29_winner_channel": decision.b29_winner_channel,
                    "b29_salience_lock": decision.b29_salience_lock,
                    "b29_decision": decision.b29_decision,
                    "b29_genetic_generation": (
                        decision.b29_genetic_generation
                    ),
                    "b29_genetic_candidate": (
                        decision.b29_genetic_candidate
                    ),
                    "b30_controller_profile": (
                        decision.b30_controller_profile
                    ),
                    "b30_gate_state": decision.b30_gate_state,
                    "b30_go_signal": decision.b30_go_signal,
                    "b30_no_go_signal": decision.b30_no_go_signal,
                    "b30_action_gate": decision.b30_action_gate,
                    "b30_gate_lock": decision.b30_gate_lock,
                    "b30_decision": decision.b30_decision,
                    "b30_genetic_generation": (
                        decision.b30_genetic_generation
                    ),
                    "b30_genetic_candidate": (
                        decision.b30_genetic_candidate
                    ),
                    "b31_controller_profile": (
                        decision.b31_controller_profile
                    ),
                    "b31_dopamine_state": (
                        decision.b31_dopamine_state
                    ),
                    "b31_reward_prediction_error": (
                        decision.b31_reward_prediction_error
                    ),
                    "b31_tonic_dopamine": (
                        decision.b31_tonic_dopamine
                    ),
                    "b31_phasic_dopamine": (
                        decision.b31_phasic_dopamine
                    ),
                    "b31_gate_bias": decision.b31_gate_bias,
                    "b31_dopamine_lock": decision.b31_dopamine_lock,
                    "b31_decision": decision.b31_decision,
                    "b31_genetic_generation": (
                        decision.b31_genetic_generation
                    ),
                    "b31_genetic_candidate": (
                        decision.b31_genetic_candidate
                    ),
                    "b32_controller_profile": (
                        decision.b32_controller_profile
                    ),
                    "b32_critic_value": decision.b32_critic_value,
                    "b32_actor_advantage": (
                        decision.b32_actor_advantage
                    ),
                    "b32_value_error": decision.b32_value_error,
                    "b32_policy_bias": decision.b32_policy_bias,
                    "b32_value_lock": decision.b32_value_lock,
                    "b32_decision": decision.b32_decision,
                    "b32_genetic_generation": (
                        decision.b32_genetic_generation
                    ),
                    "b32_genetic_candidate": (
                        decision.b32_genetic_candidate
                    ),
                    "b33_controller_profile": (
                        decision.b33_controller_profile
                    ),
                    "b33_td_error": decision.b33_td_error,
                    "b33_bootstrap_value": decision.b33_bootstrap_value,
                    "b33_reward_trace": decision.b33_reward_trace,
                    "b33_actor_update": decision.b33_actor_update,
                    "b33_td_lock": decision.b33_td_lock,
                    "b33_decision": decision.b33_decision,
                    "b33_genetic_generation": (
                        decision.b33_genetic_generation
                    ),
                    "b33_genetic_candidate": (
                        decision.b33_genetic_candidate
                    ),
                    "b34_controller_profile": (
                        decision.b34_controller_profile
                    ),
                    "b34_eligibility_trace": (
                        decision.b34_eligibility_trace
                    ),
                    "b34_credit_assignment": (
                        decision.b34_credit_assignment
                    ),
                    "b34_synaptic_tag": decision.b34_synaptic_tag,
                    "b34_decay_memory": decision.b34_decay_memory,
                    "b34_credit_lock": decision.b34_credit_lock,
                    "b34_decision": decision.b34_decision,
                    "b34_genetic_generation": (
                        decision.b34_genetic_generation
                    ),
                    "b34_genetic_candidate": (
                        decision.b34_genetic_candidate
                    ),
                    "b35_controller_profile": (
                        decision.b35_controller_profile
                    ),
                    "b35_forward_value": decision.b35_forward_value,
                    "b35_transition_error": (
                        decision.b35_transition_error
                    ),
                    "b35_model_confidence": (
                        decision.b35_model_confidence
                    ),
                    "b35_prediction_memory": (
                        decision.b35_prediction_memory
                    ),
                    "b35_model_lock": decision.b35_model_lock,
                    "b35_decision": decision.b35_decision,
                    "b35_genetic_generation": (
                        decision.b35_genetic_generation
                    ),
                    "b35_genetic_candidate": (
                        decision.b35_genetic_candidate
                    ),
                    "b36_controller_profile": (
                        decision.b36_controller_profile
                    ),
                    "b36_latent_state": decision.b36_latent_state,
                    "b36_belief_error": decision.b36_belief_error,
                    "b36_state_confidence": (
                        decision.b36_state_confidence
                    ),
                    "b36_context_memory": decision.b36_context_memory,
                    "b36_belief_lock": decision.b36_belief_lock,
                    "b36_decision": decision.b36_decision,
                    "b36_genetic_generation": (
                        decision.b36_genetic_generation
                    ),
                    "b36_genetic_candidate": (
                        decision.b36_genetic_candidate
                    ),
                    "b37_controller_profile": (
                        decision.b37_controller_profile
                    ),
                    "b37_external_state_factor": (
                        decision.b37_external_state_factor
                    ),
                    "b37_internal_state_factor": (
                        decision.b37_internal_state_factor
                    ),
                    "b37_factor_alignment": (
                        decision.b37_factor_alignment
                    ),
                    "b37_factor_confidence": (
                        decision.b37_factor_confidence
                    ),
                    "b37_factor_lock": decision.b37_factor_lock,
                    "b37_decision": decision.b37_decision,
                    "b37_genetic_generation": (
                        decision.b37_genetic_generation
                    ),
                    "b37_genetic_candidate": (
                        decision.b37_genetic_candidate
                    ),
                    "b38_controller_profile": (
                        decision.b38_controller_profile
                    ),
                    "b38_external_attention": (
                        decision.b38_external_attention
                    ),
                    "b38_internal_attention": (
                        decision.b38_internal_attention
                    ),
                    "b38_attention_balance": (
                        decision.b38_attention_balance
                    ),
                    "b38_attention_gain": decision.b38_attention_gain,
                    "b38_attention_lock": decision.b38_attention_lock,
                    "b38_decision": decision.b38_decision,
                    "b38_genetic_generation": (
                        decision.b38_genetic_generation
                    ),
                    "b38_genetic_candidate": (
                        decision.b38_genetic_candidate
                    ),
                    "b39_controller_profile": (
                        decision.b39_controller_profile
                    ),
                    "b39_binding_strength": (
                        decision.b39_binding_strength
                    ),
                    "b39_cross_factor_coherence": (
                        decision.b39_cross_factor_coherence
                    ),
                    "b39_bound_context": decision.b39_bound_context,
                    "b39_binding_gain": decision.b39_binding_gain,
                    "b39_binding_lock": decision.b39_binding_lock,
                    "b39_decision": decision.b39_decision,
                    "b39_genetic_generation": (
                        decision.b39_genetic_generation
                    ),
                    "b39_genetic_candidate": (
                        decision.b39_genetic_candidate
                    ),
                    "b40_controller_profile": (
                        decision.b40_controller_profile
                    ),
                    "b40_workspace_activation": (
                        decision.b40_workspace_activation
                    ),
                    "b40_broadcast_gain": (
                        decision.b40_broadcast_gain
                    ),
                    "b40_context_availability": (
                        decision.b40_context_availability
                    ),
                    "b40_workspace_stability": (
                        decision.b40_workspace_stability
                    ),
                    "b40_workspace_lock": decision.b40_workspace_lock,
                    "b40_decision": decision.b40_decision,
                    "b40_genetic_generation": (
                        decision.b40_genetic_generation
                    ),
                    "b40_genetic_candidate": (
                        decision.b40_genetic_candidate
                    ),
                    "b41_controller_profile": (
                        decision.b41_controller_profile
                    ),
                    "b41_executive_selection": (
                        decision.b41_executive_selection
                    ),
                    "b41_inhibitory_pressure": (
                        decision.b41_inhibitory_pressure
                    ),
                    "b41_goal_context": decision.b41_goal_context,
                    "b41_executive_stability": (
                        decision.b41_executive_stability
                    ),
                    "b41_executive_lock": decision.b41_executive_lock,
                    "b41_decision": decision.b41_decision,
                    "b41_genetic_generation": (
                        decision.b41_genetic_generation
                    ),
                    "b41_genetic_candidate": (
                        decision.b41_genetic_candidate
                    ),
                    "b42_controller_profile": (
                        decision.b42_controller_profile
                    ),
                    "b42_error_signal": decision.b42_error_signal,
                    "b42_conflict_signal": decision.b42_conflict_signal,
                    "b42_performance_context": (
                        decision.b42_performance_context
                    ),
                    "b42_monitor_stability": (
                        decision.b42_monitor_stability
                    ),
                    "b42_monitor_lock": decision.b42_monitor_lock,
                    "b42_decision": decision.b42_decision,
                    "b42_genetic_generation": (
                        decision.b42_genetic_generation
                    ),
                    "b42_genetic_candidate": (
                        decision.b42_genetic_candidate
                    ),
                    "b43_controller_profile": (
                        decision.b43_controller_profile
                    ),
                    "b43_precision_signal": (
                        decision.b43_precision_signal
                    ),
                    "b43_adaptive_threshold": (
                        decision.b43_adaptive_threshold
                    ),
                    "b43_arousal_context": (
                        decision.b43_arousal_context
                    ),
                    "b43_control_stability": (
                        decision.b43_control_stability
                    ),
                    "b43_precision_lock": decision.b43_precision_lock,
                    "b43_decision": decision.b43_decision,
                    "b43_genetic_generation": (
                        decision.b43_genetic_generation
                    ),
                    "b43_genetic_candidate": (
                        decision.b43_genetic_candidate
                    ),
                    "b44_controller_profile": (
                        decision.b44_controller_profile
                    ),
                    "b44_relay_gate": decision.b44_relay_gate,
                    "b44_sensory_precision": (
                        decision.b44_sensory_precision
                    ),
                    "b44_context_relay": decision.b44_context_relay,
                    "b44_gate_stability": decision.b44_gate_stability,
                    "b44_relay_lock": decision.b44_relay_lock,
                    "b44_decision": decision.b44_decision,
                    "b44_genetic_generation": (
                        decision.b44_genetic_generation
                    ),
                    "b44_genetic_candidate": (
                        decision.b44_genetic_candidate
                    ),
                    "b45_controller_profile": (
                        decision.b45_controller_profile
                    ),
                    "b45_inhibitory_gate": (
                        decision.b45_inhibitory_gate
                    ),
                    "b45_sensory_filter": decision.b45_sensory_filter,
                    "b45_context_suppression": (
                        decision.b45_context_suppression
                    ),
                    "b45_loop_stability": decision.b45_loop_stability,
                    "b45_inhibition_lock": decision.b45_inhibition_lock,
                    "b45_decision": decision.b45_decision,
                    "b45_genetic_generation": (
                        decision.b45_genetic_generation
                    ),
                    "b45_genetic_candidate": (
                        decision.b45_genetic_candidate
                    ),
                    "b46_controller_profile": (
                        decision.b46_controller_profile
                    ),
                    "b46_feedback_gain": decision.b46_feedback_gain,
                    "b46_topdown_context": (
                        decision.b46_topdown_context
                    ),
                    "b46_prediction_match": (
                        decision.b46_prediction_match
                    ),
                    "b46_feedback_stability": (
                        decision.b46_feedback_stability
                    ),
                    "b46_feedback_lock": decision.b46_feedback_lock,
                    "b46_decision": decision.b46_decision,
                    "b46_genetic_generation": (
                        decision.b46_genetic_generation
                    ),
                    "b46_genetic_candidate": (
                        decision.b46_genetic_candidate
                    ),
                    "b47_controller_profile": (
                        decision.b47_controller_profile
                    ),
                    "b47_phase_alignment": (
                        decision.b47_phase_alignment
                    ),
                    "b47_synchrony_gain": (
                        decision.b47_synchrony_gain
                    ),
                    "b47_cross_loop_coherence": (
                        decision.b47_cross_loop_coherence
                    ),
                    "b47_phase_lock": decision.b47_phase_lock,
                    "b47_decision": decision.b47_decision,
                    "b47_genetic_generation": (
                        decision.b47_genetic_generation
                    ),
                    "b47_genetic_candidate": (
                        decision.b47_genetic_candidate
                    ),
                    "b48_controller_profile": (
                        decision.b48_controller_profile
                    ),
                    "b48_timing_error": decision.b48_timing_error,
                    "b48_predictive_timing": (
                        decision.b48_predictive_timing
                    ),
                    "b48_corrective_gain": (
                        decision.b48_corrective_gain
                    ),
                    "b48_calibration_lock": (
                        decision.b48_calibration_lock
                    ),
                    "b48_decision": decision.b48_decision,
                    "b48_genetic_generation": (
                        decision.b48_genetic_generation
                    ),
                    "b48_genetic_candidate": (
                        decision.b48_genetic_candidate
                    ),
                    "b49_controller_profile": (
                        decision.b49_controller_profile
                    ),
                    "b49_go_signal": decision.b49_go_signal,
                    "b49_no_go_signal": decision.b49_no_go_signal,
                    "b49_action_gate_balance": (
                        decision.b49_action_gate_balance
                    ),
                    "b49_selection_lock": decision.b49_selection_lock,
                    "b49_decision": decision.b49_decision,
                    "b49_genetic_generation": (
                        decision.b49_genetic_generation
                    ),
                    "b49_genetic_candidate": (
                        decision.b49_genetic_candidate
                    ),
                    "b50_controller_profile": (
                        decision.b50_controller_profile
                    ),
                    "b50_habit_strength": (
                        decision.b50_habit_strength
                    ),
                    "b50_chunk_value": decision.b50_chunk_value,
                    "b50_habit_stability": (
                        decision.b50_habit_stability
                    ),
                    "b50_chunk_lock": decision.b50_chunk_lock,
                    "b50_decision": decision.b50_decision,
                    "b50_genetic_generation": (
                        decision.b50_genetic_generation
                    ),
                    "b50_genetic_candidate": (
                        decision.b50_genetic_candidate
                    ),
                    "b51_controller_profile": (
                        decision.b51_controller_profile
                    ),
                    "b51_prediction_error": (
                        decision.b51_prediction_error
                    ),
                    "b51_dopamine_gain": decision.b51_dopamine_gain,
                    "b51_habit_modulation": (
                        decision.b51_habit_modulation
                    ),
                    "b51_modulation_lock": (
                        decision.b51_modulation_lock
                    ),
                    "b51_decision": decision.b51_decision,
                    "b51_genetic_generation": (
                        decision.b51_genetic_generation
                    ),
                    "b51_genetic_candidate": (
                        decision.b51_genetic_candidate
                    ),
                    "b52_controller_profile": (
                        decision.b52_controller_profile
                    ),
                    "b52_acetylcholine_level": (
                        decision.b52_acetylcholine_level
                    ),
                    "b52_precision_gain": decision.b52_precision_gain,
                    "b52_uncertainty_signal": (
                        decision.b52_uncertainty_signal
                    ),
                    "b52_attention_lock": (
                        decision.b52_attention_lock
                    ),
                    "b52_decision": decision.b52_decision,
                    "b52_genetic_generation": (
                        decision.b52_genetic_generation
                    ),
                    "b52_genetic_candidate": (
                        decision.b52_genetic_candidate
                    ),
                    "b53_controller_profile": (
                        decision.b53_controller_profile
                    ),
                    "b53_norepinephrine_level": (
                        decision.b53_norepinephrine_level
                    ),
                    "b53_arousal_gain": decision.b53_arousal_gain,
                    "b53_surprise_signal": (
                        decision.b53_surprise_signal
                    ),
                    "b53_gain_lock": decision.b53_gain_lock,
                    "b53_decision": decision.b53_decision,
                    "b53_genetic_generation": (
                        decision.b53_genetic_generation
                    ),
                    "b53_genetic_candidate": (
                        decision.b53_genetic_candidate
                    ),
                    "b54_controller_profile": (
                        decision.b54_controller_profile
                    ),
                    "b54_serotonin_level": (
                        decision.b54_serotonin_level
                    ),
                    "b54_patience_signal": (
                        decision.b54_patience_signal
                    ),
                    "b54_impulse_suppression": (
                        decision.b54_impulse_suppression
                    ),
                    "b54_patience_lock": decision.b54_patience_lock,
                    "b54_decision": decision.b54_decision,
                    "b54_genetic_generation": (
                        decision.b54_genetic_generation
                    ),
                    "b54_genetic_candidate": (
                        decision.b54_genetic_candidate
                    ),
                    "b55_controller_profile": (
                        decision.b55_controller_profile
                    ),
                    "b55_hypothalamic_drive": (
                        decision.b55_hypothalamic_drive
                    ),
                    "b55_satiety_signal": decision.b55_satiety_signal,
                    "b55_recovery_bias": decision.b55_recovery_bias,
                    "b55_drive_balance": decision.b55_drive_balance,
                    "b55_drive_lock": decision.b55_drive_lock,
                    "b55_decision": decision.b55_decision,
                    "b55_genetic_generation": (
                        decision.b55_genetic_generation
                    ),
                    "b55_genetic_candidate": (
                        decision.b55_genetic_candidate
                    ),
                    "b56_controller_profile": (
                        decision.b56_controller_profile
                    ),
                    "b56_cortisol_level": decision.b56_cortisol_level,
                    "b56_stress_load": decision.b56_stress_load,
                    "b56_recovery_signal": (
                        decision.b56_recovery_signal
                    ),
                    "b56_endocrine_balance": (
                        decision.b56_endocrine_balance
                    ),
                    "b56_stress_lock": decision.b56_stress_lock,
                    "b56_decision": decision.b56_decision,
                    "b56_genetic_generation": (
                        decision.b56_genetic_generation
                    ),
                    "b56_genetic_candidate": (
                        decision.b56_genetic_candidate
                    ),
                    "b57_controller_profile": (
                        decision.b57_controller_profile
                    ),
                    "b57_interoceptive_awareness": (
                        decision.b57_interoceptive_awareness
                    ),
                    "b57_visceral_salience": (
                        decision.b57_visceral_salience
                    ),
                    "b57_body_state_confidence": (
                        decision.b57_body_state_confidence
                    ),
                    "b57_awareness_balance": (
                        decision.b57_awareness_balance
                    ),
                    "b57_awareness_lock": decision.b57_awareness_lock,
                    "b57_decision": decision.b57_decision,
                    "b57_genetic_generation": (
                        decision.b57_genetic_generation
                    ),
                    "b57_genetic_candidate": (
                        decision.b57_genetic_candidate
                    ),
                    "b58_controller_profile": (
                        decision.b58_controller_profile
                    ),
                    "b58_conflict_signal": decision.b58_conflict_signal,
                    "b58_error_likelihood": (
                        decision.b58_error_likelihood
                    ),
                    "b58_control_allocation": (
                        decision.b58_control_allocation
                    ),
                    "b58_resolution_balance": (
                        decision.b58_resolution_balance
                    ),
                    "b58_conflict_lock": decision.b58_conflict_lock,
                    "b58_decision": decision.b58_decision,
                    "b58_genetic_generation": (
                        decision.b58_genetic_generation
                    ),
                    "b58_genetic_candidate": (
                        decision.b58_genetic_candidate
                    ),
                    "b59_controller_profile": (
                        decision.b59_controller_profile
                    ),
                    "b59_goal_context": decision.b59_goal_context,
                    "b59_working_set_stability": (
                        decision.b59_working_set_stability
                    ),
                    "b59_task_set_confidence": (
                        decision.b59_task_set_confidence
                    ),
                    "b59_executive_balance": (
                        decision.b59_executive_balance
                    ),
                    "b59_executive_lock": decision.b59_executive_lock,
                    "b59_decision": decision.b59_decision,
                    "b59_genetic_generation": (
                        decision.b59_genetic_generation
                    ),
                    "b59_genetic_candidate": (
                        decision.b59_genetic_candidate
                    ),
                    "b60_controller_profile": (
                        decision.b60_controller_profile
                    ),
                    "b60_outcome_value": decision.b60_outcome_value,
                    "b60_reversal_signal": decision.b60_reversal_signal,
                    "b60_goal_value_confidence": (
                        decision.b60_goal_value_confidence
                    ),
                    "b60_value_balance": decision.b60_value_balance,
                    "b60_value_lock": decision.b60_value_lock,
                    "b60_decision": decision.b60_decision,
                    "b60_genetic_generation": (
                        decision.b60_genetic_generation
                    ),
                    "b60_genetic_candidate": (
                        decision.b60_genetic_candidate
                    ),
                    "b61_controller_profile": (
                        decision.b61_controller_profile
                    ),
                    "b61_safety_value": decision.b61_safety_value,
                    "b61_threat_value": decision.b61_threat_value,
                    "b61_safety_confidence": (
                        decision.b61_safety_confidence
                    ),
                    "b61_affective_balance": (
                        decision.b61_affective_balance
                    ),
                    "b61_safety_lock": decision.b61_safety_lock,
                    "b61_decision": decision.b61_decision,
                    "b61_genetic_generation": (
                        decision.b61_genetic_generation
                    ),
                    "b61_genetic_candidate": (
                        decision.b61_genetic_candidate
                    ),
                    "b62_controller_profile": (
                        decision.b62_controller_profile
                    ),
                    "b62_defensive_mode": decision.b62_defensive_mode,
                    "b62_freeze_pressure": (
                        decision.b62_freeze_pressure
                    ),
                    "b62_flee_pressure": decision.b62_flee_pressure,
                    "b62_shelter_bias": decision.b62_shelter_bias,
                    "b62_defense_balance": (
                        decision.b62_defense_balance
                    ),
                    "b62_defense_lock": decision.b62_defense_lock,
                    "b62_decision": decision.b62_decision,
                    "b62_genetic_generation": (
                        decision.b62_genetic_generation
                    ),
                    "b62_genetic_candidate": (
                        decision.b62_genetic_candidate
                    ),
                    "semantic_action": decision.semantic_action,
                    "semantic_action_idx": int(
                        decision.semantic_action_idx
                    ),
                    "learned_semantic_action": (
                        decision.learned_semantic_action
                    ),
                    "learned_semantic_action_idx": int(
                        decision.learned_semantic_action_idx
                    ),
                    "semantic_action_source": (
                        decision.semantic_action_source
                    ),
                    "semantic_action_reason": (
                        decision.semantic_action_reason
                    ),
                    "semantic_override_count": int(
                        decision.semantic_override_count
                    ),
                    "semantic_logits": decision.semantic_logits.round(
                        6
                    ).tolist(),
                    "semantic_policy": decision.semantic_policy.round(
                        6
                    ).tolist(),
                    "bridge_primitive_action": (
                        decision.bridge_primitive_action
                    ),
                    "bridge_reason": decision.bridge_reason,
                    "blocked_mask": dict(decision.blocked_mask),
                    "food_delta_used": round(
                        float(decision.food_delta_used),
                        6,
                    ),
                    "shelter_delta_used": round(
                        float(decision.shelter_delta_used),
                        6,
                    ),
                    "external_override_count": int(
                        decision.external_override_count
                    ),
                },
            }
            if self.brain.config.is_b_series
            else {}
        ),
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
