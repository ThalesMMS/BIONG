from __future__ import annotations

from ._config_model import *


def brain_ablation_config_to_summary(self) -> dict[str, object]:
    """
    Produce a JSON-friendly summary of this ablation configuration.

    Returns:
        summary (Dict[str, object]): Mapping of configuration fields suitable for JSON serialization. Keys:
            - name: str
            - architecture: str
            - architecture_description: str
            - module_dropout: float
            - enable_reflexes: bool
            - enable_auxiliary_targets: bool
            - use_learned_arbitration: bool
            - enable_deterministic_guards: bool
            - enable_food_direction_bias: bool
            - warm_start_scale: float
            - gate_adjustment_bounds: list[float]
            - credit_strategy: str
            - disabled_modules: list[str]
            - recurrent_modules: list[str]
            - is_recurrent: bool
            - reflex_scale: float
            - module_reflex_scales: dict[str, float]
            - capacity_profile_name: str
            - capacity_profile: str
            - capacity_profile_version: str | int
            - capacity_scale_factor: float
            - module_hidden_dims: dict[str, int]
            - direct_policy_hidden_dims: list[int]
            - action_center_hidden_dim: int
            - arbitration_hidden_dim: int
            - motor_hidden_dim: int
            - monolithic_hidden_dim: int
    """
    resolved_capacity_profile = resolve_capacity_profile(self.capacity_profile)
    return {
        "name": self.name,
        "architecture": self.architecture,
        "architecture_description": architecture_description(self.architecture),
        "module_dropout": self.module_dropout,
        "enable_reflexes": self.enable_reflexes,
        "enable_auxiliary_targets": self.enable_auxiliary_targets,
        "use_learned_arbitration": self.use_learned_arbitration,
        "enable_deterministic_guards": self.enable_deterministic_guards,
        "enable_food_direction_bias": self.enable_food_direction_bias,
        "warm_start_scale": self.warm_start_scale,
        "gate_adjustment_bounds": list(self.gate_adjustment_bounds),
        "credit_strategy": self.credit_strategy,
        "route_mask_threshold": self.route_mask_threshold,
        "disabled_modules": list(self.disabled_modules),
        "recurrent_modules": list(self.recurrent_modules),
        "is_recurrent": self.is_recurrent,
        "reflex_scale": self.reflex_scale,
        "module_reflex_scales": dict(self.module_reflex_scales),
        "capacity_profile_name": resolved_capacity_profile.name,
        "capacity_profile": resolved_capacity_profile.name,
        "capacity_profile_version": resolved_capacity_profile.version,
        "capacity_scale_factor": resolved_capacity_profile.scale_factor,
        "module_hidden_dims": dict(self.module_hidden_dims),
        "direct_policy_hidden_dims": list(self.direct_policy_hidden_dims),
        "direct_policy_recurrent": bool(self.direct_policy_recurrent),
        "direct_policy_phase_head": bool(self.direct_policy_phase_head),
        "direct_policy_event_attention": bool(
            self.direct_policy_event_attention
        ),
        "direct_policy_event_buffer_size": int(
            self.direct_policy_event_buffer_size
        ),
        "direct_policy_option_head": bool(self.direct_policy_option_head),
        "direct_policy_owned_option_controller": bool(
            self.direct_policy_owned_option_controller
        ),
        "direct_policy_option_ttl": int(self.direct_policy_option_ttl),
        "direct_policy_affordance_head": bool(
            self.direct_policy_affordance_head
        ),
        "direct_policy_affordance_feedback": bool(
            self.direct_policy_affordance_feedback
        ),
        "direct_policy_geometry_head": bool(
            self.direct_policy_geometry_head
        ),
        "direct_policy_shelter_column_head": bool(
            self.direct_policy_shelter_column_head
        ),
        "direct_policy_shelter_position_head": bool(
            self.direct_policy_shelter_position_head
        ),
        "direct_policy_local_affordance_inputs": bool(
            self.direct_policy_local_affordance_inputs
        ),
        "direct_policy_local_spatial_inputs": bool(
            self.direct_policy_local_spatial_inputs
        ),
        "direct_policy_local_transition_inputs": bool(
            self.direct_policy_local_transition_inputs
        ),
        "direct_policy_local_transition_rollout_inputs": bool(
            self.direct_policy_local_transition_rollout_inputs
        ),
        "direct_policy_local_geodesic_inputs": bool(
            self.direct_policy_local_geodesic_inputs
        ),
        "direct_policy_transition_prediction_head": bool(
            self.direct_policy_transition_prediction_head
        ),
        "direct_policy_transition_prediction_feedback": bool(
            self.direct_policy_transition_prediction_feedback
        ),
        "direct_policy_transition_rollout_prediction_head": bool(
            self.direct_policy_transition_rollout_prediction_head
        ),
        "direct_policy_transition_rollout_prediction_feedback": bool(
            self.direct_policy_transition_rollout_prediction_feedback
        ),
        "direct_policy_handoff_teacher": bool(
            self.direct_policy_handoff_teacher
        ),
        "direct_policy_handoff_option_teacher": bool(
            self.direct_policy_handoff_option_teacher
        ),
        "direct_policy_post_rest_action_teacher": bool(
            self.direct_policy_post_rest_action_teacher
        ),
        "direct_policy_post_rest_release_sequence_teacher": bool(
            self.direct_policy_post_rest_release_sequence_teacher
        ),
        "direct_policy_post_rest_release_sequence_replay_boost": bool(
            self.direct_policy_post_rest_release_sequence_replay_boost
        ),
        "direct_policy_post_rest_release_sequence_distill": bool(
            self.direct_policy_post_rest_release_sequence_distill
        ),
        "direct_policy_post_rest_probe_distillation": bool(
            self.direct_policy_post_rest_probe_distillation
        ),
        "direct_policy_post_rest_probe_sequence_distillation": bool(
            self.direct_policy_post_rest_probe_sequence_distillation
        ),
        "direct_policy_post_rest_probe_family_distillation": bool(
            self.direct_policy_post_rest_probe_family_distillation
        ),
        "direct_policy_post_rest_probe_handoff_distillation": bool(
            self.direct_policy_post_rest_probe_handoff_distillation
        ),
        "direct_policy_post_rest_probe_trajectory_distillation": bool(
            self.direct_policy_post_rest_probe_trajectory_distillation
        ),
        "direct_policy_post_rest_probe_cycle_distillation": bool(
            self.direct_policy_post_rest_probe_cycle_distillation
        ),
        "direct_policy_post_rest_probe_trace_distillation": bool(
            self.direct_policy_post_rest_probe_trace_distillation
        ),
        "direct_policy_post_rest_probe_rollout_distillation": bool(
            self.direct_policy_post_rest_probe_rollout_distillation
        ),
        "direct_policy_post_rest_probe_frontier_teacher_distillation": bool(
            self.direct_policy_post_rest_probe_frontier_teacher_distillation
        ),
        "direct_policy_post_rest_probe_replayable_teacher_distillation": bool(
            self.direct_policy_post_rest_probe_replayable_teacher_distillation
        ),
        "direct_policy_continuation_replay_passes": int(
            self.direct_policy_continuation_replay_passes
        ),
        "direct_policy_continuation_replay_lr_scale": float(
            self.direct_policy_continuation_replay_lr_scale
        ),
        "direct_policy_continuation_margin_weight": float(
            self.direct_policy_continuation_margin_weight
        ),
        "direct_policy_phase_option_feedback": bool(
            self.direct_policy_phase_option_feedback
        ),
        "direct_policy_option_transition_feedback": bool(
            self.direct_policy_option_transition_feedback
        ),
        "direct_policy_option_termination_cooldown": bool(
            self.direct_policy_option_termination_cooldown
        ),
        "direct_policy_option_action_head": bool(
            self.direct_policy_option_action_head
        ),
        "direct_policy_option_decoder_state": bool(
            self.direct_policy_option_decoder_state
        ),
        "direct_policy_option_recurrent_dynamics": bool(
            self.direct_policy_option_recurrent_dynamics
        ),
        "direct_policy_option_sequence_head": bool(
            self.direct_policy_option_sequence_head
        ),
        "direct_policy_option_decoder_recurrent_state": bool(
            self.direct_policy_option_decoder_recurrent_state
        ),
        "direct_policy_option_action_transition_state": bool(
            self.direct_policy_option_action_transition_state
        ),
        "direct_policy_option_action_controller_state": bool(
            self.direct_policy_option_action_controller_state
        ),
        "direct_policy_option_action_token_decoder": bool(
            self.direct_policy_option_action_token_decoder
        ),
        "direct_policy_option_action_recurrent_core": bool(
            self.direct_policy_option_action_recurrent_core
        ),
        "direct_policy_option_action_separate_recurrent_head": bool(
            self.direct_policy_option_action_separate_recurrent_head
        ),
        "direct_policy_option_action_separate_policy_path": bool(
            self.direct_policy_option_action_separate_policy_path
        ),
        "direct_policy_option_action_separate_backbone": bool(
            self.direct_policy_option_action_separate_backbone
        ),
        "direct_policy_executive_physiology_option_gating": bool(
            self.direct_policy_executive_physiology_option_gating
        ),
        "direct_policy_executive_affordance_action_gating": bool(
            self.direct_policy_executive_affordance_action_gating
        ),
        "direct_policy_executive_option_action_masking": bool(
            self.direct_policy_executive_option_action_masking
        ),
        "direct_policy_executive_event_release_latching": bool(
            self.direct_policy_executive_event_release_latching
        ),
        "direct_policy_executive_event_release_action_commitment": bool(
            self.direct_policy_executive_event_release_action_commitment
        ),
        "direct_policy_executive_release_phase_state": bool(
            self.direct_policy_executive_release_phase_state
        ),
        "direct_policy_executive_release_progression": bool(
            self.direct_policy_executive_release_progression
        ),
        "direct_policy_executive_release_exit_contract": bool(
            self.direct_policy_executive_release_exit_contract
        ),
        "direct_policy_executive_release_substate_progression": bool(
            self.direct_policy_executive_release_substate_progression
        ),
        "direct_policy_executive_post_exit_continuation": bool(
            self.direct_policy_executive_post_exit_continuation
        ),
        "direct_policy_executive_post_exit_food_guidance": bool(
            self.direct_policy_executive_post_exit_food_guidance
        ),
        "direct_policy_executive_post_exit_food_commitment": bool(
            self.direct_policy_executive_post_exit_food_commitment
        ),
        "direct_policy_executive_post_exit_food_progression": bool(
            self.direct_policy_executive_post_exit_food_progression
        ),
        "direct_policy_executive_post_exit_food_heading_progression": bool(
            self.direct_policy_executive_post_exit_food_heading_progression
        ),
        "direct_policy_executive_post_exit_smell_progression": bool(
            self.direct_policy_executive_post_exit_smell_progression
        ),
        "direct_policy_executive_post_exit_corridor_progression": bool(
            self.direct_policy_executive_post_exit_corridor_progression
        ),
        "direct_policy_executive_post_exit_corridor_affordance_progression": bool(
            self.direct_policy_executive_post_exit_corridor_affordance_progression
        ),
        "direct_policy_executive_post_food_return": bool(
            self.direct_policy_executive_post_food_return
        ),
        "direct_policy_executive_post_food_vector_return": bool(
            self.direct_policy_executive_post_food_vector_return
        ),
        "direct_policy_executive_post_food_path_return": bool(
            self.direct_policy_executive_post_food_path_return
        ),
        "action_center_hidden_dim": int(self.action_center_hidden_dim or 0),
        "arbitration_hidden_dim": int(self.arbitration_hidden_dim or 0),
        "motor_hidden_dim": int(self.motor_hidden_dim or 0),
        "integration_hidden_dim": int(self.integration_hidden_dim or 0),
        "monolithic_hidden_dim": self.monolithic_hidden_dim,
        "b_level": int(self.b_level),
        "b_mode": str(self.b_mode),
        "b_hidden_dim": int(self.b_hidden_dim or 0),
        "b_parent_level": self.b_parent_level,
        "b_transfer_source_checkpoint": self.b_transfer_source_checkpoint,
        "b_transfer_min_coverage": float(self.b_transfer_min_coverage),
        "b_transfer_allow_low_coverage": bool(
            self.b_transfer_allow_low_coverage
        ),
        "b_controller_profile": self.b_controller_profile,
        "b_controller_params": dict(self.b_controller_params),
    }


def brain_ablation_config_from_summary(cls, summary: Mapping[str, object]) -> "BrainAblationConfig":
    """
    Create a BrainAblationConfig instance from a mapping produced by to_summary() or a compatible summary.

    The mapping may contain any subset of configuration keys; missing or invalid entries are replaced with sensible defaults. Notable behaviors:
    - "gate_adjustment_bounds" is accepted from any non-string Sequence; otherwise defaults to (0.5, 1.5).
    - "capacity_profile_name" is taken from "capacity_profile_name", then "capacity_profile", then "current".
    - Numeric fields ("module_dropout", "warm_start_scale", "reflex_scale") are coerced to float when present; default values are applied otherwise.
    - Collections are normalized: "disabled_modules" and "recurrent_modules" become tuples, "module_reflex_scales" and "module_hidden_dims" become dicts.
    - Other boolean and string options are read with sensible defaults.

    Parameters:
        summary (Mapping[str, object]): Mapping of configuration values (typically from to_summary()).

    Returns:
        BrainAblationConfig: A validated and normalized configuration instance built from the provided summary.

    Raises:
        ValueError: If `summary` is not a mapping.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("BrainAblationConfig summary must be a mapping.")
    gate_adjustment_bounds_raw = summary.get("gate_adjustment_bounds", (0.5, 1.5))
    if not isinstance(gate_adjustment_bounds_raw, Sequence) or isinstance(
        gate_adjustment_bounds_raw,
        (str, bytes),
    ):
        gate_adjustment_bounds_raw = (0.5, 1.5)
    capacity_profile_name = str(
        summary.get(
            "capacity_profile_name",
            summary.get("capacity_profile", "current"),
        )
        or "current"
    )
    module_dropout_value = summary.get("module_dropout")
    warm_start_scale_value = summary.get("warm_start_scale")
    reflex_scale_value = summary.get("reflex_scale")
    route_mask_threshold_value = summary.get("route_mask_threshold")
    legacy_integration_hidden_dim = summary.get("integration_hidden_dim")
    return cls(
        name=str(summary.get("name", "custom") or "custom"),
        architecture=str(summary.get("architecture", "modular") or "modular"),
        module_dropout=(
            float(module_dropout_value)
            if module_dropout_value is not None
            else 0.05
        ),
        enable_reflexes=bool(summary.get("enable_reflexes", True)),
        enable_auxiliary_targets=bool(
            summary.get("enable_auxiliary_targets", True)
        ),
        use_learned_arbitration=bool(
            summary.get("use_learned_arbitration", True)
        ),
        enable_deterministic_guards=bool(
            summary.get("enable_deterministic_guards", False)
        ),
        enable_food_direction_bias=bool(
            summary.get("enable_food_direction_bias", False)
        ),
        warm_start_scale=(
            float(warm_start_scale_value)
            if warm_start_scale_value is not None
            else 1.0
        ),
        gate_adjustment_bounds=tuple(gate_adjustment_bounds_raw),
        credit_strategy=str(summary.get("credit_strategy", "broadcast") or "broadcast"),
        route_mask_threshold=(
            float(route_mask_threshold_value)
            if route_mask_threshold_value is not None
            else 0.05
        ),
        disabled_modules=tuple(summary.get("disabled_modules", ()) or ()),
        recurrent_modules=tuple(summary.get("recurrent_modules", ()) or ()),
        reflex_scale=(
            float(reflex_scale_value)
            if reflex_scale_value is not None
            else 1.0
        ),
        module_reflex_scales=dict(summary.get("module_reflex_scales", {}) or {}),
        capacity_profile_name=capacity_profile_name,
        module_hidden_dims=dict(summary.get("module_hidden_dims", {}) or {}),
        direct_policy_hidden_dims=tuple(
            summary.get("direct_policy_hidden_dims", ()) or ()
        ),
        direct_policy_recurrent=bool(
            summary.get("direct_policy_recurrent", False)
        ),
        direct_policy_phase_head=bool(
            summary.get("direct_policy_phase_head", False)
        ),
        direct_policy_event_attention=bool(
            summary.get("direct_policy_event_attention", False)
        ),
        direct_policy_event_buffer_size=int(
            summary.get("direct_policy_event_buffer_size", 0)
        ),
        direct_policy_option_head=bool(
            summary.get("direct_policy_option_head", False)
        ),
        direct_policy_owned_option_controller=bool(
            summary.get("direct_policy_owned_option_controller", False)
        ),
        direct_policy_option_ttl=int(
            summary.get("direct_policy_option_ttl", 0)
        ),
        direct_policy_affordance_head=bool(
            summary.get("direct_policy_affordance_head", False)
        ),
        direct_policy_affordance_feedback=bool(
            summary.get("direct_policy_affordance_feedback", False)
        ),
        direct_policy_geometry_head=bool(
            summary.get("direct_policy_geometry_head", False)
        ),
        direct_policy_shelter_column_head=bool(
            summary.get("direct_policy_shelter_column_head", False)
        ),
        direct_policy_shelter_position_head=bool(
            summary.get("direct_policy_shelter_position_head", False)
        ),
        direct_policy_local_affordance_inputs=bool(
            summary.get("direct_policy_local_affordance_inputs", False)
        ),
        direct_policy_local_spatial_inputs=bool(
            summary.get("direct_policy_local_spatial_inputs", False)
        ),
        direct_policy_local_transition_inputs=bool(
            summary.get("direct_policy_local_transition_inputs", False)
        ),
        direct_policy_local_transition_rollout_inputs=bool(
            summary.get("direct_policy_local_transition_rollout_inputs", False)
        ),
        direct_policy_local_geodesic_inputs=bool(
            summary.get("direct_policy_local_geodesic_inputs", False)
        ),
        direct_policy_transition_prediction_head=bool(
            summary.get("direct_policy_transition_prediction_head", False)
        ),
        direct_policy_transition_prediction_feedback=bool(
            summary.get("direct_policy_transition_prediction_feedback", False)
        ),
        direct_policy_transition_rollout_prediction_head=bool(
            summary.get("direct_policy_transition_rollout_prediction_head", False)
        ),
        direct_policy_transition_rollout_prediction_feedback=bool(
            summary.get(
                "direct_policy_transition_rollout_prediction_feedback", False
            )
        ),
        direct_policy_handoff_teacher=bool(
            summary.get("direct_policy_handoff_teacher", False)
        ),
        direct_policy_handoff_option_teacher=bool(
            summary.get("direct_policy_handoff_option_teacher", False)
        ),
        direct_policy_post_rest_action_teacher=bool(
            summary.get("direct_policy_post_rest_action_teacher", False)
        ),
        direct_policy_post_rest_release_sequence_teacher=bool(
            summary.get(
                "direct_policy_post_rest_release_sequence_teacher",
                False,
            )
        ),
        direct_policy_post_rest_release_sequence_replay_boost=bool(
            summary.get(
                "direct_policy_post_rest_release_sequence_replay_boost",
                False,
            )
        ),
        direct_policy_post_rest_release_sequence_distill=bool(
            summary.get(
                "direct_policy_post_rest_release_sequence_distill",
                False,
            )
        ),
        direct_policy_post_rest_probe_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_sequence_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_sequence_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_family_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_family_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_handoff_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_handoff_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_trajectory_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_trajectory_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_cycle_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_cycle_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_trace_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_trace_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_rollout_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_rollout_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_frontier_teacher_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_frontier_teacher_distillation",
                False,
            )
        ),
        direct_policy_post_rest_probe_replayable_teacher_distillation=bool(
            summary.get(
                "direct_policy_post_rest_probe_replayable_teacher_distillation",
                False,
            )
        ),
        direct_policy_continuation_replay_passes=int(
            summary.get("direct_policy_continuation_replay_passes", 0)
        ),
        direct_policy_continuation_replay_lr_scale=float(
            summary.get("direct_policy_continuation_replay_lr_scale", 0.0)
        ),
        direct_policy_continuation_margin_weight=float(
            summary.get("direct_policy_continuation_margin_weight", 0.0)
        ),
        direct_policy_phase_option_feedback=bool(
            summary.get("direct_policy_phase_option_feedback", False)
        ),
        direct_policy_option_transition_feedback=bool(
            summary.get("direct_policy_option_transition_feedback", False)
        ),
        direct_policy_option_termination_cooldown=bool(
            summary.get("direct_policy_option_termination_cooldown", False)
        ),
        direct_policy_option_action_head=bool(
            summary.get("direct_policy_option_action_head", False)
        ),
        direct_policy_option_decoder_state=bool(
            summary.get("direct_policy_option_decoder_state", False)
        ),
        direct_policy_option_recurrent_dynamics=bool(
            summary.get("direct_policy_option_recurrent_dynamics", False)
        ),
        direct_policy_option_sequence_head=bool(
            summary.get("direct_policy_option_sequence_head", False)
        ),
        direct_policy_option_decoder_recurrent_state=bool(
            summary.get("direct_policy_option_decoder_recurrent_state", False)
        ),
        direct_policy_option_action_transition_state=bool(
            summary.get("direct_policy_option_action_transition_state", False)
        ),
        direct_policy_option_action_controller_state=bool(
            summary.get("direct_policy_option_action_controller_state", False)
        ),
        direct_policy_option_action_token_decoder=bool(
            summary.get("direct_policy_option_action_token_decoder", False)
        ),
        direct_policy_option_action_recurrent_core=bool(
            summary.get("direct_policy_option_action_recurrent_core", False)
        ),
        direct_policy_option_action_separate_recurrent_head=bool(
            summary.get(
                "direct_policy_option_action_separate_recurrent_head",
                False,
            )
        ),
        direct_policy_option_action_separate_policy_path=bool(
            summary.get(
                "direct_policy_option_action_separate_policy_path",
                False,
            )
        ),
        direct_policy_option_action_separate_backbone=bool(
            summary.get(
                "direct_policy_option_action_separate_backbone",
                False,
            )
        ),
        direct_policy_executive_physiology_option_gating=bool(
            summary.get(
                "direct_policy_executive_physiology_option_gating",
                False,
            )
        ),
        direct_policy_executive_affordance_action_gating=bool(
            summary.get(
                "direct_policy_executive_affordance_action_gating",
                False,
            )
        ),
        direct_policy_executive_option_action_masking=bool(
            summary.get(
                "direct_policy_executive_option_action_masking",
                False,
            )
        ),
        direct_policy_executive_event_release_latching=bool(
            summary.get(
                "direct_policy_executive_event_release_latching",
                False,
            )
        ),
        direct_policy_executive_event_release_action_commitment=bool(
            summary.get(
                "direct_policy_executive_event_release_action_commitment",
                False,
            )
        ),
        direct_policy_executive_release_phase_state=bool(
            summary.get(
                "direct_policy_executive_release_phase_state",
                False,
            )
        ),
        direct_policy_executive_release_progression=bool(
            summary.get(
                "direct_policy_executive_release_progression",
                False,
            )
        ),
        direct_policy_executive_release_exit_contract=bool(
            summary.get(
                "direct_policy_executive_release_exit_contract",
                False,
            )
        ),
        direct_policy_executive_release_substate_progression=bool(
            summary.get(
                "direct_policy_executive_release_substate_progression",
                False,
            )
        ),
        direct_policy_executive_post_exit_continuation=bool(
            summary.get(
                "direct_policy_executive_post_exit_continuation",
                False,
            )
        ),
        direct_policy_executive_post_exit_food_guidance=bool(
            summary.get(
                "direct_policy_executive_post_exit_food_guidance",
                False,
            )
        ),
        direct_policy_executive_post_exit_food_commitment=bool(
            summary.get(
                "direct_policy_executive_post_exit_food_commitment",
                False,
            )
        ),
        direct_policy_executive_post_exit_food_progression=bool(
            summary.get(
                "direct_policy_executive_post_exit_food_progression",
                False,
            )
        ),
        direct_policy_executive_post_exit_food_heading_progression=bool(
            summary.get(
                "direct_policy_executive_post_exit_food_heading_progression",
                False,
            )
        ),
        direct_policy_executive_post_exit_smell_progression=bool(
            summary.get(
                "direct_policy_executive_post_exit_smell_progression",
                False,
            )
        ),
        direct_policy_executive_post_exit_corridor_progression=bool(
            summary.get(
                "direct_policy_executive_post_exit_corridor_progression",
                False,
            )
        ),
        direct_policy_executive_post_exit_corridor_affordance_progression=bool(
            summary.get(
                "direct_policy_executive_post_exit_corridor_affordance_progression",
                False,
            )
        ),
        direct_policy_executive_post_food_return=bool(
            summary.get(
                "direct_policy_executive_post_food_return",
                False,
            )
        ),
        direct_policy_executive_post_food_vector_return=bool(
            summary.get(
                "direct_policy_executive_post_food_vector_return",
                False,
            )
        ),
        direct_policy_executive_post_food_path_return=bool(
            summary.get(
                "direct_policy_executive_post_food_path_return",
                False,
            )
        ),
        action_center_hidden_dim=summary.get(
            "action_center_hidden_dim",
            legacy_integration_hidden_dim,
        ),
        arbitration_hidden_dim=summary.get(
            "arbitration_hidden_dim",
            legacy_integration_hidden_dim,
        ),
        motor_hidden_dim=summary.get(
            "motor_hidden_dim",
            legacy_integration_hidden_dim,
        ),
        b_level=int(summary.get("b_level", 0) or 0),
        b_mode=str(summary.get("b_mode", "current_bridge") or "current_bridge"),
        b_hidden_dim=(
            int(summary.get("b_hidden_dim"))
            if summary.get("b_hidden_dim") is not None
            else None
        ),
        b_parent_level=(
            int(summary.get("b_parent_level"))
            if summary.get("b_parent_level") is not None
            else None
        ),
        b_transfer_source_checkpoint=(
            None
            if summary.get("b_transfer_source_checkpoint") in {None, ""}
            else str(summary.get("b_transfer_source_checkpoint"))
        ),
        b_transfer_min_coverage=float(
            summary.get("b_transfer_min_coverage", 0.50)
        ),
        b_transfer_allow_low_coverage=bool(
            summary.get("b_transfer_allow_low_coverage", False)
        ),
        b_controller_profile=(
            None
            if summary.get("b_controller_profile") in {None, ""}
            else str(summary.get("b_controller_profile"))
        ),
        b_controller_params={
            str(key): float(value)
            for key, value in dict(
                summary.get("b_controller_params", {}) or {}
            ).items()
        },
    )
