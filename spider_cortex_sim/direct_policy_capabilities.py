from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .direct_policy_affordances import (
    DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM,
    DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM,
    DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM,
)


_DIRECT_POLICY_BOOL_FIELDS = (
    "direct_policy_recurrent",
    "direct_policy_phase_head",
    "direct_policy_event_attention",
    "direct_policy_option_head",
    "direct_policy_owned_option_controller",
    "direct_policy_affordance_head",
    "direct_policy_affordance_feedback",
    "direct_policy_geometry_head",
    "direct_policy_shelter_column_head",
    "direct_policy_shelter_position_head",
    "direct_policy_local_affordance_inputs",
    "direct_policy_local_spatial_inputs",
    "direct_policy_local_transition_inputs",
    "direct_policy_local_transition_rollout_inputs",
    "direct_policy_local_geodesic_inputs",
    "direct_policy_transition_prediction_head",
    "direct_policy_transition_prediction_feedback",
    "direct_policy_transition_rollout_prediction_head",
    "direct_policy_transition_rollout_prediction_feedback",
    "direct_policy_handoff_teacher",
    "direct_policy_handoff_option_teacher",
    "direct_policy_post_rest_action_teacher",
    "direct_policy_post_rest_release_sequence_teacher",
    "direct_policy_post_rest_release_sequence_replay_boost",
    "direct_policy_post_rest_release_sequence_distill",
    "direct_policy_post_rest_probe_distillation",
    "direct_policy_post_rest_probe_sequence_distillation",
    "direct_policy_post_rest_probe_family_distillation",
    "direct_policy_post_rest_probe_handoff_distillation",
    "direct_policy_post_rest_probe_trajectory_distillation",
    "direct_policy_post_rest_probe_cycle_distillation",
    "direct_policy_post_rest_probe_trace_distillation",
    "direct_policy_post_rest_probe_rollout_distillation",
    "direct_policy_post_rest_probe_frontier_teacher_distillation",
    "direct_policy_post_rest_probe_replayable_teacher_distillation",
    "direct_policy_phase_option_feedback",
    "direct_policy_option_transition_feedback",
    "direct_policy_option_termination_cooldown",
    "direct_policy_option_action_head",
    "direct_policy_option_decoder_state",
    "direct_policy_option_recurrent_dynamics",
    "direct_policy_option_sequence_head",
    "direct_policy_option_decoder_recurrent_state",
    "direct_policy_option_action_transition_state",
    "direct_policy_option_action_controller_state",
    "direct_policy_option_action_token_decoder",
    "direct_policy_option_action_recurrent_core",
    "direct_policy_option_action_separate_recurrent_head",
    "direct_policy_option_action_separate_policy_path",
    "direct_policy_option_action_separate_backbone",
    "direct_policy_executive_physiology_option_gating",
    "direct_policy_executive_affordance_action_gating",
    "direct_policy_executive_option_action_masking",
    "direct_policy_executive_event_release_latching",
    "direct_policy_executive_event_release_action_commitment",
    "direct_policy_executive_release_phase_state",
    "direct_policy_executive_release_progression",
    "direct_policy_executive_release_exit_contract",
    "direct_policy_executive_release_substate_progression",
    "direct_policy_executive_post_exit_continuation",
    "direct_policy_executive_post_exit_food_guidance",
    "direct_policy_executive_post_exit_food_commitment",
    "direct_policy_executive_post_exit_food_progression",
    "direct_policy_executive_post_exit_food_heading_progression",
    "direct_policy_executive_post_exit_smell_progression",
    "direct_policy_executive_post_exit_corridor_progression",
    "direct_policy_executive_post_exit_corridor_affordance_progression",
    "direct_policy_executive_post_food_return",
    "direct_policy_executive_post_food_vector_return",
    "direct_policy_executive_post_food_path_return",
)

_DIRECT_POLICY_INT_FIELDS = (
    "direct_policy_event_buffer_size",
    "direct_policy_option_ttl",
    "direct_policy_continuation_replay_passes",
)

_DIRECT_POLICY_FLOAT_FIELDS = (
    "direct_policy_continuation_replay_lr_scale",
    "direct_policy_continuation_margin_weight",
)

_DIRECT_POLICY_METADATA_FIELDS = (
    "direct_policy_recurrent",
    "direct_policy_hidden_dims",
    "direct_policy_phase_head",
    "direct_policy_event_attention",
    "direct_policy_event_buffer_size",
    "direct_policy_option_head",
    "direct_policy_owned_option_controller",
    "direct_policy_option_ttl",
    "direct_policy_affordance_head",
    "direct_policy_affordance_feedback",
    "direct_policy_geometry_head",
    "direct_policy_shelter_column_head",
    "direct_policy_shelter_position_head",
    "direct_policy_local_affordance_inputs",
    "direct_policy_local_spatial_inputs",
    "direct_policy_local_transition_inputs",
    "direct_policy_local_transition_rollout_inputs",
    "direct_policy_local_geodesic_inputs",
    "direct_policy_transition_prediction_head",
    "direct_policy_transition_prediction_feedback",
    "direct_policy_transition_rollout_prediction_head",
    "direct_policy_transition_rollout_prediction_feedback",
    "direct_policy_handoff_teacher",
    "direct_policy_handoff_option_teacher",
    "direct_policy_post_rest_action_teacher",
    "direct_policy_post_rest_release_sequence_teacher",
    "direct_policy_post_rest_release_sequence_replay_boost",
    "direct_policy_post_rest_release_sequence_distill",
    "direct_policy_post_rest_probe_distillation",
    "direct_policy_post_rest_probe_sequence_distillation",
    "direct_policy_post_rest_probe_family_distillation",
    "direct_policy_post_rest_probe_handoff_distillation",
    "direct_policy_post_rest_probe_trajectory_distillation",
    "direct_policy_post_rest_probe_cycle_distillation",
    "direct_policy_post_rest_probe_trace_distillation",
    "direct_policy_post_rest_probe_rollout_distillation",
    "direct_policy_post_rest_probe_frontier_teacher_distillation",
    "direct_policy_post_rest_probe_replayable_teacher_distillation",
    "direct_policy_continuation_replay_passes",
    "direct_policy_continuation_replay_lr_scale",
    "direct_policy_continuation_margin_weight",
    "direct_policy_phase_option_feedback",
    "direct_policy_option_transition_feedback",
    "direct_policy_option_termination_cooldown",
    "direct_policy_option_action_head",
    "direct_policy_option_decoder_state",
    "direct_policy_option_recurrent_dynamics",
    "direct_policy_option_sequence_head",
    "direct_policy_option_decoder_recurrent_state",
    "direct_policy_option_action_transition_state",
    "direct_policy_option_action_controller_state",
    "direct_policy_option_action_token_decoder",
    "direct_policy_option_action_recurrent_core",
    "direct_policy_option_action_separate_recurrent_head",
    "direct_policy_option_action_separate_policy_path",
    "direct_policy_option_action_separate_backbone",
    "direct_policy_executive_physiology_option_gating",
    "direct_policy_executive_affordance_action_gating",
    "direct_policy_executive_option_action_masking",
    "direct_policy_executive_event_release_latching",
    "direct_policy_executive_event_release_action_commitment",
    "direct_policy_executive_release_phase_state",
    "direct_policy_executive_release_progression",
    "direct_policy_executive_release_exit_contract",
    "direct_policy_executive_release_substate_progression",
    "direct_policy_executive_post_exit_continuation",
    "direct_policy_executive_post_exit_food_guidance",
    "direct_policy_executive_post_exit_food_commitment",
    "direct_policy_executive_post_exit_food_progression",
    "direct_policy_executive_post_exit_food_heading_progression",
    "direct_policy_executive_post_exit_smell_progression",
    "direct_policy_executive_post_exit_corridor_progression",
    "direct_policy_executive_post_exit_corridor_affordance_progression",
    "direct_policy_executive_post_food_return",
    "direct_policy_executive_post_food_vector_return",
    "direct_policy_executive_post_food_path_return",
)

_POSITION_FEEDBACK_KWARG_FIELDS = {
    "phase_option_feedback": "direct_policy_phase_option_feedback",
    "option_transition_feedback": "direct_policy_option_transition_feedback",
    "option_termination_cooldown": "direct_policy_option_termination_cooldown",
    "option_action_head": "direct_policy_option_action_head",
    "option_decoder_state": "direct_policy_option_decoder_state",
    "option_recurrent_dynamics": "direct_policy_option_recurrent_dynamics",
    "option_sequence_head": "direct_policy_option_sequence_head",
    "option_decoder_recurrent_state": "direct_policy_option_decoder_recurrent_state",
    "option_action_transition_state": "direct_policy_option_action_transition_state",
    "option_action_controller_state": "direct_policy_option_action_controller_state",
    "option_action_token_decoder": "direct_policy_option_action_token_decoder",
    "option_action_recurrent_core": "direct_policy_option_action_recurrent_core",
    "option_action_separate_recurrent_head": (
        "direct_policy_option_action_separate_recurrent_head"
    ),
    "option_action_separate_policy_path": (
        "direct_policy_option_action_separate_policy_path"
    ),
    "option_action_separate_backbone": (
        "direct_policy_option_action_separate_backbone"
    ),
    "executive_physiology_option_gating": (
        "direct_policy_executive_physiology_option_gating"
    ),
    "executive_affordance_action_gating": (
        "direct_policy_executive_affordance_action_gating"
    ),
    "executive_option_action_masking": (
        "direct_policy_executive_option_action_masking"
    ),
    "executive_event_release_latching": (
        "direct_policy_executive_event_release_latching"
    ),
    "executive_event_release_action_commitment": (
        "direct_policy_executive_event_release_action_commitment"
    ),
    "executive_release_phase_state": "direct_policy_executive_release_phase_state",
    "executive_release_progression": "direct_policy_executive_release_progression",
    "executive_release_exit_contract": "direct_policy_executive_release_exit_contract",
    "executive_release_substate_progression": (
        "direct_policy_executive_release_substate_progression"
    ),
    "executive_post_exit_continuation": (
        "direct_policy_executive_post_exit_continuation"
    ),
    "executive_post_exit_food_guidance": (
        "direct_policy_executive_post_exit_food_guidance"
    ),
    "executive_post_exit_food_commitment": (
        "direct_policy_executive_post_exit_food_commitment"
    ),
    "executive_post_exit_food_progression": (
        "direct_policy_executive_post_exit_food_progression"
    ),
    "executive_post_exit_food_heading_progression": (
        "direct_policy_executive_post_exit_food_heading_progression"
    ),
    "executive_post_exit_smell_progression": (
        "direct_policy_executive_post_exit_smell_progression"
    ),
    "executive_post_exit_corridor_progression": (
        "direct_policy_executive_post_exit_corridor_progression"
    ),
    "executive_post_exit_corridor_affordance_progression": (
        "direct_policy_executive_post_exit_corridor_affordance_progression"
    ),
    "executive_post_food_return": "direct_policy_executive_post_food_return",
    "executive_post_food_vector_return": (
        "direct_policy_executive_post_food_vector_return"
    ),
    "executive_post_food_path_return": (
        "direct_policy_executive_post_food_path_return"
    ),
    "transition_prediction_head": "direct_policy_transition_prediction_head",
    "transition_prediction_feedback": "direct_policy_transition_prediction_feedback",
    "transition_rollout_prediction_head": (
        "direct_policy_transition_rollout_prediction_head"
    ),
    "transition_rollout_prediction_feedback": (
        "direct_policy_transition_rollout_prediction_feedback"
    ),
}


@dataclass(frozen=True)
class _DirectPolicyCapabilityGroup:
    fields: Mapping[str, object]

    def _bool(self, name: str) -> bool:
        return bool(self.fields.get(name, False))

    def _int(self, name: str) -> int:
        return int(self.fields.get(name, 0))

    def _float(self, name: str) -> float:
        return float(self.fields.get(name, 0.0))


@dataclass(frozen=True)
class DirectPolicyNetworkCapabilities(_DirectPolicyCapabilityGroup):
    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.fields.get("direct_policy_hidden_dims", ()))

    @property
    def recurrent(self) -> bool:
        return self._bool("direct_policy_recurrent")

    @property
    def event_attention(self) -> bool:
        return self._bool("direct_policy_event_attention")

    @property
    def event_buffer_size(self) -> int:
        return self._int("direct_policy_event_buffer_size")

    @property
    def option_head(self) -> bool:
        return self._bool("direct_policy_option_head")

    @property
    def owned_option_controller(self) -> bool:
        return self._bool("direct_policy_owned_option_controller")

    @property
    def option_ttl(self) -> int:
        return self._int("direct_policy_option_ttl")


@dataclass(frozen=True)
class DirectPolicyHeadCapabilities(_DirectPolicyCapabilityGroup):
    @property
    def phase(self) -> bool:
        return self._bool("direct_policy_phase_head")

    @property
    def affordance(self) -> bool:
        return self._bool("direct_policy_affordance_head")

    @property
    def affordance_feedback(self) -> bool:
        return self._bool("direct_policy_affordance_feedback")

    @property
    def geometry(self) -> bool:
        return self._bool("direct_policy_geometry_head")

    @property
    def shelter_column(self) -> bool:
        return self._bool("direct_policy_shelter_column_head")

    @property
    def shelter_position(self) -> bool:
        return self._bool("direct_policy_shelter_position_head")

    @property
    def transition_prediction(self) -> bool:
        return self._bool("direct_policy_transition_prediction_head")

    @property
    def transition_rollout_prediction(self) -> bool:
        return self._bool("direct_policy_transition_rollout_prediction_head")


@dataclass(frozen=True)
class DirectPolicyLocalInputCapabilities(_DirectPolicyCapabilityGroup):
    @property
    def affordance(self) -> bool:
        return self._bool("direct_policy_local_affordance_inputs")

    @property
    def spatial(self) -> bool:
        return self._bool("direct_policy_local_spatial_inputs")

    @property
    def transition(self) -> bool:
        return self._bool("direct_policy_local_transition_inputs")

    @property
    def transition_rollout(self) -> bool:
        return self._bool("direct_policy_local_transition_rollout_inputs")

    @property
    def geodesic(self) -> bool:
        return self._bool("direct_policy_local_geodesic_inputs")

    def input_dim(self) -> int:
        total = 0
        if self.affordance:
            total += DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
        if self.spatial:
            total += DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
        if self.transition:
            total += DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM
        if self.transition_rollout:
            total += DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM
        if self.geodesic:
            total += DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM
        return total


@dataclass(frozen=True)
class DirectPolicyTeacherCapabilities(_DirectPolicyCapabilityGroup):
    @property
    def handoff(self) -> bool:
        return self._bool("direct_policy_handoff_teacher")

    @property
    def handoff_option(self) -> bool:
        return self._bool("direct_policy_handoff_option_teacher")

    @property
    def continuation_margin_weight(self) -> float:
        return self._float("direct_policy_continuation_margin_weight")


@dataclass(frozen=True)
class DirectPolicyCapabilities:
    fields: Mapping[str, object]

    @property
    def network(self) -> DirectPolicyNetworkCapabilities:
        return DirectPolicyNetworkCapabilities(self.fields)

    @property
    def heads(self) -> DirectPolicyHeadCapabilities:
        return DirectPolicyHeadCapabilities(self.fields)

    @property
    def local_inputs(self) -> DirectPolicyLocalInputCapabilities:
        return DirectPolicyLocalInputCapabilities(self.fields)

    @property
    def teachers(self) -> DirectPolicyTeacherCapabilities:
        return DirectPolicyTeacherCapabilities(self.fields)

    def position_feedback_network_kwargs(self, *, phase_output_dim: int) -> dict[str, object]:
        kwargs = {
            keyword: bool(self.fields.get(field_name, False))
            for keyword, field_name in _POSITION_FEEDBACK_KWARG_FIELDS.items()
        }
        kwargs["phase_output_dim"] = phase_output_dim if self.heads.phase else 0
        return kwargs

    def architecture_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for name in _DIRECT_POLICY_METADATA_FIELDS:
            value = self.fields[name]
            metadata[name] = list(value) if name == "direct_policy_hidden_dims" else value
        return metadata


def derive_direct_policy_capabilities(config: object) -> DirectPolicyCapabilities:
    fields: dict[str, object] = {
        "direct_policy_hidden_dims": tuple(
            int(v) for v in getattr(config, "direct_policy_hidden_dims", ())
        )
    }
    fields.update(
        {
            name: bool(getattr(config, name, False))
            for name in _DIRECT_POLICY_BOOL_FIELDS
        }
    )
    fields.update(
        {name: int(getattr(config, name, 0)) for name in _DIRECT_POLICY_INT_FIELDS}
    )
    fields.update(
        {
            name: float(getattr(config, name, 0.0))
            for name in _DIRECT_POLICY_FLOAT_FIELDS
        }
    )
    return DirectPolicyCapabilities(fields)


def get_direct_policy_capabilities(owner: object) -> DirectPolicyCapabilities:
    capabilities = getattr(owner, "direct_policy_capabilities", None)
    if isinstance(capabilities, DirectPolicyCapabilities):
        return capabilities
    return derive_direct_policy_capabilities(getattr(owner, "config"))
