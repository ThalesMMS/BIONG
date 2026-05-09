from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

from ..capacity_profiles import CapacityProfile, resolve_capacity_profile
from ..interfaces import MODULE_INTERFACES


MODULE_NAMES: tuple[str, ...] = tuple(spec.name for spec in MODULE_INTERFACES)
COARSE_ROLLUP_MODULES: tuple[str, ...] = (
    "perception_center",
    "homeostasis_center",
    "threat_center",
)
A4_FINE_MODULES: tuple[str, ...] = (
    "visual_cortex",
    "sensory_cortex",
    "hunger_center",
    "sleep_center",
    "alert_center",
)
MONOLITHIC_POLICY_NAME: str = "monolithic_policy"
TRUE_MONOLITHIC_POLICY_NAME: str = "true_monolithic_policy"
PROPOSAL_SOURCE_NAMES: tuple[str, ...] = (
    *MODULE_NAMES,
    MONOLITHIC_POLICY_NAME,
    TRUE_MONOLITHIC_POLICY_NAME,
)
REFLEX_MODULE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in MODULE_INTERFACES if spec.role == "proposal"
)
MULTI_PREDATOR_SCENARIOS: tuple[str, ...] = (
    "visual_olfactory_pincer",
    "olfactory_ambush",
    "visual_hunter_open_field",
)
VISUAL_PREDATOR_SCENARIOS: tuple[str, ...] = (
    "visual_olfactory_pincer",
    "visual_hunter_open_field",
)
OLFACTORY_PREDATOR_SCENARIOS: tuple[str, ...] = (
    "visual_olfactory_pincer",
    "olfactory_ambush",
)
MULTI_PREDATOR_SCENARIO_GROUPS: dict[str, tuple[str, ...]] = {
    "multi_predator_ecology": MULTI_PREDATOR_SCENARIOS,
    "visual_predator_scenarios": VISUAL_PREDATOR_SCENARIOS,
    "olfactory_predator_scenarios": OLFACTORY_PREDATOR_SCENARIOS,
}


def _normalize_module_names(
    names: Sequence[str],
    *,
    preserve_order: bool,
    invalid_msg: str,
    monolithic_msg: str,
    architecture: str,
) -> tuple[str, ...]:
    """
    Normalize and validate a sequence of module names with architecture-specific constraints.
    
    Parameters:
        names: Sequence of raw module name values; each entry will be converted to str.
        preserve_order: If True, preserve the first-occurrence order when deduplicating;
            if False, deduplicate and return names sorted.
        invalid_msg: Prefix for the ValueError message when unknown names are present.
        monolithic_msg: Error message used when a non-empty result is not allowed under the
            current monolithic architecture.
        architecture: Architecture identifier; if `"monolithic"` or `"true_monolithic"`, a
            non-empty normalized result is rejected.
    
    Returns:
        tuple[str, ...]: A normalized tuple of unique, validated module names.
    
    Raises:
        ValueError: If any normalized name is not in MODULE_NAMES, or if the architecture is
            `"monolithic"` or `"true_monolithic"` and the resulting tuple is non-empty.
    """
    if preserve_order:
        normalized = tuple(dict.fromkeys(str(n) for n in names))
    else:
        normalized = tuple(sorted({str(n) for n in names}))
    invalid = [n for n in normalized if n not in MODULE_NAMES]
    if invalid:
        raise ValueError(f"{invalid_msg}: {invalid}.")
    if architecture in {"monolithic", "true_monolithic"} and normalized:
        raise ValueError(monolithic_msg)
    return normalized


def architecture_description(architecture: str) -> str:
    """
    Produce a short human-readable label describing the architecture.
    
    Parameters:
        architecture (str): Architecture identifier; expected values include "modular", "monolithic", and "true_monolithic".
    
    Returns:
        A brief label describing the architecture: "true monolithic direct control" for "true_monolithic", "monolithic proposer + action/motor pipeline" for "monolithic", and "full modular with arbitration" otherwise.
    """
    architecture = str(architecture)
    if architecture == "true_monolithic":
        return "true monolithic direct control"
    if architecture == "monolithic":
        return "monolithic proposer + action/motor pipeline"
    return "full modular with arbitration"


@dataclass(frozen=True)
class BrainAblationConfig:
    name: str = "custom"
    architecture: str = "modular"
    module_dropout: float = 0.05
    enable_reflexes: bool = True
    enable_auxiliary_targets: bool = True
    use_learned_arbitration: bool = True
    enable_deterministic_guards: bool = False
    enable_food_direction_bias: bool = False
    warm_start_scale: float = 1.0
    gate_adjustment_bounds: tuple[float, float] = (0.5, 1.5)
    credit_strategy: str = "broadcast"
    route_mask_threshold: float = 0.05
    disabled_modules: tuple[str, ...] = ()
    recurrent_modules: tuple[str, ...] = ()
    reflex_scale: float = 1.0
    module_reflex_scales: dict[str, float] = field(default_factory=dict)
    capacity_profile_name: str = "current"
    capacity_profile: str | CapacityProfile | None = None
    module_hidden_dims: dict[str, int] = field(default_factory=dict)
    direct_policy_hidden_dims: tuple[int, ...] = ()
    direct_policy_recurrent: bool = False
    direct_policy_phase_head: bool = False
    direct_policy_event_attention: bool = False
    direct_policy_event_buffer_size: int = 0
    direct_policy_option_head: bool = False
    direct_policy_option_ttl: int = 0
    direct_policy_affordance_head: bool = False
    direct_policy_affordance_feedback: bool = False
    direct_policy_geometry_head: bool = False
    direct_policy_shelter_column_head: bool = False
    direct_policy_shelter_position_head: bool = False
    direct_policy_local_affordance_inputs: bool = False
    direct_policy_local_spatial_inputs: bool = False
    direct_policy_local_transition_inputs: bool = False
    direct_policy_local_transition_rollout_inputs: bool = False
    direct_policy_local_geodesic_inputs: bool = False
    direct_policy_transition_prediction_head: bool = False
    direct_policy_transition_prediction_feedback: bool = False
    direct_policy_transition_rollout_prediction_head: bool = False
    direct_policy_transition_rollout_prediction_feedback: bool = False
    direct_policy_handoff_teacher: bool = False
    direct_policy_handoff_option_teacher: bool = False
    direct_policy_post_rest_action_teacher: bool = False
    direct_policy_post_rest_release_sequence_teacher: bool = False
    direct_policy_post_rest_release_sequence_replay_boost: bool = False
    direct_policy_post_rest_release_sequence_distill: bool = False
    direct_policy_post_rest_probe_distillation: bool = False
    direct_policy_post_rest_probe_sequence_distillation: bool = False
    direct_policy_post_rest_probe_family_distillation: bool = False
    direct_policy_post_rest_probe_handoff_distillation: bool = False
    direct_policy_post_rest_probe_trajectory_distillation: bool = False
    direct_policy_post_rest_probe_cycle_distillation: bool = False
    direct_policy_post_rest_probe_trace_distillation: bool = False
    direct_policy_post_rest_probe_rollout_distillation: bool = False
    direct_policy_post_rest_probe_frontier_teacher_distillation: bool = False
    direct_policy_post_rest_probe_replayable_teacher_distillation: bool = False
    direct_policy_continuation_replay_passes: int = 0
    direct_policy_continuation_replay_lr_scale: float = 0.0
    direct_policy_continuation_margin_weight: float = 0.0
    direct_policy_phase_option_feedback: bool = False
    direct_policy_option_transition_feedback: bool = False
    direct_policy_option_termination_cooldown: bool = False
    direct_policy_option_action_head: bool = False
    direct_policy_option_decoder_state: bool = False
    direct_policy_option_recurrent_dynamics: bool = False
    direct_policy_option_sequence_head: bool = False
    direct_policy_option_decoder_recurrent_state: bool = False
    direct_policy_option_action_transition_state: bool = False
    direct_policy_option_action_controller_state: bool = False
    direct_policy_option_action_token_decoder: bool = False
    direct_policy_option_action_recurrent_core: bool = False
    direct_policy_option_action_separate_recurrent_head: bool = False
    direct_policy_option_action_separate_policy_path: bool = False
    direct_policy_option_action_separate_backbone: bool = False
    direct_policy_executive_physiology_option_gating: bool = False
    direct_policy_executive_affordance_action_gating: bool = False
    direct_policy_executive_option_action_masking: bool = False
    direct_policy_executive_event_release_latching: bool = False
    direct_policy_executive_event_release_action_commitment: bool = False
    direct_policy_executive_release_phase_state: bool = False
    direct_policy_executive_release_progression: bool = False
    direct_policy_executive_release_exit_contract: bool = False
    direct_policy_executive_release_substate_progression: bool = False
    direct_policy_executive_post_exit_continuation: bool = False
    direct_policy_executive_post_exit_food_guidance: bool = False
    direct_policy_executive_post_exit_food_commitment: bool = False
    direct_policy_executive_post_exit_food_progression: bool = False
    direct_policy_executive_post_exit_food_heading_progression: bool = False
    direct_policy_executive_post_exit_smell_progression: bool = False
    direct_policy_executive_post_exit_corridor_progression: bool = False
    direct_policy_executive_post_exit_corridor_affordance_progression: bool = False
    direct_policy_executive_post_food_return: bool = False
    direct_policy_executive_post_food_vector_return: bool = False
    direct_policy_executive_post_food_path_return: bool = False
    action_center_hidden_dim: int | None = None
    arbitration_hidden_dim: int | None = None
    motor_hidden_dim: int | None = None
    integration_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        """
        Validate and normalize dataclass fields after initialization.
        
        Normalizes and persists these fields on the frozen dataclass:
        - `architecture`: coerced to `str` and must be one of "modular", "monolithic", or "true_monolithic".
        - `module_dropout`: coerced to `float`, must be finite, and must be in [0.0, 1.0].
        - `use_learned_arbitration`: coerced to `bool`.
        - `enable_deterministic_guards`: coerced to `bool`.
        - `enable_food_direction_bias`: coerced to `bool`.
        - `warm_start_scale`: coerced to `float`, must be finite and in [0.0, 1.0].
        - `gate_adjustment_bounds`: coerced to two finite floats ordered as min < max.
        - `credit_strategy`: coerced to `str` and must be "broadcast",
          "local_only", "counterfactual", or "route_mask".
        - `reflex_scale`: coerced to `float`, must be finite and >= 0.0.
        - `disabled_modules`: coerced to `str`, deduplicated, sorted into a `tuple`, and each name must be in `MODULE_NAMES`.
        - `recurrent_modules`: coerced to `str`, deduplicated while preserving order into a `tuple`, and each name must be in `MODULE_NAMES`.
        - `module_reflex_scales`: keys coerced to `str`, values coerced to `float`, sorted by key into a `dict`; each key must be in `REFLEX_MODULE_NAMES`, each value must be finite and >= 0.0.
        
        Raises:
            ValueError: If `architecture` is not "modular", "monolithic", or "true_monolithic";
                        if `module_dropout` or any `module_reflex_scales` value is not finite;
                        if `module_dropout` is outside [0.0, 1.0];
                        if `warm_start_scale` is outside [0.0, 1.0];
                        if `gate_adjustment_bounds` does not contain two finite ordered values;
                        if `reflex_scale` or any `module_reflex_scales` value is negative;
                        if `disabled_modules`, `recurrent_modules`, or `module_reflex_scales` contain unknown module names;
                        or if `architecture` is "monolithic" or "true_monolithic" while `disabled_modules`, `recurrent_modules`, or `module_reflex_scales` are non-empty.
        """
        architecture = str(self.architecture)
        if architecture not in {"modular", "monolithic", "true_monolithic"}:
            raise ValueError(f"Invalid ablation architecture: {architecture!r}.")
        object.__setattr__(self, "architecture", architecture)
        module_dropout = float(self.module_dropout)
        if not math.isfinite(module_dropout):
            raise ValueError("module_dropout must be finite.")
        if module_dropout < 0.0 or module_dropout > 1.0:
            raise ValueError("module_dropout must be in [0.0, 1.0].")
        object.__setattr__(self, "module_dropout", module_dropout)
        object.__setattr__(
            self,
            "use_learned_arbitration",
            bool(self.use_learned_arbitration),
        )
        object.__setattr__(
            self,
            "enable_deterministic_guards",
            bool(self.enable_deterministic_guards),
        )
        object.__setattr__(
            self,
            "enable_food_direction_bias",
            bool(self.enable_food_direction_bias),
        )
        capacity_profile_spec = (
            self.capacity_profile
            if self.capacity_profile is not None
            else self.capacity_profile_name
        )
        resolved_capacity_profile = resolve_capacity_profile(capacity_profile_spec)
        object.__setattr__(
            self,
            "capacity_profile_name",
            resolved_capacity_profile.name,
        )
        object.__setattr__(self, "capacity_profile", resolved_capacity_profile)
        normalized_module_hidden_dims = {
            name: int(hidden_dim)
            for name, hidden_dim in resolved_capacity_profile.module_hidden_dims.items()
        }
        for name, hidden_dim in dict(self.module_hidden_dims).items():
            normalized_module_hidden_dims[str(name)] = int(hidden_dim)
        invalid_hidden_dim_modules = sorted(
            name
            for name in normalized_module_hidden_dims
            if name not in MODULE_NAMES
        )
        if invalid_hidden_dim_modules:
            raise ValueError(
                "Invalid modules in module_hidden_dims: "
                f"{invalid_hidden_dim_modules}."
            )
        if any(hidden_dim <= 0 for hidden_dim in normalized_module_hidden_dims.values()):
            raise ValueError("module_hidden_dims must contain only positive integers.")
        object.__setattr__(
            self,
            "module_hidden_dims",
            {
                name: int(hidden_dim)
                for name, hidden_dim in sorted(normalized_module_hidden_dims.items())
            },
        )
        direct_policy_hidden_dims = tuple(
            int(hidden_dim) for hidden_dim in tuple(self.direct_policy_hidden_dims)
        )
        if direct_policy_hidden_dims and architecture != "true_monolithic":
            raise ValueError(
                "direct_policy_hidden_dims is only supported for true_monolithic architectures."
            )
        if any(hidden_dim <= 0 for hidden_dim in direct_policy_hidden_dims):
            raise ValueError(
                "direct_policy_hidden_dims must contain only positive integers."
            )
        object.__setattr__(
            self,
            "direct_policy_hidden_dims",
            direct_policy_hidden_dims,
        )
        object.__setattr__(
            self,
            "direct_policy_recurrent",
            bool(self.direct_policy_recurrent),
        )
        object.__setattr__(
            self,
            "direct_policy_phase_head",
            bool(self.direct_policy_phase_head),
        )
        object.__setattr__(
            self,
            "direct_policy_event_attention",
            bool(self.direct_policy_event_attention),
        )
        direct_policy_event_buffer_size = int(self.direct_policy_event_buffer_size)
        if bool(self.direct_policy_event_attention):
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_event_attention is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_recurrent):
                raise ValueError(
                    "direct_policy_event_attention requires direct_policy_recurrent=True."
                )
            if direct_policy_event_buffer_size <= 0:
                direct_policy_event_buffer_size = 8
        if direct_policy_event_buffer_size < 0:
            raise ValueError("direct_policy_event_buffer_size must be non-negative.")
        object.__setattr__(
            self,
            "direct_policy_event_buffer_size",
            direct_policy_event_buffer_size,
        )
        object.__setattr__(
            self,
            "direct_policy_option_head",
            bool(self.direct_policy_option_head),
        )
        direct_policy_option_ttl = int(self.direct_policy_option_ttl)
        if bool(self.direct_policy_option_head):
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_head is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_recurrent):
                raise ValueError(
                    "direct_policy_option_head requires direct_policy_recurrent=True."
                )
            if not bool(self.direct_policy_event_attention):
                raise ValueError(
                    "direct_policy_option_head requires direct_policy_event_attention=True."
                )
            if direct_policy_option_ttl <= 0:
                direct_policy_option_ttl = 4
        if direct_policy_option_ttl < 0:
            raise ValueError("direct_policy_option_ttl must be non-negative.")
        object.__setattr__(
            self,
            "direct_policy_option_ttl",
            direct_policy_option_ttl,
        )
        object.__setattr__(
            self,
            "direct_policy_affordance_head",
            bool(self.direct_policy_affordance_head),
        )
        if bool(self.direct_policy_affordance_head):
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_affordance_head is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_recurrent):
                raise ValueError(
                    "direct_policy_affordance_head requires direct_policy_recurrent=True."
                )
            if not bool(self.direct_policy_event_attention):
                raise ValueError(
                    "direct_policy_affordance_head requires direct_policy_event_attention=True."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_affordance_head requires direct_policy_option_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_affordance_feedback",
            bool(self.direct_policy_affordance_feedback),
        )
        if bool(self.direct_policy_affordance_feedback):
            if not bool(self.direct_policy_affordance_head):
                raise ValueError(
                    "direct_policy_affordance_feedback requires direct_policy_affordance_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_geometry_head",
            bool(self.direct_policy_geometry_head),
        )
        if bool(self.direct_policy_geometry_head):
            if not bool(self.direct_policy_affordance_head):
                raise ValueError(
                    "direct_policy_geometry_head requires direct_policy_affordance_head=True."
                )
            if not bool(self.direct_policy_affordance_feedback):
                raise ValueError(
                    "direct_policy_geometry_head requires direct_policy_affordance_feedback=True."
                )
        object.__setattr__(
            self,
            "direct_policy_shelter_column_head",
            bool(self.direct_policy_shelter_column_head),
        )
        if bool(self.direct_policy_shelter_column_head):
            if not bool(self.direct_policy_affordance_head):
                raise ValueError(
                    "direct_policy_shelter_column_head requires direct_policy_affordance_head=True."
                )
            if not bool(self.direct_policy_affordance_feedback):
                raise ValueError(
                    "direct_policy_shelter_column_head requires direct_policy_affordance_feedback=True."
                )
            if not bool(self.direct_policy_geometry_head):
                raise ValueError(
                    "direct_policy_shelter_column_head requires direct_policy_geometry_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_shelter_position_head",
            bool(self.direct_policy_shelter_position_head),
        )
        if bool(self.direct_policy_shelter_position_head):
            if not bool(self.direct_policy_affordance_head):
                raise ValueError(
                    "direct_policy_shelter_position_head requires direct_policy_affordance_head=True."
                )
            if not bool(self.direct_policy_affordance_feedback):
                raise ValueError(
                    "direct_policy_shelter_position_head requires direct_policy_affordance_feedback=True."
                )
            if not bool(self.direct_policy_geometry_head):
                raise ValueError(
                    "direct_policy_shelter_position_head requires direct_policy_geometry_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_local_affordance_inputs",
            bool(self.direct_policy_local_affordance_inputs),
        )
        if bool(self.direct_policy_local_affordance_inputs):
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_local_affordance_inputs is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_shelter_position_head):
                raise ValueError(
                    "direct_policy_local_affordance_inputs requires direct_policy_shelter_position_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_local_spatial_inputs",
            bool(self.direct_policy_local_spatial_inputs),
        )
        if bool(self.direct_policy_local_spatial_inputs):
            if not bool(self.direct_policy_local_affordance_inputs):
                raise ValueError(
                    "direct_policy_local_spatial_inputs requires direct_policy_local_affordance_inputs=True."
                )
        object.__setattr__(
            self,
            "direct_policy_local_transition_inputs",
            bool(self.direct_policy_local_transition_inputs),
        )
        if bool(self.direct_policy_local_transition_inputs):
            if not bool(self.direct_policy_local_spatial_inputs):
                raise ValueError(
                    "direct_policy_local_transition_inputs requires direct_policy_local_spatial_inputs=True."
                )
        object.__setattr__(
            self,
            "direct_policy_local_transition_rollout_inputs",
            bool(self.direct_policy_local_transition_rollout_inputs),
        )
        if bool(self.direct_policy_local_transition_rollout_inputs):
            if not bool(self.direct_policy_local_transition_inputs):
                raise ValueError(
                    "direct_policy_local_transition_rollout_inputs requires direct_policy_local_transition_inputs=True."
                )
        object.__setattr__(
            self,
            "direct_policy_local_geodesic_inputs",
            bool(self.direct_policy_local_geodesic_inputs),
        )
        if bool(self.direct_policy_local_geodesic_inputs):
            if not bool(self.direct_policy_local_transition_rollout_inputs):
                raise ValueError(
                    "direct_policy_local_geodesic_inputs requires direct_policy_local_transition_rollout_inputs=True."
                )
        object.__setattr__(
            self,
            "direct_policy_transition_prediction_head",
            bool(self.direct_policy_transition_prediction_head),
        )
        if bool(self.direct_policy_transition_prediction_head):
            if not bool(self.direct_policy_local_transition_inputs):
                raise ValueError(
                    "direct_policy_transition_prediction_head requires direct_policy_local_transition_inputs=True."
                )
        object.__setattr__(
            self,
            "direct_policy_transition_prediction_feedback",
            bool(self.direct_policy_transition_prediction_feedback),
        )
        if bool(self.direct_policy_transition_prediction_feedback):
            if not bool(self.direct_policy_transition_prediction_head):
                raise ValueError(
                    "direct_policy_transition_prediction_feedback requires direct_policy_transition_prediction_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_transition_rollout_prediction_head",
            bool(self.direct_policy_transition_rollout_prediction_head),
        )
        if bool(self.direct_policy_transition_rollout_prediction_head):
            if not bool(self.direct_policy_local_transition_rollout_inputs):
                raise ValueError(
                    "direct_policy_transition_rollout_prediction_head requires direct_policy_local_transition_rollout_inputs=True."
                )
        object.__setattr__(
            self,
            "direct_policy_transition_rollout_prediction_feedback",
            bool(self.direct_policy_transition_rollout_prediction_feedback),
        )
        if bool(self.direct_policy_transition_rollout_prediction_feedback):
            if not bool(self.direct_policy_transition_rollout_prediction_head):
                raise ValueError(
                    "direct_policy_transition_rollout_prediction_feedback requires direct_policy_transition_rollout_prediction_head=True."
                )
        if bool(self.direct_policy_phase_head) and bool(self.direct_policy_option_head):
            if not bool(self.direct_policy_shelter_position_head):
                raise ValueError(
                    "direct_policy_phase_head with direct_policy_option_head requires direct_policy_shelter_position_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_handoff_teacher",
            bool(self.direct_policy_handoff_teacher),
        )
        if bool(self.direct_policy_handoff_teacher):
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_handoff_teacher is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_shelter_position_head):
                raise ValueError(
                    "direct_policy_handoff_teacher requires direct_policy_shelter_position_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_handoff_option_teacher",
            bool(self.direct_policy_handoff_option_teacher),
        )
        if bool(self.direct_policy_handoff_option_teacher):
            if not bool(self.direct_policy_handoff_teacher):
                raise ValueError(
                    "direct_policy_handoff_option_teacher requires direct_policy_handoff_teacher=True."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_handoff_option_teacher requires direct_policy_option_head=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_action_teacher",
            bool(self.direct_policy_post_rest_action_teacher),
        )
        if bool(self.direct_policy_post_rest_action_teacher):
            if not bool(self.direct_policy_handoff_teacher):
                raise ValueError(
                    "direct_policy_post_rest_action_teacher requires direct_policy_handoff_teacher=True."
                )
            if not bool(self.direct_policy_option_action_separate_backbone):
                raise ValueError(
                    "direct_policy_post_rest_action_teacher requires direct_policy_option_action_separate_backbone=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_release_sequence_teacher",
            bool(self.direct_policy_post_rest_release_sequence_teacher),
        )
        if bool(self.direct_policy_post_rest_release_sequence_teacher):
            if not bool(self.direct_policy_post_rest_action_teacher):
                raise ValueError(
                    "direct_policy_post_rest_release_sequence_teacher requires direct_policy_post_rest_action_teacher=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_release_sequence_replay_boost",
            bool(self.direct_policy_post_rest_release_sequence_replay_boost),
        )
        if bool(self.direct_policy_post_rest_release_sequence_replay_boost):
            if not bool(self.direct_policy_post_rest_release_sequence_teacher):
                raise ValueError(
                    "direct_policy_post_rest_release_sequence_replay_boost requires direct_policy_post_rest_release_sequence_teacher=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_release_sequence_distill",
            bool(self.direct_policy_post_rest_release_sequence_distill),
        )
        if bool(self.direct_policy_post_rest_release_sequence_distill):
            if not bool(self.direct_policy_post_rest_release_sequence_teacher):
                raise ValueError(
                    "direct_policy_post_rest_release_sequence_distill requires direct_policy_post_rest_release_sequence_teacher=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_distillation",
            bool(self.direct_policy_post_rest_probe_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_distillation):
            if not bool(self.direct_policy_post_rest_release_sequence_teacher):
                raise ValueError(
                    "direct_policy_post_rest_probe_distillation requires direct_policy_post_rest_release_sequence_teacher=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_sequence_distillation",
            bool(self.direct_policy_post_rest_probe_sequence_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_sequence_distillation):
            if not bool(self.direct_policy_post_rest_probe_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_sequence_distillation requires direct_policy_post_rest_probe_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_family_distillation",
            bool(self.direct_policy_post_rest_probe_family_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_family_distillation):
            if not bool(self.direct_policy_post_rest_probe_sequence_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_family_distillation requires direct_policy_post_rest_probe_sequence_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_handoff_distillation",
            bool(self.direct_policy_post_rest_probe_handoff_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_handoff_distillation):
            if not bool(self.direct_policy_post_rest_probe_family_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_handoff_distillation requires direct_policy_post_rest_probe_family_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_trajectory_distillation",
            bool(self.direct_policy_post_rest_probe_trajectory_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_trajectory_distillation):
            if not bool(self.direct_policy_post_rest_probe_handoff_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_trajectory_distillation requires direct_policy_post_rest_probe_handoff_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_cycle_distillation",
            bool(self.direct_policy_post_rest_probe_cycle_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_cycle_distillation):
            if not bool(self.direct_policy_post_rest_probe_trajectory_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_cycle_distillation requires direct_policy_post_rest_probe_trajectory_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_trace_distillation",
            bool(self.direct_policy_post_rest_probe_trace_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_trace_distillation):
            if not bool(self.direct_policy_post_rest_probe_cycle_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_trace_distillation requires direct_policy_post_rest_probe_cycle_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_rollout_distillation",
            bool(self.direct_policy_post_rest_probe_rollout_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_rollout_distillation):
            if not bool(self.direct_policy_post_rest_probe_trace_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_rollout_distillation requires direct_policy_post_rest_probe_trace_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_frontier_teacher_distillation",
            bool(self.direct_policy_post_rest_probe_frontier_teacher_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_frontier_teacher_distillation):
            if not bool(self.direct_policy_post_rest_probe_rollout_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_frontier_teacher_distillation requires direct_policy_post_rest_probe_rollout_distillation=True."
                )
        object.__setattr__(
            self,
            "direct_policy_post_rest_probe_replayable_teacher_distillation",
            bool(self.direct_policy_post_rest_probe_replayable_teacher_distillation),
        )
        if bool(self.direct_policy_post_rest_probe_replayable_teacher_distillation):
            if not bool(self.direct_policy_post_rest_probe_rollout_distillation):
                raise ValueError(
                    "direct_policy_post_rest_probe_replayable_teacher_distillation requires direct_policy_post_rest_probe_rollout_distillation=True."
                )
        continuation_replay_passes = int(
            self.direct_policy_continuation_replay_passes
        )
        if continuation_replay_passes < 0:
            raise ValueError(
                "direct_policy_continuation_replay_passes must be >= 0."
            )
        object.__setattr__(
            self,
            "direct_policy_continuation_replay_passes",
            continuation_replay_passes,
        )
        continuation_replay_lr_scale = float(
            self.direct_policy_continuation_replay_lr_scale
        )
        if not math.isfinite(continuation_replay_lr_scale):
            raise ValueError(
                "direct_policy_continuation_replay_lr_scale must be finite."
            )
        if continuation_replay_lr_scale < 0.0:
            raise ValueError(
                "direct_policy_continuation_replay_lr_scale must be >= 0.0."
            )
        object.__setattr__(
            self,
            "direct_policy_continuation_replay_lr_scale",
            continuation_replay_lr_scale,
        )
        continuation_margin_weight = float(
            self.direct_policy_continuation_margin_weight
        )
        if not math.isfinite(continuation_margin_weight):
            raise ValueError(
                "direct_policy_continuation_margin_weight must be finite."
            )
        if continuation_margin_weight < 0.0:
            raise ValueError(
                "direct_policy_continuation_margin_weight must be >= 0.0."
            )
        object.__setattr__(
            self,
            "direct_policy_continuation_margin_weight",
            continuation_margin_weight,
        )
        phase_option_feedback = bool(self.direct_policy_phase_option_feedback)
        object.__setattr__(
            self,
            "direct_policy_phase_option_feedback",
            phase_option_feedback,
        )
        option_transition_feedback = bool(
            self.direct_policy_option_transition_feedback
        )
        object.__setattr__(
            self,
            "direct_policy_option_transition_feedback",
            option_transition_feedback,
        )
        option_termination_cooldown = bool(
            self.direct_policy_option_termination_cooldown
        )
        object.__setattr__(
            self,
            "direct_policy_option_termination_cooldown",
            option_termination_cooldown,
        )
        option_action_head = bool(self.direct_policy_option_action_head)
        object.__setattr__(
            self,
            "direct_policy_option_action_head",
            option_action_head,
        )
        option_decoder_state = bool(self.direct_policy_option_decoder_state)
        object.__setattr__(
            self,
            "direct_policy_option_decoder_state",
            option_decoder_state,
        )
        option_recurrent_dynamics = bool(
            self.direct_policy_option_recurrent_dynamics
        )
        object.__setattr__(
            self,
            "direct_policy_option_recurrent_dynamics",
            option_recurrent_dynamics,
        )
        option_sequence_head = bool(self.direct_policy_option_sequence_head)
        object.__setattr__(
            self,
            "direct_policy_option_sequence_head",
            option_sequence_head,
        )
        option_decoder_recurrent_state = bool(
            self.direct_policy_option_decoder_recurrent_state
        )
        object.__setattr__(
            self,
            "direct_policy_option_decoder_recurrent_state",
            option_decoder_recurrent_state,
        )
        option_action_transition_state = bool(
            self.direct_policy_option_action_transition_state
        )
        object.__setattr__(
            self,
            "direct_policy_option_action_transition_state",
            option_action_transition_state,
        )
        option_action_controller_state = bool(
            self.direct_policy_option_action_controller_state
        )
        object.__setattr__(
            self,
            "direct_policy_option_action_controller_state",
            option_action_controller_state,
        )
        option_action_token_decoder = bool(
            self.direct_policy_option_action_token_decoder
        )
        object.__setattr__(
            self,
            "direct_policy_option_action_token_decoder",
            option_action_token_decoder,
        )
        option_action_recurrent_core = bool(
            self.direct_policy_option_action_recurrent_core
        )
        object.__setattr__(
            self,
            "direct_policy_option_action_recurrent_core",
            option_action_recurrent_core,
        )
        option_action_separate_recurrent_head = bool(
            self.direct_policy_option_action_separate_recurrent_head
        )
        object.__setattr__(
            self,
            "direct_policy_option_action_separate_recurrent_head",
            option_action_separate_recurrent_head,
        )
        option_action_separate_policy_path = bool(
            self.direct_policy_option_action_separate_policy_path
        )
        object.__setattr__(
            self,
            "direct_policy_option_action_separate_policy_path",
            option_action_separate_policy_path,
        )
        option_action_separate_backbone = bool(
            self.direct_policy_option_action_separate_backbone
        )
        object.__setattr__(
            self,
            "direct_policy_option_action_separate_backbone",
            option_action_separate_backbone,
        )
        executive_physiology_option_gating = bool(
            self.direct_policy_executive_physiology_option_gating
        )
        object.__setattr__(
            self,
            "direct_policy_executive_physiology_option_gating",
            executive_physiology_option_gating,
        )
        executive_affordance_action_gating = bool(
            self.direct_policy_executive_affordance_action_gating
        )
        object.__setattr__(
            self,
            "direct_policy_executive_affordance_action_gating",
            executive_affordance_action_gating,
        )
        executive_option_action_masking = bool(
            self.direct_policy_executive_option_action_masking
        )
        object.__setattr__(
            self,
            "direct_policy_executive_option_action_masking",
            executive_option_action_masking,
        )
        executive_event_release_latching = bool(
            self.direct_policy_executive_event_release_latching
        )
        object.__setattr__(
            self,
            "direct_policy_executive_event_release_latching",
            executive_event_release_latching,
        )
        executive_event_release_action_commitment = bool(
            self.direct_policy_executive_event_release_action_commitment
        )
        object.__setattr__(
            self,
            "direct_policy_executive_event_release_action_commitment",
            executive_event_release_action_commitment,
        )
        executive_release_phase_state = bool(
            self.direct_policy_executive_release_phase_state
        )
        object.__setattr__(
            self,
            "direct_policy_executive_release_phase_state",
            executive_release_phase_state,
        )
        executive_release_progression = bool(
            self.direct_policy_executive_release_progression
        )
        object.__setattr__(
            self,
            "direct_policy_executive_release_progression",
            executive_release_progression,
        )
        executive_release_exit_contract = bool(
            self.direct_policy_executive_release_exit_contract
        )
        object.__setattr__(
            self,
            "direct_policy_executive_release_exit_contract",
            executive_release_exit_contract,
        )
        executive_release_substate_progression = bool(
            self.direct_policy_executive_release_substate_progression
        )
        object.__setattr__(
            self,
            "direct_policy_executive_release_substate_progression",
            executive_release_substate_progression,
        )
        executive_post_exit_continuation = bool(
            self.direct_policy_executive_post_exit_continuation
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_continuation",
            executive_post_exit_continuation,
        )
        executive_post_exit_food_guidance = bool(
            self.direct_policy_executive_post_exit_food_guidance
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_food_guidance",
            executive_post_exit_food_guidance,
        )
        executive_post_exit_food_commitment = bool(
            self.direct_policy_executive_post_exit_food_commitment
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_food_commitment",
            executive_post_exit_food_commitment,
        )
        executive_post_exit_food_progression = bool(
            self.direct_policy_executive_post_exit_food_progression
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_food_progression",
            executive_post_exit_food_progression,
        )
        executive_post_exit_food_heading_progression = bool(
            self.direct_policy_executive_post_exit_food_heading_progression
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_food_heading_progression",
            executive_post_exit_food_heading_progression,
        )
        executive_post_exit_smell_progression = bool(
            self.direct_policy_executive_post_exit_smell_progression
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_smell_progression",
            executive_post_exit_smell_progression,
        )
        executive_post_exit_corridor_progression = bool(
            self.direct_policy_executive_post_exit_corridor_progression
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_corridor_progression",
            executive_post_exit_corridor_progression,
        )
        executive_post_exit_corridor_affordance_progression = bool(
            self.direct_policy_executive_post_exit_corridor_affordance_progression
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_exit_corridor_affordance_progression",
            executive_post_exit_corridor_affordance_progression,
        )
        executive_post_food_return = bool(
            self.direct_policy_executive_post_food_return
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_food_return",
            executive_post_food_return,
        )
        executive_post_food_vector_return = bool(
            self.direct_policy_executive_post_food_vector_return
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_food_vector_return",
            executive_post_food_vector_return,
        )
        executive_post_food_path_return = bool(
            self.direct_policy_executive_post_food_path_return
        )
        object.__setattr__(
            self,
            "direct_policy_executive_post_food_path_return",
            executive_post_food_path_return,
        )
        if continuation_replay_passes > 0:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_continuation_replay_passes is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_handoff_option_teacher):
                raise ValueError(
                    "direct_policy_continuation_replay_passes requires direct_policy_handoff_option_teacher=True."
                )
            if continuation_replay_lr_scale <= 0.0:
                raise ValueError(
                    "direct_policy_continuation_replay_lr_scale must be > 0.0 when continuation replay is enabled."
                )
        if continuation_margin_weight > 0.0:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_continuation_margin_weight is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_handoff_option_teacher):
                raise ValueError(
                    "direct_policy_continuation_margin_weight requires direct_policy_handoff_option_teacher=True."
                )
            if not bool(self.direct_policy_phase_head):
                raise ValueError(
                    "direct_policy_continuation_margin_weight requires direct_policy_phase_head=True."
                )
        if phase_option_feedback:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_phase_option_feedback is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_phase_head):
                raise ValueError(
                    "direct_policy_phase_option_feedback requires direct_policy_phase_head=True."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_phase_option_feedback requires direct_policy_option_head=True."
                )
        if option_transition_feedback:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_transition_feedback is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_transition_feedback requires direct_policy_option_head=True."
                )
        if option_termination_cooldown:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_termination_cooldown is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_termination_cooldown requires direct_policy_option_head=True."
                )
        if option_action_head:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_head is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_head requires direct_policy_option_head=True."
                )
        if option_decoder_state:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_decoder_state is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_decoder_state requires direct_policy_option_head=True."
                )
        if option_recurrent_dynamics:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_recurrent_dynamics is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_recurrent_dynamics requires direct_policy_option_head=True."
                )
        if option_sequence_head:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_sequence_head is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_sequence_head requires direct_policy_option_head=True."
                )
        if option_decoder_recurrent_state:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_decoder_recurrent_state is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_decoder_recurrent_state requires direct_policy_option_head=True."
                )
        if option_action_transition_state:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_transition_state is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_transition_state requires direct_policy_option_head=True."
                )
        if option_action_controller_state:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_controller_state is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_controller_state requires direct_policy_option_head=True."
                )
        if option_action_token_decoder:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_token_decoder is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_token_decoder requires direct_policy_option_head=True."
                )
        if option_action_recurrent_core:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_recurrent_core is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_recurrent_core requires direct_policy_option_head=True."
                )
        if option_action_separate_recurrent_head:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_separate_recurrent_head is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_separate_recurrent_head requires direct_policy_option_head=True."
                )
            if not option_action_recurrent_core:
                raise ValueError(
                    "direct_policy_option_action_separate_recurrent_head requires direct_policy_option_action_recurrent_core=True."
                )
        if option_action_separate_policy_path:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_separate_policy_path is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_separate_policy_path requires direct_policy_option_head=True."
                )
            if not option_action_recurrent_core:
                raise ValueError(
                    "direct_policy_option_action_separate_policy_path requires direct_policy_option_action_recurrent_core=True."
                )
        if option_action_separate_backbone:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_option_action_separate_backbone is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_option_action_separate_backbone requires direct_policy_option_head=True."
                )
            if not option_action_separate_policy_path:
                raise ValueError(
                    "direct_policy_option_action_separate_backbone requires direct_policy_option_action_separate_policy_path=True."
                )
        if executive_physiology_option_gating:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_physiology_option_gating is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_executive_physiology_option_gating requires direct_policy_option_head=True."
                )
        if executive_affordance_action_gating:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_affordance_action_gating is only supported for true_monolithic architectures."
                )
            if not bool(self.direct_policy_option_head):
                raise ValueError(
                    "direct_policy_executive_affordance_action_gating requires direct_policy_option_head=True."
                )
            if not bool(self.direct_policy_affordance_head):
                raise ValueError(
                    "direct_policy_executive_affordance_action_gating requires direct_policy_affordance_head=True."
                )
            if not bool(self.direct_policy_geometry_head):
                raise ValueError(
                    "direct_policy_executive_affordance_action_gating requires direct_policy_geometry_head=True."
                )
            if not bool(self.direct_policy_shelter_position_head):
                raise ValueError(
                    "direct_policy_executive_affordance_action_gating requires direct_policy_shelter_position_head=True."
                )
        if executive_option_action_masking:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_option_action_masking is only supported for true_monolithic architectures."
                )
            if not executive_affordance_action_gating:
                raise ValueError(
                    "direct_policy_executive_option_action_masking requires direct_policy_executive_affordance_action_gating=True."
                )
        if executive_event_release_latching:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_event_release_latching is only supported for true_monolithic architectures."
                )
            if not executive_physiology_option_gating:
                raise ValueError(
                    "direct_policy_executive_event_release_latching requires direct_policy_executive_physiology_option_gating=True."
                )
            if not executive_affordance_action_gating:
                raise ValueError(
                    "direct_policy_executive_event_release_latching requires direct_policy_executive_affordance_action_gating=True."
                )
        if executive_event_release_action_commitment:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_event_release_action_commitment is only supported for true_monolithic architectures."
                )
            if not executive_event_release_latching:
                raise ValueError(
                    "direct_policy_executive_event_release_action_commitment requires direct_policy_executive_event_release_latching=True."
                )
        if executive_release_phase_state:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_release_phase_state is only supported for true_monolithic architectures."
                )
            if not executive_event_release_action_commitment:
                raise ValueError(
                    "direct_policy_executive_release_phase_state requires direct_policy_executive_event_release_action_commitment=True."
                )
        if executive_release_progression:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_release_progression is only supported for true_monolithic architectures."
                )
            if not executive_release_phase_state:
                raise ValueError(
                    "direct_policy_executive_release_progression requires direct_policy_executive_release_phase_state=True."
                )
        if executive_release_exit_contract:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_release_exit_contract is only supported for true_monolithic architectures."
                )
            if not executive_release_phase_state:
                raise ValueError(
                    "direct_policy_executive_release_exit_contract requires direct_policy_executive_release_phase_state=True."
                )
        if executive_release_substate_progression:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_release_substate_progression is only supported for true_monolithic architectures."
                )
            if not executive_release_phase_state:
                raise ValueError(
                    "direct_policy_executive_release_substate_progression requires direct_policy_executive_release_phase_state=True."
                )
            if not executive_release_exit_contract:
                raise ValueError(
                    "direct_policy_executive_release_substate_progression requires direct_policy_executive_release_exit_contract=True."
                )
        if executive_post_exit_continuation:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_continuation is only supported for true_monolithic architectures."
                )
            if not executive_release_substate_progression:
                raise ValueError(
                    "direct_policy_executive_post_exit_continuation requires direct_policy_executive_release_substate_progression=True."
                )
        if executive_post_exit_food_guidance:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_food_guidance is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_continuation:
                raise ValueError(
                    "direct_policy_executive_post_exit_food_guidance requires direct_policy_executive_post_exit_continuation=True."
                )
        if executive_post_exit_food_commitment:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_food_commitment is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_food_guidance:
                raise ValueError(
                    "direct_policy_executive_post_exit_food_commitment requires direct_policy_executive_post_exit_food_guidance=True."
                )
        if executive_post_exit_food_progression:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_food_progression is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_food_guidance:
                raise ValueError(
                    "direct_policy_executive_post_exit_food_progression requires direct_policy_executive_post_exit_food_guidance=True."
                )
        if executive_post_exit_food_heading_progression:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_food_heading_progression is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_food_guidance:
                raise ValueError(
                    "direct_policy_executive_post_exit_food_heading_progression requires direct_policy_executive_post_exit_food_guidance=True."
                )
        if executive_post_exit_smell_progression:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_smell_progression is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_food_guidance:
                raise ValueError(
                    "direct_policy_executive_post_exit_smell_progression requires direct_policy_executive_post_exit_food_guidance=True."
                )
        if executive_post_exit_corridor_progression:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_corridor_progression is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_continuation:
                raise ValueError(
                    "direct_policy_executive_post_exit_corridor_progression requires direct_policy_executive_post_exit_continuation=True."
                )
        if executive_post_exit_corridor_affordance_progression:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_exit_corridor_affordance_progression is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_corridor_progression:
                raise ValueError(
                    "direct_policy_executive_post_exit_corridor_affordance_progression requires direct_policy_executive_post_exit_corridor_progression=True."
                )
        if executive_post_food_return:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_food_return is only supported for true_monolithic architectures."
                )
            if not executive_post_exit_continuation:
                raise ValueError(
                    "direct_policy_executive_post_food_return requires direct_policy_executive_post_exit_continuation=True."
                )
        if executive_post_food_vector_return:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_food_vector_return is only supported for true_monolithic architectures."
                )
            if not executive_post_food_return:
                raise ValueError(
                    "direct_policy_executive_post_food_vector_return requires direct_policy_executive_post_food_return=True."
                )
        if executive_post_food_path_return:
            if architecture != "true_monolithic":
                raise ValueError(
                    "direct_policy_executive_post_food_path_return is only supported for true_monolithic architectures."
                )
            if not executive_post_food_return:
                raise ValueError(
                    "direct_policy_executive_post_food_path_return requires direct_policy_executive_post_food_return=True."
                )
        legacy_integration_hidden_dim = (
            None
            if self.integration_hidden_dim is None
            else int(self.integration_hidden_dim)
        )
        action_center_hidden_dim = (
            resolved_capacity_profile.action_center_hidden_dim
            if self.action_center_hidden_dim is None
            else int(self.action_center_hidden_dim)
        )
        arbitration_hidden_dim = (
            resolved_capacity_profile.arbitration_hidden_dim
            if self.arbitration_hidden_dim is None
            else int(self.arbitration_hidden_dim)
        )
        motor_hidden_dim = (
            resolved_capacity_profile.motor_hidden_dim
            if self.motor_hidden_dim is None
            else int(self.motor_hidden_dim)
        )
        if legacy_integration_hidden_dim is not None:
            if self.action_center_hidden_dim is None:
                action_center_hidden_dim = legacy_integration_hidden_dim
            if self.arbitration_hidden_dim is None:
                arbitration_hidden_dim = legacy_integration_hidden_dim
            if self.motor_hidden_dim is None:
                motor_hidden_dim = legacy_integration_hidden_dim
        hidden_dim_fields = {
            "action_center_hidden_dim": action_center_hidden_dim,
            "arbitration_hidden_dim": arbitration_hidden_dim,
            "motor_hidden_dim": motor_hidden_dim,
        }
        non_positive_hidden_dims = [
            name for name, value in hidden_dim_fields.items() if int(value) <= 0
        ]
        if non_positive_hidden_dims:
            raise ValueError(
                "integration hidden dims must be positive for "
                f"{non_positive_hidden_dims}."
            )
        integration_hidden_dim = (
            legacy_integration_hidden_dim
            if legacy_integration_hidden_dim is not None
            else action_center_hidden_dim
        )
        for _field, _value in (
            ("action_center_hidden_dim", action_center_hidden_dim),
            ("arbitration_hidden_dim", arbitration_hidden_dim),
            ("motor_hidden_dim", motor_hidden_dim),
            ("integration_hidden_dim", integration_hidden_dim),
        ):
            object.__setattr__(self, _field, int(_value))
        warm_start_scale = float(self.warm_start_scale)
        if not math.isfinite(warm_start_scale):
            raise ValueError("warm_start_scale must be finite.")
        if warm_start_scale < 0.0 or warm_start_scale > 1.0:
            raise ValueError("warm_start_scale must be in [0.0, 1.0].")
        object.__setattr__(self, "warm_start_scale", warm_start_scale)
        try:
            gate_adjustment_bounds = tuple(
                float(value)
                for value in self.gate_adjustment_bounds
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("gate_adjustment_bounds must contain two finite values.") from exc
        if len(gate_adjustment_bounds) != 2:
            raise ValueError("gate_adjustment_bounds must contain exactly two values.")
        gate_adjustment_min, gate_adjustment_max = gate_adjustment_bounds
        if (
            not math.isfinite(gate_adjustment_min)
            or not math.isfinite(gate_adjustment_max)
        ):
            raise ValueError("gate_adjustment_bounds must contain only finite values.")
        if gate_adjustment_min >= gate_adjustment_max:
            raise ValueError("gate_adjustment_bounds must be ordered as min < max.")
        object.__setattr__(self, "gate_adjustment_bounds", gate_adjustment_bounds)
        credit_strategy = str(self.credit_strategy)
        if credit_strategy not in {
            "broadcast",
            "local_only",
            "counterfactual",
            "route_mask",
        }:
            raise ValueError(f"Invalid credit_strategy: {credit_strategy!r}.")
        object.__setattr__(self, "credit_strategy", credit_strategy)
        route_mask_threshold = float(self.route_mask_threshold)
        if not math.isfinite(route_mask_threshold):
            raise ValueError("route_mask_threshold must be finite.")
        if route_mask_threshold < 0.0:
            raise ValueError("route_mask_threshold must be non-negative.")
        object.__setattr__(self, "route_mask_threshold", route_mask_threshold)
        reflex_scale = float(self.reflex_scale)
        if not math.isfinite(reflex_scale):
            raise ValueError("reflex_scale must be finite.")
        if reflex_scale < 0.0:
            raise ValueError("reflex_scale must be non-negative.")
        object.__setattr__(self, "reflex_scale", reflex_scale)

        object.__setattr__(self, "disabled_modules", _normalize_module_names(
            self.disabled_modules,
            preserve_order=False,
            invalid_msg="Invalid ablation modules",
            monolithic_msg="Monolithic baselines do not accept disabled modules.",
            architecture=architecture,
        ))
        object.__setattr__(self, "recurrent_modules", _normalize_module_names(
            self.recurrent_modules,
            preserve_order=True,
            invalid_msg="Invalid recurrent modules",
            monolithic_msg="Monolithic baselines do not accept recurrent modules.",
            architecture=architecture,
        ))
        normalized_reflex_scales = {
            name: float(value)
            for name, value in sorted(
                (str(name), value)
                for name, value in dict(self.module_reflex_scales).items()
            )
        }
        invalid_reflex_modules = [
            name for name in normalized_reflex_scales if name not in REFLEX_MODULE_NAMES
        ]
        if invalid_reflex_modules:
            raise ValueError(
                f"Invalid modules in module_reflex_scales: {invalid_reflex_modules}."
            )
        non_finite_reflex_scales = [
            name
            for name, value in normalized_reflex_scales.items()
            if not math.isfinite(value)
        ]
        if non_finite_reflex_scales:
            raise ValueError(
                "module_reflex_scales must contain only finite values."
            )
        negative_reflex_scales = [
            name
            for name, value in normalized_reflex_scales.items()
            if value < 0.0
        ]
        if negative_reflex_scales:
            raise ValueError(
                "module_reflex_scales must contain only non-negative values."
            )
        if architecture in {"monolithic", "true_monolithic"} and normalized_reflex_scales:
            raise ValueError(
                "Monolithic baselines do not accept per-module module_reflex_scales."
            )
        object.__setattr__(self, "module_reflex_scales", normalized_reflex_scales)

    @property
    def is_modular(self) -> bool:
        """
        Whether the configuration uses the modular architecture.
        
        Returns:
            `true` if `architecture` equals "modular", `false` otherwise.
        """
        return self.architecture == "modular"

    @property
    def is_monolithic(self) -> bool:
        """
        Indicates whether the configuration uses the monolithic architecture.
        
        Returns:
            `true` if the configuration's architecture is "monolithic", `false` otherwise.
        """
        return self.architecture == "monolithic"

    @property
    def is_true_monolithic(self) -> bool:
        """
        Indicates whether the configuration uses the true-monolithic architecture.
        
        Returns:
            True if the architecture is 'true_monolithic', False otherwise.
        """
        return self.architecture == "true_monolithic"

    @property
    def uses_local_credit_only(self) -> bool:
        """
        Whether the configuration uses local-only credit assignment.
        
        Returns:
            `true` if `credit_strategy` is "local_only", `false` otherwise.
        """
        return self.credit_strategy == "local_only"

    @property
    def uses_counterfactual_credit(self) -> bool:
        """
        Indicates whether the ablation uses counterfactual credit.
        
        Returns:
            True if `credit_strategy` is "counterfactual", False otherwise.
        """
        return self.credit_strategy == "counterfactual"

    @property
    def uses_route_mask_credit(self) -> bool:
        return self.credit_strategy == "route_mask"

    @property
    def is_recurrent(self) -> bool:
        """
        Whether any module proposer uses recurrent state.
        
        Returns:
            `true` if `recurrent_modules` is non-empty, `false` otherwise.
        """
        return bool(self.recurrent_modules) or bool(self.direct_policy_recurrent)

    @property
    def monolithic_hidden_dim(self) -> int:
        """
        Total hidden dimension produced by summing all per-module hidden sizes.
        
        Returns:
            int: Sum of all module hidden dimensions.
        """
        if self.direct_policy_hidden_dims:
            return int(sum(self.direct_policy_hidden_dims))
        return int(sum(self.module_hidden_dims.values()))

    def to_summary(self) -> Dict[str, object]:
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
        }

    @classmethod
    def from_summary(
        cls,
        summary: Mapping[str, object],
    ) -> "BrainAblationConfig":
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
        )


def _arbitration_fields(
    *,
    use_learned_arbitration: bool = True,
    enable_deterministic_guards: bool = False,
    enable_food_direction_bias: bool = False,
    warm_start_scale: float = 1.0,
    gate_adjustment_bounds: tuple[float, float] = (0.5, 1.5),
) -> dict[str, object]:
    """
    Collects arbitration-related configuration values into a dictionary.
    
    Parameters:
        use_learned_arbitration (bool): Whether learned arbitration is enabled.
        enable_deterministic_guards (bool): Whether deterministic guard behavior is enabled.
        enable_food_direction_bias (bool): Whether a bias toward food direction is applied.
        warm_start_scale (float): Scaling factor applied during warm-start of arbitration.
        gate_adjustment_bounds (tuple[float, float]): Lower and upper bounds used to adjust gate values (min, max).
    
    Returns:
        dict[str, object]: Mapping of arbitration field names to the provided values:
            {
                "use_learned_arbitration": ...,
                "enable_deterministic_guards": ...,
                "enable_food_direction_bias": ...,
                "warm_start_scale": ...,
                "gate_adjustment_bounds": ...,
            }
    """
    return {
        "use_learned_arbitration": use_learned_arbitration,
        "enable_deterministic_guards": enable_deterministic_guards,
        "enable_food_direction_bias": enable_food_direction_bias,
        "warm_start_scale": warm_start_scale,
        "gate_adjustment_bounds": gate_adjustment_bounds,
    }


def default_brain_config(
    *,
    module_dropout: float = 0.05,
    capacity_profile: str | CapacityProfile | None = None,
) -> BrainAblationConfig:
    """
    Provide the canonical "modular_full" brain ablation configuration.
    
    Parameters:
        module_dropout (float): Module-level dropout probability to apply to the configuration.
        capacity_profile (str | CapacityProfile | None): Capacity profile to
            apply to the returned configuration, or None to use the default.
    
    Returns:
        BrainAblationConfig: A `modular_full` A4 configuration with the
        requested capacity profile, reflexes and auxiliary targets enabled,
        deterministic guards and food-direction bias disabled, route-mask credit
        strategy, coarse rollup modules disabled, and no recurrent modules.
    """
    return BrainAblationConfig(
        name="modular_full",
        architecture="modular",
        module_dropout=module_dropout,
        enable_reflexes=True,
        enable_auxiliary_targets=True,
        **_arbitration_fields(),
        credit_strategy="route_mask",
        disabled_modules=COARSE_ROLLUP_MODULES,
        recurrent_modules=(),
        reflex_scale=1.0,
        module_reflex_scales={},
        capacity_profile=capacity_profile,
    )

__all__ = [name for name in globals() if not name.startswith("__")]
