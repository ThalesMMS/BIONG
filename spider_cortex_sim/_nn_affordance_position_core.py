from __future__ import annotations

from ._nn_affordance_geometry import *
from ._nn_affordance_position_gating import _RecurrentOptionAffordancePositionGatingMixin
from ._nn_affordance_position_forward import _RecurrentOptionAffordancePositionForwardMixin
from ._nn_affordance_position_backward import _RecurrentOptionAffordancePositionBackwardMixin


class RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
    _RecurrentOptionAffordancePositionGatingMixin,
    _RecurrentOptionAffordancePositionForwardMixin,
    _RecurrentOptionAffordancePositionBackwardMixin,
    RecurrentOptionAffordanceTopologyFeedbackTrueMonolithicNetwork,
):
    shelter_position_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        phase_output_dim: int = 0,
        phase_option_feedback: bool = False,
        option_transition_feedback: bool = False,
        option_termination_cooldown: bool = False,
        option_action_head: bool = False,
        option_decoder_state: bool = False,
        option_recurrent_dynamics: bool = False,
        option_sequence_head: bool = False,
        option_decoder_recurrent_state: bool = False,
        option_action_transition_state: bool = False,
        option_action_controller_state: bool = False,
        option_action_token_decoder: bool = False,
        option_action_recurrent_core: bool = False,
        option_action_separate_recurrent_head: bool = False,
        option_action_separate_policy_path: bool = False,
        option_action_separate_backbone: bool = False,
        executive_physiology_option_gating: bool = False,
        executive_affordance_action_gating: bool = False,
        executive_option_action_masking: bool = False,
        executive_event_release_latching: bool = False,
        executive_event_release_action_commitment: bool = False,
        executive_release_phase_state: bool = False,
        executive_release_progression: bool = False,
        executive_release_exit_contract: bool = False,
        executive_release_substate_progression: bool = False,
        executive_post_exit_continuation: bool = False,
        executive_post_exit_food_guidance: bool = False,
        executive_post_exit_food_commitment: bool = False,
        executive_post_exit_food_progression: bool = False,
        executive_post_exit_food_heading_progression: bool = False,
        executive_post_exit_smell_progression: bool = False,
        executive_post_exit_corridor_progression: bool = False,
        executive_post_exit_corridor_affordance_progression: bool = False,
        executive_post_food_return: bool = False,
        executive_post_food_vector_return: bool = False,
        executive_post_food_path_return: bool = False,
        transition_prediction_head: bool = False,
        transition_prediction_feedback: bool = False,
        transition_rollout_prediction_head: bool = False,
        transition_rollout_prediction_feedback: bool = False,
        name: str = "true_monolithic_policy",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            event_buffer_size=event_buffer_size,
            option_ttl=option_ttl,
            name=name,
        )
        self.phase_output_dim = int(phase_output_dim)
        self.phase_option_feedback = bool(
            phase_option_feedback and self.phase_output_dim > 0
        )
        self.option_transition_feedback = bool(option_transition_feedback)
        self.option_termination_cooldown = bool(option_termination_cooldown)
        self.option_action_head = bool(option_action_head)
        self.option_decoder_state = bool(option_decoder_state)
        self.option_recurrent_dynamics = bool(option_recurrent_dynamics)
        self.option_sequence_head = bool(option_sequence_head)
        self.option_decoder_recurrent_state = bool(option_decoder_recurrent_state)
        self.option_action_transition_state = bool(option_action_transition_state)
        self.option_action_controller_state = bool(option_action_controller_state)
        self.option_action_token_decoder = bool(option_action_token_decoder)
        self.option_action_recurrent_core = bool(option_action_recurrent_core)
        self.option_action_separate_recurrent_head = bool(
            option_action_separate_recurrent_head
        )
        self.option_action_separate_policy_path = bool(
            option_action_separate_policy_path
        )
        self.option_action_separate_backbone = bool(
            option_action_separate_backbone
        )
        self.executive_physiology_option_gating = bool(
            executive_physiology_option_gating and self.option_dim > 0
        )
        self.executive_affordance_action_gating = bool(
            executive_affordance_action_gating
            and self.option_dim > 0
            and self.output_dim > 0
        )
        self.executive_option_action_masking = bool(
            executive_option_action_masking and self.executive_affordance_action_gating
        )
        self.executive_event_release_latching = bool(
            executive_event_release_latching
            and self.executive_physiology_option_gating
            and self.executive_affordance_action_gating
        )
        self.executive_event_release_action_commitment = bool(
            executive_event_release_action_commitment
            and self.executive_event_release_latching
        )
        self.executive_release_phase_state = bool(
            executive_release_phase_state
            and self.executive_event_release_action_commitment
        )
        self.executive_release_progression = bool(
            executive_release_progression and self.executive_release_phase_state
        )
        self.executive_release_exit_contract = bool(
            executive_release_exit_contract and self.executive_release_phase_state
        )
        self.executive_release_substate_progression = bool(
            executive_release_substate_progression
            and self.executive_release_exit_contract
        )
        self.executive_post_exit_continuation = bool(
            executive_post_exit_continuation
            and self.executive_release_substate_progression
        )
        self.executive_post_exit_food_guidance = bool(
            executive_post_exit_food_guidance
            and self.executive_post_exit_continuation
        )
        self.executive_post_exit_food_commitment = bool(
            executive_post_exit_food_commitment
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_food_progression = bool(
            executive_post_exit_food_progression
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_food_heading_progression = bool(
            executive_post_exit_food_heading_progression
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_smell_progression = bool(
            executive_post_exit_smell_progression
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_corridor_progression = bool(
            executive_post_exit_corridor_progression
            and self.executive_post_exit_continuation
        )
        self.executive_post_exit_corridor_affordance_progression = bool(
            executive_post_exit_corridor_affordance_progression
            and self.executive_post_exit_corridor_progression
        )
        self.executive_post_food_return = bool(
            executive_post_food_return and self.executive_post_exit_continuation
        )
        self.executive_post_food_vector_return = bool(
            executive_post_food_vector_return and self.executive_post_food_return
        )
        self.executive_post_food_path_return = bool(
            executive_post_food_path_return and self.executive_post_food_return
        )
        self.transition_prediction_head = bool(transition_prediction_head)
        self.transition_prediction_feedback = bool(
            transition_prediction_feedback and self.transition_prediction_head
        )
        self.transition_rollout_prediction_head = bool(
            transition_rollout_prediction_head
        )
        self.transition_rollout_prediction_feedback = bool(
            transition_rollout_prediction_feedback
            and self.transition_rollout_prediction_head
        )
        self.option_cooldowns = np.zeros(self.option_dim, dtype=int)
        self.decoder_action_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_backbone_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_policy_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_controller_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_token_state = np.zeros(self.hidden_dim, dtype=float)
        self.previous_action_idx = -1
        self.executive_release_steps_remaining = 0
        self.executive_post_exit_steps_remaining = 0
        self.executive_post_exit_corridor_steps_remaining = 0
        self.executive_post_food_return_steps_remaining = 0
        self.executive_runtime_meta: dict[str, object] = {}
        self.executive_post_food_path_history: list[int] = []
        self.executive_post_food_return_queue: list[int] = []
        self.shelter_position_feature_dim = int(
            self.output_dim * self.shelter_position_dim
        )
        self.affordance_feature_dim = int(
            self.output_dim
            + self.output_dim * self.affordance_role_dim
            + self.geometry_feature_dim
            + self.shelter_position_feature_dim
        )
        self.transition_prediction_feature_dim = int(
            DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM
        )
        self.transition_rollout_prediction_feature_dim = int(
            DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM
        )
        self.W2_shelter_position = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.shelter_position_feature_dim, self.hidden_dim),
        )
        self.b2_shelter_position = np.zeros(
            self.shelter_position_feature_dim,
            dtype=float,
        )
        self.W2_transition_prediction = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.transition_prediction_feature_dim, self.hidden_dim),
            )
            if self.transition_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_transition_prediction = (
            np.zeros(self.transition_prediction_feature_dim, dtype=float)
            if self.transition_prediction_head
            else np.zeros(0, dtype=float)
        )
        self.W_transition_prediction_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.transition_prediction_feature_dim),
                size=(self.hidden_dim, self.transition_prediction_feature_dim),
            )
            if self.transition_prediction_feedback
            else np.zeros((0, self.transition_prediction_feature_dim), dtype=float)
        )
        self.b_transition_prediction_feedback = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.transition_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_transition_rollout_prediction = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(
                    self.transition_rollout_prediction_feature_dim,
                    self.hidden_dim,
                ),
            )
            if self.transition_rollout_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_transition_rollout_prediction = (
            np.zeros(
                self.transition_rollout_prediction_feature_dim,
                dtype=float,
            )
            if self.transition_rollout_prediction_head
            else np.zeros(0, dtype=float)
        )
        self.W_transition_rollout_prediction_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.transition_rollout_prediction_feature_dim),
                size=(self.hidden_dim, self.transition_rollout_prediction_feature_dim),
            )
            if self.transition_rollout_prediction_feedback
            else np.zeros(
                (0, self.transition_rollout_prediction_feature_dim),
                dtype=float,
            )
        )
        self.b_transition_rollout_prediction_feedback = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.transition_rollout_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_phase = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.phase_output_dim, self.hidden_dim),
            )
            if self.phase_output_dim > 0
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_phase = (
            np.zeros(self.phase_output_dim, dtype=float)
            if self.phase_output_dim > 0
            else np.zeros(0, dtype=float)
        )
        self.W2_phase_option_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.phase_output_dim),
                size=(self.option_dim, self.phase_output_dim),
            )
            if self.phase_option_feedback
            else np.zeros((self.option_dim, 0), dtype=float)
        )
        self.b2_phase_option_feedback = (
            np.zeros(self.option_dim, dtype=float)
            if self.phase_option_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_option_transition_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.option_dim),
                size=(self.option_dim, self.option_dim),
            )
            if self.option_transition_feedback
            else np.zeros((self.option_dim, 0), dtype=float)
        )
        self.b2_option_transition_feedback = (
            np.zeros(self.option_dim, dtype=float)
            if self.option_transition_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_option_action_head = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.output_dim, self.hidden_dim),
            )
            if self.option_action_head
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        self.b2_option_action_head = (
            np.zeros((self.option_dim, self.output_dim), dtype=float)
            if self.option_action_head
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.W_option_decoder_state = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_decoder_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.b_option_decoder_state = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_decoder_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_recurrent_dynamics = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.b_option_recurrent_dynamics = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W2_option_sequence_head = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(
                    self.option_dim,
                    self.option_ttl,
                    self.output_dim,
                    self.hidden_dim,
                ),
            )
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim, self.hidden_dim), dtype=float)
        )
        self.b2_option_sequence_head = (
            np.zeros((self.option_dim, self.option_ttl, self.output_dim), dtype=float)
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim), dtype=float)
        )
        self.W_option_decoder_recurrent_state = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.b_option_decoder_recurrent_state = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_action_transition_state = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_transition_state = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_action_controller_decoder = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_controller_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_controller_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_controller = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W2_option_action_controller_head = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.output_dim, self.hidden_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        self.b2_option_action_controller_head = (
            np.zeros((self.option_dim, self.output_dim), dtype=float)
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.W_option_action_token_decoder = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_token_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_token_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_token = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_action_policy_decoder = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_policy_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_policy_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_policy = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W2_action_policy_core = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.output_dim, self.hidden_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_action_policy_core = (
            np.zeros(self.output_dim, dtype=float)
            if self.option_action_recurrent_core
            else np.zeros(0, dtype=float)
        )
        self.W_action_policy_path_input = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_action_policy_path_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_action_policy_path_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.hidden_dim, self.output_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.b_action_policy_path = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.option_action_separate_policy_path
            else np.zeros(0, dtype=float)
        )
        self.W2_action_policy_path = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.output_dim, self.hidden_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_action_policy_path = (
            np.zeros(self.output_dim, dtype=float)
            if self.option_action_separate_policy_path
            else np.zeros(0, dtype=float)
        )
        self.W_action_backbone_input = (
            rng.normal(
                0.0,
                _weight_scale(self.input_dim),
                size=(self.hidden_dim, self.input_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.input_dim), dtype=float)
        )
        self.W_action_backbone_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_action_backbone_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.hidden_dim, self.output_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.b_action_backbone = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.option_action_separate_backbone
            else np.zeros(0, dtype=float)
        )
        self.W2_action_backbone = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.output_dim, self.hidden_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_action_backbone = (
            np.zeros(self.output_dim, dtype=float)
            if self.option_action_separate_backbone
            else np.zeros(0, dtype=float)
        )
        self.W_affordance_feedback = rng.normal(
            0.0,
            _weight_scale(self.affordance_feature_dim),
            size=(self.hidden_dim, self.affordance_feature_dim),
        )
        self.b_affordance_feedback = np.zeros(self.hidden_dim, dtype=float)
        self.last_affordance_summary["shelter_position_logits"] = []

    def reset_hidden_state(self) -> None:
        super().reset_hidden_state()
        self.option_cooldowns.fill(0)
        self.decoder_action_state.fill(0.0)
        self.action_backbone_state.fill(0.0)
        self.action_policy_state.fill(0.0)
        self.action_controller_state.fill(0.0)
        self.action_token_state.fill(0.0)
        self.previous_action_idx = -1
        self.executive_release_steps_remaining = 0
        self.executive_post_exit_steps_remaining = 0
        self.executive_post_exit_corridor_steps_remaining = 0
        self.executive_post_food_return_steps_remaining = 0
        self.executive_runtime_meta = {}
        self.executive_post_food_path_history = []
        self.executive_post_food_return_queue = []
        self.last_affordance_summary["shelter_position_logits"] = []

    def set_runtime_observation_meta(self, meta: dict[str, object] | None) -> None:
        self.executive_runtime_meta = dict(meta or {})

    def get_runtime_state(self) -> dict[str, object]:
        runtime_state = super().get_runtime_state()
        runtime_state["option_cooldowns"] = self.option_cooldowns.copy()
        runtime_state["decoder_action_state"] = self.decoder_action_state.copy()
        runtime_state["action_backbone_state"] = self.action_backbone_state.copy()
        runtime_state["action_policy_state"] = self.action_policy_state.copy()
        runtime_state["action_controller_state"] = self.action_controller_state.copy()
        runtime_state["action_token_state"] = self.action_token_state.copy()
        runtime_state["previous_action_idx"] = int(self.previous_action_idx)
        runtime_state["executive_release_steps_remaining"] = int(
            self.executive_release_steps_remaining
        )
        runtime_state["executive_post_exit_steps_remaining"] = int(
            self.executive_post_exit_steps_remaining
        )
        runtime_state["executive_post_exit_corridor_steps_remaining"] = int(
            self.executive_post_exit_corridor_steps_remaining
        )
        runtime_state["executive_post_food_return_steps_remaining"] = int(
            self.executive_post_food_return_steps_remaining
        )
        runtime_state["executive_post_food_path_history"] = list(
            self.executive_post_food_path_history
        )
        runtime_state["executive_post_food_return_queue"] = list(
            self.executive_post_food_return_queue
        )
        return runtime_state

    def set_runtime_state(self, runtime_state: dict[str, object]) -> None:
        super().set_runtime_state(runtime_state)
        cooldowns = runtime_state.get("option_cooldowns")
        decoder_action_state = runtime_state.get("decoder_action_state")
        action_backbone_state = runtime_state.get("action_backbone_state")
        action_policy_state = runtime_state.get("action_policy_state")
        action_controller_state = runtime_state.get("action_controller_state")
        action_token_state = runtime_state.get("action_token_state")
        previous_action_idx = runtime_state.get("previous_action_idx", -1)
        executive_release_steps_remaining = runtime_state.get(
            "executive_release_steps_remaining",
            0,
        )
        executive_post_exit_steps_remaining = runtime_state.get(
            "executive_post_exit_steps_remaining",
            0,
        )
        executive_post_exit_corridor_steps_remaining = runtime_state.get(
            "executive_post_exit_corridor_steps_remaining",
            0,
        )
        executive_post_food_return_steps_remaining = runtime_state.get(
            "executive_post_food_return_steps_remaining",
            0,
        )
        executive_post_food_path_history = runtime_state.get(
            "executive_post_food_path_history",
            [],
        )
        executive_post_food_return_queue = runtime_state.get(
            "executive_post_food_return_queue",
            [],
        )
        if cooldowns is None:
            self.option_cooldowns.fill(0)
        else:
            cooldown_array = np.asarray(cooldowns, dtype=int)
            if cooldown_array.shape != (self.option_dim,):
                raise ValueError(
                    f"{self.name}: option_cooldowns expected {(self.option_dim,)}, received {cooldown_array.shape}"
                )
            self.option_cooldowns = np.maximum(0, cooldown_array)
        self.executive_release_steps_remaining = max(
            0,
            int(executive_release_steps_remaining),
        )
        self.executive_post_exit_steps_remaining = max(
            0,
            int(executive_post_exit_steps_remaining),
        )
        self.executive_post_exit_corridor_steps_remaining = max(
            0,
            int(executive_post_exit_corridor_steps_remaining),
        )
        self.executive_post_food_return_steps_remaining = max(
            0,
            int(executive_post_food_return_steps_remaining),
        )
        self.executive_post_food_path_history = [
            int(action_idx)
            for action_idx in list(executive_post_food_path_history)
            if int(action_idx) in _LOCAL_ACTION_TO_POLICY_INDEX.values()
        ]
        self.executive_post_food_return_queue = [
            int(action_idx)
            for action_idx in list(executive_post_food_return_queue)
            if int(action_idx) in _LOCAL_ACTION_TO_POLICY_INDEX.values()
        ]
        if decoder_action_state is None:
            self.decoder_action_state.fill(0.0)
            return
        decoder_action_state_array = np.asarray(decoder_action_state, dtype=float)
        if decoder_action_state_array.shape != (self.hidden_dim,):
            raise ValueError(
                f"{self.name}: decoder_action_state expected {(self.hidden_dim,)}, received {decoder_action_state_array.shape}"
            )
        self.decoder_action_state = decoder_action_state_array.copy()
        if action_backbone_state is None:
            self.action_backbone_state.fill(0.0)
        else:
            action_backbone_state_array = np.asarray(
                action_backbone_state,
                dtype=float,
            )
            if action_backbone_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_backbone_state expected {(self.hidden_dim,)}, received {action_backbone_state_array.shape}"
                )
            self.action_backbone_state = action_backbone_state_array.copy()
        if action_policy_state is None:
            self.action_policy_state.fill(0.0)
        else:
            action_policy_state_array = np.asarray(action_policy_state, dtype=float)
            if action_policy_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_policy_state expected {(self.hidden_dim,)}, received {action_policy_state_array.shape}"
                )
            self.action_policy_state = action_policy_state_array.copy()
        if action_controller_state is None:
            self.action_controller_state.fill(0.0)
        else:
            action_controller_state_array = np.asarray(
                action_controller_state,
                dtype=float,
            )
            if action_controller_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_controller_state expected {(self.hidden_dim,)}, received {action_controller_state_array.shape}"
                )
            self.action_controller_state = action_controller_state_array.copy()
        if action_token_state is None:
            self.action_token_state.fill(0.0)
        else:
            action_token_state_array = np.asarray(
                action_token_state,
                dtype=float,
            )
            if action_token_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_token_state expected {(self.hidden_dim,)}, received {action_token_state_array.shape}"
                )
            self.action_token_state = action_token_state_array.copy()
        self.previous_action_idx = int(previous_action_idx)

    def record_executed_action(self, action_idx: int) -> None:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self.output_dim:
            raise ValueError(
                f"{self.name}: action_idx expected in [0, {self.output_dim}), received {action_idx}"
            )
        locomotion_indices = set(_LOCAL_ACTION_TO_POLICY_INDEX.values())
        if (
            self.executive_post_food_path_return
            and action_idx in locomotion_indices
            and self.executive_post_food_return_steps_remaining <= 0
            and self.executive_post_exit_steps_remaining > 0
            and len(self.executive_post_food_path_history) < 16
        ):
            self.executive_post_food_path_history.append(action_idx)
        if (
            self.executive_post_food_path_return
            and self.executive_post_food_return_queue
            and action_idx == int(self.executive_post_food_return_queue[0])
        ):
            self.executive_post_food_return_queue.pop(0)
        self.previous_action_idx = action_idx

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["phase_output_dim"] = self.phase_output_dim
        state["phase_option_feedback"] = self.phase_option_feedback
        state["option_transition_feedback"] = self.option_transition_feedback
        state["option_termination_cooldown"] = self.option_termination_cooldown
        state["option_action_head"] = self.option_action_head
        state["option_decoder_state"] = self.option_decoder_state
        state["option_recurrent_dynamics"] = self.option_recurrent_dynamics
        state["option_sequence_head"] = self.option_sequence_head
        state["option_decoder_recurrent_state"] = self.option_decoder_recurrent_state
        state["option_action_transition_state"] = self.option_action_transition_state
        state["option_action_controller_state"] = self.option_action_controller_state
        state["option_action_token_decoder"] = self.option_action_token_decoder
        state["option_action_recurrent_core"] = self.option_action_recurrent_core
        state["option_action_separate_recurrent_head"] = (
            self.option_action_separate_recurrent_head
        )
        state["option_action_separate_policy_path"] = (
            self.option_action_separate_policy_path
        )
        state["option_action_separate_backbone"] = (
            self.option_action_separate_backbone
        )
        state["transition_prediction_head"] = self.transition_prediction_head
        state["transition_prediction_feedback"] = (
            self.transition_prediction_feedback
        )
        state["transition_rollout_prediction_head"] = (
            self.transition_rollout_prediction_head
        )
        state["transition_rollout_prediction_feedback"] = (
            self.transition_rollout_prediction_feedback
        )
        state["shelter_position_head"] = True
        state["shelter_position_dim"] = self.shelter_position_dim
        state["shelter_position_feature_dim"] = self.shelter_position_feature_dim
        state["transition_prediction_feature_dim"] = (
            self.transition_prediction_feature_dim
        )
        state["transition_rollout_prediction_feature_dim"] = (
            self.transition_rollout_prediction_feature_dim
        )
        state["W2_shelter_position"] = self.W2_shelter_position.copy()
        state["b2_shelter_position"] = self.b2_shelter_position.copy()
        if self.transition_prediction_head:
            state["W2_transition_prediction"] = (
                self.W2_transition_prediction.copy()
            )
            state["b2_transition_prediction"] = (
                self.b2_transition_prediction.copy()
            )
        if self.transition_prediction_feedback:
            state["W_transition_prediction_feedback"] = (
                self.W_transition_prediction_feedback.copy()
            )
            state["b_transition_prediction_feedback"] = (
                self.b_transition_prediction_feedback.copy()
            )
        if self.transition_rollout_prediction_head:
            state["W2_transition_rollout_prediction"] = (
                self.W2_transition_rollout_prediction.copy()
            )
            state["b2_transition_rollout_prediction"] = (
                self.b2_transition_rollout_prediction.copy()
            )
        if self.transition_rollout_prediction_feedback:
            state["W_transition_rollout_prediction_feedback"] = (
                self.W_transition_rollout_prediction_feedback.copy()
            )
            state["b_transition_rollout_prediction_feedback"] = (
                self.b_transition_rollout_prediction_feedback.copy()
            )
        if self.phase_output_dim > 0:
            state["W2_phase"] = self.W2_phase.copy()
            state["b2_phase"] = self.b2_phase.copy()
        if self.phase_option_feedback:
            state["W2_phase_option_feedback"] = self.W2_phase_option_feedback.copy()
            state["b2_phase_option_feedback"] = self.b2_phase_option_feedback.copy()
        if self.option_transition_feedback:
            state["W2_option_transition_feedback"] = (
                self.W2_option_transition_feedback.copy()
            )
            state["b2_option_transition_feedback"] = (
                self.b2_option_transition_feedback.copy()
            )
        if self.option_action_head:
            state["W2_option_action_head"] = self.W2_option_action_head.copy()
            state["b2_option_action_head"] = self.b2_option_action_head.copy()
        if self.option_decoder_state:
            state["W_option_decoder_state"] = self.W_option_decoder_state.copy()
            state["b_option_decoder_state"] = self.b_option_decoder_state.copy()
        if self.option_recurrent_dynamics:
            state["W_option_recurrent_dynamics"] = (
                self.W_option_recurrent_dynamics.copy()
            )
            state["b_option_recurrent_dynamics"] = (
                self.b_option_recurrent_dynamics.copy()
            )
        if self.option_sequence_head:
            state["W2_option_sequence_head"] = self.W2_option_sequence_head.copy()
            state["b2_option_sequence_head"] = self.b2_option_sequence_head.copy()
        if self.option_decoder_recurrent_state:
            state["W_option_decoder_recurrent_state"] = (
                self.W_option_decoder_recurrent_state.copy()
            )
            state["b_option_decoder_recurrent_state"] = (
                self.b_option_decoder_recurrent_state.copy()
            )
        if self.option_action_transition_state:
            state["W_option_action_transition_state"] = (
                self.W_option_action_transition_state.copy()
            )
            state["b_option_action_transition_state"] = (
                self.b_option_action_transition_state.copy()
            )
        if self.option_action_controller_state:
            state["W_option_action_controller_decoder"] = (
                self.W_option_action_controller_decoder.copy()
            )
            state["W_option_action_controller_prev"] = (
                self.W_option_action_controller_prev.copy()
            )
            state["W_option_action_controller_action"] = (
                self.W_option_action_controller_action.copy()
            )
            state["b_option_action_controller"] = (
                self.b_option_action_controller.copy()
            )
            state["W2_option_action_controller_head"] = (
                self.W2_option_action_controller_head.copy()
            )
            state["b2_option_action_controller_head"] = (
                self.b2_option_action_controller_head.copy()
            )
        if self.option_action_token_decoder:
            state["W_option_action_token_decoder"] = (
                self.W_option_action_token_decoder.copy()
            )
            state["W_option_action_token_prev"] = (
                self.W_option_action_token_prev.copy()
            )
            state["W_option_action_token_action"] = (
                self.W_option_action_token_action.copy()
            )
            state["b_option_action_token"] = self.b_option_action_token.copy()
        if self.option_action_recurrent_core:
            state["W_option_action_policy_decoder"] = (
                self.W_option_action_policy_decoder.copy()
            )
            state["W_option_action_policy_prev"] = (
                self.W_option_action_policy_prev.copy()
            )
            state["W_option_action_policy_action"] = (
                self.W_option_action_policy_action.copy()
            )
            state["b_option_action_policy"] = self.b_option_action_policy.copy()
            state["W2_action_policy_core"] = self.W2_action_policy_core.copy()
            state["b2_action_policy_core"] = self.b2_action_policy_core.copy()
        if self.option_action_separate_policy_path:
            state["W_action_policy_path_input"] = (
                self.W_action_policy_path_input.copy()
            )
            state["W_action_policy_path_prev"] = (
                self.W_action_policy_path_prev.copy()
            )
            state["W_action_policy_path_action"] = (
                self.W_action_policy_path_action.copy()
            )
            state["b_action_policy_path"] = self.b_action_policy_path.copy()
            state["W2_action_policy_path"] = self.W2_action_policy_path.copy()
            state["b2_action_policy_path"] = self.b2_action_policy_path.copy()
        if self.option_action_separate_backbone:
            state["W_action_backbone_input"] = self.W_action_backbone_input.copy()
            state["W_action_backbone_prev"] = self.W_action_backbone_prev.copy()
            state["W_action_backbone_action"] = self.W_action_backbone_action.copy()
            state["b_action_backbone"] = self.b_action_backbone.copy()
            state["W2_action_backbone"] = self.W2_action_backbone.copy()
            state["b2_action_backbone"] = self.b2_action_backbone.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys=(
                set(super().state_dict().keys())
                | {
                    "phase_output_dim",
                    "phase_option_feedback",
                    "option_transition_feedback",
                    "option_termination_cooldown",
                    "option_action_head",
                    "option_decoder_state",
                    "option_recurrent_dynamics",
                    "option_sequence_head",
                    "option_decoder_recurrent_state",
                    "option_action_transition_state",
                    "option_action_controller_state",
                    "option_action_token_decoder",
                    "option_action_recurrent_core",
                    "option_action_separate_recurrent_head",
                    "option_action_separate_policy_path",
                    "option_action_separate_backbone",
                    "transition_prediction_head",
                    "transition_prediction_feedback",
                    "transition_rollout_prediction_head",
                    "transition_rollout_prediction_feedback",
                    "shelter_position_head",
                    "shelter_position_dim",
                    "shelter_position_feature_dim",
                    "transition_prediction_feature_dim",
                    "transition_rollout_prediction_feature_dim",
                    "W2_shelter_position",
                    "b2_shelter_position",
                }
                | (
                    {
                        "W2_transition_prediction",
                        "b2_transition_prediction",
                    }
                    if self.transition_prediction_head
                    else set()
                )
                | (
                    {
                        "W_transition_prediction_feedback",
                        "b_transition_prediction_feedback",
                    }
                    if self.transition_prediction_feedback
                    else set()
                )
                | (
                    {
                        "W2_transition_rollout_prediction",
                        "b2_transition_rollout_prediction",
                    }
                    if self.transition_rollout_prediction_head
                    else set()
                )
                | (
                    {
                        "W_transition_rollout_prediction_feedback",
                        "b_transition_rollout_prediction_feedback",
                    }
                    if self.transition_rollout_prediction_feedback
                    else set()
                )
                | (
                    {"W2_phase", "b2_phase"}
                    if self.phase_output_dim > 0
                    else set()
                )
                | (
                    {"W2_phase_option_feedback", "b2_phase_option_feedback"}
                    if self.phase_option_feedback
                    else set()
                )
                | (
                    {
                        "W2_option_transition_feedback",
                        "b2_option_transition_feedback",
                    }
                    if self.option_transition_feedback
                    else set()
                )
                | (
                    {
                        "W2_option_action_head",
                        "b2_option_action_head",
                    }
                    if self.option_action_head
                    else set()
                )
                | (
                    {
                        "W_option_decoder_state",
                        "b_option_decoder_state",
                    }
                    if self.option_decoder_state
                    else set()
                )
                | (
                    {
                        "W_option_recurrent_dynamics",
                        "b_option_recurrent_dynamics",
                    }
                    if self.option_recurrent_dynamics
                    else set()
                )
                | (
                    {
                        "W2_option_sequence_head",
                        "b2_option_sequence_head",
                    }
                    if self.option_sequence_head
                    else set()
                )
                | (
                    {
                        "W_option_decoder_recurrent_state",
                        "b_option_decoder_recurrent_state",
                    }
                    if self.option_decoder_recurrent_state
                    else set()
                )
                | (
                    {
                        "W_option_action_transition_state",
                        "b_option_action_transition_state",
                    }
                    if self.option_action_transition_state
                    else set()
                )
                | (
                    {
                        "W_option_action_controller_decoder",
                        "W_option_action_controller_prev",
                        "W_option_action_controller_action",
                        "b_option_action_controller",
                        "W2_option_action_controller_head",
                        "b2_option_action_controller_head",
                    }
                    if self.option_action_controller_state
                    else set()
                )
                | (
                    {
                        "W_option_action_token_decoder",
                        "W_option_action_token_prev",
                        "W_option_action_token_action",
                        "b_option_action_token",
                    }
                    if self.option_action_token_decoder
                    else set()
                )
                | (
                    {
                        "W_option_action_policy_decoder",
                        "W_option_action_policy_prev",
                        "W_option_action_policy_action",
                        "b_option_action_policy",
                        "W2_action_policy_core",
                        "b2_action_policy_core",
                    }
                    if self.option_action_recurrent_core
                    else set()
                )
                | (
                    {
                        "W_action_policy_path_input",
                        "W_action_policy_path_prev",
                        "W_action_policy_path_action",
                        "b_action_policy_path",
                        "W2_action_policy_path",
                        "b2_action_policy_path",
                    }
                    if self.option_action_separate_policy_path
                    else set()
                )
                | (
                    {
                        "W_action_backbone_input",
                        "W_action_backbone_prev",
                        "W_action_backbone_action",
                        "b_action_backbone",
                        "W2_action_backbone",
                        "b2_action_backbone",
                    }
                    if self.option_action_separate_backbone
                    else set()
                )
            ),
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "phase_output_dim": self.phase_output_dim,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
                "option_head": True,
                "option_ttl": self.option_ttl,
                "option_dim": self.option_dim,
                "phase_option_feedback": self.phase_option_feedback,
                "option_transition_feedback": self.option_transition_feedback,
                "option_termination_cooldown": self.option_termination_cooldown,
                "option_action_head": self.option_action_head,
                "option_decoder_state": self.option_decoder_state,
                "option_recurrent_dynamics": self.option_recurrent_dynamics,
                "option_sequence_head": self.option_sequence_head,
                "option_decoder_recurrent_state": (
                    self.option_decoder_recurrent_state
                ),
                "option_action_transition_state": (
                    self.option_action_transition_state
                ),
                "option_action_controller_state": (
                    self.option_action_controller_state
                ),
                "option_action_token_decoder": (
                    self.option_action_token_decoder
                ),
                "option_action_recurrent_core": (
                    self.option_action_recurrent_core
                ),
                "option_action_separate_recurrent_head": (
                    self.option_action_separate_recurrent_head
                ),
                "option_action_separate_policy_path": (
                    self.option_action_separate_policy_path
                ),
                "option_action_separate_backbone": (
                    self.option_action_separate_backbone
                ),
                "transition_prediction_head": self.transition_prediction_head,
                "transition_prediction_feedback": (
                    self.transition_prediction_feedback
                ),
                "transition_rollout_prediction_head": (
                    self.transition_rollout_prediction_head
                ),
                "transition_rollout_prediction_feedback": (
                    self.transition_rollout_prediction_feedback
                ),
                "affordance_head": True,
                "affordance_role_dim": self.affordance_role_dim,
                "affordance_feedback": True,
                "geometry_head": True,
                "geometry_dim": self.geometry_dim,
                "geometry_feature_dim": self.geometry_feature_dim,
                "shelter_position_head": True,
                "shelter_position_dim": self.shelter_position_dim,
                "shelter_position_feature_dim": self.shelter_position_feature_dim,
                "transition_prediction_feature_dim": (
                    self.transition_prediction_feature_dim
                ),
                "transition_rollout_prediction_feature_dim": (
                    self.transition_rollout_prediction_feature_dim
                ),
            },
            name=self.name,
        )
        super().load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "phase_output_dim",
                    "phase_option_feedback",
                    "option_transition_feedback",
                    "option_termination_cooldown",
                    "option_action_head",
                    "option_decoder_state",
                    "option_recurrent_dynamics",
                    "option_sequence_head",
                    "option_decoder_recurrent_state",
                    "option_action_transition_state",
                    "option_action_controller_state",
                    "option_action_token_decoder",
                    "option_action_recurrent_core",
                    "option_action_separate_recurrent_head",
                    "option_action_separate_policy_path",
                    "option_action_separate_backbone",
                    "transition_prediction_head",
                    "transition_prediction_feedback",
                    "transition_rollout_prediction_head",
                    "transition_rollout_prediction_feedback",
                    "shelter_position_head",
                    "shelter_position_dim",
                    "shelter_position_feature_dim",
                    "transition_prediction_feature_dim",
                    "transition_rollout_prediction_feature_dim",
                    "W2_shelter_position",
                    "b2_shelter_position",
                    "W2_transition_prediction",
                    "b2_transition_prediction",
                    "W_transition_prediction_feedback",
                    "b_transition_prediction_feedback",
                    "W2_transition_rollout_prediction",
                    "b2_transition_rollout_prediction",
                    "W_transition_rollout_prediction_feedback",
                    "b_transition_rollout_prediction_feedback",
                    "W2_phase",
                    "b2_phase",
                    "W2_phase_option_feedback",
                    "b2_phase_option_feedback",
                    "W2_option_transition_feedback",
                    "b2_option_transition_feedback",
                    "W2_option_action_head",
                    "b2_option_action_head",
                    "W_option_decoder_state",
                    "b_option_decoder_state",
                    "W_option_recurrent_dynamics",
                    "b_option_recurrent_dynamics",
                    "W2_option_sequence_head",
                    "b2_option_sequence_head",
                    "W_option_decoder_recurrent_state",
                    "b_option_decoder_recurrent_state",
                    "W_option_action_transition_state",
                    "b_option_action_transition_state",
                    "W_option_action_controller_decoder",
                    "W_option_action_controller_prev",
                    "W_option_action_controller_action",
                    "b_option_action_controller",
                    "W2_option_action_controller_head",
                    "b2_option_action_controller_head",
                    "W_option_action_token_decoder",
                    "W_option_action_token_prev",
                    "W_option_action_token_action",
                    "b_option_action_token",
                    "W_option_action_policy_decoder",
                    "W_option_action_policy_prev",
                    "W_option_action_policy_action",
                    "b_option_action_policy",
                    "W2_action_policy_core",
                    "b2_action_policy_core",
                    "W_action_policy_path_input",
                    "W_action_policy_path_prev",
                    "W_action_policy_path_action",
                    "b_action_policy_path",
                    "W2_action_policy_path",
                    "b2_action_policy_path",
                    "W_action_backbone_input",
                    "W_action_backbone_prev",
                    "W_action_backbone_action",
                    "b_action_backbone",
                    "W2_action_backbone",
                    "b2_action_backbone",
                }
            }
        )
        self.W2_shelter_position = _coerce_state_array(
            state,
            "W2_shelter_position",
            (self.shelter_position_feature_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_shelter_position = _coerce_state_array(
            state,
            "b2_shelter_position",
            (self.shelter_position_feature_dim,),
            name=self.name,
        )
        if self.transition_prediction_head:
            self.W2_transition_prediction = _coerce_state_array(
                state,
                "W2_transition_prediction",
                (self.transition_prediction_feature_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_transition_prediction = _coerce_state_array(
                state,
                "b2_transition_prediction",
                (self.transition_prediction_feature_dim,),
                name=self.name,
            )
        if self.transition_prediction_feedback:
            self.W_transition_prediction_feedback = _coerce_state_array(
                state,
                "W_transition_prediction_feedback",
                (self.hidden_dim, self.transition_prediction_feature_dim),
                name=self.name,
            )
            self.b_transition_prediction_feedback = _coerce_state_array(
                state,
                "b_transition_prediction_feedback",
                (self.hidden_dim,),
                name=self.name,
            )
        if self.transition_rollout_prediction_head:
            self.W2_transition_rollout_prediction = _coerce_state_array(
                state,
                "W2_transition_rollout_prediction",
                (
                    self.transition_rollout_prediction_feature_dim,
                    self.hidden_dim,
                ),
                name=self.name,
            )
            self.b2_transition_rollout_prediction = _coerce_state_array(
                state,
                "b2_transition_rollout_prediction",
                (self.transition_rollout_prediction_feature_dim,),
                name=self.name,
            )
        if self.transition_rollout_prediction_feedback:
            self.W_transition_rollout_prediction_feedback = _coerce_state_array(
                state,
                "W_transition_rollout_prediction_feedback",
                (
                    self.hidden_dim,
                    self.transition_rollout_prediction_feature_dim,
                ),
                name=self.name,
            )
            self.b_transition_rollout_prediction_feedback = _coerce_state_array(
                state,
                "b_transition_rollout_prediction_feedback",
                (self.hidden_dim,),
                name=self.name,
            )
        if self.phase_output_dim > 0:
            self.W2_phase = _coerce_state_array(
                state,
                "W2_phase",
                (self.phase_output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_phase = _coerce_state_array(
                state,
                "b2_phase",
                (self.phase_output_dim,),
                name=self.name,
            )
        if self.phase_option_feedback:
            self.W2_phase_option_feedback = _coerce_state_array(
                state,
                "W2_phase_option_feedback",
                (self.option_dim, self.phase_output_dim),
                name=self.name,
            )
            self.b2_phase_option_feedback = _coerce_state_array(
                state,
                "b2_phase_option_feedback",
                (self.option_dim,),
                name=self.name,
            )
        if self.option_transition_feedback:
            self.W2_option_transition_feedback = _coerce_state_array(
                state,
                "W2_option_transition_feedback",
                (self.option_dim, self.option_dim),
                name=self.name,
            )
            self.b2_option_transition_feedback = _coerce_state_array(
                state,
                "b2_option_transition_feedback",
                (self.option_dim,),
                name=self.name,
            )
        if self.option_action_head:
            self.W2_option_action_head = _coerce_state_array(
                state,
                "W2_option_action_head",
                (self.option_dim, self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_option_action_head = _coerce_state_array(
                state,
                "b2_option_action_head",
                (self.option_dim, self.output_dim),
                name=self.name,
            )
        if self.option_decoder_state:
            self.W_option_decoder_state = _coerce_state_array(
                state,
                "W_option_decoder_state",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.b_option_decoder_state = _coerce_state_array(
                state,
                "b_option_decoder_state",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_recurrent_dynamics:
            self.W_option_recurrent_dynamics = _coerce_state_array(
                state,
                "W_option_recurrent_dynamics",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.b_option_recurrent_dynamics = _coerce_state_array(
                state,
                "b_option_recurrent_dynamics",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_sequence_head:
            self.W2_option_sequence_head = _coerce_state_array(
                state,
                "W2_option_sequence_head",
                (self.option_dim, self.option_ttl, self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_option_sequence_head = _coerce_state_array(
                state,
                "b2_option_sequence_head",
                (self.option_dim, self.option_ttl, self.output_dim),
                name=self.name,
            )
        if self.option_decoder_recurrent_state:
            self.W_option_decoder_recurrent_state = _coerce_state_array(
                state,
                "W_option_decoder_recurrent_state",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.b_option_decoder_recurrent_state = _coerce_state_array(
                state,
                "b_option_decoder_recurrent_state",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_action_transition_state:
            self.W_option_action_transition_state = _coerce_state_array(
                state,
                "W_option_action_transition_state",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_transition_state = _coerce_state_array(
                state,
                "b_option_action_transition_state",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_action_controller_state:
            self.W_option_action_controller_decoder = _coerce_state_array(
                state,
                "W_option_action_controller_decoder",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_controller_prev = _coerce_state_array(
                state,
                "W_option_action_controller_prev",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_controller_action = _coerce_state_array(
                state,
                "W_option_action_controller_action",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_controller = _coerce_state_array(
                state,
                "b_option_action_controller",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
            self.W2_option_action_controller_head = _coerce_state_array(
                state,
                "W2_option_action_controller_head",
                (self.option_dim, self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_option_action_controller_head = _coerce_state_array(
                state,
                "b2_option_action_controller_head",
                (self.option_dim, self.output_dim),
                name=self.name,
            )
        if self.option_action_token_decoder:
            self.W_option_action_token_decoder = _coerce_state_array(
                state,
                "W_option_action_token_decoder",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_token_prev = _coerce_state_array(
                state,
                "W_option_action_token_prev",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_token_action = _coerce_state_array(
                state,
                "W_option_action_token_action",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_token = _coerce_state_array(
                state,
                "b_option_action_token",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_action_recurrent_core:
            self.W_option_action_policy_decoder = _coerce_state_array(
                state,
                "W_option_action_policy_decoder",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_policy_prev = _coerce_state_array(
                state,
                "W_option_action_policy_prev",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_policy_action = _coerce_state_array(
                state,
                "W_option_action_policy_action",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_policy = _coerce_state_array(
                state,
                "b_option_action_policy",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
            self.W2_action_policy_core = _coerce_state_array(
                state,
                "W2_action_policy_core",
                (self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_action_policy_core = _coerce_state_array(
                state,
                "b2_action_policy_core",
                (self.output_dim,),
                name=self.name,
            )
        if self.option_action_separate_policy_path:
            self.W_action_policy_path_input = _coerce_state_array(
                state,
                "W_action_policy_path_input",
                (self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_action_policy_path_prev = _coerce_state_array(
                state,
                "W_action_policy_path_prev",
                (self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_action_policy_path_action = _coerce_state_array(
                state,
                "W_action_policy_path_action",
                (self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_action_policy_path = _coerce_state_array(
                state,
                "b_action_policy_path",
                (self.hidden_dim,),
                name=self.name,
            )
            self.W2_action_policy_path = _coerce_state_array(
                state,
                "W2_action_policy_path",
                (self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_action_policy_path = _coerce_state_array(
                state,
                "b2_action_policy_path",
                (self.output_dim,),
                name=self.name,
            )
        if self.option_action_separate_backbone:
            self.W_action_backbone_input = _coerce_state_array(
                state,
                "W_action_backbone_input",
                (self.hidden_dim, self.input_dim),
                name=self.name,
            )
            self.W_action_backbone_prev = _coerce_state_array(
                state,
                "W_action_backbone_prev",
                (self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_action_backbone_action = _coerce_state_array(
                state,
                "W_action_backbone_action",
                (self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_action_backbone = _coerce_state_array(
                state,
                "b_action_backbone",
                (self.hidden_dim,),
                name=self.name,
            )
            self.W2_action_backbone = _coerce_state_array(
                state,
                "W2_action_backbone",
                (self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_action_backbone = _coerce_state_array(
                state,
                "b2_action_backbone",
                (self.output_dim,),
                name=self.name,
            )
        self.last_affordance_summary["shelter_position_logits"] = []

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_option,
            self.b2_option,
            self.W2_phase,
            self.b2_phase,
            self.option_action_bias,
            self.W2_affordance_blocked,
            self.b2_affordance_blocked,
            self.W2_affordance_role,
            self.b2_affordance_role,
            self.W2_geometry,
            self.b2_geometry,
            self.W2_shelter_position,
            self.b2_shelter_position,
            self.W2_transition_prediction,
            self.b2_transition_prediction,
            self.W_transition_prediction_feedback,
            self.b_transition_prediction_feedback,
            self.W_affordance_feedback,
            self.b_affordance_feedback,
            self.W2_policy_feedback,
            self.b2_policy_feedback,
            self.W2_option_feedback,
            self.b2_option_feedback,
            self.W2_phase_option_feedback,
            self.b2_phase_option_feedback,
            self.W2_option_transition_feedback,
            self.b2_option_transition_feedback,
            self.W2_option_action_head,
            self.b2_option_action_head,
            self.W_option_decoder_state,
            self.b_option_decoder_state,
            self.W_option_recurrent_dynamics,
            self.b_option_recurrent_dynamics,
            self.W2_option_sequence_head,
            self.b2_option_sequence_head,
            self.W_option_decoder_recurrent_state,
            self.b_option_decoder_recurrent_state,
            self.W_option_action_transition_state,
            self.b_option_action_transition_state,
            self.W_option_action_controller_decoder,
            self.W_option_action_controller_prev,
            self.W_option_action_controller_action,
            self.b_option_action_controller,
            self.W2_option_action_controller_head,
            self.b2_option_action_controller_head,
            self.W_option_action_token_decoder,
            self.W_option_action_token_prev,
            self.W_option_action_token_action,
            self.b_option_action_token,
            self.W_option_action_policy_decoder,
            self.W_option_action_policy_prev,
            self.W_option_action_policy_action,
            self.b_option_action_policy,
            self.W2_action_policy_core,
            self.b2_action_policy_core,
            self.W_action_backbone_input,
            self.W_action_backbone_prev,
            self.W_action_backbone_action,
            self.b_action_backbone,
            self.W2_action_backbone,
            self.b2_action_backbone,
            self.W_action_policy_path_input,
            self.W_action_policy_path_prev,
            self.W_action_policy_path_action,
            self.b_action_policy_path,
            self.W2_action_policy_path,
            self.b2_action_policy_path,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            super().count_parameters()
            + self.W2_shelter_position.size
            + self.b2_shelter_position.size
            + self.W2_transition_prediction.size
            + self.b2_transition_prediction.size
            + self.W_transition_prediction_feedback.size
            + self.b_transition_prediction_feedback.size
            + self.W2_phase.size
            + self.b2_phase.size
            + self.W2_phase_option_feedback.size
            + self.b2_phase_option_feedback.size
            + self.W2_option_transition_feedback.size
            + self.b2_option_transition_feedback.size
            + self.W2_option_action_head.size
            + self.b2_option_action_head.size
            + self.W_option_decoder_state.size
            + self.b_option_decoder_state.size
            + self.W_option_recurrent_dynamics.size
            + self.b_option_recurrent_dynamics.size
            + self.W2_option_sequence_head.size
            + self.b2_option_sequence_head.size
            + self.W_option_decoder_recurrent_state.size
            + self.b_option_decoder_recurrent_state.size
            + self.W_option_action_transition_state.size
            + self.b_option_action_transition_state.size
            + self.W_option_action_controller_decoder.size
            + self.W_option_action_controller_prev.size
            + self.W_option_action_controller_action.size
            + self.b_option_action_controller.size
            + self.W2_option_action_controller_head.size
            + self.b2_option_action_controller_head.size
            + self.W_option_action_token_decoder.size
            + self.W_option_action_token_prev.size
            + self.W_option_action_token_action.size
            + self.b_option_action_token.size
            + self.W_option_action_policy_decoder.size
            + self.W_option_action_policy_prev.size
            + self.W_option_action_policy_action.size
            + self.b_option_action_policy.size
            + self.W2_action_policy_core.size
            + self.b2_action_policy_core.size
            + self.W_action_backbone_input.size
            + self.W_action_backbone_prev.size
            + self.W_action_backbone_action.size
            + self.b_action_backbone.size
            + self.W2_action_backbone.size
            + self.b2_action_backbone.size
            + self.W_action_policy_path_input.size
            + self.W_action_policy_path_prev.size
            + self.W_action_policy_path_action.size
            + self.b_action_policy_path.size
            + self.W2_action_policy_path.size
            + self.b2_action_policy_path.size
        )


__all__ = [name for name in globals() if not name.startswith("__")]
