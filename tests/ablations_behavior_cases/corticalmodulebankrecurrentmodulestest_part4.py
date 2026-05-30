from __future__ import annotations

from .shared import *
from .corticalmodulebankrecurrentmodulestest_helpers import CorticalModuleBankRecurrentModulesTestHelpers



class CorticalModuleBankRecurrentModulesTestPart4(CorticalModuleBankRecurrentModulesTestHelpers, unittest.TestCase):
    def test_option_action_separate_recurrent_head_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=79,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_policy_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_action_policy_core.fill(0.0)
        recurrent.b2_action_policy_core.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_option_action_policy_decoder.fill(0.0)
        recurrent.W_option_action_policy_prev.fill(0.0)
        recurrent.W_option_action_policy_action.fill(0.0)
        recurrent.b_option_action_policy.fill(0.0)
        recurrent.b2_action_policy_core[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_action_policy_core[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_action_policy_core[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_option_action_policy_action[
            OPTION_NAMES.index("FORAGE"),
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_separate_policy_path_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=83,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_policy_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_action_policy_path.fill(0.0)
        recurrent.b2_action_policy_path.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_action_policy_path_input.fill(0.0)
        recurrent.W_action_policy_path_prev.fill(0.0)
        recurrent.W_action_policy_path_action.fill(0.0)
        recurrent.b_action_policy_path.fill(0.0)
        recurrent.b2_action_policy_path[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_action_policy_path[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_action_policy_path[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_action_policy_path_action[
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_separate_backbone_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=87,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_backbone_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_action_backbone.fill(0.0)
        recurrent.b2_action_backbone.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_action_backbone_input.fill(0.0)
        recurrent.W_action_backbone_prev.fill(0.0)
        recurrent.W_action_backbone_action.fill(0.0)
        recurrent.b_action_backbone.fill(0.0)
        recurrent.b2_action_backbone[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_action_backbone[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_action_backbone[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_action_backbone_action[
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_direct_policy_handoff_teacher_requires_position_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_handoff_teacher requires direct_policy_shelter_position_head=True",
        ):
            BrainAblationConfig(
                name="invalid_teacher_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_handoff_teacher=True,
            )

    def test_direct_policy_handoff_option_teacher_requires_handoff_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_handoff_option_teacher requires direct_policy_handoff_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_teacher_option_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_option_teacher=True,
            )

    def test_direct_policy_continuation_replay_requires_handoff_option_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_continuation_replay_passes requires direct_policy_handoff_option_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_teacher_replay_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            )

    def test_direct_policy_phase_head_with_option_head_requires_position_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_phase_head with direct_policy_option_head requires direct_policy_shelter_position_head=True",
        ):
            BrainAblationConfig(
                name="invalid_phase_option_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
            )

    def test_direct_policy_continuation_margin_requires_phase_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_continuation_margin_weight requires direct_policy_phase_head=True",
        ):
            BrainAblationConfig(
                name="invalid_continuation_margin_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_margin_weight=1.0,
            )

    def test_direct_policy_phase_option_feedback_requires_phase_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_phase_option_feedback requires direct_policy_phase_head=True",
        ):
            BrainAblationConfig(
                name="invalid_phase_option_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_phase_option_feedback=True,
            )

    def test_direct_policy_option_transition_feedback_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_transition_feedback requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_transition_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_transition_feedback=True,
            )

    def test_direct_policy_option_termination_cooldown_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_termination_cooldown requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_termination_cooldown_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_termination_cooldown=True,
            )

    def test_direct_policy_option_action_head_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_head requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_head=True,
            )

    def test_direct_policy_option_decoder_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_decoder_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_decoder_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_decoder_state=True,
            )

    def test_direct_policy_option_recurrent_dynamics_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_recurrent_dynamics requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_recurrent_dynamics_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_recurrent_dynamics=True,
            )

    def test_direct_policy_option_sequence_head_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_sequence_head requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_sequence_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_sequence_head=True,
            )

    def test_direct_policy_option_decoder_recurrent_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_decoder_recurrent_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_decoder_recurrent_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_decoder_recurrent_state=True,
            )

    def test_direct_policy_option_action_transition_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_transition_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_transition_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_transition_state=True,
            )

    def test_direct_policy_option_action_controller_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_controller_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_controller_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_controller_state=True,
            )

    def test_direct_policy_option_action_token_decoder_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_token_decoder requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_token_decoder_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_token_decoder=True,
            )

    def test_direct_policy_option_action_recurrent_core_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_recurrent_core requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_recurrent_core_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_recurrent_core=True,
            )

    def test_direct_policy_option_action_separate_recurrent_head_requires_action_recurrent_core(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_separate_recurrent_head requires direct_policy_option_action_recurrent_core=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_separate_recurrent_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_option_action_separate_recurrent_head=True,
            )

    def test_direct_policy_option_action_separate_policy_path_requires_action_recurrent_core(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_separate_policy_path requires direct_policy_option_action_recurrent_core=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_separate_policy_path_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_option_action_separate_policy_path=True,
            )

    def test_direct_policy_option_action_separate_backbone_requires_separate_policy_path(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_separate_backbone requires direct_policy_option_action_separate_policy_path=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_separate_backbone_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_backbone=True,
            )

    def test_direct_policy_post_rest_action_teacher_requires_separate_backbone(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_action_teacher requires direct_policy_option_action_separate_backbone=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_action_teacher_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_post_rest_action_teacher=True,
            )

    def test_direct_policy_post_rest_release_sequence_teacher_requires_post_rest_action_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_release_sequence_teacher requires direct_policy_post_rest_action_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_release_sequence_teacher_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_release_sequence_teacher=True,
            )

    def test_direct_policy_post_rest_release_sequence_replay_boost_requires_release_sequence_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_release_sequence_replay_boost requires direct_policy_post_rest_release_sequence_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_release_sequence_replay_boost_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_replay_boost=True,
            )

    def test_direct_policy_post_rest_release_sequence_distill_requires_release_sequence_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_release_sequence_distill requires direct_policy_post_rest_release_sequence_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_release_sequence_distill_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_distill=True,
            )

    def test_direct_policy_post_rest_probe_distillation_requires_release_sequence_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_distillation requires direct_policy_post_rest_release_sequence_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
            )

    def test_direct_policy_post_rest_probe_sequence_distillation_requires_probe_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_sequence_distillation requires direct_policy_post_rest_probe_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_sequence_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
            )

    def test_direct_policy_post_rest_probe_family_distillation_requires_probe_sequence_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_family_distillation requires direct_policy_post_rest_probe_sequence_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_family_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
            )

    def test_direct_policy_post_rest_probe_handoff_distillation_requires_probe_family_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_handoff_distillation requires direct_policy_post_rest_probe_family_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_handoff_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
            )

    def test_direct_policy_post_rest_probe_trajectory_distillation_requires_probe_handoff_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_trajectory_distillation requires direct_policy_post_rest_probe_handoff_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_trajectory_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
            )

    def test_direct_policy_post_rest_probe_cycle_distillation_requires_probe_trajectory_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_cycle_distillation requires direct_policy_post_rest_probe_trajectory_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_cycle_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
            )

    def test_direct_policy_post_rest_probe_trace_distillation_requires_probe_cycle_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_trace_distillation requires direct_policy_post_rest_probe_cycle_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_trace_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_trace_distillation=True,
            )

    def test_direct_policy_post_rest_probe_rollout_distillation_requires_probe_trace_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_rollout_distillation requires direct_policy_post_rest_probe_trace_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_rollout_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
                direct_policy_post_rest_probe_rollout_distillation=True,
            )

    def test_direct_policy_post_rest_probe_frontier_teacher_distillation_requires_probe_rollout_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_frontier_teacher_distillation requires direct_policy_post_rest_probe_rollout_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_frontier_teacher_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
                direct_policy_post_rest_probe_trace_distillation=True,
                direct_policy_post_rest_probe_frontier_teacher_distillation=True,
            )

    def test_direct_policy_post_rest_probe_replayable_teacher_distillation_requires_probe_rollout_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_replayable_teacher_distillation requires direct_policy_post_rest_probe_rollout_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_replayable_teacher_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
                direct_policy_post_rest_probe_trace_distillation=True,
                direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            )

    def test_direct_policy_local_affordance_inputs_require_position_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_affordance_inputs requires direct_policy_shelter_position_head=True",
        ):
            BrainAblationConfig(
                name="invalid_local_affordance_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_local_affordance_inputs=True,
            )

    def test_direct_policy_local_affordance_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_affordance_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
        )
        brain = SpiderBrain(seed=13, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM,
        )
        affordance_tail = vector[-DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM:]
        np.testing.assert_allclose(affordance_tail[:4], np.array([0.0, 0.0, 1.0, 0.0]))

    def test_direct_policy_local_spatial_inputs_require_affordance_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_spatial_inputs requires direct_policy_local_affordance_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_spatial_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_spatial_inputs=True,
            )

    def test_direct_policy_local_spatial_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_spatial_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
        )
        brain = SpiderBrain(seed=17, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM,
        )
        spatial_tail = vector[-DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM:]
        np.testing.assert_allclose(
            spatial_tail[:9],
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )

    def test_direct_policy_local_transition_inputs_require_spatial_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_transition_inputs requires direct_policy_local_spatial_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_transition_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_transition_inputs=True,
            )

    def test_direct_policy_local_transition_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_transition_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
        )
        brain = SpiderBrain(seed=19, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            "local_transition_consequences": {
                "STAY": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
                "MOVE_UP": {"food_dist_delta": 1.0, "shelter_dist_delta": 1.0, "predator_dist_delta": -1.0, "next_cell_has_food": False},
                "MOVE_DOWN": {"food_dist_delta": -1.0, "shelter_dist_delta": -1.0, "predator_dist_delta": 1.0, "next_cell_has_food": False},
                "MOVE_LEFT": {"food_dist_delta": 0.0, "shelter_dist_delta": 1.0, "predator_dist_delta": 0.0, "next_cell_has_food": True},
                "MOVE_RIGHT": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM,
        )
        transition_tail = vector[-DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM:]
        np.testing.assert_allclose(
            transition_tail[:8],
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0]),
        )
