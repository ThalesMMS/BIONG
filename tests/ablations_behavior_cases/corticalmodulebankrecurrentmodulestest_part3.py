from __future__ import annotations

from .shared import *
from .corticalmodulebankrecurrentmodulestest_helpers import CorticalModuleBankRecurrentModulesTestHelpers



class CorticalModuleBankRecurrentModulesTestPart3(CorticalModuleBankRecurrentModulesTestHelpers, unittest.TestCase):
    def test_true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=54,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy",
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
                direct_policy_option_action_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=56,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy",
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
                direct_policy_option_sequence_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=60,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy",
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
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=62,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy",
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
                direct_policy_option_action_transition_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=66,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy",
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
                direct_policy_option_action_controller_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=70,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy",
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
                direct_policy_option_action_token_decoder=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=74,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy",
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
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=78,
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
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=82,
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
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=86,
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
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_phase_option_feedback_affects_selected_option(self) -> None:
        brain = SpiderBrain(
            seed=41,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy",
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
                direct_policy_phase_option_feedback=True,
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
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W2_phase.fill(0.0)
        recurrent.b2_phase.fill(0.0)
        recurrent.W2_phase_option_feedback.fill(0.0)
        recurrent.b2_phase_option_feedback.fill(0.0)
        recurrent.b2_phase[int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])] = 8.0
        recurrent.W2_phase_option_feedback[
            OPTION_NAMES.index("FORAGE"),
            int(PHASE_TO_INDEX["POST_REST_REACTIVATE"]),
        ] = 6.0
        recurrent.W2_phase_option_feedback[
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
            int(PHASE_TO_INDEX["POST_REST_REACTIVATE"]),
        ] = -6.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.phase_prediction, "POST_REST_REACTIVATE")
        self.assertEqual(decision.selected_option, "FORAGE")

    def test_option_transition_feedback_affects_selected_option(self) -> None:
        brain = SpiderBrain(
            seed=43,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy",
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
                direct_policy_option_transition_feedback=True,
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
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 2
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W2_option_transition_feedback.fill(0.0)
        recurrent.b2_option_transition_feedback.fill(0.0)
        recurrent.W2_option_transition_feedback[
            OPTION_NAMES.index("FORAGE"),
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
        ] = 6.0
        recurrent.W2_option_transition_feedback[
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
        ] = -6.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")

    def test_option_transition_branch_holds_option_until_ttl_expires(self) -> None:
        brain = SpiderBrain(
            seed=44,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy",
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
                direct_policy_option_transition_feedback=True,
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
        decisions = [
            brain.act_inference(_build_observation(), sample=False)
            for _ in range(5)
        ]
        selected_option = decisions[0].selected_option
        self.assertEqual(decisions[0].option_age, 0)
        self.assertEqual(decisions[0].option_termination_reason, "initial_selection")
        for expected_age, decision in enumerate(decisions[1:4], start=1):
            self.assertEqual(decision.selected_option, selected_option)
            self.assertEqual(decision.option_age, expected_age)
            self.assertEqual(decision.option_termination_reason, "active")
        self.assertEqual(decisions[4].option_age, 0)
        self.assertEqual(decisions[4].option_termination_reason, "ttl_expired")

    def test_option_termination_cooldown_blocks_immediate_reselection(self) -> None:
        brain = SpiderBrain(
            seed=46,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy",
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
                direct_policy_option_termination_cooldown=True,
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
        recurrent.current_option_idx = OPTION_NAMES.index("REST")
        recurrent.current_option_age = 2
        recurrent.current_option_steps_remaining = 1
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.b2_option[OPTION_NAMES.index("REST")] = 8.0
        recurrent.b2_option[OPTION_NAMES.index("FORAGE")] = 6.0
        brain.set_direct_policy_event_clock(9)
        brain.record_direct_policy_event(
            "RECOVERY_COMPLETED",
            features=np.array([1.0, 0.5, 0.1, 1.0, 0.0], dtype=float),
            tick=9,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_termination_reason, "recovery_completed")
        self.assertEqual(decision.selected_option, "FORAGE")

    def test_option_action_head_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=48,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy",
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
                direct_policy_option_action_head=True,
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
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.b2_option[OPTION_NAMES.index("FORAGE")] = 8.0
        recurrent.W2_option_action_head.fill(0.0)
        recurrent.b2_option_action_head.fill(0.0)
        recurrent.b2_option_action_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
        ] = 6.0
        recurrent.b2_option_action_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = -6.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_decoder_state_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=50,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy",
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
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.b_h[0] = 1.0
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.b2_option[OPTION_NAMES.index("FORAGE")] = 8.0
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.b_option_decoder_state[OPTION_NAMES.index("FORAGE"), 0] = 4.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_recurrent_dynamics_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=53,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy",
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
        recurrent.hidden_state[0] = 1.0
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_option_recurrent_dynamics[
            OPTION_NAMES.index("FORAGE"),
            0,
            0,
        ] = 4.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_sequence_head_affects_selected_action_by_option_age(self) -> None:
        brain = SpiderBrain(
            seed=57,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy",
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
                direct_policy_option_sequence_head=True,
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
        recurrent.hidden_state[0] = 1.0
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W2_option_sequence_head.fill(0.0)
        recurrent.b2_option_sequence_head.fill(0.0)
        recurrent.b2_option_sequence_head[
            OPTION_NAMES.index("FORAGE"),
            2,
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
        ] = 3.0
        recurrent.b2_option_sequence_head[
            OPTION_NAMES.index("FORAGE"),
            2,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = -3.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.option_age, 2)
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_decoder_recurrent_state_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=61,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy",
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
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.decoder_action_state.fill(0.0)
        recurrent.decoder_action_state[0] = 1.0
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
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
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_option_decoder_recurrent_state[
            OPTION_NAMES.index("FORAGE"),
            0,
            0,
        ] = 4.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.option_age, 2)
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_transition_state_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=63,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy",
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
                direct_policy_option_action_transition_state=True,
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
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
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
        recurrent.W_option_action_transition_state.fill(0.0)
        recurrent.b_option_action_transition_state.fill(0.0)
        recurrent.b2_policy[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("MOVE_RIGHT"), 0] = 1.0
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("STAY"), 0] = -1.0
        recurrent.W_option_action_transition_state[
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

    def test_option_action_controller_state_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=67,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy",
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
                direct_policy_option_action_controller_state=True,
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
        recurrent.action_controller_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
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
        recurrent.W_option_action_controller_decoder.fill(0.0)
        recurrent.W_option_action_controller_prev.fill(0.0)
        recurrent.W_option_action_controller_action.fill(0.0)
        recurrent.b_option_action_controller.fill(0.0)
        recurrent.W2_option_action_controller_head.fill(0.0)
        recurrent.b2_option_action_controller_head.fill(0.0)
        recurrent.b2_policy[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W_option_action_controller_action[
            OPTION_NAMES.index("FORAGE"),
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        recurrent.W2_option_action_controller_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_option_action_controller_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_token_decoder_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=71,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy",
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
                direct_policy_option_action_token_decoder=True,
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
        recurrent.action_token_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
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
        recurrent.W_option_action_token_decoder.fill(0.0)
        recurrent.W_option_action_token_prev.fill(0.0)
        recurrent.W_option_action_token_action.fill(0.0)
        recurrent.b_option_action_token.fill(0.0)
        recurrent.b2_policy[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("MOVE_RIGHT"), 0] = 1.0
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("STAY"), 0] = -1.0
        recurrent.W_option_action_token_action[
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

    def test_option_action_recurrent_core_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=75,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy",
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
