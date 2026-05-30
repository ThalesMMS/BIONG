from __future__ import annotations

from .shared import *



class TrueMonolithicRecurrentCheckpointingTestPart3(unittest.TestCase):
    def test_transition_prediction_branch_round_trip_restores_transition_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_transition_prediction_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_prediction_head=True,
            direct_policy_transition_prediction_feedback=True,
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=57, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=58, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_transition_prediction,
            recurrent.W2_transition_prediction,
        )
        np.testing.assert_allclose(
            loaded.b2_transition_prediction,
            recurrent.b2_transition_prediction,
        )
        np.testing.assert_allclose(
            loaded.W_transition_prediction_feedback,
            recurrent.W_transition_prediction_feedback,
        )
        np.testing.assert_allclose(
            loaded.b_transition_prediction_feedback,
            recurrent.b_transition_prediction_feedback,
        )

    def test_transition_rollout_prediction_branch_round_trip_restores_transition_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_transition_rollout_prediction_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_rollout_prediction_head=True,
            direct_policy_transition_rollout_prediction_feedback=True,
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=59, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=60, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_transition_rollout_prediction,
            recurrent.W2_transition_rollout_prediction,
        )
        np.testing.assert_allclose(
            loaded.b2_transition_rollout_prediction,
            recurrent.b2_transition_rollout_prediction,
        )
        np.testing.assert_allclose(
            loaded.W_transition_rollout_prediction_feedback,
            recurrent.W_transition_rollout_prediction_feedback,
        )
        np.testing.assert_allclose(
            loaded.b_transition_rollout_prediction_feedback,
            recurrent.b_transition_rollout_prediction_feedback,
        )
