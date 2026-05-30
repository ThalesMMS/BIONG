from __future__ import annotations

from .shared import *



class TrueMonolithicRecurrentCheckpointingTestPart1(unittest.TestCase):
    def test_save_load_round_trip_resets_live_hidden_state_and_restores_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_recurrent_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
        )
        source = SpiderBrain(seed=31, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=32, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        recurrent.hidden_state[:] = 0.75
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.hidden_state, np.zeros(loaded.hidden_dim, dtype=float))
        np.testing.assert_allclose(loaded.W_xh, recurrent.W_xh)
        np.testing.assert_allclose(loaded.W_hh, recurrent.W_hh)
        np.testing.assert_allclose(loaded.W2_policy, recurrent.W2_policy)

    def test_phase_head_round_trip_restores_phase_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_recurrent_phase_policy",
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
        )
        source = SpiderBrain(seed=33, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=34, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.W2_phase, recurrent.W2_phase)
        np.testing.assert_allclose(loaded.b2_phase, recurrent.b2_phase)

    def test_position_phase_replay_round_trip_restores_phase_and_option_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_teacher_option_replay_policy",
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
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=39, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=40, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.W2_option, recurrent.W2_option)
        np.testing.assert_allclose(loaded.b2_option, recurrent.b2_option)
        np.testing.assert_allclose(loaded.W2_phase, recurrent.W2_phase)
        np.testing.assert_allclose(loaded.b2_phase, recurrent.b2_phase)

    def test_position_phase_option_feedback_replay_round_trip_restores_phase_feedback_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=41, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=42, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_phase_option_feedback,
            recurrent.W2_phase_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_phase_option_feedback,
            recurrent.b2_phase_option_feedback,
        )

    def test_position_phase_option_transition_replay_round_trip_restores_transition_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=43, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=44, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_option_transition_feedback,
            recurrent.W2_option_transition_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_option_transition_feedback,
            recurrent.b2_option_transition_feedback,
        )

    def test_position_phase_option_action_replay_round_trip_restores_action_head_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=45, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=46, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_option_action_head,
            recurrent.W2_option_action_head,
        )
        np.testing.assert_allclose(
            loaded.b2_option_action_head,
            recurrent.b2_option_action_head,
        )

    def test_position_phase_option_decoder_replay_round_trip_restores_decoder_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=47, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=48, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_decoder_state,
            recurrent.W_option_decoder_state,
        )
        np.testing.assert_allclose(
            loaded.b_option_decoder_state,
            recurrent.b_option_decoder_state,
        )

    def test_position_phase_option_dynamics_replay_round_trip_restores_recurrent_dynamics_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=49, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=50, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_recurrent_dynamics,
            recurrent.W_option_recurrent_dynamics,
        )
        np.testing.assert_allclose(
            loaded.b_option_recurrent_dynamics,
            recurrent.b_option_recurrent_dynamics,
        )

    def test_position_phase_option_dynamics_sequence_replay_round_trip_restores_sequence_head_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=58, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=59, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_option_sequence_head,
            recurrent.W2_option_sequence_head,
        )
        np.testing.assert_allclose(
            loaded.b2_option_sequence_head,
            recurrent.b2_option_sequence_head,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_replay_round_trip_restores_decoder_recurrent_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=62, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=63, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_decoder_recurrent_state,
            recurrent.W_option_decoder_recurrent_state,
        )
        np.testing.assert_allclose(
            loaded.b_option_decoder_recurrent_state,
            recurrent.b_option_decoder_recurrent_state,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_transition_replay_round_trip_restores_action_transition_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=64, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=65, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_transition_state,
            recurrent.W_option_action_transition_state,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_transition_state,
            recurrent.b_option_action_transition_state,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_controller_replay_round_trip_restores_action_controller_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=68, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=69, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_controller_decoder,
            recurrent.W_option_action_controller_decoder,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_controller_prev,
            recurrent.W_option_action_controller_prev,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_controller_action,
            recurrent.W_option_action_controller_action,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_controller,
            recurrent.b_option_action_controller,
        )
        np.testing.assert_allclose(
            loaded.W2_option_action_controller_head,
            recurrent.W2_option_action_controller_head,
        )
        np.testing.assert_allclose(
            loaded.b2_option_action_controller_head,
            recurrent.b2_option_action_controller_head,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_token_replay_round_trip_restores_action_token_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=72, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=73, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_token_decoder,
            recurrent.W_option_action_token_decoder,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_token_prev,
            recurrent.W_option_action_token_prev,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_token_action,
            recurrent.W_option_action_token_action,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_token,
            recurrent.b_option_action_token,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_core_replay_round_trip_restores_action_core_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=76, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=77, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_policy_decoder,
            recurrent.W_option_action_policy_decoder,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_policy_prev,
            recurrent.W_option_action_policy_prev,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_policy_action,
            recurrent.W_option_action_policy_action,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_policy,
            recurrent.b_option_action_policy,
        )
        np.testing.assert_allclose(
            loaded.W2_action_policy_core,
            recurrent.W2_action_policy_core,
        )
        np.testing.assert_allclose(
            loaded.b2_action_policy_core,
            recurrent.b2_action_policy_core,
        )

    def test_position_phase_option_dynamics_separate_action_recurrent_head_replay_round_trip_restores_action_core_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=80, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=81, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_policy_decoder,
            recurrent.W_option_action_policy_decoder,
        )
        np.testing.assert_allclose(
            loaded.W2_action_policy_core,
            recurrent.W2_action_policy_core,
        )

    def test_position_phase_option_dynamics_separate_action_policy_path_replay_round_trip_restores_action_policy_path_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=84, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=85, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_action_policy_path_input,
            recurrent.W_action_policy_path_input,
        )
        np.testing.assert_allclose(
            loaded.W_action_policy_path_prev,
            recurrent.W_action_policy_path_prev,
        )
        np.testing.assert_allclose(
            loaded.W_action_policy_path_action,
            recurrent.W_action_policy_path_action,
        )
        np.testing.assert_allclose(
            loaded.b_action_policy_path,
            recurrent.b_action_policy_path,
        )
        np.testing.assert_allclose(
            loaded.W2_action_policy_path,
            recurrent.W2_action_policy_path,
        )
        np.testing.assert_allclose(
            loaded.b2_action_policy_path,
            recurrent.b2_action_policy_path,
        )

    def test_position_phase_option_dynamics_separate_action_backbone_replay_round_trip_restores_action_backbone_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=88, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=89, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_action_backbone_input,
            recurrent.W_action_backbone_input,
        )
        np.testing.assert_allclose(
            loaded.W_action_backbone_prev,
            recurrent.W_action_backbone_prev,
        )
        np.testing.assert_allclose(
            loaded.W_action_backbone_action,
            recurrent.W_action_backbone_action,
        )
        np.testing.assert_allclose(
            loaded.b_action_backbone,
            recurrent.b_action_backbone,
        )
        np.testing.assert_allclose(
            loaded.W2_action_backbone,
            recurrent.W2_action_backbone,
        )
        np.testing.assert_allclose(
            loaded.b2_action_backbone,
            recurrent.b2_action_backbone,
        )

    def test_executive_option_policy_round_trip_restores_feedback_and_action_backbone_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
        )
        source = SpiderBrain(seed=90, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=91, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_phase_option_feedback,
            recurrent.W2_phase_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_phase_option_feedback,
            recurrent.b2_phase_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_option_transition_feedback,
            recurrent.W2_option_transition_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_option_transition_feedback,
            recurrent.b2_option_transition_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_action_backbone,
            recurrent.W2_action_backbone,
        )
        np.testing.assert_allclose(
            loaded.b2_action_backbone,
            recurrent.b2_action_backbone,
        )

    def test_executive_option_physiology_gating_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_physiology_gated_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
        )
        source = SpiderBrain(seed=92, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=93, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_physiology_option_gating)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_physiology_option_gating)

    def test_executive_option_release_round_trip_preserves_action_gate_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
        )
        source = SpiderBrain(seed=94, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=95, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_affordance_action_gating)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_affordance_action_gating)

    def test_executive_option_masked_release_round_trip_preserves_mask_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_masked_release_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
        )
        source = SpiderBrain(seed=96, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=97, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_option_action_masking)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_option_action_masking)

    def test_executive_option_event_release_round_trip_preserves_latch_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_event_release_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
        )
        source = SpiderBrain(seed=98, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=99, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_event_release_latching)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_event_release_latching)

    def test_executive_option_transition_release_round_trip_preserves_commitment_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_transition_release_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
            direct_policy_executive_event_release_action_commitment=True,
        )
        source = SpiderBrain(seed=100, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=101, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_event_release_action_commitment)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_event_release_action_commitment)

    def test_executive_option_release_phase_round_trip_preserves_phase_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_phase_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
            direct_policy_executive_event_release_action_commitment=True,
            direct_policy_executive_release_phase_state=True,
        )
        source = SpiderBrain(seed=102, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=103, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_phase_state)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_phase_state)

    def test_executive_option_release_progression_round_trip_preserves_progression_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_progression_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
            direct_policy_executive_event_release_action_commitment=True,
            direct_policy_executive_release_phase_state=True,
            direct_policy_executive_release_progression=True,
        )
        source = SpiderBrain(seed=104, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=105, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_progression)

    def test_executive_option_release_exit_contract_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_exit_contract_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
            direct_policy_executive_event_release_action_commitment=True,
            direct_policy_executive_release_phase_state=True,
            direct_policy_executive_release_exit_contract=True,
        )
        source = SpiderBrain(seed=106, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=107, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_exit_contract)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_exit_contract)

    def test_executive_option_release_substate_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_substate_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
            direct_policy_executive_event_release_action_commitment=True,
            direct_policy_executive_release_phase_state=True,
            direct_policy_executive_release_exit_contract=True,
            direct_policy_executive_release_substate_progression=True,
        )
        source = SpiderBrain(seed=108, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=109, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_substate_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_substate_progression)

    def test_executive_option_post_exit_continuation_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_continuation_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
            direct_policy_executive_event_release_action_commitment=True,
            direct_policy_executive_release_phase_state=True,
            direct_policy_executive_release_exit_contract=True,
            direct_policy_executive_release_substate_progression=True,
            direct_policy_executive_post_exit_continuation=True,
        )
        source = SpiderBrain(seed=110, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=111, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_continuation)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_continuation)

    def test_executive_option_post_exit_food_guidance_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_food_guidance_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_executive_physiology_option_gating=True,
            direct_policy_executive_affordance_action_gating=True,
            direct_policy_executive_option_action_masking=True,
            direct_policy_executive_event_release_latching=True,
            direct_policy_executive_event_release_action_commitment=True,
            direct_policy_executive_release_phase_state=True,
            direct_policy_executive_release_exit_contract=True,
            direct_policy_executive_release_substate_progression=True,
            direct_policy_executive_post_exit_continuation=True,
            direct_policy_executive_post_exit_food_guidance=True,
        )
        source = SpiderBrain(seed=113, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=114, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_guidance)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_guidance)
