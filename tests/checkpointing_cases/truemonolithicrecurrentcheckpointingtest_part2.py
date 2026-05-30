from __future__ import annotations

from .shared import *



class TrueMonolithicRecurrentCheckpointingTestPart2(unittest.TestCase):
    def test_executive_option_post_exit_food_commitment_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_food_commitment_policy",
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
            direct_policy_executive_post_exit_food_commitment=True,
        )
        source = SpiderBrain(seed=116, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=117, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_commitment)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_commitment)

    def test_executive_option_post_exit_food_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_food_progression_policy",
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
            direct_policy_executive_post_exit_food_progression=True,
        )
        source = SpiderBrain(seed=119, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=120, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_progression)

    def test_executive_option_post_exit_food_heading_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_food_heading_progression_policy",
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
            direct_policy_executive_post_exit_food_heading_progression=True,
        )
        source = SpiderBrain(seed=122, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=123, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_heading_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_heading_progression)

    def test_executive_option_post_exit_smell_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_smell_progression_policy",
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
            direct_policy_executive_post_exit_smell_progression=True,
        )
        source = SpiderBrain(seed=125, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=126, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_smell_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_smell_progression)

    def test_executive_option_post_exit_corridor_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_corridor_progression_policy",
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
            direct_policy_executive_post_exit_corridor_progression=True,
        )
        source = SpiderBrain(seed=225, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=226, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_corridor_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_corridor_progression)

    def test_executive_option_post_exit_corridor_affordance_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_corridor_affordance_progression_policy",
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
            direct_policy_executive_post_exit_corridor_progression=True,
            direct_policy_executive_post_exit_corridor_affordance_progression=True,
        )
        source = SpiderBrain(seed=227, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=228, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_corridor_affordance_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_corridor_affordance_progression)

    def test_executive_option_post_food_return_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_food_return_policy",
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
            direct_policy_executive_post_exit_corridor_progression=True,
            direct_policy_executive_post_exit_corridor_affordance_progression=True,
            direct_policy_executive_post_food_return=True,
        )
        source = SpiderBrain(seed=229, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=230, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_food_return)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_food_return)

    def test_executive_option_post_food_vector_return_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_food_vector_return_policy",
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
            direct_policy_executive_post_exit_corridor_progression=True,
            direct_policy_executive_post_exit_corridor_affordance_progression=True,
            direct_policy_executive_post_food_return=True,
            direct_policy_executive_post_food_vector_return=True,
        )
        source = SpiderBrain(seed=231, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=232, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_food_vector_return)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_food_vector_return)

    def test_executive_option_post_food_path_return_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_food_path_return_policy",
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
            direct_policy_executive_post_exit_corridor_progression=True,
            direct_policy_executive_post_exit_corridor_affordance_progression=True,
            direct_policy_executive_post_food_return=True,
            direct_policy_executive_post_food_path_return=True,
        )
        source = SpiderBrain(seed=233, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=234, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_food_path_return)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_food_path_return)

    def test_event_attention_round_trip_restores_attention_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_event_attention_policy",
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
        )
        source = SpiderBrain(seed=35, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=36, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.W_query, recurrent.W_query)
        np.testing.assert_allclose(loaded.W_key, recurrent.W_key)
        np.testing.assert_allclose(loaded.W_value, recurrent.W_value)
        np.testing.assert_allclose(
            loaded.event_type_embeddings,
            recurrent.event_type_embeddings,
        )

    def test_option_head_round_trip_restores_option_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_policy",
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
        )
        source = SpiderBrain(seed=37, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=38, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        recurrent.hidden_state[:] = 0.25
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.hidden_state, np.zeros(loaded.hidden_dim, dtype=float))
        np.testing.assert_allclose(loaded.W2_option, recurrent.W2_option)
        np.testing.assert_allclose(loaded.b2_option, recurrent.b2_option)
        np.testing.assert_allclose(
            loaded.option_action_bias,
            recurrent.option_action_bias,
        )

    def test_affordance_head_round_trip_restores_affordance_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_policy",
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
        np.testing.assert_allclose(
            loaded.W2_affordance_blocked,
            recurrent.W2_affordance_blocked,
        )
        np.testing.assert_allclose(
            loaded.b2_affordance_blocked,
            recurrent.b2_affordance_blocked,
        )
        np.testing.assert_allclose(
            loaded.W2_affordance_role,
            recurrent.W2_affordance_role,
        )
        np.testing.assert_allclose(
            loaded.b2_affordance_role,
            recurrent.b2_affordance_role,
        )

    def test_affordance_feedback_round_trip_restores_feedback_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_feedback_policy",
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
            loaded.W_affordance_feedback,
            recurrent.W_affordance_feedback,
        )
        np.testing.assert_allclose(
            loaded.b_affordance_feedback,
            recurrent.b_affordance_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_policy_feedback,
            recurrent.W2_policy_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_policy_feedback,
            recurrent.b2_policy_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_option_feedback,
            recurrent.W2_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_option_feedback,
            recurrent.b2_option_feedback,
        )

    def test_geometry_head_round_trip_restores_geometry_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_geometry_policy",
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
            loaded.W2_geometry,
            recurrent.W2_geometry,
        )
        np.testing.assert_allclose(
            loaded.b2_geometry,
            recurrent.b2_geometry,
        )

    def test_topology_head_round_trip_restores_shelter_column_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_topology_policy",
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
            direct_policy_shelter_column_head=True,
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
            loaded.W2_shelter_column,
            recurrent.W2_shelter_column,
        )
        np.testing.assert_allclose(
            loaded.b2_shelter_column,
            recurrent.b2_shelter_column,
        )

    def test_position_head_round_trip_restores_shelter_position_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_policy",
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
            loaded.W2_shelter_position,
            recurrent.W2_shelter_position,
        )
        np.testing.assert_allclose(
            loaded.b2_shelter_position,
            recurrent.b2_shelter_position,
        )

    def test_local_affordance_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_affordance_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
        source = SpiderBrain(seed=49, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=50, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_spatial_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_spatial_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
        source = SpiderBrain(seed=51, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=52, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_transition_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
        source = SpiderBrain(seed=53, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=54, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_transition_rollout_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_rollout_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
        source = SpiderBrain(seed=55, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=56, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_local_geodesic_inputs=True,
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
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_margin_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_replayable_teacher_distill_option_margin_replay_policy",
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
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
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
            direct_policy_continuation_margin_weight=1.0,
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
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_option_transition_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_option_transition_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
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
            direct_policy_option_transition_feedback=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=61, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=62, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_phase_option_feedback_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_phase_option_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
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
            direct_policy_phase_option_feedback=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=63, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=64, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_option_sequence_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_option_sequence_head_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
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
            direct_policy_option_sequence_head=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=65, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=66, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_trace_distill_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_trace_distill_option_replay_policy",
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
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
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
        source = SpiderBrain(seed=67, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=68, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_action_token_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_action_token_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
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
            direct_policy_option_action_token_decoder=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=69, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=70, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))
