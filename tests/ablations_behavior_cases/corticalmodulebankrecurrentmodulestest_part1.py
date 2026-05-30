from __future__ import annotations

from .shared import *
from .corticalmodulebankrecurrentmodulestest_helpers import CorticalModuleBankRecurrentModulesTestHelpers



class CorticalModuleBankRecurrentModulesTestPart1(CorticalModuleBankRecurrentModulesTestHelpers, unittest.TestCase):
    def test_recurrent_module_uses_recurrent_network(self) -> None:
        bank = self._make_bank(("alert_center",))
        self.assertIsInstance(bank.modules["alert_center"], RecurrentProposalNetwork)
        self.assertNotIsInstance(bank.modules["hunger_center"], RecurrentProposalNetwork)

    def test_has_recurrent_modules_reflects_configuration(self) -> None:
        self.assertFalse(self._make_bank(()).has_recurrent_modules)
        self.assertTrue(self._make_bank(("alert_center",)).has_recurrent_modules)

    def test_unknown_recurrent_module_raises_value_error(self) -> None:
        rng = np.random.default_rng(44)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                recurrent_modules=("nonexistent_module",),
            )

    def test_reset_hidden_states_clears_recurrent_modules(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 1.0
        bank.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_snapshot_and_restore_hidden_states_round_trip(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.25
        snapshot = bank.snapshot_hidden_states()
        recurrent.hidden_state[:] = 0.75
        bank.restore_hidden_states(snapshot)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.25, dtype=float),
        )

    def test_spider_brain_reset_hidden_states_delegates_to_module_bank(self) -> None:
        brain = SpiderBrain(seed=5, module_dropout=0.0)
        brain.module_bank = self._make_bank(("alert_center",))
        recurrent = brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 1.0
        brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_spider_brain_estimate_value_does_not_commit_hidden_state(self) -> None:
        brain = SpiderBrain(
            seed=5,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="full_modular_recurrent_test",
                module_dropout=0.0,
                disabled_modules=(),
            ),
        )
        brain.module_bank = self._make_bank(("alert_center",))
        recurrent = brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.5
        observation = _build_observation()
        brain.estimate_value(observation)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.5, dtype=float),
        )

    def test_recurrent_hidden_state_accumulates_within_episode_and_resets_between_episodes(self) -> None:
        config = BrainAblationConfig(
            name="recurrent_integration",
            architecture="modular",
            module_dropout=0.0,
            recurrent_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=17, max_steps=6, brain_config=config)
        observation = sim.world.reset(seed=17)
        sim.brain.reset_hidden_states()
        recurrent = sim.brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

        first_step = sim.brain.act(observation, sample=False)
        hidden_after_first = recurrent.hidden_state.copy()
        self.assertGreater(float(np.linalg.norm(hidden_after_first)), 0.0)

        next_observation, _, _, _ = sim.world.step(first_step.action_idx)
        sim.brain.act(next_observation, sample=False)
        hidden_after_second = recurrent.hidden_state.copy()
        self.assertGreater(float(np.linalg.norm(hidden_after_second)), 0.0)
        self.assertFalse(np.allclose(hidden_after_second, hidden_after_first))

        sim.world.reset(seed=18)
        sim.brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_snapshot_returns_empty_dict_when_no_recurrent_modules(self) -> None:
        bank = self._make_bank(())
        snapshot = bank.snapshot_hidden_states()
        self.assertEqual(snapshot, {})

    def test_restore_raises_key_error_for_unknown_module(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        snapshot = {"nonexistent_module": np.zeros(recurrent.hidden_dim)}
        with self.assertRaises(KeyError):
            bank.restore_hidden_states(snapshot)

    def test_restore_raises_value_error_for_non_recurrent_module(self) -> None:
        bank = self._make_bank(("alert_center",))
        # hunger_center is feed-forward; it has no hidden_state attribute
        non_recurrent = bank.modules["hunger_center"]
        self.assertNotIsInstance(non_recurrent, RecurrentProposalNetwork)
        snapshot = {"hunger_center": np.zeros(10)}
        with self.assertRaises(ValueError):
            bank.restore_hidden_states(snapshot)

    def test_multiple_recurrent_modules_all_get_recurrent_networks(self) -> None:
        bank = self._make_bank(("alert_center", "sleep_center", "hunger_center"))
        for name in ("alert_center", "sleep_center", "hunger_center"):
            self.assertIsInstance(bank.modules[name], RecurrentProposalNetwork,
                                  f"{name} should be RecurrentProposalNetwork")
        # Feed-forward modules should remain feed-forward
        for name in ("visual_cortex", "sensory_cortex"):
            self.assertNotIsInstance(bank.modules[name], RecurrentProposalNetwork,
                                     f"{name} should not be RecurrentProposalNetwork")

    def test_snapshot_returns_copies_not_live_references(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.3
        snapshot = bank.snapshot_hidden_states()
        # Mutate live hidden state after snapshot
        recurrent.hidden_state[:] = 0.9
        # Snapshot should retain the value at snapshot time
        np.testing.assert_allclose(
            snapshot["alert_center"],
            np.full(recurrent.hidden_dim, 0.3, dtype=float),
        )

    def test_reset_hidden_states_does_not_affect_feedforward_module_identity(self) -> None:
        # Resetting hidden states should not raise and must leave feed-forward modules intact
        bank = self._make_bank(("alert_center",))
        ff_before = bank.modules["hunger_center"]
        bank.reset_hidden_states()
        ff_after = bank.modules["hunger_center"]
        self.assertIs(ff_before, ff_after)

    def test_spider_brain_reset_hidden_states_is_noop_with_no_module_bank(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.module_bank = None
        # Should not raise
        brain.reset_hidden_states()

    def test_run_episode_resets_recurrent_hidden_state_at_start(self) -> None:
        config = BrainAblationConfig(
            name="recurrent_episode_reset",
            architecture="modular",
            module_dropout=0.0,
            recurrent_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=19, max_steps=3, brain_config=config)
        recurrent = sim.brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        dirty_state = np.full(recurrent.hidden_dim, 99.0, dtype=float)
        recurrent.hidden_state[:] = dirty_state
        reset_snapshots: list[np.ndarray] = []
        original_reset_hidden_states = sim.brain.reset_hidden_states

        def wrapped_reset_hidden_states() -> None:
            original_reset_hidden_states()
            reset_snapshots.append(recurrent.get_hidden_state())

        with mock.patch.object(
            sim.brain,
            "reset_hidden_states",
            side_effect=wrapped_reset_hidden_states,
        ) as reset_mock:
            sim.run_episode(0, training=False, sample=False)

        self.assertEqual(reset_mock.call_count, 1)
        self.assertEqual(len(reset_snapshots), 1)
        np.testing.assert_allclose(
            reset_snapshots[0],
            np.zeros(recurrent.hidden_dim, dtype=float),
        )
        self.assertFalse(np.allclose(reset_snapshots[0], dirty_state))
        self.assertTrue(np.all(np.isfinite(recurrent.hidden_state)))

    def test_true_monolithic_recurrent_policy_uses_recurrent_direct_network(self) -> None:
        brain = SpiderBrain(
            seed=23,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        self.assertIsInstance(brain.true_monolithic_policy, RecurrentTrueMonolithicNetwork)

    def test_true_monolithic_recurrent_reset_hidden_states_clears_hidden_state(self) -> None:
        brain = SpiderBrain(
            seed=24,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        recurrent.hidden_state[:] = 1.0
        brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_true_monolithic_recurrent_estimate_value_does_not_commit_hidden_state(self) -> None:
        brain = SpiderBrain(
            seed=25,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        recurrent.hidden_state[:] = 0.5
        observation = _build_observation()
        brain.estimate_value(observation)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.5, dtype=float),
        )

    def test_true_monolithic_recurrent_inference_is_deterministic_given_same_state(self) -> None:
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
        brain = SpiderBrain(seed=26, module_dropout=0.0, config=config)
        observation = _build_observation()
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        seeded_state = np.linspace(-0.2, 0.2, recurrent.hidden_dim, dtype=float)
        recurrent.set_hidden_state(seeded_state)
        first = brain.act_inference(observation, sample=False)
        hidden_after_first = recurrent.get_hidden_state()
        recurrent.set_hidden_state(seeded_state)
        second = brain.act_inference(observation, sample=False)
        hidden_after_second = recurrent.get_hidden_state()
        np.testing.assert_allclose(first.total_logits, second.total_logits)
        self.assertEqual(first.action_idx, second.action_idx)
        np.testing.assert_allclose(hidden_after_first, hidden_after_second)

    def test_true_monolithic_recurrent_phase_policy_exposes_phase_predictions(self) -> None:
        brain = SpiderBrain(
            seed=27,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)
        self.assertGreaterEqual(decision.phase_prediction_confidence, 0.0)
        self.assertLessEqual(decision.phase_prediction_confidence, 1.0)

    def test_true_monolithic_event_attention_policy_exposes_top_event(self) -> None:
        brain = SpiderBrain(
            seed=28,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentEventAttentionTrueMonolithicNetwork,
        )
        brain.set_direct_policy_event_clock(4)
        brain.record_direct_policy_event(
            "REST_STARTED",
            features=np.array([1.0, 0.4, 0.1, 1.0, 0.0], dtype=float),
            tick=4,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.event_attention_top_type, "REST_STARTED")
        self.assertEqual(decision.event_attention_top_age, 0)
        self.assertGreaterEqual(decision.event_attention_entropy, 0.0)

    def test_true_monolithic_option_policy_exposes_option_commitment(self) -> None:
        brain = SpiderBrain(
            seed=29,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentOptionTrueMonolithicNetwork)
        brain.set_direct_policy_event_clock(6)
        brain.record_direct_policy_event(
            "RECOVERY_COMPLETED",
            features=np.array([1.0, 0.6, 0.1, 1.0, 0.0], dtype=float),
            tick=6,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertIn(decision.selected_option, OPTION_NAMES)
        self.assertEqual(decision.option_age, 0)
        self.assertEqual(decision.option_termination_reason, "initial_selection")
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(
            recurrent.last_option_summary["selected_option"],
            decision.selected_option,
        )

    def test_true_monolithic_option_affordance_policy_exposes_affordance_logits(self) -> None:
        brain = SpiderBrain(
            seed=30,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.affordance_blocked_logits.shape,
            (len(LOCOMOTION_ACTIONS),),
        )
        self.assertEqual(
            decision.affordance_role_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 4,),
        )

    def test_true_monolithic_option_affordance_feedback_policy_exposes_affordance_logits(self) -> None:
        brain = SpiderBrain(
            seed=31,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertIn(decision.selected_option, OPTION_NAMES)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(
            decision.affordance_blocked_logits.shape,
            (len(LOCOMOTION_ACTIONS),),
        )
        self.assertEqual(
            decision.affordance_role_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 4,),
        )

    def test_true_monolithic_option_affordance_geometry_policy_exposes_geometry_logits(self) -> None:
        brain = SpiderBrain(
            seed=32,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceGeometryFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.geometry_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 3,),
        )

    def test_true_monolithic_option_affordance_topology_policy_exposes_shelter_column_logits(self) -> None:
        brain = SpiderBrain(
            seed=33,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceTopologyFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.shelter_column_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 4,),
        )

    def test_true_monolithic_option_affordance_position_policy_exposes_shelter_position_logits(self) -> None:
        brain = SpiderBrain(
            seed=34,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.shelter_position_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 10,),
        )

    def test_true_monolithic_option_affordance_position_teacher_policy_uses_position_network(self) -> None:
        brain = SpiderBrain(
            seed=35,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_teacher_policy",
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
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )

    def test_true_monolithic_option_affordance_position_teacher_option_policy_uses_position_network(self) -> None:
        brain = SpiderBrain(
            seed=36,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_teacher_option_policy",
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
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )

    def test_true_monolithic_executive_option_guarded_policy_skips_food_direction_bias(self) -> None:
        common_fields = dict(
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
        control_brain = SpiderBrain(
            seed=301,
            module_dropout=0.0,
            config=BrainAblationConfig(name="control_executive_bias_on", **common_fields),
        )
        guarded_brain = SpiderBrain(
            seed=301,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_guarded_policy",
                **common_fields,
            ),
        )
        observation = _build_observation()
        control_net = control_brain.true_monolithic_policy
        guarded_net = guarded_brain.true_monolithic_policy
        self.assertIsNotNone(control_net)
        self.assertIsNotNone(guarded_net)

        def _fixed_forward(net):
            policy_logits = np.array(
                [-2.0, -2.0, -2.0, -2.0, 0.5, -2.0, -2.0, -2.0, -2.0],
                dtype=float,
            )
            option_logits = np.zeros(net.option_dim, dtype=float)
            phase_logits = np.zeros(net.phase_output_dim, dtype=float)
            return policy_logits, 0.0, option_logits, phase_logits

        control_net.forward = lambda *_args, **_kwargs: _fixed_forward(control_net)
        guarded_net.forward = lambda *_args, **_kwargs: _fixed_forward(guarded_net)
        control_brain._threat_escape_bias_action = lambda _obs: None
        control_brain._sleep_rest_bias_action = lambda _obs: None
        control_brain._food_direction_bias_action = lambda _obs: "MOVE_RIGHT"
        guarded_brain._threat_escape_bias_action = lambda _obs: None
        guarded_brain._sleep_rest_bias_action = lambda _obs: None
        guarded_brain._food_direction_bias_action = lambda _obs: "MOVE_RIGHT"

        control_decision = control_brain.act(observation, sample=False)
        guarded_decision = guarded_brain.act(observation, sample=False)

        self.assertEqual(LOCOMOTION_ACTIONS[control_decision.action_idx], "MOVE_RIGHT")
        self.assertEqual(LOCOMOTION_ACTIONS[guarded_decision.action_idx], "STAY")

    def test_executive_physiology_option_gating_prefers_rest_inside_shelter(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)

        def _build_monolithic_input() -> np.ndarray:
            vectors = []
            for spec in MODULE_INTERFACES:
                mapping = {name: 0.0 for name in spec.signal_names}
                if spec.name == "sleep_center":
                    mapping.update(
                        {
                            "fatigue": 0.35,
                            "hunger": 0.1,
                            "on_shelter": 1.0,
                            "night": 1.0,
                            "sleep_phase_level": 0.4,
                            "rest_streak_norm": 0.6,
                            "sleep_debt": 0.3,
                            "shelter_role_level": 1.0,
                            "shelter_memory_age": 0.0,
                        }
                    )
                vectors.append(spec.vector_from_mapping(mapping))
            return np.concatenate(vectors, axis=0)

        def _build_network(*, gated: bool):
            net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
                input_dim=input_dim,
                hidden_dim=8,
                output_dim=len(LOCOMOTION_ACTIONS),
                rng=np.random.default_rng(7),
                event_buffer_size=2,
                option_ttl=4,
                executive_physiology_option_gating=gated,
                name="test_true_monolithic_policy",
            )
            net.W_xh.fill(0.0)
            net.W_hh.fill(0.0)
            net.b_h.fill(0.0)
            net.W2_option.fill(0.0)
            net.b2_option.fill(0.0)
            net.W2_option_feedback.fill(0.0)
            net.b2_option_feedback.fill(0.0)
            net.W_affordance_feedback.fill(0.0)
            net.b_affordance_feedback.fill(0.0)
            net.b2_option[OPTION_NAMES.index("ESCAPE")] = 4.0
            return net

        x = _build_monolithic_input()
        ungated = _build_network(gated=False)
        gated = _build_network(gated=True)
        ungated.forward(x, store_cache=False)
        gated.forward(x, store_cache=False)

        self.assertEqual(ungated.last_option_summary["selected_option"], "ESCAPE")
        self.assertEqual(gated.last_option_summary["selected_option"], "REST")

    def test_executive_release_action_gating_prefers_move_up_after_rest(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)

        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.28,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.5,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 1.0,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(8),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            name="test_true_monolithic_policy",
        )
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_DOWN"]] = 5.0
        net.b2_option[OPTION_NAMES.index("POST_REST_REACTIVATE")] = 6.0
        net.W2_shelter_position.fill(0.0)
        net.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        net.b2_shelter_position[offset + outside_idx] = 6.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(net.last_option_summary["selected_option"], "POST_REST_REACTIVATE")
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_UP",
        )

    def test_executive_masked_release_action_gating_prefers_stay_for_rest(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.35,
                        "hunger": 0.1,
                        "on_shelter": 1.0,
                        "night": 1.0,
                        "sleep_phase_level": 0.4,
                        "rest_streak_norm": 0.6,
                        "sleep_debt": 0.3,
                        "shelter_role_level": 1.0,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(9),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            name="test_true_monolithic_policy",
        )
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_UP"]] = 12.0
        net.b2_option[OPTION_NAMES.index("REST")] = 6.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(net.last_option_summary["selected_option"], "REST")
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "STAY",
        )

    def test_executive_event_release_latch_promotes_post_rest_move_up(self) -> None:
        brain = SpiderBrain(
            seed=47,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("DEEPEN_IN_SHELTER")
        recurrent.current_option_age = 2
        recurrent.current_option_steps_remaining = 1
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["ORIENT_DOWN"]] = 6.0
        recurrent.b2_option[OPTION_NAMES.index("DEEPEN_IN_SHELTER")] = 8.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        recurrent.b2_shelter_position[offset + outside_idx] = 6.0
        brain.set_direct_policy_event_clock(7)
        brain.record_direct_policy_event(
            "DEEP_SLEEP_REACHED",
            features=np.array([1.0, 0.4, 0.0, 1.0, 0.0], dtype=float),
            tick=7,
        )
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.22,
                    "on_shelter": 1.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 1.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertEqual(decision.option_termination_reason, "deep_shelter_reached")
        self.assertEqual(decision.selected_option, "POST_REST_REACTIVATE")
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_transition_release_commitment_forces_move_up(self) -> None:
        brain = SpiderBrain(
            seed=49,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.executive_release_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 9.0
        recurrent.b2_option[OPTION_NAMES.index("RETURN_TO_SHELTER")] = 8.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        recurrent.b2_shelter_position[offset + outside_idx] = 6.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.22,
                    "on_shelter": 1.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 1.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertIn(
            decision.selected_option,
            {"RETURN_TO_SHELTER", "POST_REST_REACTIVATE"},
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_release_phase_state_arms_move_up_at_start(self) -> None:
        brain = SpiderBrain(
            seed=50,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["ORIENT_UP"]] = 9.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        recurrent.b2_shelter_position[offset + outside_idx] = 6.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 1.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.0,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 1.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertGreater(recurrent.executive_release_steps_remaining, 0)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_release_progression_prefers_move_right_at_entrance(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.18,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.0,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 0.33,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(10),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            executive_event_release_latching=True,
            executive_event_release_action_commitment=True,
            executive_release_phase_state=True,
            executive_release_progression=True,
            name="test_true_monolithic_policy",
        )
        net.executive_release_steps_remaining = 2
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_RIGHT"]] = 12.0
        net.W2_shelter_position.fill(0.0)
        net.b2_shelter_position.fill(0.0)
        stay_idx = ACTION_TO_INDEX["STAY"]
        entrance_center_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("entrance_center")
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        net.b2_shelter_position[stay_idx * pos_dim + entrance_center_idx] = 6.0
        net.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 6.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_RIGHT",
        )

    def test_executive_release_exit_contract_prefers_move_left_at_entrance(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.18,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.0,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 0.33,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(11),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            executive_event_release_latching=True,
            executive_event_release_action_commitment=True,
            executive_release_phase_state=True,
            executive_release_exit_contract=True,
            name="test_true_monolithic_policy",
        )
        net.executive_release_steps_remaining = 2
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_RIGHT"]] = 12.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_LEFT",
        )

    def test_executive_release_substate_progression_keeps_move_left_at_entrance_with_last_step(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.18,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.0,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 1.0 / 3.0,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(12),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            executive_event_release_latching=True,
            executive_event_release_action_commitment=True,
            executive_release_phase_state=True,
            executive_release_exit_contract=True,
            executive_release_substate_progression=True,
            name="test_true_monolithic_policy",
        )
        net.executive_release_steps_remaining = 1
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_RIGHT"]] = 12.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_LEFT",
        )

    def test_executive_post_exit_continuation_preserves_post_rest_option_after_exit(self) -> None:
        brain = SpiderBrain(
            seed=112,
            module_dropout=0.0,
            config=BrainAblationConfig(
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
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 3
        recurrent.current_option_steps_remaining = 1
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 8.0
        recurrent.record_event(
            "SHELTER_EXIT",
            features=np.zeros(recurrent.event_feature_dim, dtype=float),
            tick=0,
        )
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertEqual(decision.selected_option, "POST_REST_REACTIVATE")
        self.assertGreater(recurrent.executive_post_exit_steps_remaining, 0)
