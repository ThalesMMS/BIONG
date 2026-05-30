from __future__ import annotations

from .shared import *



class BSeriesCheckpointTestPart3(unittest.TestCase):
    def test_b10_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b9_waypoint_planner_source(tmpdir)
            config = build_b10_prospective_replay_config(
                B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=52,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b9_controller_profile",
            "b10_controller_profile",
            "b10_replay_state",
            "b10_prospective_value",
            "b10_rollout_depth",
            "b10_replay_memory",
            "b10_plan_commitment",
            "b10_abort_signal",
            "b10_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 10)
        self.assertEqual(first["b_parent_level"], 9)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b10_controller_profile"], "prospective_replay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b11_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b10_prospective_replay_source(tmpdir)
            config = build_b11_confidence_arbiter_config(
                B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=53,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b10_controller_profile",
            "b11_controller_profile",
            "b11_confidence_state",
            "b11_plan_confidence",
            "b11_uncertainty",
            "b11_neuromod_signal",
            "b11_confidence_lock",
            "b11_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 11)
        self.assertEqual(first["b_parent_level"], 10)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B11_CONFIDENCE_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(first["b11_controller_profile"], "confidence_arbiter")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b12_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b11_confidence_arbiter_source(tmpdir)
            config = build_b12_predictive_attention_config(
                B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=54,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b11_controller_profile",
            "b12_controller_profile",
            "b12_attention_state",
            "b12_prediction_error",
            "b12_attention_gain",
            "b12_expected_progress",
            "b12_search_lock",
            "b12_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 12)
        self.assertEqual(first["b_parent_level"], 11)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b12_controller_profile"], "predictive_attention")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b13_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b12_predictive_attention_source(tmpdir)
            config = build_b13_local_affordance_search_config(
                B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=55,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b12_controller_profile",
            "b13_controller_profile",
            "b13_search_state",
            "b13_local_route_score",
            "b13_affordance_samples",
            "b13_search_memory",
            "b13_dead_end_score",
            "b13_search_lock",
            "b13_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 13)
        self.assertEqual(first["b_parent_level"], 12)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B13_LOCAL_SEARCH_SELECTION_SOURCE,
        )
        self.assertEqual(first["b13_controller_profile"], "local_affordance_search")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b14_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b13_local_affordance_search_source(tmpdir)
            config = build_b14_affordance_uncertainty_config(
                B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=56,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b13_controller_profile",
            "b14_controller_profile",
            "b14_uncertainty_state",
            "b14_affordance_confidence",
            "b14_uncertainty",
            "b14_risk_adjusted_score",
            "b14_commitment_lock",
            "b14_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 14)
        self.assertEqual(first["b_parent_level"], 13)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b14_controller_profile"], "affordance_uncertainty")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b15_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b14_affordance_uncertainty_source(tmpdir)
            config = build_b15_option_critic_config(
                B15_OPTION_CRITIC_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=57,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b14_controller_profile",
            "b15_controller_profile",
            "b15_option_state",
            "b15_option_value",
            "b15_termination_pressure",
            "b15_persistence_score",
            "b15_option_lock",
            "b15_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 15)
        self.assertEqual(first["b_parent_level"], 14)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B15_OPTION_CRITIC_SELECTION_SOURCE,
        )
        self.assertEqual(first["b15_controller_profile"], "option_critic")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b16_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b15_option_critic_source(tmpdir)
            config = build_b16_option_ensemble_config(
                B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=58,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b15_controller_profile",
            "b16_controller_profile",
            "b16_ensemble_state",
            "b16_continue_vote",
            "b16_return_vote",
            "b16_option_votes",
            "b16_consensus_score",
            "b16_conflict_score",
            "b16_ensemble_lock",
            "b16_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 16)
        self.assertEqual(first["b_parent_level"], 15)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B16_OPTION_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b16_controller_profile"], "option_ensemble")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b17_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b16_option_ensemble_source(tmpdir)
            config = build_b17_neuromodulated_ensemble_config(
                B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=59,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b16_controller_profile",
            "b17_controller_profile",
            "b17_modulator_state",
            "b17_arousal_signal",
            "b17_homeostatic_gain",
            "b17_option_gain",
            "b17_conflict_release",
            "b17_modulation_lock",
            "b17_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 17)
        self.assertEqual(first["b_parent_level"], 16)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b17_controller_profile"], "neuromodulated_ensemble")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b18_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b17_neuromodulated_ensemble_source(tmpdir)
            config = build_b18_eligibility_trace_config(
                B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=60,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b17_controller_profile",
            "b18_controller_profile",
            "b18_trace_state",
            "b18_eligibility_trace",
            "b18_reward_prediction_proxy",
            "b18_stability_bias",
            "b18_switch_pressure",
            "b18_trace_lock",
            "b18_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 18)
        self.assertEqual(first["b_parent_level"], 17)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B18_ELIGIBILITY_TRACE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b18_controller_profile"], "eligibility_trace")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b19_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b18_eligibility_trace_source(tmpdir)
            config = build_b19_episodic_meta_memory_config(
                B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=61,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b18_controller_profile",
            "b19_controller_profile",
            "b19_memory_state",
            "b19_episode_memory",
            "b19_consolidation_score",
            "b19_stability_vote",
            "b19_switch_suppression",
            "b19_memory_lock",
            "b19_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 19)
        self.assertEqual(first["b_parent_level"], 18)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B19_EPISODIC_META_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b19_controller_profile"], "episodic_meta_memory")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b20_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b19_episodic_meta_memory_source(tmpdir)
            config = build_b20_working_memory_gate_config(
                B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=62,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b19_controller_profile",
            "b20_controller_profile",
            "b20_buffer_state",
            "b20_working_buffer",
            "b20_context_binding",
            "b20_gate_vote",
            "b20_release_vote",
            "b20_buffer_lock",
            "b20_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 20)
        self.assertEqual(first["b_parent_level"], 19)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B20_WORKING_MEMORY_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b20_controller_profile"], "working_memory_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b21_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b20_working_memory_gate_source(tmpdir)
            config = build_b21_hippocampal_replay_config(
                B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=63,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b20_controller_profile",
            "b21_controller_profile",
            "b21_replay_state",
            "b21_sequence_memory",
            "b21_replay_score",
            "b21_route_commitment",
            "b21_abort_prediction",
            "b21_replay_lock",
            "b21_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 21)
        self.assertEqual(first["b_parent_level"], 20)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b21_controller_profile"], "hippocampal_replay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b22_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b21_hippocampal_replay_source(tmpdir)
            config = build_b22_prospective_replay_config(
                B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=64,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b21_controller_profile",
            "b22_controller_profile",
            "b22_sim_state",
            "b22_prospective_sim",
            "b22_forward_model_score",
            "b22_viability_projection",
            "b22_abort_projection",
            "b22_sim_lock",
            "b22_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 22)
        self.assertEqual(first["b_parent_level"], 21)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b22_controller_profile"], "prospective_map_replay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b23_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b22_prospective_replay_source(tmpdir)
            config = build_b23_conflict_monitor_config(
                B23_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=65,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b22_controller_profile",
            "b23_controller_profile",
            "b23_conflict_state",
            "b23_prediction_error",
            "b23_conflict_memory",
            "b23_stability_vote",
            "b23_abort_bias",
            "b23_monitor_lock",
            "b23_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 23)
        self.assertEqual(first["b_parent_level"], 22)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B23_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b23_controller_profile"], "conflict_monitor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b24_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b23_conflict_monitor_source(tmpdir)
            config = build_b24_precision_conflict_config(
                B24_PRECISION_CONFLICT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=66,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b23_controller_profile",
            "b24_controller_profile",
            "b24_precision_state",
            "b24_precision_memory",
            "b24_precision_vote",
            "b24_uncertainty_pressure",
            "b24_abort_precision",
            "b24_precision_lock",
            "b24_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 24)
        self.assertEqual(first["b_parent_level"], 23)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B24_PRECISION_CONFLICT_SELECTION_SOURCE,
        )
        self.assertEqual(first["b24_controller_profile"], "precision_conflict")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b25_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b24_precision_conflict_source(tmpdir)
            config = build_b25_metacognitive_confidence_config(
                B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=67,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b24_controller_profile",
            "b25_controller_profile",
            "b25_metacognitive_state",
            "b25_confidence_memory",
            "b25_confidence_vote",
            "b25_doubt_pressure",
            "b25_control_gain",
            "b25_meta_lock",
            "b25_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 25)
        self.assertEqual(first["b_parent_level"], 24)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b25_controller_profile"], "metacognitive_confidence")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b26_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b25_metacognitive_confidence_source(tmpdir)
            config = build_b26_allostatic_prediction_config(
                B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=68,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b25_controller_profile",
            "b26_controller_profile",
            "b26_allostatic_state",
            "b26_prediction_error",
            "b26_setpoint_pressure",
            "b26_control_vote",
            "b26_stability_lock",
            "b26_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 26)
        self.assertEqual(first["b_parent_level"], 25)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b26_controller_profile"], "allostatic_prediction")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b27_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b26_allostatic_prediction_source(tmpdir)
            config = build_b27_arousal_gain_config(
                B27_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=69,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b26_controller_profile",
            "b27_controller_profile",
            "b27_arousal_state",
            "b27_arousal_level",
            "b27_gain_modulation",
            "b27_stress_pressure",
            "b27_arousal_lock",
            "b27_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 27)
        self.assertEqual(first["b_parent_level"], 26)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B27_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(first["b27_controller_profile"], "arousal_gain")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b28_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b27_arousal_gain_source(tmpdir)
            config = build_b28_interoceptive_attention_config(
                B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=70,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b27_controller_profile",
            "b28_controller_profile",
            "b28_attention_state",
            "b28_interoceptive_focus",
            "b28_attention_gain",
            "b28_distractor_pressure",
            "b28_attention_lock",
            "b28_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 28)
        self.assertEqual(first["b_parent_level"], 27)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b28_controller_profile"], "interoceptive_attention")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b29_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b28_interoceptive_attention_source(tmpdir)
            config = build_b29_salience_competition_config(
                B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=71,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b28_controller_profile",
            "b29_controller_profile",
            "b29_salience_state",
            "b29_threat_salience",
            "b29_homeostatic_salience",
            "b29_corridor_salience",
            "b29_winner_channel",
            "b29_salience_lock",
            "b29_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 29)
        self.assertEqual(first["b_parent_level"], 28)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B29_SALIENCE_COMPETITION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b29_controller_profile"], "salience_competition")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b30_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b29_salience_competition_source(tmpdir)
            config = build_b30_basal_ganglia_gate_config(
                B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=72,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b29_controller_profile",
            "b30_controller_profile",
            "b30_gate_state",
            "b30_go_signal",
            "b30_no_go_signal",
            "b30_action_gate",
            "b30_gate_lock",
            "b30_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 30)
        self.assertEqual(first["b_parent_level"], 29)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b30_controller_profile"], "basal_ganglia_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b31_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b30_basal_ganglia_gate_source(tmpdir)
            config = build_b31_dopamine_prediction_error_config(
                B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=73,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b30_controller_profile",
            "b31_controller_profile",
            "b31_dopamine_state",
            "b31_reward_prediction_error",
            "b31_tonic_dopamine",
            "b31_phasic_dopamine",
            "b31_gate_bias",
            "b31_dopamine_lock",
            "b31_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 31)
        self.assertEqual(first["b_parent_level"], 30)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b31_controller_profile"], "dopamine_prediction_error")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b32_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b31_dopamine_prediction_error_source(tmpdir)
            config = build_b32_actor_critic_value_config(
                B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=74,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b31_controller_profile",
            "b32_controller_profile",
            "b32_critic_value",
            "b32_actor_advantage",
            "b32_value_error",
            "b32_policy_bias",
            "b32_value_lock",
            "b32_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 32)
        self.assertEqual(first["b_parent_level"], 31)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b32_controller_profile"], "actor_critic_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b33_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b32_actor_critic_value_source(tmpdir)
            config = build_b33_td_error_decomposition_config(
                B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=75,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b32_controller_profile",
            "b33_controller_profile",
            "b33_td_error",
            "b33_bootstrap_value",
            "b33_reward_trace",
            "b33_actor_update",
            "b33_td_lock",
            "b33_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 33)
        self.assertEqual(first["b_parent_level"], 32)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b33_controller_profile"], "td_error_decomposition")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b34_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b33_td_error_decomposition_source(tmpdir)
            config = build_b34_eligibility_credit_config(
                B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=76,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b33_controller_profile",
            "b34_controller_profile",
            "b34_eligibility_trace",
            "b34_credit_assignment",
            "b34_synaptic_tag",
            "b34_decay_memory",
            "b34_credit_lock",
            "b34_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 34)
        self.assertEqual(first["b_parent_level"], 33)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE,
        )
        self.assertEqual(first["b34_controller_profile"], "eligibility_credit")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b35_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b34_eligibility_credit_source(tmpdir)
            config = build_b35_forward_model_value_config(
                B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=77,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b34_controller_profile",
            "b35_controller_profile",
            "b35_forward_value",
            "b35_transition_error",
            "b35_model_confidence",
            "b35_prediction_memory",
            "b35_model_lock",
            "b35_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 35)
        self.assertEqual(first["b_parent_level"], 34)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b35_controller_profile"], "forward_model_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b36_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b35_forward_model_value_source(tmpdir)
            config = build_b36_latent_belief_state_config(
                B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=78,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b35_controller_profile",
            "b36_controller_profile",
            "b36_latent_state",
            "b36_belief_error",
            "b36_state_confidence",
            "b36_context_memory",
            "b36_belief_lock",
            "b36_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 36)
        self.assertEqual(first["b_parent_level"], 35)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B36_LATENT_BELIEF_STATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b36_controller_profile"], "latent_belief_state")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)
