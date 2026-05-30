from __future__ import annotations

from .shared import *



class BSeriesCheckpointTestPart4(unittest.TestCase):
    def test_b37_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b36_latent_belief_state_source(tmpdir)
            config = build_b37_state_factor_gate_config(
                B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=79,
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
            "b36_controller_profile",
            "b37_controller_profile",
            "b37_external_state_factor",
            "b37_internal_state_factor",
            "b37_factor_alignment",
            "b37_factor_confidence",
            "b37_factor_lock",
            "b37_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 37)
        self.assertEqual(first["b_parent_level"], 36)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B37_STATE_FACTOR_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b37_controller_profile"], "state_factor_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b38_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b37_state_factor_gate_source(tmpdir)
            config = build_b38_factor_attention_config(
                B38_FACTOR_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=80,
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
            "b37_controller_profile",
            "b38_controller_profile",
            "b38_external_attention",
            "b38_internal_attention",
            "b38_attention_balance",
            "b38_attention_gain",
            "b38_attention_lock",
            "b38_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 38)
        self.assertEqual(first["b_parent_level"], 37)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B38_FACTOR_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b38_controller_profile"], "factor_attention")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b39_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b38_factor_attention_source(tmpdir)
            config = build_b39_attention_binding_config(
                B39_ATTENTION_BINDING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=81,
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
            "b38_controller_profile",
            "b39_controller_profile",
            "b39_binding_strength",
            "b39_cross_factor_coherence",
            "b39_bound_context",
            "b39_binding_gain",
            "b39_binding_lock",
            "b39_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 39)
        self.assertEqual(first["b_parent_level"], 38)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B39_ATTENTION_BINDING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b39_controller_profile"], "attention_binding")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b40_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b39_attention_binding_source(tmpdir)
            config = build_b40_global_workspace_config(
                B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=82,
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
            "b39_controller_profile",
            "b40_controller_profile",
            "b40_workspace_activation",
            "b40_broadcast_gain",
            "b40_context_availability",
            "b40_workspace_stability",
            "b40_workspace_lock",
            "b40_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 40)
        self.assertEqual(first["b_parent_level"], 39)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B40_GLOBAL_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b40_controller_profile"], "global_workspace")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b41_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b40_global_workspace_source(tmpdir)
            config = build_b41_executive_workspace_config(
                B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=83,
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
            "b40_controller_profile",
            "b41_controller_profile",
            "b41_executive_selection",
            "b41_inhibitory_pressure",
            "b41_goal_context",
            "b41_executive_stability",
            "b41_executive_lock",
            "b41_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 41)
        self.assertEqual(first["b_parent_level"], 40)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b41_controller_profile"], "executive_workspace")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b42_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b41_executive_workspace_source(tmpdir)
            config = build_b42_error_monitor_config(
                B42_ERROR_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=84,
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
            "b41_controller_profile",
            "b42_controller_profile",
            "b42_error_signal",
            "b42_conflict_signal",
            "b42_performance_context",
            "b42_monitor_stability",
            "b42_monitor_lock",
            "b42_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 42)
        self.assertEqual(first["b_parent_level"], 41)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B42_ERROR_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b42_controller_profile"], "error_monitor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b43_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b42_error_monitor_source(tmpdir)
            config = build_b43_adaptive_precision_config(
                B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=85,
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
            "b42_controller_profile",
            "b43_controller_profile",
            "b43_precision_signal",
            "b43_adaptive_threshold",
            "b43_arousal_context",
            "b43_control_stability",
            "b43_precision_lock",
            "b43_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 43)
        self.assertEqual(first["b_parent_level"], 42)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B43_ADAPTIVE_PRECISION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b43_controller_profile"], "adaptive_precision")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b44_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b43_adaptive_precision_source(tmpdir)
            config = build_b44_thalamic_relay_config(
                B44_THALAMIC_RELAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=86,
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
            "b43_controller_profile",
            "b44_controller_profile",
            "b44_relay_gate",
            "b44_sensory_precision",
            "b44_context_relay",
            "b44_gate_stability",
            "b44_relay_lock",
            "b44_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 44)
        self.assertEqual(first["b_parent_level"], 43)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B44_THALAMIC_RELAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b44_controller_profile"], "thalamic_relay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b45_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b44_thalamic_relay_source(tmpdir)
            config = build_b45_reticular_inhibition_config(
                B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=87,
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
            "b44_controller_profile",
            "b45_controller_profile",
            "b45_inhibitory_gate",
            "b45_sensory_filter",
            "b45_context_suppression",
            "b45_loop_stability",
            "b45_inhibition_lock",
            "b45_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 45)
        self.assertEqual(first["b_parent_level"], 44)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B45_RETICULAR_INHIBITION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b45_controller_profile"], "reticular_inhibition")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b46_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b45_reticular_inhibition_source(tmpdir)
            config = build_b46_corticothalamic_feedback_config(
                B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=88,
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
            "b45_controller_profile",
            "b46_controller_profile",
            "b46_feedback_gain",
            "b46_topdown_context",
            "b46_prediction_match",
            "b46_feedback_stability",
            "b46_feedback_lock",
            "b46_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 46)
        self.assertEqual(first["b_parent_level"], 45)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE,
        )
        self.assertEqual(first["b46_controller_profile"], "corticothalamic_feedback")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b47_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b46_corticothalamic_feedback_source(tmpdir)
            config = build_b47_oscillatory_synchrony_config(
                B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=89,
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
            "b46_controller_profile",
            "b47_controller_profile",
            "b47_phase_alignment",
            "b47_synchrony_gain",
            "b47_cross_loop_coherence",
            "b47_phase_lock",
            "b47_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 47)
        self.assertEqual(first["b_parent_level"], 46)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b47_controller_profile"], "oscillatory_synchrony")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b48_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b47_oscillatory_synchrony_source(tmpdir)
            config = build_b48_cerebellar_timing_config(
                B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=90,
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
            "b47_controller_profile",
            "b48_controller_profile",
            "b48_timing_error",
            "b48_predictive_timing",
            "b48_corrective_gain",
            "b48_calibration_lock",
            "b48_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 48)
        self.assertEqual(first["b_parent_level"], 47)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B48_CEREBELLAR_TIMING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b48_controller_profile"], "cerebellar_timing")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b49_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b48_cerebellar_timing_source(tmpdir)
            config = build_b49_striatal_action_gate_config(
                B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=91,
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
            "b48_controller_profile",
            "b49_controller_profile",
            "b49_go_signal",
            "b49_no_go_signal",
            "b49_action_gate_balance",
            "b49_selection_lock",
            "b49_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 49)
        self.assertEqual(first["b_parent_level"], 48)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b49_controller_profile"], "striatal_action_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b50_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b49_striatal_action_gate_source(tmpdir)
            config = build_b50_habit_chunking_config(
                B50_HABIT_CHUNKING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=92,
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
            "b49_controller_profile",
            "b50_controller_profile",
            "b50_habit_strength",
            "b50_chunk_value",
            "b50_habit_stability",
            "b50_chunk_lock",
            "b50_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 50)
        self.assertEqual(first["b_parent_level"], 49)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B50_HABIT_CHUNKING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b50_controller_profile"], "habit_chunking")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b51_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b50_habit_chunking_source(tmpdir)
            config = build_b51_dopaminergic_habit_modulation_config(
                B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=93,
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
            "b50_controller_profile",
            "b51_controller_profile",
            "b51_prediction_error",
            "b51_dopamine_gain",
            "b51_habit_modulation",
            "b51_modulation_lock",
            "b51_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 51)
        self.assertEqual(first["b_parent_level"], 50)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE,
        )
        self.assertEqual(
            first["b51_controller_profile"],
            "dopaminergic_habit_modulation",
        )
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b52_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b51_dopaminergic_habit_source(tmpdir)
            config = build_b52_cholinergic_precision_gate_config(
                B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=94,
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
            "b51_controller_profile",
            "b52_controller_profile",
            "b52_acetylcholine_level",
            "b52_precision_gain",
            "b52_uncertainty_signal",
            "b52_attention_lock",
            "b52_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 52)
        self.assertEqual(first["b_parent_level"], 51)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b52_controller_profile"], "cholinergic_precision_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b53_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b52_cholinergic_precision_source(tmpdir)
            config = build_b53_noradrenergic_arousal_gain_config(
                B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=95,
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
            "b52_controller_profile",
            "b53_controller_profile",
            "b53_norepinephrine_level",
            "b53_arousal_gain",
            "b53_surprise_signal",
            "b53_gain_lock",
            "b53_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 53)
        self.assertEqual(first["b_parent_level"], 52)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(first["b53_controller_profile"], "noradrenergic_arousal_gain")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b54_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b53_noradrenergic_arousal_source(tmpdir)
            config = build_b54_serotonergic_patience_gate_config(
                B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=96,
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
            "b53_controller_profile",
            "b54_controller_profile",
            "b54_serotonin_level",
            "b54_patience_signal",
            "b54_impulse_suppression",
            "b54_patience_lock",
            "b54_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 54)
        self.assertEqual(first["b_parent_level"], 53)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b54_controller_profile"], "serotonergic_patience_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b55_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b54_serotonergic_patience_source(tmpdir)
            config = build_b55_hypothalamic_drive_coupling_config(
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=97,
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
            "b54_controller_profile",
            "b55_controller_profile",
            "b55_hypothalamic_drive",
            "b55_satiety_signal",
            "b55_recovery_bias",
            "b55_drive_balance",
            "b55_drive_lock",
            "b55_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 55)
        self.assertEqual(first["b_parent_level"], 54)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b55_controller_profile"], "hypothalamic_drive_coupling")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b56_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b55_hypothalamic_drive_source(tmpdir)
            config = build_b56_hpa_stress_axis_config(
                B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=98,
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
            "b55_controller_profile",
            "b56_controller_profile",
            "b56_cortisol_level",
            "b56_stress_load",
            "b56_recovery_signal",
            "b56_endocrine_balance",
            "b56_stress_lock",
            "b56_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 56)
        self.assertEqual(first["b_parent_level"], 55)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B56_HPA_STRESS_AXIS_SELECTION_SOURCE,
        )
        self.assertEqual(first["b56_controller_profile"], "hpa_stress_axis")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b57_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b56_hpa_stress_source(tmpdir)
            config = build_b57_insular_interoceptive_awareness_config(
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=99,
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
            "b56_controller_profile",
            "b57_controller_profile",
            "b57_interoceptive_awareness",
            "b57_visceral_salience",
            "b57_body_state_confidence",
            "b57_awareness_balance",
            "b57_awareness_lock",
            "b57_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 57)
        self.assertEqual(first["b_parent_level"], 56)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE,
        )
        self.assertEqual(
            first["b57_controller_profile"],
            "insular_interoceptive_awareness",
        )
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b58_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b57_insular_interoceptive_source(tmpdir)
            config = build_b58_acc_conflict_monitor_config(
                B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=100,
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
            "b57_controller_profile",
            "b58_controller_profile",
            "b58_conflict_signal",
            "b58_error_likelihood",
            "b58_control_allocation",
            "b58_resolution_balance",
            "b58_conflict_lock",
            "b58_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 58)
        self.assertEqual(first["b_parent_level"], 57)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b58_controller_profile"], "acc_conflict_monitor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b59_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b58_acc_conflict_source(tmpdir)
            config = build_b59_prefrontal_goal_context_config(
                B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=101,
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
            "b58_controller_profile",
            "b59_controller_profile",
            "b59_goal_context",
            "b59_working_set_stability",
            "b59_task_set_confidence",
            "b59_executive_balance",
            "b59_executive_lock",
            "b59_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 59)
        self.assertEqual(first["b_parent_level"], 58)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE,
        )
        self.assertEqual(first["b59_controller_profile"], "prefrontal_goal_context")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b60_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b59_prefrontal_goal_source(tmpdir)
            config = build_b60_orbitofrontal_outcome_value_config(
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=102,
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
            "b59_controller_profile",
            "b60_controller_profile",
            "b60_outcome_value",
            "b60_reversal_signal",
            "b60_goal_value_confidence",
            "b60_value_balance",
            "b60_value_lock",
            "b60_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 60)
        self.assertEqual(first["b_parent_level"], 59)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b60_controller_profile"], "orbitofrontal_outcome_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b61_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b60_orbitofrontal_value_source(tmpdir)
            config = build_b61_amygdala_safety_value_config(
                B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=103,
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
            "b60_controller_profile",
            "b61_controller_profile",
            "b61_safety_value",
            "b61_threat_value",
            "b61_safety_confidence",
            "b61_affective_balance",
            "b61_safety_lock",
            "b61_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 61)
        self.assertEqual(first["b_parent_level"], 60)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b61_controller_profile"], "amygdala_safety_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b62_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b61_amygdala_safety_source(tmpdir)
            config = build_b62_defensive_mode_selector_config(
                B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=104,
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
            "b61_controller_profile",
            "b62_controller_profile",
            "b62_defensive_mode",
            "b62_freeze_pressure",
            "b62_flee_pressure",
            "b62_shelter_bias",
            "b62_defense_balance",
            "b62_defense_lock",
            "b62_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 62)
        self.assertEqual(first["b_parent_level"], 61)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b62_controller_profile"], "defensive_mode_selector")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)
