from __future__ import annotations

from .shared import *


class BSeriesCheckpointTestPart5(unittest.TestCase):
    def test_b78_trace_fields_and_primitive_contract(self) -> None:
        build_b78 = getattr(
            b_series_evolution_module,
            "build_b78_vestibular_balance_config",
            None,
        )
        self.assertIsNotNone(build_b78)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b77_olivary_error_source(tmpdir)
            config = build_b78(
                "b78_vestibular_balance_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=120,
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
            "b77_controller_profile",
            "b78_controller_profile",
            "b78_balance_error",
            "b78_head_stabilization",
            "b78_locomotor_confidence",
            "b78_slip_risk",
            "b78_balance_lock",
            "b78_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 78)
        self.assertEqual(first["b_parent_level"], 77)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            "b78_vestibular_balance_controller",
        )
        self.assertEqual(first["b78_controller_profile"], "vestibular_balance")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b77_trace_fields_and_primitive_contract(self) -> None:
        build_b77 = getattr(
            b_series_evolution_module,
            "build_b77_olivary_error_config",
            None,
        )
        self.assertIsNotNone(build_b77)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b76_cerebellar_stride_source(tmpdir)
            config = build_b77(
                "b77_olivary_error_correction_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=119,
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
            "b76_controller_profile",
            "b77_controller_profile",
            "b77_prediction_error",
            "b77_climbing_fiber_drive",
            "b77_corrective_timing",
            "b77_stability_confidence",
            "b77_error_lock",
            "b77_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 77)
        self.assertEqual(first["b_parent_level"], 76)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            "b77_olivary_error_controller",
        )
        self.assertEqual(first["b77_controller_profile"], "olivary_error_correction")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b76_trace_fields_and_primitive_contract(self) -> None:
        build_b76 = getattr(
            b_series_evolution_module,
            "build_b76_cerebellar_stride_config",
            None,
        )
        self.assertIsNotNone(build_b76)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b75_basal_thalamic_release_source(tmpdir)
            config = build_b76(
                "b76_cerebellar_stride_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=118,
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
            "b75_controller_profile",
            "b76_controller_profile",
            "b76_stride_timing_signal",
            "b76_error_correction",
            "b76_burst_smoothing",
            "b76_stride_gate",
            "b76_stride_lock",
            "b76_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 76)
        self.assertEqual(first["b_parent_level"], 75)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            "b76_cerebellar_stride_controller",
        )
        self.assertEqual(first["b76_controller_profile"], "cerebellar_stride_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b75_trace_fields_and_primitive_contract(self) -> None:
        build_b75 = getattr(
            b_series_evolution_module,
            "build_b75_basal_thalamic_release_config",
            None,
        )
        self.assertIsNotNone(build_b75)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b74_thalamic_rebound_source(tmpdir)
            config = build_b75(
                "b75_basal_thalamic_release_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=117,
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
            "b74_controller_profile",
            "b75_controller_profile",
            "b75_release_timing_drive",
            "b75_go_disinhibition",
            "b75_nogo_brake",
            "b75_burst_window",
            "b75_release_lock",
            "b75_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 75)
        self.assertEqual(first["b_parent_level"], 74)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            "b75_basal_thalamic_release_controller",
        )
        self.assertEqual(first["b75_controller_profile"], "basal_thalamic_release")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b74_trace_fields_and_primitive_contract(self) -> None:
        build_b74 = getattr(
            b_series_evolution_module,
            "build_b74_thalamic_rebound_config",
            None,
        )
        self.assertIsNotNone(build_b74)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b73_reticular_inhibition_source(tmpdir)
            config = build_b74(
                "b74_thalamic_rebound_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=116,
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
            "b73_controller_profile",
            "b74_controller_profile",
            "b74_rebound_potential",
            "b74_inhibition_aftereffect",
            "b74_release_window",
            "b74_rebound_lock",
            "b74_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 74)
        self.assertEqual(first["b_parent_level"], 73)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            "b74_thalamic_rebound_controller",
        )
        self.assertEqual(first["b74_controller_profile"], "thalamic_rebound_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b73_trace_fields_and_primitive_contract(self) -> None:
        build_b73 = getattr(
            b_series_evolution_module,
            "build_b73_reticular_inhibition_config",
            None,
        )
        self.assertIsNotNone(build_b73)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b72_pulvinar_attention_source(tmpdir)
            config = build_b73(
                "b73_reticular_inhibition_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=115,
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
            "b72_controller_profile",
            "b73_controller_profile",
            "b73_inhibitory_tone",
            "b73_surround_suppression",
            "b73_release_drive",
            "b73_inhibition_lock",
            "b73_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 73)
        self.assertEqual(first["b_parent_level"], 72)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            "b73_reticular_inhibition_controller",
        )
        self.assertEqual(first["b73_controller_profile"], "reticular_inhibition_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)
