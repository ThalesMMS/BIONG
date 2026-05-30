from __future__ import annotations

from .shared import *



class BSeriesCheckpointTestPart2(unittest.TestCase):
    def test_b30_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b29_salience_competition_source(tmpdir)
            variants = (
                (B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME, 1.0),
                (B30_GO_NOGO_BALANCE_H48_POLICY_NAME, 1.0),
                (B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME, 1.0),
                (B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME, 0.85),
                (B30_GENETIC_ACTION_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b30_basal_ganglia_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=187 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 30)
                self.assertEqual(report["parent_level"], 29)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b31_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b30_basal_ganglia_gate_source(tmpdir)
            variants = (
                (B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME, 1.0),
                (B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME, 1.0),
                (B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME, 1.0),
                (B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME, 0.85),
                (B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b31_dopamine_prediction_error_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=192 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 31)
                self.assertEqual(report["parent_level"], 30)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b32_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b31_dopamine_prediction_error_source(tmpdir)
            variants = (
                (B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME, 1.0),
                (B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME, 1.0),
                (B32_CRITIC_STABILITY_H48_POLICY_NAME, 1.0),
                (B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME, 0.85),
                (B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b32_actor_critic_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=197 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 32)
                self.assertEqual(report["parent_level"], 31)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b33_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b32_actor_critic_value_source(tmpdir)
            variants = (
                (B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME, 1.0),
                (B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME, 1.0),
                (B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME, 1.0),
                (B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME, 0.85),
                (B33_GENETIC_TD_VALUE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b33_td_error_decomposition_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=202 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 33)
                self.assertEqual(report["parent_level"], 32)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b34_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b33_td_error_decomposition_source(tmpdir)
            variants = (
                (B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME, 1.0),
                (B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME, 1.0),
                (B34_SYNAPTIC_TAGGING_H48_POLICY_NAME, 1.0),
                (B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME, 0.85),
                (B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b34_eligibility_credit_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=207 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 34)
                self.assertEqual(report["parent_level"], 33)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b35_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b34_eligibility_credit_source(tmpdir)
            variants = (
                (B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME, 1.0),
                (B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME, 1.0),
                (B35_MODEL_CONFIDENCE_H48_POLICY_NAME, 1.0),
                (B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME, 0.85),
                (B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b35_forward_model_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=212 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 35)
                self.assertEqual(report["parent_level"], 34)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b36_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b35_forward_model_value_source(tmpdir)
            variants = (
                (B36_LATENT_BELIEF_STATE_H48_POLICY_NAME, 1.0),
                (B36_BELIEF_ERROR_GATE_H48_POLICY_NAME, 1.0),
                (B36_CONTEXT_INFERENCE_H48_POLICY_NAME, 1.0),
                (B36_LATENT_BELIEF_STATE_H56_POLICY_NAME, 0.85),
                (B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b36_latent_belief_state_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=217 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 36)
                self.assertEqual(report["parent_level"], 35)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b37_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b36_latent_belief_state_source(tmpdir)
            variants = (
                (B37_STATE_FACTOR_GATE_H48_POLICY_NAME, 1.0),
                (B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME, 1.0),
                (B37_FACTOR_CONFIDENCE_H48_POLICY_NAME, 1.0),
                (B37_STATE_FACTOR_GATE_H56_POLICY_NAME, 0.85),
                (B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b37_state_factor_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=222 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 37)
                self.assertEqual(report["parent_level"], 36)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b38_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b37_state_factor_gate_source(tmpdir)
            variants = (
                (B38_FACTOR_ATTENTION_H48_POLICY_NAME, 1.0),
                (B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B38_FACTOR_ATTENTION_H56_POLICY_NAME, 0.85),
                (B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b38_factor_attention_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=227 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 38)
                self.assertEqual(report["parent_level"], 37)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b39_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b38_factor_attention_source(tmpdir)
            variants = (
                (B39_ATTENTION_BINDING_H48_POLICY_NAME, 1.0),
                (B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME, 1.0),
                (B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME, 1.0),
                (B39_ATTENTION_BINDING_H56_POLICY_NAME, 0.85),
                (B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b39_attention_binding_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=232 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 39)
                self.assertEqual(report["parent_level"], 38)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b40_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b39_attention_binding_source(tmpdir)
            variants = (
                (B40_GLOBAL_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B40_SENSORY_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B40_CONTEXT_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B40_GLOBAL_WORKSPACE_H56_POLICY_NAME, 0.85),
                (B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b40_global_workspace_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=237 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 40)
                self.assertEqual(report["parent_level"], 39)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b41_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b40_global_workspace_source(tmpdir)
            variants = (
                (B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B41_INHIBITORY_CONTROL_H48_POLICY_NAME, 1.0),
                (B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME, 1.0),
                (B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME, 0.85),
                (B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b41_executive_workspace_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=242 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 41)
                self.assertEqual(report["parent_level"], 40)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b42_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b41_executive_workspace_source(tmpdir)
            variants = (
                (B42_ERROR_MONITOR_H48_POLICY_NAME, 1.0),
                (B42_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
                (B42_PERFORMANCE_MONITOR_H48_POLICY_NAME, 1.0),
                (B42_ERROR_MONITOR_H56_POLICY_NAME, 0.85),
                (B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b42_error_monitor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=247 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 42)
                self.assertEqual(report["parent_level"], 41)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b43_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b42_error_monitor_source(tmpdir)
            variants = (
                (B43_ADAPTIVE_PRECISION_H48_POLICY_NAME, 1.0),
                (B43_AROUSAL_PRECISION_H48_POLICY_NAME, 1.0),
                (B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME, 1.0),
                (B43_ADAPTIVE_PRECISION_H56_POLICY_NAME, 0.85),
                (B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b43_adaptive_precision_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=252 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 43)
                self.assertEqual(report["parent_level"], 42)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b44_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b43_adaptive_precision_source(tmpdir)
            variants = (
                (B44_THALAMIC_RELAY_H48_POLICY_NAME, 1.0),
                (B44_SENSORY_RELAY_H48_POLICY_NAME, 1.0),
                (B44_CONTEXT_RELAY_H48_POLICY_NAME, 1.0),
                (B44_THALAMIC_RELAY_H56_POLICY_NAME, 0.85),
                (B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b44_thalamic_relay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=257 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 44)
                self.assertEqual(report["parent_level"], 43)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b45_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b44_thalamic_relay_source(tmpdir)
            variants = (
                (B45_RETICULAR_INHIBITION_H48_POLICY_NAME, 1.0),
                (B45_SENSORY_INHIBITION_H48_POLICY_NAME, 1.0),
                (B45_CONTEXT_INHIBITION_H48_POLICY_NAME, 1.0),
                (B45_RETICULAR_INHIBITION_H56_POLICY_NAME, 0.85),
                (B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b45_reticular_inhibition_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=262 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 45)
                self.assertEqual(report["parent_level"], 44)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b46_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b45_reticular_inhibition_source(tmpdir)
            variants = (
                (B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME, 1.0),
                (B46_FEEDBACK_GAIN_H48_POLICY_NAME, 1.0),
                (B46_CONTEXT_FEEDBACK_H48_POLICY_NAME, 1.0),
                (B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME, 0.85),
                (B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b46_corticothalamic_feedback_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=267 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 46)
                self.assertEqual(report["parent_level"], 45)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b47_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b46_corticothalamic_feedback_source(tmpdir)
            variants = (
                (B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME, 1.0),
                (B47_PHASE_LOCKING_H48_POLICY_NAME, 1.0),
                (B47_COHERENCE_GATE_H48_POLICY_NAME, 1.0),
                (B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME, 0.85),
                (B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b47_oscillatory_synchrony_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=272 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 47)
                self.assertEqual(report["parent_level"], 46)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b48_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b47_oscillatory_synchrony_source(tmpdir)
            variants = (
                (B48_CEREBELLAR_TIMING_H48_POLICY_NAME, 1.0),
                (B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME, 1.0),
                (B48_PREDICTIVE_TIMING_H48_POLICY_NAME, 1.0),
                (B48_CEREBELLAR_TIMING_H56_POLICY_NAME, 0.85),
                (B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b48_cerebellar_timing_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=277 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 48)
                self.assertEqual(report["parent_level"], 47)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b49_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b48_cerebellar_timing_source(tmpdir)
            variants = (
                (B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME, 1.0),
                (B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME, 1.0),
                (B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME, 0.85),
                (B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b49_striatal_action_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=282 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 49)
                self.assertEqual(report["parent_level"], 48)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b50_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b49_striatal_action_gate_source(tmpdir)
            variants = (
                (B50_HABIT_CHUNKING_H48_POLICY_NAME, 1.0),
                (B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME, 1.0),
                (B50_HABIT_STABILITY_H48_POLICY_NAME, 1.0),
                (B50_HABIT_CHUNKING_H56_POLICY_NAME, 0.85),
                (B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b50_habit_chunking_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=287 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 50)
                self.assertEqual(report["parent_level"], 49)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b51_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b50_habit_chunking_source(tmpdir)
            variants = (
                (B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME, 1.0),
                (B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME, 1.0),
                (B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME, 1.0),
                (B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME, 0.85),
                (B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b51_dopaminergic_habit_modulation_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=292 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 51)
                self.assertEqual(report["parent_level"], 50)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b52_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b51_dopaminergic_habit_source(tmpdir)
            variants = (
                (B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME, 1.0),
                (B52_ATTENTION_GAIN_H48_POLICY_NAME, 1.0),
                (B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME, 1.0),
                (B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME, 0.85),
                (B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b52_cholinergic_precision_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=297 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 52)
                self.assertEqual(report["parent_level"], 51)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b53_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b52_cholinergic_precision_source(tmpdir)
            variants = (
                (B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME, 1.0),
                (B53_SURPRISE_GAIN_H48_POLICY_NAME, 1.0),
                (B53_STRESS_PRECISION_H48_POLICY_NAME, 1.0),
                (B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME, 0.85),
                (B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b53_noradrenergic_arousal_gain_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=302 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 53)
                self.assertEqual(report["parent_level"], 52)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b54_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b53_noradrenergic_arousal_source(tmpdir)
            variants = (
                (B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B54_PATIENCE_BALANCE_H48_POLICY_NAME, 1.0),
                (B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME, 0.85),
                (B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b54_serotonergic_patience_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=307 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 54)
                self.assertEqual(report["parent_level"], 53)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b55_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b54_serotonergic_patience_source(tmpdir)
            variants = (
                (B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME, 1.0),
                (B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME, 1.0),
                (B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME, 1.0),
                (B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME, 0.85),
                (B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b55_hypothalamic_drive_coupling_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=312 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 55)
                self.assertEqual(report["parent_level"], 54)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b56_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b55_hypothalamic_drive_source(tmpdir)
            variants = (
                (B56_HPA_STRESS_AXIS_H48_POLICY_NAME, 1.0),
                (B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME, 1.0),
                (B56_STRESS_LOAD_GATE_H48_POLICY_NAME, 1.0),
                (B56_HPA_STRESS_AXIS_H56_POLICY_NAME, 0.85),
                (B56_GENETIC_HPA_STRESS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b56_hpa_stress_axis_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=317 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 56)
                self.assertEqual(report["parent_level"], 55)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b57_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b56_hpa_stress_source(tmpdir)
            variants = (
                (B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME, 1.0),
                (B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME, 1.0),
                (B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME, 0.85),
                (B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b57_insular_interoceptive_awareness_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=322 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 57)
                self.assertEqual(report["parent_level"], 56)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b58_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b57_insular_interoceptive_source(tmpdir)
            variants = (
                (B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
                (B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME, 1.0),
                (B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME, 0.85),
                (B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b58_acc_conflict_monitor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=327 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 58)
                self.assertEqual(report["parent_level"], 57)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b59_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b58_acc_conflict_source(tmpdir)
            variants = (
                (B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME, 1.0),
                (B59_WORKING_SET_STABILITY_H48_POLICY_NAME, 1.0),
                (B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME, 1.0),
                (B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME, 0.85),
                (B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b59_prefrontal_goal_context_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=332 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 59)
                self.assertEqual(report["parent_level"], 58)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b60_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b59_prefrontal_goal_source(tmpdir)
            variants = (
                (B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME, 1.0),
                (B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME, 1.0),
                (B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME, 1.0),
                (B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME, 0.85),
                (B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b60_orbitofrontal_outcome_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=337 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 60)
                self.assertEqual(report["parent_level"], 59)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b61_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b60_orbitofrontal_value_source(tmpdir)
            variants = (
                (B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME, 1.0),
                (B61_THREAT_VALUE_TAG_H48_POLICY_NAME, 1.0),
                (B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME, 1.0),
                (B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME, 0.85),
                (B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b61_amygdala_safety_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=342 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 61)
                self.assertEqual(report["parent_level"], 60)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b62_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b61_amygdala_safety_source(tmpdir)
            variants = (
                (B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME, 1.0),
                (B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME, 1.0),
                (B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME, 1.0),
                (B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME, 0.85),
                (B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b62_defensive_mode_selector_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=352 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 62)
                self.assertEqual(report["parent_level"], 61)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b1_trace_fields_and_primitive_contract(self) -> None:
        source_config = _b0_config()
        source = SpiderBrain(seed=36, module_dropout=0.0, config=source_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_CAPACITY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=37,
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
            "b_effective_level",
            "b_mode",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "semantic_action_reason",
            "semantic_override_count",
            "bridge_primitive_action",
            "bridge_reason",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 1)
        self.assertEqual(first["b_effective_level"], "B1")
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(first["semantic_action_source"], "network_policy")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b2_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            config = build_b2_temporal_threat_config(
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=40,
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
            "b_temporal_threat_pressure",
            "b_predator_memory_pressure",
            "b_predator_trace_pressure",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 2)
        self.assertEqual(first["b_parent_level"], 1)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B2_TEMPORAL_THREAT_SELECTION_SOURCE,
        )
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b3_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=44,
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
            "b_temporal_threat_pressure",
            "b_predator_memory_pressure",
            "b_predator_trace_pressure",
            "b3_contact_cooldown",
            "b3_post_food_cooldown",
            "b3_hunger_drop",
            "b3_controller_profile",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 3)
        self.assertEqual(first["b_parent_level"], 2)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B3_CONTACT_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b3_controller_profile"], "standard")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b4_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            config = build_b4_recovery_balance_config(
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=46,
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
            "b_temporal_threat_pressure",
            "b_predator_memory_pressure",
            "b_predator_trace_pressure",
            "b3_contact_cooldown",
            "b3_post_food_cooldown",
            "b3_hunger_drop",
            "b3_controller_profile",
            "b4_controller_profile",
            "b4_recovery_pressure",
            "b4_sleep_hold",
            "b4_exit_blocked",
            "b4_hunger_release",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 4)
        self.assertEqual(first["b_parent_level"], 3)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B4_RECOVERY_BALANCE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b4_controller_profile"], "recovery_balance")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b5_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=47,
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
            "b4_controller_profile",
            "b4_recovery_pressure",
            "b5_controller_profile",
            "b5_hunger_urgency",
            "b5_sleep_pressure",
            "b5_recovery_debt",
            "b5_threat_gate",
            "b5_sleep_bout_lock",
            "b5_forage_commitment_lock",
            "b5_homeostatic_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 5)
        self.assertEqual(first["b_parent_level"], 4)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(first["b5_controller_profile"], "homeostatic_arbiter")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b6_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=48,
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
            "b5_controller_profile",
            "b6_controller_family",
            "b6_controller_profile",
            "b6_risk_pressure",
            "b6_threat_priority",
            "b6_forage_suppressed",
            "b6_corridor_commitment",
            "b6_corridor_progress_memory",
            "b6_recurrent_state",
            "b6_return_lock",
            "b6_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 6)
        self.assertEqual(first["b_parent_level"], 5)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B6_RISK_CORRIDOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b6_controller_family"], "risk_corridor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b7_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            config = build_b7_affordance_budget_config(
                B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=49,
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
            "b6_controller_family",
            "b7_controller_profile",
            "b7_affordance_state",
            "b7_energy_budget",
            "b7_budget_margin",
            "b7_food_steps_estimate",
            "b7_return_steps_estimate",
            "b7_corridor_viability",
            "b7_abort_return",
            "b7_commitment_lock",
            "b7_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 7)
        self.assertEqual(first["b_parent_level"], 6)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B7_AFFORDANCE_BUDGET_SELECTION_SOURCE,
        )
        self.assertEqual(first["b7_controller_profile"], "affordance_budget")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b8_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b7_affordance_budget_source(tmpdir)
            config = build_b8_spatial_affordance_config(
                B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=50,
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
            "b7_controller_profile",
            "b8_controller_profile",
            "b8_spatial_map_state",
            "b8_local_affordance_score",
            "b8_return_vector_strength",
            "b8_corridor_dead_end_risk",
            "b8_abort_executed",
            "b8_place_memory",
            "b8_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 8)
        self.assertEqual(first["b_parent_level"], 7)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b8_controller_profile"], "spatial_affordance_map")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b9_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b8_spatial_affordance_source(tmpdir)
            config = build_b9_waypoint_planner_config(
                B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=51,
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
            "b8_controller_profile",
            "b9_controller_profile",
            "b9_route_state",
            "b9_route_confidence",
            "b9_waypoint_lock",
            "b9_path_integrator",
            "b9_replan_signal",
            "b9_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 9)
        self.assertEqual(first["b_parent_level"], 8)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B9_WAYPOINT_PLANNER_SELECTION_SOURCE,
        )
        self.assertEqual(first["b9_controller_profile"], "waypoint_planner")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)
