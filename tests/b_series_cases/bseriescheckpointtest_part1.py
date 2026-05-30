from __future__ import annotations

from .shared import *



class BSeriesCheckpointTestPart1(unittest.TestCase):
    def test_checkpoint_save_load_preserves_b_series_metadata_and_weights(self) -> None:
        config = _b0_config()
        source = SpiderBrain(seed=21, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=22, module_dropout=0.0, config=config)
        assert source.b_series_policy is not None
        assert target.b_series_policy is not None
        source.b_series_policy.b2_policy[:] = np.arange(
            len(B_SEMANTIC_ACTIONS),
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            loaded = target.load(checkpoint)
            metadata = json.loads((checkpoint / "metadata.json").read_text())

        self.assertIn("b_series_policy", loaded)
        self.assertEqual(
            metadata["modules"]["b_series_policy"]["semantic_actions"],
            list(B_SEMANTIC_ACTIONS),
        )
        self.assertEqual(metadata["modules"]["b_series_policy"]["b_level"], 0)
        self.assertEqual(metadata["modules"]["b_series_policy"]["b_mode"], "current_bridge")
        np.testing.assert_allclose(
            source.b_series_policy.b2_policy,
            target.b_series_policy.b2_policy,
        )

    def test_b1_partial_transfer_reports_coverage(self) -> None:
        source_config = _b0_config()
        source = SpiderBrain(seed=31, module_dropout=0.0, config=source_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            b1_config = BrainAblationConfig(
                name="b1_transfer_smoke",
                architecture="b_series",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                use_learned_arbitration=False,
                enable_food_direction_bias=False,
                warm_start_scale=0.0,
                credit_strategy="broadcast",
                disabled_modules=(),
                reflex_scale=0.0,
                module_reflex_scales={},
                b_level=1,
                b_mode="current_bridge",
                b_parent_level=0,
                b_hidden_dim=40,
                b_transfer_source_checkpoint=str(checkpoint),
            )
            target = SpiderBrain(seed=32, module_dropout=0.0, config=b1_config)

        report = target.b_series_transfer_report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertGreaterEqual(float(report["coverage"]), 0.50)
        self.assertEqual(report["target_b_level"], 1)
        self.assertEqual(report["parent_level"], 0)
        self.assertIn("W1", report["partially_loaded_keys"])

    def test_b1_requires_b0_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B1_CAPACITY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b0-current",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=33, module_dropout=0.0, config=config)

    def test_b1_h48_transfer_report_records_source_and_coverage(self) -> None:
        source_config = _b0_config()
        source = SpiderBrain(seed=34, module_dropout=0.0, config=source_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_CAPACITY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            target = SpiderBrain(seed=35, module_dropout=0.0, config=config)

        report = target.b_series_transfer_report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["source_checkpoint"], str(checkpoint))
        self.assertEqual(report["target_b_level"], 1)
        self.assertEqual(report["parent_level"], 0)
        self.assertGreaterEqual(float(report["coverage"]), 0.50)
        self.assertLess(float(report["coverage"]), 1.0)
        self.assertFalse(report["allow_low_coverage"])

    def test_b2_requires_b1_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B2_TEMPORAL_THREAT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b1-threat-guard",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=38, module_dropout=0.0, config=config)

    def test_b3_requires_b2_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B3_CONTACT_MEMORY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b2-temporal-threat",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=41, module_dropout=0.0, config=config)

    def test_b4_requires_b3_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B4_RECOVERY_BALANCE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b3-recurrent-guard",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=45, module_dropout=0.0, config=config)

    def test_b5_requires_b4_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b4-genetic-recovery",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=46, module_dropout=0.0, config=config)

    def test_b6_requires_b5_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b5-genetic-homeostasis",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=47, module_dropout=0.0, config=config)

    def test_b7_requires_b6_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B7_AFFORDANCE_BUDGET_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b6-fused-risk-recurrent",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=48, module_dropout=0.0, config=config)

    def test_b8_requires_b7_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b7-affordance-budget",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=49, module_dropout=0.0, config=config)

    def test_b9_requires_b8_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B9_WAYPOINT_PLANNER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b8-spatial-affordance",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=50, module_dropout=0.0, config=config)

    def test_b10_requires_b9_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b9-waypoint-planner",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=51, module_dropout=0.0, config=config)

    def test_b11_requires_b10_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B11_CONFIDENCE_ARBITER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b10-prospective-replay",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=52, module_dropout=0.0, config=config)

    def test_b12_requires_b11_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b11-confidence-arbiter",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=53, module_dropout=0.0, config=config)

    def test_b13_requires_b12_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b12-predictive-attention",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=54, module_dropout=0.0, config=config)

    def test_b14_requires_b13_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b13-local-search",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=55, module_dropout=0.0, config=config)

    def test_b15_requires_b14_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B15_OPTION_CRITIC_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b14-affordance-uncertainty",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=56, module_dropout=0.0, config=config)

    def test_b16_requires_b15_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B16_OPTION_ENSEMBLE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b15-option-critic",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=57, module_dropout=0.0, config=config)

    def test_b17_requires_b16_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b16-option-ensemble",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=58, module_dropout=0.0, config=config)

    def test_b18_requires_b17_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B18_ELIGIBILITY_TRACE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b17-neuromodulated",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=59, module_dropout=0.0, config=config)

    def test_b19_requires_b18_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B19_EPISODIC_META_MEMORY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b18-eligibility",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=60, module_dropout=0.0, config=config)

    def test_b20_requires_b19_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B20_WORKING_MEMORY_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b19-meta-memory",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=61, module_dropout=0.0, config=config)

    def test_b21_requires_b20_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b20-working-memory",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=62, module_dropout=0.0, config=config)

    def test_b22_requires_b21_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b21-replay",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=63, module_dropout=0.0, config=config)

    def test_b23_requires_b22_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B23_CONFLICT_MONITOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b22-prospective",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=64, module_dropout=0.0, config=config)

    def test_b24_requires_b23_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B24_PRECISION_CONFLICT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b23-conflict",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=65, module_dropout=0.0, config=config)

    def test_b25_requires_b24_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b24-precision",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=66, module_dropout=0.0, config=config)

    def test_b26_requires_b25_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b25-metacog",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=67, module_dropout=0.0, config=config)

    def test_b27_requires_b26_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B27_AROUSAL_GAIN_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b26-allostasis",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=68, module_dropout=0.0, config=config)

    def test_b28_requires_b27_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b27-arousal",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=69, module_dropout=0.0, config=config)

    def test_b29_requires_b28_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B29_SALIENCE_COMPETITION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b28-attention",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=70, module_dropout=0.0, config=config)

    def test_b30_requires_b29_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b29-salience",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=71, module_dropout=0.0, config=config)

    def test_b31_requires_b30_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b30-gate",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=72, module_dropout=0.0, config=config)

    def test_b32_requires_b31_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b31-dopamine",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=73, module_dropout=0.0, config=config)

    def test_b33_requires_b32_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b32-value",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=74, module_dropout=0.0, config=config)

    def test_b34_requires_b33_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b33-td",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=75, module_dropout=0.0, config=config)

    def test_b35_requires_b34_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b34-eligibility",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=76, module_dropout=0.0, config=config)

    def test_b36_requires_b35_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B36_LATENT_BELIEF_STATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b35-forward-model",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=77, module_dropout=0.0, config=config)

    def test_b37_requires_b36_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B37_STATE_FACTOR_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b36-belief-state",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=78, module_dropout=0.0, config=config)

    def test_b38_requires_b37_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B38_FACTOR_ATTENTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b37-state-factor",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=79, module_dropout=0.0, config=config)

    def test_b39_requires_b38_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B39_ATTENTION_BINDING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b38-factor-attention",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=80, module_dropout=0.0, config=config)

    def test_b40_requires_b39_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B40_GLOBAL_WORKSPACE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b39-attention-binding",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=81, module_dropout=0.0, config=config)

    def test_b41_requires_b40_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b40-global-workspace",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=82, module_dropout=0.0, config=config)

    def test_b42_requires_b41_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B42_ERROR_MONITOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b41-executive-workspace",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=83, module_dropout=0.0, config=config)

    def test_b43_requires_b42_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B43_ADAPTIVE_PRECISION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b42-error-monitor",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=84, module_dropout=0.0, config=config)

    def test_b44_requires_b43_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B44_THALAMIC_RELAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b43-adaptive-precision",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=85, module_dropout=0.0, config=config)

    def test_b45_requires_b44_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B45_RETICULAR_INHIBITION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b44-thalamic-relay",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=86, module_dropout=0.0, config=config)

    def test_b46_requires_b45_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b45-reticular-inhibition",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=87, module_dropout=0.0, config=config)

    def test_b47_requires_b46_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b46-corticothalamic-feedback",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=88, module_dropout=0.0, config=config)

    def test_b48_requires_b47_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B48_CEREBELLAR_TIMING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b47-oscillatory-synchrony",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=89, module_dropout=0.0, config=config)

    def test_b49_requires_b48_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b48-cerebellar-timing",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=90, module_dropout=0.0, config=config)

    def test_b50_requires_b49_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B50_HABIT_CHUNKING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b49-striatal-gate",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=91, module_dropout=0.0, config=config)

    def test_b51_requires_b50_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b50-habit-chunking",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=92, module_dropout=0.0, config=config)

    def test_b52_requires_b51_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b51-dopaminergic-habit",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=93, module_dropout=0.0, config=config)

    def test_b53_requires_b52_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b52-cholinergic-precision",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=94, module_dropout=0.0, config=config)

    def test_b54_requires_b53_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b53-noradrenergic-arousal",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=95, module_dropout=0.0, config=config)

    def test_b55_requires_b54_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b54-serotonergic-patience",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=96, module_dropout=0.0, config=config)

    def test_b56_requires_b55_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B56_HPA_STRESS_AXIS_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b55-hypothalamic-drive",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=97, module_dropout=0.0, config=config)

    def test_b57_requires_b56_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b56-hpa-stress-axis",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=98, module_dropout=0.0, config=config)

    def test_b58_requires_b57_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b57-insular-interoception",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=99, module_dropout=0.0, config=config)

    def test_b59_requires_b58_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b58-acc-conflict",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=100, module_dropout=0.0, config=config)

    def test_b60_requires_b59_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b59-prefrontal-goal",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=101, module_dropout=0.0, config=config)

    def test_b61_requires_b60_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b60-orbitofrontal-value",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=102, module_dropout=0.0, config=config)

    def test_b62_requires_b61_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b61-amygdala-safety",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=103, module_dropout=0.0, config=config)

    def test_b2_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            variants = (
                (B2_TEMPORAL_THREAT_H48_POLICY_NAME, 1.0),
                (B2_TEMPORAL_THREAT_H56_POLICY_NAME, 0.85),
                (B2_TEMPORAL_THREAT_H64_POLICY_NAME, 0.75),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b2_temporal_threat_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=39 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 2)
                self.assertEqual(report["parent_level"], 1)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b3_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            variants = (
                (B3_CONTACT_MEMORY_H48_POLICY_NAME, 1.0),
                (B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME, 1.0),
                (B3_CONTACT_MEMORY_H56_POLICY_NAME, 0.85),
                (B3_RECURRENT_GUARD_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b3_contact_memory_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=42 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 3)
                self.assertEqual(report["parent_level"], 2)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b4_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            variants = (
                (B4_RECOVERY_BALANCE_H48_POLICY_NAME, 1.0),
                (B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME, 1.0),
                (B4_RECOVERY_BALANCE_H56_POLICY_NAME, 0.85),
                (B4_GENETIC_RECOVERY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b4_recovery_balance_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=52 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 4)
                self.assertEqual(report["parent_level"], 3)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b5_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            variants = (
                (B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME, 1.0),
                (B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME, 1.0),
                (B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME, 0.85),
                (B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b5_homeostatic_arbiter_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=56 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 5)
                self.assertEqual(report["parent_level"], 4)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b6_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            variants = (
                (B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME, 1.0),
                (B6_RECURRENT_CONTEXT_H48_POLICY_NAME, 1.0),
                (B6_RISK_CORRIDOR_H56_POLICY_NAME, 0.85),
                (B6_RECURRENT_CONTEXT_H56_POLICY_NAME, 0.85),
                (B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b6_risk_corridor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=60 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 6)
                self.assertEqual(report["parent_level"], 5)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b7_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            variants = (
                (B7_AFFORDANCE_BUDGET_H48_POLICY_NAME, 1.0),
                (B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME, 1.0),
                (B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME, 1.0),
                (B7_AFFORDANCE_BUDGET_H56_POLICY_NAME, 0.85),
                (B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b7_affordance_budget_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=66 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 7)
                self.assertEqual(report["parent_level"], 6)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b8_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b7_affordance_budget_source(tmpdir)
            variants = (
                (B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME, 1.0),
                (B8_RETURN_VECTOR_H48_POLICY_NAME, 1.0),
                (B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME, 1.0),
                (B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME, 0.85),
                (B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b8_spatial_affordance_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=72 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 8)
                self.assertEqual(report["parent_level"], 7)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b9_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b8_spatial_affordance_source(tmpdir)
            variants = (
                (B9_WAYPOINT_PLANNER_H48_POLICY_NAME, 1.0),
                (B9_PATH_INTEGRATION_H48_POLICY_NAME, 1.0),
                (B9_ROUTE_MEMORY_H48_POLICY_NAME, 1.0),
                (B9_WAYPOINT_PLANNER_H56_POLICY_NAME, 0.85),
                (B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b9_waypoint_planner_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=78 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 9)
                self.assertEqual(report["parent_level"], 8)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b10_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b9_waypoint_planner_source(tmpdir)
            variants = (
                (B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME, 1.0),
                (B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME, 1.0),
                (B10_REPLAY_PLANNER_H48_POLICY_NAME, 1.0),
                (B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME, 0.85),
                (B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b10_prospective_replay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=84 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 10)
                self.assertEqual(report["parent_level"], 9)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b11_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b10_prospective_replay_source(tmpdir)
            variants = (
                (B11_CONFIDENCE_ARBITER_H48_POLICY_NAME, 1.0),
                (B11_UNCERTAINTY_GATE_H48_POLICY_NAME, 1.0),
                (B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME, 1.0),
                (B11_CONFIDENCE_ARBITER_H56_POLICY_NAME, 0.85),
                (B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b11_confidence_arbiter_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=90 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 11)
                self.assertEqual(report["parent_level"], 10)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b12_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b11_confidence_arbiter_source(tmpdir)
            variants = (
                (B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME, 1.0),
                (B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME, 0.85),
                (B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b12_predictive_attention_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=96 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 12)
                self.assertEqual(report["parent_level"], 11)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b13_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b12_predictive_attention_source(tmpdir)
            variants = (
                (B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME, 1.0),
                (B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME, 1.0),
                (B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME, 1.0),
                (B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME, 0.85),
                (B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b13_local_affordance_search_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=101 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 13)
                self.assertEqual(report["parent_level"], 12)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b14_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b13_local_affordance_search_source(tmpdir)
            variants = (
                (B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME, 1.0),
                (B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME, 1.0),
                (B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME, 1.0),
                (B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME, 0.85),
                (B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b14_affordance_uncertainty_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=106 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 14)
                self.assertEqual(report["parent_level"], 13)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b15_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b14_affordance_uncertainty_source(tmpdir)
            variants = (
                (B15_OPTION_CRITIC_H48_POLICY_NAME, 1.0),
                (B15_PERSISTENCE_GATE_H48_POLICY_NAME, 1.0),
                (B15_VALUE_GATED_OPTION_H48_POLICY_NAME, 1.0),
                (B15_OPTION_CRITIC_H56_POLICY_NAME, 0.85),
                (B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b15_option_critic_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=111 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 15)
                self.assertEqual(report["parent_level"], 14)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b16_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b15_option_critic_source(tmpdir)
            variants = (
                (B16_OPTION_ENSEMBLE_H48_POLICY_NAME, 1.0),
                (B16_COMPETING_OPTIONS_H48_POLICY_NAME, 1.0),
                (B16_ACTION_SET_VOTER_H48_POLICY_NAME, 1.0),
                (B16_OPTION_ENSEMBLE_H56_POLICY_NAME, 0.85),
                (B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b16_option_ensemble_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=117 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 16)
                self.assertEqual(report["parent_level"], 15)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b17_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b16_option_ensemble_source(tmpdir)
            variants = (
                (B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME, 1.0),
                (B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME, 1.0),
                (B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME, 1.0),
                (B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME, 0.85),
                (B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b17_neuromodulated_ensemble_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=122 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 17)
                self.assertEqual(report["parent_level"], 16)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b18_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b17_neuromodulated_ensemble_source(tmpdir)
            variants = (
                (B18_ELIGIBILITY_TRACE_H48_POLICY_NAME, 1.0),
                (B18_METASTABLE_AROUSAL_H48_POLICY_NAME, 1.0),
                (B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME, 1.0),
                (B18_ELIGIBILITY_TRACE_H56_POLICY_NAME, 0.85),
                (B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b18_eligibility_trace_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=127 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 18)
                self.assertEqual(report["parent_level"], 17)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b19_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b18_eligibility_trace_source(tmpdir)
            variants = (
                (B19_EPISODIC_META_MEMORY_H48_POLICY_NAME, 1.0),
                (B19_STABILITY_MEMORY_H48_POLICY_NAME, 1.0),
                (B19_SWITCH_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B19_EPISODIC_META_MEMORY_H56_POLICY_NAME, 0.85),
                (B19_GENETIC_META_MEMORY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b19_episodic_meta_memory_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=132 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 19)
                self.assertEqual(report["parent_level"], 18)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b20_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b19_episodic_meta_memory_source(tmpdir)
            variants = (
                (B20_WORKING_MEMORY_GATE_H48_POLICY_NAME, 1.0),
                (B20_CONTEXT_BINDING_H48_POLICY_NAME, 1.0),
                (B20_STABILITY_BUFFER_H48_POLICY_NAME, 1.0),
                (B20_WORKING_MEMORY_GATE_H56_POLICY_NAME, 0.85),
                (B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b20_working_memory_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=137 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 20)
                self.assertEqual(report["parent_level"], 19)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b21_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b20_working_memory_gate_source(tmpdir)
            variants = (
                (B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME, 1.0),
                (B21_SEQUENCE_BINDING_H48_POLICY_NAME, 1.0),
                (B21_ROUTE_REHEARSAL_H48_POLICY_NAME, 1.0),
                (B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME, 0.85),
                (B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b21_hippocampal_replay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=142 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 21)
                self.assertEqual(report["parent_level"], 20)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b22_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b21_hippocampal_replay_source(tmpdir)
            variants = (
                (B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME, 1.0),
                (B22_FORWARD_MODEL_GATE_H48_POLICY_NAME, 1.0),
                (B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME, 1.0),
                (B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME, 0.85),
                (B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b22_prospective_replay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=147 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 22)
                self.assertEqual(report["parent_level"], 21)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b23_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b22_prospective_replay_source(tmpdir)
            variants = (
                (B23_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
                (B23_ERROR_GATED_REPLAY_H48_POLICY_NAME, 1.0),
                (B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME, 1.0),
                (B23_CONFLICT_MONITOR_H56_POLICY_NAME, 0.85),
                (B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b23_conflict_monitor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=152 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 23)
                self.assertEqual(report["parent_level"], 22)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b24_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b23_conflict_monitor_source(tmpdir)
            variants = (
                (B24_PRECISION_CONFLICT_H48_POLICY_NAME, 1.0),
                (B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME, 1.0),
                (B24_RELIABILITY_ABORT_H48_POLICY_NAME, 1.0),
                (B24_PRECISION_CONFLICT_H56_POLICY_NAME, 0.85),
                (B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b24_precision_conflict_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=157 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 24)
                self.assertEqual(report["parent_level"], 23)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b25_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b24_precision_conflict_source(tmpdir)
            variants = (
                (B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME, 1.0),
                (B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME, 1.0),
                (B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME, 1.0),
                (B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME, 0.85),
                (B25_GENETIC_METACOGNITION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b25_metacognitive_confidence_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=162 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 25)
                self.assertEqual(report["parent_level"], 24)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b26_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b25_metacognitive_confidence_source(tmpdir)
            variants = (
                (B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME, 1.0),
                (B26_SETPOINT_DRIFT_H48_POLICY_NAME, 1.0),
                (B26_ERROR_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME, 0.85),
                (B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b26_allostatic_prediction_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=167 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 26)
                self.assertEqual(report["parent_level"], 25)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b27_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b26_allostatic_prediction_source(tmpdir)
            variants = (
                (B27_AROUSAL_GAIN_H48_POLICY_NAME, 1.0),
                (B27_STRESS_MODULATION_H48_POLICY_NAME, 1.0),
                (B27_ENERGY_AROUSAL_H48_POLICY_NAME, 1.0),
                (B27_AROUSAL_GAIN_H56_POLICY_NAME, 0.85),
                (B27_GENETIC_AROUSAL_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b27_arousal_gain_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=172 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 27)
                self.assertEqual(report["parent_level"], 26)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b28_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b27_arousal_gain_source(tmpdir)
            variants = (
                (B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME, 1.0),
                (B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME, 1.0),
                (B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME, 0.85),
                (B28_GENETIC_ATTENTION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b28_interoceptive_attention_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=177 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 28)
                self.assertEqual(report["parent_level"], 27)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b29_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b28_interoceptive_attention_source(tmpdir)
            variants = (
                (B29_SALIENCE_COMPETITION_H48_POLICY_NAME, 1.0),
                (B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B29_SALIENCE_COMPETITION_H56_POLICY_NAME, 0.85),
                (B29_GENETIC_SALIENCE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b29_salience_competition_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=182 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 29)
                self.assertEqual(report["parent_level"], 28)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])
