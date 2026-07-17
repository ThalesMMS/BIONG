from __future__ import annotations

from .shared import *



class BSeriesRuntimeTestPart3(unittest.TestCase):
    def test_b77_olivary_error_correction_uses_b76_stride_context(self) -> None:
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
            brain = SpiderBrain(seed=103, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.06
        meta["recent_contact"] = 0.04
        for tick in range(77, 101):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.56},
                    sleep={"health": 0.56, "sleep_debt": 0.36, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.05,
                        "predator_motion_salience": 0.06,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B77-olivary-error-correction")
        self.assertEqual(
            decision.semantic_action_source,
            "b77_olivary_error_controller",
        )
        self.assertEqual(decision.b77_controller_profile, "olivary_error_correction")
        self.assertIn(
            decision.b77_decision,
            {"olivary_error_hold", "continue_error_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b77_prediction_error), 0.0)
        self.assertGreater(float(decision.b77_climbing_fiber_drive), 0.0)
        self.assertGreater(float(decision.b77_corrective_timing), 0.0)
        self.assertGreater(float(decision.b77_stability_confidence), 0.0)
        self.assertGreaterEqual(int(decision.b77_error_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b76_cerebellar_stride_gate_smooths_b75_release_context(self) -> None:
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
            brain = SpiderBrain(seed=102, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.07
        meta["recent_contact"] = 0.04
        for tick in range(76, 100):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.55},
                    sleep={"health": 0.55, "sleep_debt": 0.38, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.05,
                        "predator_motion_salience": 0.07,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B76-cerebellar-stride-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b76_cerebellar_stride_controller",
        )
        self.assertEqual(decision.b76_controller_profile, "cerebellar_stride_gate")
        self.assertIn(
            decision.b76_decision,
            {"cerebellar_stride_hold", "continue_stride_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b76_stride_timing_signal), 0.0)
        self.assertGreater(float(decision.b76_error_correction), 0.0)
        self.assertGreater(float(decision.b76_burst_smoothing), 0.0)
        self.assertGreater(float(decision.b76_stride_gate), 0.0)
        self.assertGreaterEqual(int(decision.b76_stride_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b75_basal_thalamic_release_times_b74_rebound_context(self) -> None:
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
            brain = SpiderBrain(seed=101, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.08
        meta["recent_contact"] = 0.04
        for tick in range(75, 99):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.54},
                    sleep={"health": 0.54, "sleep_debt": 0.40, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.06,
                        "predator_motion_salience": 0.08,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B75-basal-thalamic-release")
        self.assertEqual(
            decision.semantic_action_source,
            "b75_basal_thalamic_release_controller",
        )
        self.assertEqual(decision.b75_controller_profile, "basal_thalamic_release")
        self.assertIn(
            decision.b75_decision,
            {"basal_thalamic_hold", "continue_release_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b75_release_timing_drive), 0.0)
        self.assertGreater(float(decision.b75_go_disinhibition), 0.0)
        self.assertGreater(float(decision.b75_nogo_brake), 0.0)
        self.assertGreater(float(decision.b75_burst_window), 0.0)
        self.assertGreaterEqual(int(decision.b75_release_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b74_thalamic_rebound_releases_b73_inhibition_context(self) -> None:
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
            brain = SpiderBrain(seed=100, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.10
        meta["recent_contact"] = 0.05
        for tick in range(74, 97):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.52},
                    sleep={"health": 0.52, "sleep_debt": 0.44, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.07,
                        "predator_motion_salience": 0.09,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B74-thalamic-rebound-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b74_thalamic_rebound_controller",
        )
        self.assertEqual(decision.b74_controller_profile, "thalamic_rebound_gate")
        self.assertIn(
            decision.b74_decision,
            {"thalamic_rebound_hold", "continue_rebound_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b74_rebound_potential), 0.0)
        self.assertGreater(float(decision.b74_inhibition_aftereffect), 0.0)
        self.assertGreater(float(decision.b74_release_window), 0.0)
        self.assertGreaterEqual(int(decision.b74_rebound_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b73_reticular_inhibition_uses_b72_attention_filter(self) -> None:
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
            brain = SpiderBrain(seed=99, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.10
        meta["recent_contact"] = 0.05
        for tick in range(73, 96):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.52},
                    sleep={"health": 0.52, "sleep_debt": 0.44, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.07,
                        "predator_motion_salience": 0.09,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B73-reticular-inhibition-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b73_reticular_inhibition_controller",
        )
        self.assertEqual(decision.b73_controller_profile, "reticular_inhibition_gate")
        self.assertIn(
            decision.b73_decision,
            {"reticular_surround_hold", "continue_inhibition_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b73_inhibitory_tone), 0.0)
        self.assertGreater(float(decision.b73_surround_suppression), 0.0)
        self.assertGreater(float(decision.b73_release_drive), 0.0)
        self.assertGreaterEqual(int(decision.b73_inhibition_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b72_pulvinar_attention_filters_b71_orienting_context(self) -> None:
        build_b72 = getattr(
            b_series_evolution_module,
            "build_b72_pulvinar_attention_config",
            None,
        )
        self.assertIsNotNone(build_b72)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b71_tectal_orienting_source(tmpdir)
            config = build_b72(
                "b72_pulvinar_attention_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=98, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.10
        meta["recent_contact"] = 0.05
        for tick in range(72, 95):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.52},
                    sleep={"health": 0.52, "sleep_debt": 0.44, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.07,
                        "predator_motion_salience": 0.09,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B72-pulvinar-attention-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b72_pulvinar_attention_controller",
        )
        self.assertEqual(decision.b72_controller_profile, "pulvinar_attention_gate")
        self.assertIn(
            decision.b72_decision,
            {"pulvinar_filter_recenter", "continue_attention_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b72_focus_signal), 0.0)
        self.assertGreater(float(decision.b72_distractor_load), 0.0)
        self.assertGreater(float(decision.b72_filter_gain), 0.0)
        self.assertGreaterEqual(int(decision.b72_attention_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b71_tectal_orienting_uses_b70_optic_flow_context(self) -> None:
        build_b71 = getattr(
            b_series_evolution_module,
            "build_b71_tectal_orienting_config",
            None,
        )
        self.assertIsNotNone(build_b71)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b70_optic_flow_source(tmpdir)
            config = build_b71(
                "b71_tectal_orienting_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=97, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.12
        meta["recent_contact"] = 0.05
        for tick in range(71, 93):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.51},
                    sleep={"health": 0.52, "sleep_debt": 0.44, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.08,
                        "predator_motion_salience": 0.08,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B71-tectal-orienting-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b71_tectal_orienting_controller",
        )
        self.assertEqual(decision.b71_controller_profile, "tectal_orienting_gate")
        self.assertIn(
            decision.b71_decision,
            {"tectal_recenter", "continue_orienting_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b71_target_salience), 0.0)
        self.assertGreater(float(decision.b71_orienting_gain), 0.0)
        self.assertGreater(float(decision.b71_collision_veto), 0.0)
        self.assertGreaterEqual(int(decision.b71_orienting_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b70_optic_flow_uses_b69_orientation_context(self) -> None:
        build_b70 = getattr(
            b_series_evolution_module,
            "build_b70_optic_flow_config",
            None,
        )
        self.assertIsNotNone(build_b70)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b69_vestibular_orientation_source(tmpdir)
            config = build_b70(
                "b70_optic_flow_stabilization_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=96, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.12
        meta["recent_contact"] = 0.05
        for tick in range(70, 91):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.52},
                    sleep={"health": 0.53, "sleep_debt": 0.43, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.07,
                        "predator_motion_salience": 0.07,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B70-optic-flow-stabilization-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b70_optic_flow_controller",
        )
        self.assertEqual(decision.b70_controller_profile, "optic_flow_stabilization_gate")
        self.assertIn(
            decision.b70_decision,
            {"optic_flow_recenter", "continue_flow_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b70_flow_confidence), 0.0)
        self.assertGreater(float(decision.b70_lateral_drift), 0.0)
        self.assertGreater(float(decision.b70_looming_risk), 0.0)
        self.assertGreaterEqual(int(decision.b70_flow_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b69_vestibular_orientation_uses_b68_motor_context(self) -> None:
        build_b69 = getattr(
            b_series_evolution_module,
            "build_b69_vestibular_orientation_config",
            None,
        )
        self.assertIsNotNone(build_b69)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b68_motor_pacing_source(tmpdir)
            config = build_b69(
                "b69_vestibular_orientation_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=95, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.13
        meta["recent_contact"] = 0.05
        for tick in range(69, 89):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.53},
                    sleep={"health": 0.54, "sleep_debt": 0.42, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.07,
                        "predator_motion_salience": 0.06,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B69-vestibular-orientation-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b69_vestibular_orientation_controller",
        )
        self.assertEqual(decision.b69_controller_profile, "vestibular_orientation_gate")
        self.assertIn(
            decision.b69_decision,
            {"vestibular_recenter", "continue_orientation_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b69_heading_confidence), 0.0)
        self.assertGreater(float(decision.b69_turn_error), 0.0)
        self.assertGreater(float(decision.b69_orientation_stability), 0.0)
        self.assertGreaterEqual(int(decision.b69_orientation_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b68_motor_pacing_uses_b67_glial_context(self) -> None:
        build_b68 = getattr(
            b_series_evolution_module,
            "build_b68_motor_pacing_config",
            None,
        )
        self.assertIsNotNone(build_b68)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b67_glial_energy_source(tmpdir)
            config = build_b68(
                "b68_proprioceptive_pacing_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=94, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.14
        meta["recent_contact"] = 0.06
        for tick in range(68, 87):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.54},
                    sleep={"health": 0.55, "sleep_debt": 0.40, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.08,
                        "predator_motion_salience": 0.06,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B68-motor-pacing-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b68_proprioceptive_pacing_controller",
        )
        self.assertEqual(decision.b68_controller_profile, "proprioceptive_pacing_gate")
        self.assertIn(
            decision.b68_decision,
            {"motor_recovery_pace", "continue_motor_pacing_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b68_motor_reserve), 0.0)
        self.assertGreater(float(decision.b68_stride_pacing), 0.0)
        self.assertGreater(float(decision.b68_overexertion_risk), 0.0)
        self.assertGreaterEqual(int(decision.b68_pacing_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b67_glial_energy_uses_b66_immune_context(self) -> None:
        build_b67 = getattr(
            b_series_evolution_module,
            "build_b67_glial_energy_config",
            None,
        )
        self.assertIsNotNone(build_b67)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b66_immune_malaise_source(tmpdir)
            config = build_b67(
                "b67_glial_energy_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=93, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.16
        meta["recent_contact"] = 0.08
        for tick in range(67, 85):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.55},
                    sleep={"health": 0.56, "sleep_debt": 0.38, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.09,
                        "predator_motion_salience": 0.07,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B67-glial-energy-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b67_glial_homeostasis_controller",
        )
        self.assertEqual(decision.b67_controller_profile, "glial_energy_gate")
        self.assertIn(
            decision.b67_decision,
            {"glial_recovery_support", "continue_glial_recovery_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b67_glial_reserve), 0.0)
        self.assertGreater(float(decision.b67_lactate_support), 0.0)
        self.assertGreater(float(decision.b67_fatigue_pressure), 0.0)
        self.assertGreaterEqual(int(decision.b67_recovery_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b66_immune_malaise_uses_b65_assimilation_context(self) -> None:
        build_b66 = getattr(
            b_series_evolution_module,
            "build_b66_immune_malaise_config",
            None,
        )
        self.assertIsNotNone(build_b66)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b65_enteric_assimilation_source(tmpdir)
            config = build_b66(
                "b66_immune_malaise_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=92, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        meta["recent_pain"] = 0.18
        meta["recent_contact"] = 0.10
        for tick in range(66, 83):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.56},
                    sleep={"health": 0.58, "sleep_debt": 0.36, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.10,
                        "predator_motion_salience": 0.08,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B66-immune-malaise-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b66_immune_malaise_controller",
        )
        self.assertEqual(decision.b66_controller_profile, "immune_malaise_gate")
        self.assertIn(
            decision.b66_decision,
            {"immune_recovery_hold", "continue_inflammation_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b66_immune_tone), 0.0)
        self.assertGreater(float(decision.b66_malaise_drive), 0.0)
        self.assertGreater(float(decision.b66_recovery_veto), 0.0)
        self.assertGreaterEqual(int(decision.b66_inflammation_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b65_enteric_assimilation_uses_b64_recovery_context(self) -> None:
        build_b65 = getattr(
            b_series_evolution_module,
            "build_b65_enteric_assimilation_config",
            None,
        )
        self.assertIsNotNone(build_b65)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b64_vagal_recovery_source(tmpdir)
            config = build_b65(
                "b65_enteric_assimilation_gate_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=91, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 7.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        for tick in range(65, 82):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.54},
                    sleep={"health": 0.62, "sleep_debt": 0.34, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.08,
                        "predator_motion_salience": 0.05,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B65-enteric-assimilation-gate")
        self.assertEqual(
            decision.semantic_action_source,
            "b65_enteric_assimilation_controller",
        )
        self.assertEqual(decision.b65_controller_profile, "enteric_assimilation_gate")
        self.assertIn(
            decision.b65_decision,
            {"enteric_digestive_hold", "continue_digestive_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b65_enteric_tone), 0.0)
        self.assertGreater(float(decision.b65_assimilation_drive), 0.0)
        self.assertGreater(float(decision.b65_forage_readiness), 0.0)
        self.assertGreaterEqual(int(decision.b65_digestive_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b64_vagal_recovery_brake_uses_b63_escape_context(self) -> None:
        build_b64 = getattr(
            b_series_evolution_module,
            "build_b64_vagal_recovery_config",
            None,
        )
        self.assertIsNotNone(build_b64)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b63_periaqueductal_escape_source(tmpdir)
            config = build_b64(
                "b64_vagal_recovery_brake_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=90, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "inside"
        for tick in range(64, 80):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.58},
                    sleep={"health": 0.58, "sleep_debt": 0.42, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.12,
                        "predator_motion_salience": 0.08,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B64-vagal-recovery-brake")
        self.assertEqual(
            decision.semantic_action_source,
            "b64_vagal_recovery_brake_controller",
        )
        self.assertEqual(decision.b64_controller_profile, "vagal_recovery_brake")
        self.assertIn(
            decision.b64_decision,
            {"vagal_recovery_hold", "continue_recovery_brake"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b64_recovery_tone), 0.0)
        self.assertGreater(float(decision.b64_vagal_brake), 0.0)
        self.assertGreater(float(decision.b64_post_escape_bias), 0.0)
        self.assertGreaterEqual(int(decision.b64_recovery_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b63_periaqueductal_escape_sequences_b62_defense(self) -> None:
        build_b63 = getattr(
            b_series_evolution_module,
            "build_b63_periaqueductal_escape_config",
            None,
        )
        self.assertIsNotNone(build_b63)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b62_defensive_mode_source(tmpdir)
            config = build_b63(
                "b63_periaqueductal_escape_sequence_h48_bridge_policy",
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=89, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 9.0
        meta["shelter_dist"] = 1.0
        meta["shelter_role"] = "outside"
        for tick in range(63, 80):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.80},
                    sleep={"health": 0.55, "sleep_debt": 0.28, "on_shelter": 0.0},
                    threat={
                        "predator_smell_strength": 0.34,
                        "predator_motion_salience": 0.24,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B63-periaqueductal-escape")
        self.assertEqual(
            decision.semantic_action_source,
            "b63_periaqueductal_escape_controller",
        )
        self.assertEqual(decision.b63_controller_profile, "periaqueductal_escape_sequence")
        self.assertIn(
            decision.b63_decision,
            {
                "pag_escape_to_shelter",
                "pag_freeze_then_escape",
                "continue_escape_sequence",
            },
        )
        self.assertIn(
            decision.b63_escape_phase,
            {"flight", "freeze", "sequence_lock"},
        )
        self.assertGreater(float(decision.b63_escape_urgency), 0.0)
        self.assertGreater(float(decision.b63_sequence_pressure), 0.0)
        self.assertGreaterEqual(int(decision.b63_escape_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b62_defensive_mode_uses_b61_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b61_amygdala_safety_source(tmpdir)
            config = build_b62_defensive_mode_selector_config(
                B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=88, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 9.0
        meta["shelter_dist"] = 1.0
        meta["shelter_role"] = "outside"
        for tick in range(62, 76):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.82},
                    sleep={"health": 0.60, "sleep_debt": 0.25, "on_shelter": 0.0},
                    threat={
                        "predator_smell_strength": 0.30,
                        "predator_motion_salience": 0.20,
                    },
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B62_DEFENSIVE_MODE_SELECTOR_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b62_controller_profile, "defensive_mode_selector")
        self.assertIn(
            decision.b62_decision,
            {"defensive_flee_to_shelter", "continue_defense_lock"},
        )
        self.assertIn(decision.b62_defensive_mode, {"flee_to_shelter", "continue_defense_lock"})
        self.assertGreaterEqual(int(decision.b62_defense_lock), 1)
        self.assertGreater(float(decision.b62_flee_pressure), 0.0)
        self.assertGreater(float(decision.b62_shelter_bias), 0.0)
        self.assertNotEqual(float(decision.b62_defense_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b2_temporal_threat_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            config = build_b2_temporal_threat_config(
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=18, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["STAY"]
        ] = 10.0
        meta = dict(_bridge_observation()["meta"])
        meta["on_shelter"] = False
        meta["shelter_role"] = "outside"
        meta["memory_vectors"] = {
            "predator": {"dx": 0.4, "dy": 0.0, "age": 0.1, "ttl": 10}
        }

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.45},
                sleep={"health": 1.0, "on_shelter": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL)
        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertEqual(
            decision.semantic_action_source,
            B2_TEMPORAL_THREAT_SELECTION_SOURCE,
        )
        self.assertGreaterEqual(float(decision.b_temporal_threat_pressure), 0.70)
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b2_predator_trace_vetoes_hunger_release_from_shelter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            config = build_b2_temporal_threat_config(
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=18, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta.update(
            {
                "on_shelter": True,
                "shelter_role": "deep",
                "shelter_role_level": 1.0,
                "percept_traces": {
                    "predator": {
                        "strength": 0.95,
                        "freshness": 0.95,
                        "certainty": 0.95,
                    }
                },
            }
        )

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.65},
                sleep={
                    "health": 1.0,
                    "on_shelter": 1.0,
                    "shelter_role_level": 1.0,
                },
            ),
            sample=False,
        )

        self.assertEqual(float(decision.b_predator_trace_pressure), 0.95)
        self.assertEqual(decision.semantic_action, "STAY")
        self.assertEqual(decision.semantic_action_reason, "b2_temporal_threat_hold_deep")

    def test_b2_safe_hunger_release_requires_safe_current_threat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            config = build_b2_temporal_threat_config(
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=18, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta.update(
            {
                "on_shelter": True,
                "shelter_role": "deep",
                "shelter_role_level": 1.0,
                "percept_traces": {
                    "predator": {
                        "strength": 0.50,
                        "freshness": 0.50,
                        "certainty": 0.50,
                    }
                },
            }
        )

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.65},
                sleep={
                    "health": 1.0,
                    "on_shelter": 1.0,
                    "shelter_role_level": 1.0,
                },
            ),
            sample=False,
        )

        self.assertEqual(float(decision.b_predator_trace_pressure), 0.50)
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(
            decision.semantic_action_reason,
            "b2_temporal_threat_safe_hunger_release",
        )

        threatened_decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.65},
                sleep={
                    "health": 1.0,
                    "on_shelter": 1.0,
                    "shelter_role_level": 1.0,
                },
                threat={"predator_visible": 0.95},
            ),
            sample=False,
        )

        self.assertEqual(float(threatened_decision.b_current_threat_pressure), 0.95)
        self.assertEqual(threatened_decision.semantic_action, "STAY")
        self.assertEqual(
            threatened_decision.semantic_action_reason,
            "b2_temporal_threat_hold_deep",
        )

    def test_b3_contact_memory_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=19, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["STAY"]
        ] = 10.0
        brain.set_direct_policy_event_clock(5)

        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45},
                sleep={"health": 0.70, "on_shelter": 0.0},
                threat={
                    "recent_contact": 1.0,
                    "recent_pain": 0.5,
                    "predator_smell_strength": 0.6,
                },
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B3_CONTACT_MEMORY_EFFECTIVE_LEVEL)
        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertEqual(
            decision.semantic_action_source,
            B3_CONTACT_MEMORY_SELECTION_SOURCE,
        )
        self.assertGreater(int(decision.b3_contact_cooldown), 0)
        self.assertEqual(decision.b3_controller_profile, "standard")
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b3_contact_memory_cooldowns_reset_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=20, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(8)
        first = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45},
                sleep={"health": 0.70, "on_shelter": 0.0},
                threat={"recent_contact": 1.0},
            ),
            sample=False,
        )
        self.assertGreater(int(first.b3_contact_cooldown), 0)

        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45, "on_food": 0.0},
                sleep={"health": 1.0, "on_shelter": 0.0},
                threat={"recent_contact": 0.0, "recent_pain": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(int(second.b3_contact_cooldown), 0)
        self.assertEqual(int(second.b3_post_food_cooldown), 0)
