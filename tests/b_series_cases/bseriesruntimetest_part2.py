from __future__ import annotations

from .shared import *



class BSeriesRuntimeTestPart2(unittest.TestCase):
    def test_b29_salience_competition_uses_b28_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b28_interoceptive_attention_source(tmpdir)
            config = build_b29_salience_competition_config(
                B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=55, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(29)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B29_SALIENCE_COMPETITION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B29_SALIENCE_COMPETITION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b29_controller_profile, "salience_competition")
        self.assertIn(
            decision.b29_decision,
            {"salience_competition_continue", "continue_salience_lock"},
        )
        self.assertNotEqual(decision.b29_salience_state, "non_corridor")
        self.assertIn(decision.b29_winner_channel, {"corridor", "homeostasis", "threat"})
        self.assertGreaterEqual(int(decision.b29_salience_lock), 1)
        self.assertGreater(float(decision.b29_corridor_salience), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b30_basal_ganglia_gate_uses_b29_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b29_salience_competition_source(tmpdir)
            config = build_b30_basal_ganglia_gate_config(
                B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=56, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(30)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B30_BASAL_GANGLIA_GATE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b30_controller_profile, "basal_ganglia_gate")
        self.assertIn(
            decision.b30_decision,
            {"basal_gate_go", "continue_basal_gate_lock"},
        )
        self.assertNotEqual(decision.b30_gate_state, "non_corridor")
        self.assertIn(decision.b30_action_gate, {"go", "no_go"})
        self.assertGreaterEqual(int(decision.b30_gate_lock), 1)
        self.assertGreater(float(decision.b30_go_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b31_dopamine_prediction_error_uses_b30_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b30_basal_ganglia_gate_source(tmpdir)
            config = build_b31_dopamine_prediction_error_config(
                B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=57, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(31)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B31_DOPAMINE_PREDICTION_ERROR_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b31_controller_profile, "dopamine_prediction_error")
        self.assertIn(
            decision.b31_decision,
            {"dopamine_gate_go", "continue_dopamine_lock"},
        )
        self.assertNotEqual(decision.b31_dopamine_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b31_dopamine_lock), 1)
        self.assertGreater(float(decision.b31_gate_bias), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b32_actor_critic_value_uses_b31_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b31_dopamine_prediction_error_source(tmpdir)
            config = build_b32_actor_critic_value_config(
                B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=58, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(32)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B32_ACTOR_CRITIC_VALUE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b32_controller_profile, "actor_critic_value")
        self.assertIn(
            decision.b32_decision,
            {"actor_critic_commit", "continue_value_lock"},
        )
        self.assertGreaterEqual(int(decision.b32_value_lock), 1)
        self.assertGreater(float(decision.b32_actor_advantage), 0.0)
        self.assertGreater(float(decision.b32_policy_bias), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b33_td_error_decomposition_uses_b32_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b32_actor_critic_value_source(tmpdir)
            config = build_b33_td_error_decomposition_config(
                B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=59, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(33)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B33_TD_ERROR_DECOMPOSITION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b33_controller_profile, "td_error_decomposition")
        self.assertIn(decision.b33_decision, {"td_error_commit", "continue_td_lock"})
        self.assertGreaterEqual(int(decision.b33_td_lock), 1)
        self.assertGreater(float(decision.b33_td_error), 0.0)
        self.assertGreater(float(decision.b33_actor_update), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b34_eligibility_credit_uses_b33_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b33_td_error_decomposition_source(tmpdir)
            config = build_b34_eligibility_credit_config(
                B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=60, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(34)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B34_ELIGIBILITY_CREDIT_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b34_controller_profile, "eligibility_credit")
        self.assertIn(
            decision.b34_decision,
            {"eligibility_credit_commit", "continue_eligibility_lock"},
        )
        self.assertGreaterEqual(int(decision.b34_credit_lock), 1)
        self.assertGreater(float(decision.b34_eligibility_trace), 0.0)
        self.assertGreater(float(decision.b34_credit_assignment), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b35_forward_model_uses_b34_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b34_eligibility_credit_source(tmpdir)
            config = build_b35_forward_model_value_config(
                B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=61, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(35)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B35_FORWARD_MODEL_VALUE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b35_controller_profile, "forward_model_value")
        self.assertIn(
            decision.b35_decision,
            {"forward_model_commit", "continue_model_lock"},
        )
        self.assertGreaterEqual(int(decision.b35_model_lock), 1)
        self.assertGreater(float(decision.b35_forward_value), 0.0)
        self.assertGreater(float(decision.b35_prediction_memory), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b36_latent_belief_state_uses_b35_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b35_forward_model_value_source(tmpdir)
            config = build_b36_latent_belief_state_config(
                B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=62, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(36)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B36_LATENT_BELIEF_STATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B36_LATENT_BELIEF_STATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b36_controller_profile, "latent_belief_state")
        self.assertIn(
            decision.b36_decision,
            {"latent_belief_commit", "continue_belief_lock"},
        )
        self.assertGreaterEqual(int(decision.b36_belief_lock), 1)
        self.assertGreater(float(decision.b36_latent_state), 0.0)
        self.assertGreater(float(decision.b36_belief_error), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b37_state_factor_gate_uses_b36_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b36_latent_belief_state_source(tmpdir)
            config = build_b37_state_factor_gate_config(
                B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=63, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(37, 40):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B37_STATE_FACTOR_GATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B37_STATE_FACTOR_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b37_controller_profile, "state_factor_gate")
        self.assertIn(
            decision.b37_decision,
            {"state_factor_commit", "continue_factor_lock"},
        )
        self.assertGreaterEqual(int(decision.b37_factor_lock), 1)
        self.assertGreater(abs(float(decision.b37_external_state_factor)), 0.0)
        self.assertGreater(float(decision.b37_internal_state_factor), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b38_factor_attention_uses_b37_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b37_state_factor_gate_source(tmpdir)
            config = build_b38_factor_attention_config(
                B38_FACTOR_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=64, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(38, 42):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B38_FACTOR_ATTENTION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B38_FACTOR_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b38_controller_profile, "factor_attention")
        self.assertIn(
            decision.b38_decision,
            {"factor_attention_commit", "continue_attention_lock"},
        )
        self.assertGreaterEqual(int(decision.b38_attention_lock), 1)
        self.assertGreater(float(decision.b38_internal_attention), 0.0)
        self.assertGreater(float(decision.b38_attention_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b39_attention_binding_uses_b38_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b38_factor_attention_source(tmpdir)
            config = build_b39_attention_binding_config(
                B39_ATTENTION_BINDING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=65, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(39, 43):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B39_ATTENTION_BINDING_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B39_ATTENTION_BINDING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b39_controller_profile, "attention_binding")
        self.assertIn(
            decision.b39_decision,
            {"attention_binding_commit", "continue_binding_lock"},
        )
        self.assertGreaterEqual(int(decision.b39_binding_lock), 1)
        self.assertGreater(float(decision.b39_binding_strength), 0.0)
        self.assertGreater(float(decision.b39_binding_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b40_global_workspace_uses_b39_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b39_attention_binding_source(tmpdir)
            config = build_b40_global_workspace_config(
                B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=66, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(40, 45):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B40_GLOBAL_WORKSPACE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B40_GLOBAL_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b40_controller_profile, "global_workspace")
        self.assertIn(
            decision.b40_decision,
            {"global_workspace_commit", "continue_workspace_lock"},
        )
        self.assertGreaterEqual(int(decision.b40_workspace_lock), 1)
        self.assertGreater(float(decision.b40_workspace_activation), 0.0)
        self.assertGreater(float(decision.b40_broadcast_gain), 0.0)
        self.assertGreater(float(decision.b40_context_availability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b41_executive_workspace_uses_b40_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b40_global_workspace_source(tmpdir)
            config = build_b41_executive_workspace_config(
                B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=67, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(41, 46):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B41_EXECUTIVE_WORKSPACE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b41_controller_profile, "executive_workspace")
        self.assertIn(
            decision.b41_decision,
            {"executive_workspace_select", "continue_executive_lock"},
        )
        self.assertGreaterEqual(int(decision.b41_executive_lock), 1)
        self.assertGreater(float(decision.b41_executive_selection), 0.0)
        self.assertGreater(float(decision.b41_goal_context), 0.0)
        self.assertGreater(float(decision.b41_executive_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b42_error_monitor_uses_b41_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b41_executive_workspace_source(tmpdir)
            config = build_b42_error_monitor_config(
                B42_ERROR_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=68, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(42, 47):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B42_ERROR_MONITOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B42_ERROR_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b42_controller_profile, "error_monitor")
        self.assertIn(
            decision.b42_decision,
            {"error_monitor_commit", "continue_monitor_lock"},
        )
        self.assertGreaterEqual(int(decision.b42_monitor_lock), 1)
        self.assertGreaterEqual(float(decision.b42_error_signal), 0.0)
        self.assertGreaterEqual(float(decision.b42_conflict_signal), 0.0)
        self.assertGreater(float(decision.b42_performance_context), 0.0)
        self.assertGreater(float(decision.b42_monitor_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b43_adaptive_precision_uses_b42_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b42_error_monitor_source(tmpdir)
            config = build_b43_adaptive_precision_config(
                B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=69, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(43, 48):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B43_ADAPTIVE_PRECISION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B43_ADAPTIVE_PRECISION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b43_controller_profile, "adaptive_precision")
        self.assertIn(
            decision.b43_decision,
            {"adaptive_precision_commit", "continue_precision_lock"},
        )
        self.assertGreaterEqual(int(decision.b43_precision_lock), 1)
        self.assertGreater(float(decision.b43_precision_signal), 0.0)
        self.assertGreater(float(decision.b43_adaptive_threshold), 0.0)
        self.assertGreaterEqual(float(decision.b43_arousal_context), 0.0)
        self.assertGreater(float(decision.b43_control_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b44_thalamic_relay_uses_b43_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b43_adaptive_precision_source(tmpdir)
            config = build_b44_thalamic_relay_config(
                B44_THALAMIC_RELAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=70, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(44, 49):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B44_THALAMIC_RELAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B44_THALAMIC_RELAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b44_controller_profile, "thalamic_relay")
        self.assertIn(
            decision.b44_decision,
            {"thalamic_relay_commit", "continue_relay_lock"},
        )
        self.assertGreaterEqual(int(decision.b44_relay_lock), 1)
        self.assertGreater(float(decision.b44_relay_gate), 0.0)
        self.assertGreater(float(decision.b44_sensory_precision), 0.0)
        self.assertGreater(float(decision.b44_context_relay), 0.0)
        self.assertGreater(float(decision.b44_gate_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b45_reticular_inhibition_uses_b44_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b44_thalamic_relay_source(tmpdir)
            config = build_b45_reticular_inhibition_config(
                B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=71, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(45, 50):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B45_RETICULAR_INHIBITION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B45_RETICULAR_INHIBITION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b45_controller_profile, "reticular_inhibition")
        self.assertIn(
            decision.b45_decision,
            {"reticular_inhibition_commit", "continue_inhibition_lock"},
        )
        self.assertGreaterEqual(int(decision.b45_inhibition_lock), 1)
        self.assertGreater(float(decision.b45_inhibitory_gate), 0.0)
        self.assertGreater(float(decision.b45_sensory_filter), 0.0)
        self.assertGreater(float(decision.b45_context_suppression), 0.0)
        self.assertGreater(float(decision.b45_loop_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b46_corticothalamic_feedback_uses_b45_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b45_reticular_inhibition_source(tmpdir)
            config = build_b46_corticothalamic_feedback_config(
                B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=72, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(46, 51):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B46_CORTICOTHALAMIC_FEEDBACK_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b46_controller_profile, "corticothalamic_feedback")
        self.assertIn(
            decision.b46_decision,
            {"corticothalamic_feedback_commit", "continue_feedback_lock"},
        )
        self.assertGreaterEqual(int(decision.b46_feedback_lock), 1)
        self.assertGreater(float(decision.b46_feedback_gain), 0.0)
        self.assertGreater(float(decision.b46_topdown_context), 0.0)
        self.assertGreater(float(decision.b46_prediction_match), 0.0)
        self.assertGreater(float(decision.b46_feedback_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b47_oscillatory_synchrony_uses_b46_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b46_corticothalamic_feedback_source(tmpdir)
            config = build_b47_oscillatory_synchrony_config(
                B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=73, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(47, 53):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B47_OSCILLATORY_SYNCHRONY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b47_controller_profile, "oscillatory_synchrony")
        self.assertIn(
            decision.b47_decision,
            {"oscillatory_synchrony_commit", "continue_phase_lock"},
        )
        self.assertGreaterEqual(int(decision.b47_phase_lock), 1)
        self.assertGreater(float(decision.b47_phase_alignment), 0.0)
        self.assertGreater(float(decision.b47_synchrony_gain), 0.0)
        self.assertGreater(float(decision.b47_cross_loop_coherence), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b48_cerebellar_timing_uses_b47_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b47_oscillatory_synchrony_source(tmpdir)
            config = build_b48_cerebellar_timing_config(
                B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=74, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(48, 54):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B48_CEREBELLAR_TIMING_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B48_CEREBELLAR_TIMING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b48_controller_profile, "cerebellar_timing")
        self.assertIn(
            decision.b48_decision,
            {"cerebellar_timing_commit", "continue_calibration_lock"},
        )
        self.assertGreaterEqual(int(decision.b48_calibration_lock), 1)
        self.assertGreater(float(decision.b48_timing_error), 0.0)
        self.assertGreater(float(decision.b48_predictive_timing), 0.0)
        self.assertGreater(float(decision.b48_corrective_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b49_striatal_action_gate_uses_b48_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b48_cerebellar_timing_source(tmpdir)
            config = build_b49_striatal_action_gate_config(
                B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=75, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(49, 55):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B49_STRIATAL_ACTION_GATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b49_controller_profile, "striatal_action_gate")
        self.assertIn(
            decision.b49_decision,
            {"striatal_gate_commit", "continue_selection_lock"},
        )
        self.assertGreaterEqual(int(decision.b49_selection_lock), 1)
        self.assertGreater(float(decision.b49_go_signal), 0.0)
        self.assertGreaterEqual(float(decision.b49_no_go_signal), 0.0)
        self.assertGreater(float(decision.b49_action_gate_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b50_habit_chunking_uses_b49_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b49_striatal_action_gate_source(tmpdir)
            config = build_b50_habit_chunking_config(
                B50_HABIT_CHUNKING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=76, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(50, 56):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B50_HABIT_CHUNKING_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B50_HABIT_CHUNKING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b50_controller_profile, "habit_chunking")
        self.assertIn(
            decision.b50_decision,
            {"habit_chunk_commit", "continue_habit_chunk"},
        )
        self.assertGreaterEqual(int(decision.b50_chunk_lock), 1)
        self.assertGreater(float(decision.b50_habit_strength), 0.0)
        self.assertGreater(float(decision.b50_chunk_value), 0.0)
        self.assertGreater(float(decision.b50_habit_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b51_dopaminergic_habit_uses_b50_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b50_habit_chunking_source(tmpdir)
            config = build_b51_dopaminergic_habit_modulation_config(
                B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=77, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(51, 57):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B51_DOPAMINERGIC_HABIT_MODULATION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE,
        )
        self.assertEqual(
            decision.b51_controller_profile,
            "dopaminergic_habit_modulation",
        )
        self.assertIn(
            decision.b51_decision,
            {"dopamine_habit_commit", "continue_dopamine_modulation"},
        )
        self.assertGreaterEqual(int(decision.b51_modulation_lock), 1)
        self.assertGreater(float(decision.b51_prediction_error), 0.0)
        self.assertGreater(float(decision.b51_dopamine_gain), 0.0)
        self.assertGreater(float(decision.b51_habit_modulation), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b52_cholinergic_precision_uses_b51_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b51_dopaminergic_habit_source(tmpdir)
            config = build_b52_cholinergic_precision_gate_config(
                B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=78, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(52, 58):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B52_CHOLINERGIC_PRECISION_GATE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b52_controller_profile, "cholinergic_precision_gate")
        self.assertIn(
            decision.b52_decision,
            {"cholinergic_precision_commit", "continue_precision_attention"},
        )
        self.assertGreaterEqual(int(decision.b52_attention_lock), 1)
        self.assertGreater(float(decision.b52_acetylcholine_level), 0.0)
        self.assertGreater(float(decision.b52_precision_gain), 0.0)
        self.assertGreater(float(decision.b52_uncertainty_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b53_noradrenergic_arousal_uses_b52_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b52_cholinergic_precision_source(tmpdir)
            config = build_b53_noradrenergic_arousal_gain_config(
                B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=79, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(53, 59):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B53_NORADRENERGIC_AROUSAL_GAIN_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b53_controller_profile, "noradrenergic_arousal_gain")
        self.assertIn(
            decision.b53_decision,
            {"noradrenergic_arousal_commit", "continue_arousal_gain"},
        )
        self.assertGreaterEqual(int(decision.b53_gain_lock), 1)
        self.assertGreater(float(decision.b53_norepinephrine_level), 0.0)
        self.assertGreater(float(decision.b53_arousal_gain), 0.0)
        self.assertGreater(float(decision.b53_surprise_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b54_serotonergic_patience_uses_b53_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b53_noradrenergic_arousal_source(tmpdir)
            config = build_b54_serotonergic_patience_gate_config(
                B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=80, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(54, 61):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B54_SEROTONERGIC_PATIENCE_GATE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b54_controller_profile, "serotonergic_patience_gate")
        self.assertIn(
            decision.b54_decision,
            {"serotonergic_patience_commit", "continue_patience_lock"},
        )
        self.assertGreaterEqual(int(decision.b54_patience_lock), 1)
        self.assertGreater(float(decision.b54_serotonin_level), 0.0)
        self.assertGreater(float(decision.b54_patience_signal), 0.0)
        self.assertGreaterEqual(float(decision.b54_impulse_suppression), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b55_hypothalamic_drive_uses_b54_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b54_serotonergic_patience_source(tmpdir)
            config = build_b55_hypothalamic_drive_coupling_config(
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=81, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(55, 63):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B55_HYPOTHALAMIC_DRIVE_COUPLING_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b55_controller_profile, "hypothalamic_drive_coupling")
        self.assertIn(
            decision.b55_decision,
            {"hypothalamic_drive_commit", "continue_drive_lock"},
        )
        self.assertGreaterEqual(int(decision.b55_drive_lock), 1)
        self.assertGreater(float(decision.b55_hypothalamic_drive), 0.0)
        self.assertGreater(float(decision.b55_satiety_signal), 0.0)
        self.assertGreater(float(decision.b55_recovery_bias), 0.0)
        self.assertNotEqual(float(decision.b55_drive_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b56_hpa_stress_axis_uses_b55_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b55_hypothalamic_drive_source(tmpdir)
            config = build_b56_hpa_stress_axis_config(
                B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=82, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(56, 65):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B56_HPA_STRESS_AXIS_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B56_HPA_STRESS_AXIS_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b56_controller_profile, "hpa_stress_axis")
        self.assertIn(
            decision.b56_decision,
            {"hpa_stress_axis_commit", "continue_stress_axis_lock"},
        )
        self.assertGreaterEqual(int(decision.b56_stress_lock), 1)
        self.assertGreater(float(decision.b56_cortisol_level), 0.0)
        self.assertGreater(float(decision.b56_stress_load), 0.0)
        self.assertGreater(float(decision.b56_recovery_signal), 0.0)
        self.assertNotEqual(float(decision.b56_endocrine_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b57_insular_interoception_uses_b56_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b56_hpa_stress_source(tmpdir)
            config = build_b57_insular_interoceptive_awareness_config(
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=83, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(57, 67):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE,
        )
        self.assertEqual(
            decision.b57_controller_profile,
            "insular_interoceptive_awareness",
        )
        self.assertIn(
            decision.b57_decision,
            {"interoceptive_awareness_commit", "continue_awareness_lock"},
        )
        self.assertGreaterEqual(int(decision.b57_awareness_lock), 1)
        self.assertGreater(float(decision.b57_interoceptive_awareness), 0.0)
        self.assertGreater(float(decision.b57_visceral_salience), 0.0)
        self.assertGreater(float(decision.b57_body_state_confidence), 0.0)
        self.assertNotEqual(float(decision.b57_awareness_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b58_acc_conflict_monitor_uses_b57_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b57_insular_interoceptive_source(tmpdir)
            config = build_b58_acc_conflict_monitor_config(
                B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=84, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(58, 69):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B58_ACC_CONFLICT_MONITOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b58_controller_profile, "acc_conflict_monitor")
        self.assertIn(
            decision.b58_decision,
            {"acc_conflict_commit", "continue_conflict_lock"},
        )
        self.assertGreaterEqual(int(decision.b58_conflict_lock), 1)
        self.assertGreater(float(decision.b58_conflict_signal), 0.0)
        self.assertGreater(float(decision.b58_error_likelihood), 0.0)
        self.assertGreater(float(decision.b58_control_allocation), 0.0)
        self.assertNotEqual(float(decision.b58_resolution_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b59_prefrontal_goal_context_uses_b58_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b58_acc_conflict_source(tmpdir)
            config = build_b59_prefrontal_goal_context_config(
                B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=85, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(59, 71):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B59_PREFRONTAL_GOAL_CONTEXT_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b59_controller_profile, "prefrontal_goal_context")
        self.assertIn(
            decision.b59_decision,
            {"prefrontal_goal_commit", "continue_executive_lock"},
        )
        self.assertGreaterEqual(int(decision.b59_executive_lock), 1)
        self.assertGreater(float(decision.b59_goal_context), 0.0)
        self.assertGreater(float(decision.b59_working_set_stability), 0.0)
        self.assertGreater(float(decision.b59_task_set_confidence), 0.0)
        self.assertNotEqual(float(decision.b59_executive_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b60_orbitofrontal_value_uses_b59_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b59_prefrontal_goal_source(tmpdir)
            config = build_b60_orbitofrontal_outcome_value_config(
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=86, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(60, 73):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B60_ORBITOFRONTAL_OUTCOME_VALUE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b60_controller_profile, "orbitofrontal_outcome_value")
        self.assertIn(
            decision.b60_decision,
            {"orbitofrontal_value_commit", "continue_value_lock"},
        )
        self.assertGreaterEqual(int(decision.b60_value_lock), 1)
        self.assertNotEqual(float(decision.b60_outcome_value), 0.0)
        self.assertGreaterEqual(float(decision.b60_reversal_signal), 0.0)
        self.assertGreater(float(decision.b60_goal_value_confidence), 0.0)
        self.assertNotEqual(float(decision.b60_value_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b61_amygdala_safety_uses_b60_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b60_orbitofrontal_value_source(tmpdir)
            config = build_b61_amygdala_safety_value_config(
                B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=87, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(61, 75):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B61_AMYGDALA_SAFETY_VALUE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b61_controller_profile, "amygdala_safety_value")
        self.assertIn(
            decision.b61_decision,
            {"amygdala_safety_commit", "continue_safety_lock"},
        )
        self.assertGreaterEqual(int(decision.b61_safety_lock), 1)
        self.assertGreater(float(decision.b61_safety_value), 0.0)
        self.assertGreaterEqual(float(decision.b61_threat_value), 0.0)
        self.assertGreater(float(decision.b61_safety_confidence), 0.0)
        self.assertNotEqual(float(decision.b61_affective_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)
