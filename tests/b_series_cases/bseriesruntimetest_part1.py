from __future__ import annotations

from .shared import *



class BSeriesRuntimeTestPart1(unittest.TestCase):
    def test_b0_current_bridge_never_emits_semantic_action_to_world(self) -> None:
        brain = SpiderBrain(seed=11, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["SLEEP"]
        ] = 10.0

        decision = brain.act_inference(
            _brain_observation(
                hunger={
                    "hunger": 0.82,
                    "food_visible": 1.0,
                    "food_certainty": 1.0,
                },
                sleep={"health": 1.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "SLEEP")
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.b_effective_level, B_CURRENT_BRIDGE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B_CURRENT_BRIDGE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertEqual(decision.bridge_primitive_action, "MOVE_RIGHT")
        self.assertEqual(ACTIONS[decision.action_idx], "MOVE_RIGHT")
        self.assertNotIn(decision.semantic_action, ACTIONS)
        self.assertEqual(decision.external_override_count, 0)

    def test_b0_current_rest_phase_maps_to_stay_without_final_bias(self) -> None:
        brain = SpiderBrain(seed=12, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["MOVE_TO_FOOD"]
        ] = 10.0

        meta = _bridge_observation()["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        geodesics["MOVE_LEFT"] = {
            "exit_geodesic_delta": 0.0,
            "deep_geodesic_delta": 0.0,
            "next_on_exit_target": False,
            "next_on_deep_target": True,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = brain.act_inference(
            _brain_observation(
                meta,
                sleep={
                    "fatigue": 0.90,
                    "hunger": 0.12,
                    "on_shelter": 1.0,
                    "night": 1.0,
                    "health": 1.0,
                    "sleep_debt": 0.90,
                    "shelter_role_level": 1.0,
                },
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.bridge_primitive_action, "STAY")
        self.assertEqual(ACTIONS[decision.action_idx], "STAY")
        self.assertEqual(decision.semantic_override_count, 1)

    def test_b0_current_critical_hunger_can_leave_deep_shelter_under_residual_smell(self) -> None:
        brain = SpiderBrain(seed=13, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[B_SEMANTIC_ACTION_TO_INDEX["STAY"]] = 10.0

        meta = _bridge_observation()["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        geodesics["MOVE_LEFT"] = {
            "exit_geodesic_delta": 0.0,
            "deep_geodesic_delta": 0.0,
            "next_on_exit_target": False,
            "next_on_deep_target": True,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.92},
                sleep={
                    "fatigue": 0.12,
                    "hunger": 0.92,
                    "on_shelter": 1.0,
                    "health": 1.0,
                    "shelter_role_level": 1.0,
                },
                threat={"predator_smell_strength": 0.70},
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.bridge_primitive_action, "MOVE_UP")
        self.assertEqual(ACTIONS[decision.action_idx], "MOVE_UP")

    def test_b1_uses_network_policy_without_b0_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SpiderBrain(seed=14, module_dropout=0.0, config=_b0_config())
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_CAPACITY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=15, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["SLEEP"]
        ] = 10.0

        decision = brain.act_inference(
            _brain_observation(
                hunger={
                    "hunger": 0.82,
                    "food_visible": 1.0,
                    "food_certainty": 1.0,
                },
                sleep={"health": 1.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, "B1")
        self.assertEqual(decision.learned_semantic_action, "SLEEP")
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.semantic_action_source, "network_policy")
        self.assertEqual(decision.semantic_override_count, 0)
        self.assertEqual(decision.bridge_primitive_action, "STAY")
        self.assertEqual(ACTIONS[decision.action_idx], "STAY")

    def test_b1_threat_guard_uses_transfer_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SpiderBrain(seed=16, module_dropout=0.0, config=_b0_config())
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_THREAT_GUARD_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=17, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["STAY"]
        ] = 10.0

        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45},
                sleep={"health": 1.0, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.70},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B1_THREAT_GUARD_EFFECTIVE_LEVEL)
        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertEqual(
            decision.semantic_action_source,
            B1_THREAT_GUARD_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b3_recurrent_guard_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_RECURRENT_GUARD_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=23, module_dropout=0.0, config=config)

        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.90},
                sleep={"health": 0.70, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.6},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B3_RECURRENT_GUARD_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B3_RECURRENT_GUARD_SELECTION_SOURCE,
        )
        self.assertIn(decision.bridge_primitive_action, ACTIONS)
        self.assertEqual(decision.b3_controller_profile, "recurrent_guard_strict_until_60")

    def test_b4_recovery_balance_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            config = build_b4_recovery_balance_config(
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=24, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["MOVE_TO_FOOD"]
        ] = 10.0
        brain.set_direct_policy_event_clock(20)
        brain.act_inference(
            _brain_observation(hunger={"hunger": 0.90}),
            sample=False,
        )
        brain.set_direct_policy_event_clock(21)
        meta = dict(_bridge_observation()["meta"])
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.50},
                sleep={
                    "health": 0.35,
                    "fatigue": 0.70,
                    "sleep_debt": 0.70,
                    "on_shelter": 1.0,
                    "shelter_role_level": 1.0,
                },
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B4_RECOVERY_BALANCE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.b4_controller_profile, "recovery_balance")
        self.assertGreaterEqual(float(decision.b4_recovery_pressure), 0.60)
        self.assertTrue(decision.b4_sleep_hold)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b4_genetic_recovery_records_genetic_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            config = build_b4_recovery_balance_config(
                B4_GENETIC_RECOVERY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
                controller_profile="genetic_recovery",
                controller_params={"ga_generation": 2, "ga_candidate": 3},
            )
            brain = SpiderBrain(seed=25, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(20)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.50},
                sleep={"health": 0.35, "on_shelter": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.semantic_action_source,
            B4_GENETIC_RECOVERY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b4_genetic_generation, 2)
        self.assertEqual(decision.b4_genetic_candidate, 3)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b5_homeostatic_arbiter_uses_transfer_memory_and_primitive_bridge(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=26, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["MOVE_TO_FOOD"]
        ] = 10.0
        brain.set_direct_policy_event_clock(69)
        brain.act_inference(
            _brain_observation(hunger={"hunger": 0.90}),
            sample=False,
        )
        brain.set_direct_policy_event_clock(70)
        meta = dict(_bridge_observation()["meta"])
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.30},
                sleep={
                    "health": 0.40,
                    "fatigue": 0.88,
                    "sleep_debt": 0.86,
                    "on_shelter": 1.0,
                },
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.b5_controller_profile, "homeostatic_arbiter")
        self.assertGreaterEqual(float(decision.b5_sleep_pressure), 0.80)
        self.assertGreaterEqual(int(decision.b5_sleep_bout_lock), 1)
        self.assertEqual(decision.b5_homeostatic_decision, "sleep_bout_hold")
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b5_genetic_homeostasis_records_genetic_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
                controller_profile="genetic_homeostasis",
                controller_params={"ga_generation": 3, "ga_candidate": 4},
            )
            brain = SpiderBrain(seed=27, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(0)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.92},
                sleep={"health": 0.80, "fatigue": 0.20, "sleep_debt": 0.20},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.semantic_action_source,
            B5_GENETIC_HOMEOSTASIS_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b5_genetic_generation, 3)
        self.assertEqual(decision.b5_genetic_candidate, 4)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b5_homeostatic_locks_reset_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=28, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(10)
        first = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.92},
                sleep={"health": 0.85, "fatigue": 0.20, "sleep_debt": 0.20},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )
        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.20},
                sleep={"health": 0.85, "fatigue": 0.20, "sleep_debt": 0.20},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertGreaterEqual(int(first.b5_forage_commitment_lock), 1)
        self.assertEqual(int(second.b5_forage_commitment_lock), 0)

    def test_b6_risk_corridor_uses_b5_transfer_and_action_center_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=29, module_dropout=0.0, config=config)

        bus = MessageBus()
        bus.set_tick(20)
        brain.set_direct_policy_event_clock(20)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.85},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 1.0, "predator_certainty": 1.0},
            ),
            bus=bus,
            sample=False,
        )
        action_center_messages = [
            message for message in bus.topic_messages("action.selection")
            if message.sender == "action_center"
        ]

        self.assertEqual(decision.b_effective_level, B6_RISK_CORRIDOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B6_RISK_CORRIDOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b6_controller_family, "risk_corridor")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertGreater(float(decision.b6_risk_pressure), 0.9)
        self.assertEqual(float(decision.b6_threat_priority), 1.0)
        self.assertEqual(float(decision.b6_forage_suppressed), 1.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)
        self.assertEqual(len(action_center_messages), 1)
        self.assertEqual(
            action_center_messages[0].payload["winning_valence"],
            "threat",
        )
        self.assertLess(
            action_center_messages[0].payload["module_gates"]["hunger_center"],
            0.5,
        )

    def test_b6_recurrent_memory_resets_locks_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=30, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(12)
        first = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.85},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 1.0, "predator_certainty": 1.0},
            ),
            sample=False,
        )
        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.30},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 0.0, "predator_certainty": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            first.semantic_action_source,
            B6_RECURRENT_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(first.b_effective_level, B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL)
        self.assertGreaterEqual(int(first.b6_return_lock), 1)
        self.assertEqual(int(second.b6_return_lock), 0)

    def test_b6_fused_controller_records_fused_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
                controller_profile="fused_risk_recurrent",
                controller_params={"ga_generation": 1, "ga_candidate": 2},
            )
            brain = SpiderBrain(seed=31, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.50},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 1.0, "predator_certainty": 1.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b6_controller_family, "fused_risk_recurrent")
        self.assertEqual(decision.b6_genetic_generation, 1)
        self.assertEqual(decision.b6_genetic_candidate, 2)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b7_affordance_budget_uses_b6_transfer_and_records_viability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            config = build_b7_affordance_budget_config(
                B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=32, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 10.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.30, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B7_AFFORDANCE_BUDGET_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B7_AFFORDANCE_BUDGET_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b7_controller_profile, "affordance_budget")
        self.assertEqual(decision.b7_decision, "abort_return_unviable")
        self.assertTrue(decision.b7_abort_return)
        self.assertIsNotNone(decision.b7_budget_margin)
        self.assertEqual(decision.b7_food_steps_estimate, 10.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b7_affordance_locks_reset_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            config = build_b7_affordance_budget_config(
                B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=33, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 5.0
        meta["shelter_dist"] = 1.0
        brain.set_direct_policy_event_clock(12)
        first = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.90, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )
        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.20},
                sleep={"health": 0.90, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(first.b7_decision, "continue_viable")
        self.assertGreaterEqual(int(first.b7_commitment_lock), 1)
        self.assertEqual(int(second.b7_commitment_lock), 0)

    def test_b8_spatial_affordance_uses_b7_transfer_and_records_map(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b7_affordance_budget_source(tmpdir)
            config = build_b8_spatial_affordance_config(
                B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=34, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B8_SPATIAL_AFFORDANCE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b8_controller_profile, "spatial_affordance_map")
        self.assertEqual(decision.b8_decision, "corridor_continue_mapped")
        self.assertEqual(decision.b8_spatial_map_state, "food_vector_available")
        self.assertIsNotNone(decision.b8_local_affordance_score)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b9_waypoint_planner_uses_b8_transfer_and_records_route(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b8_spatial_affordance_source(tmpdir)
            config = build_b9_waypoint_planner_config(
                B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=35, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B9_WAYPOINT_PLANNER_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B9_WAYPOINT_PLANNER_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b9_controller_profile, "waypoint_planner")
        self.assertEqual(decision.b9_decision, "commit_food_waypoint")
        self.assertEqual(decision.b9_route_state, "food_waypoint_locked")
        self.assertGreaterEqual(int(decision.b9_waypoint_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b10_prospective_replay_uses_b9_transfer_and_records_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b9_waypoint_planner_source(tmpdir)
            config = build_b10_prospective_replay_config(
                B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=36, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B10_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b10_controller_profile, "prospective_replay")
        self.assertEqual(decision.b10_decision, "commit_replayed_route")
        self.assertEqual(decision.b10_replay_state, "prospective_food_plan")
        self.assertGreaterEqual(int(decision.b10_plan_commitment), 1)
        self.assertGreater(float(decision.b10_prospective_value), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b11_confidence_arbiter_uses_b10_transfer_and_records_confidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b10_prospective_replay_source(tmpdir)
            config = build_b11_confidence_arbiter_config(
                B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=37, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B11_CONFIDENCE_ARBITER_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B11_CONFIDENCE_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b11_controller_profile, "confidence_arbiter")
        self.assertEqual(decision.b11_decision, "commit_confident_plan")
        self.assertEqual(decision.b11_confidence_state, "high_confidence_plan")
        self.assertGreaterEqual(int(decision.b11_confidence_lock), 1)
        self.assertGreater(float(decision.b11_neuromod_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b12_predictive_attention_uses_b11_transfer_and_records_attention(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b11_confidence_arbiter_source(tmpdir)
            config = build_b12_predictive_attention_config(
                B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=38, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B12_PREDICTIVE_ATTENTION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b12_controller_profile, "predictive_attention")
        self.assertEqual(decision.b12_decision, "commit_attended_affordance")
        self.assertEqual(decision.b12_attention_state, "attended_food_affordance")
        self.assertGreaterEqual(int(decision.b12_search_lock), 1)
        self.assertGreater(float(decision.b12_attention_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b13_local_search_uses_b12_transfer_and_records_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b12_predictive_attention_source(tmpdir)
            config = build_b13_local_affordance_search_config(
                B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=39, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(16)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B13_LOCAL_SEARCH_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B13_LOCAL_SEARCH_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b13_controller_profile, "local_affordance_search")
        self.assertEqual(decision.b13_decision, "commit_local_affordance_search")
        self.assertEqual(decision.b13_search_state, "local_route_viable")
        self.assertGreaterEqual(int(decision.b13_search_lock), 1)
        self.assertGreater(float(decision.b13_local_route_score), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b14_uncertainty_uses_b13_transfer_and_records_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b13_local_affordance_search_source(tmpdir)
            config = build_b14_affordance_uncertainty_config(
                B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=40, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(17)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B14_AFFORDANCE_UNCERTAINTY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b14_controller_profile, "affordance_uncertainty")
        self.assertEqual(decision.b14_decision, "commit_confident_affordance")
        self.assertEqual(decision.b14_uncertainty_state, "confidence_calibrated_route")
        self.assertGreaterEqual(int(decision.b14_commitment_lock), 1)
        self.assertGreater(float(decision.b14_affordance_confidence), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b15_option_critic_uses_b14_transfer_and_records_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b14_affordance_uncertainty_source(tmpdir)
            config = build_b15_option_critic_config(
                B15_OPTION_CRITIC_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=41, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B15_OPTION_CRITIC_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B15_OPTION_CRITIC_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b15_controller_profile, "option_critic")
        self.assertEqual(decision.b15_decision, "persist_food_option")
        self.assertEqual(decision.b15_option_state, "option_persist_food_route")
        self.assertGreaterEqual(int(decision.b15_option_lock), 1)
        self.assertGreater(float(decision.b15_option_value), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b16_option_ensemble_uses_b15_transfer_and_records_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b15_option_critic_source(tmpdir)
            config = build_b16_option_ensemble_config(
                B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=42, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B16_OPTION_ENSEMBLE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B16_OPTION_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b16_controller_profile, "option_ensemble")
        self.assertEqual(decision.b16_decision, "ensemble_continue_option")
        self.assertEqual(decision.b16_ensemble_state, "ensemble_continue_consensus")
        self.assertGreaterEqual(int(decision.b16_ensemble_lock), 1)
        self.assertGreater(float(decision.b16_continue_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b17_neuromodulated_ensemble_uses_b16_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b16_option_ensemble_source(tmpdir)
            config = build_b17_neuromodulated_ensemble_config(
                B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=43, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
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
            B17_NEUROMODULATED_ENSEMBLE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b17_controller_profile, "neuromodulated_ensemble")
        self.assertEqual(decision.b17_decision, "neuromodulated_continue")
        self.assertEqual(decision.b17_modulator_state, "modulated_continue")
        self.assertGreaterEqual(int(decision.b17_modulation_lock), 1)
        self.assertGreater(float(decision.b17_arousal_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b18_eligibility_trace_uses_b17_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b17_neuromodulated_ensemble_source(tmpdir)
            config = build_b18_eligibility_trace_config(
                B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=44, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B18_ELIGIBILITY_TRACE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B18_ELIGIBILITY_TRACE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b18_controller_profile, "eligibility_trace")
        self.assertEqual(decision.b18_decision, "eligibility_stabilize_option")
        self.assertEqual(decision.b18_trace_state, "trace_stabilizes_option")
        self.assertGreaterEqual(int(decision.b18_trace_lock), 1)
        self.assertGreater(float(decision.b18_eligibility_trace), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b19_episodic_meta_memory_uses_b18_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b18_eligibility_trace_source(tmpdir)
            config = build_b19_episodic_meta_memory_config(
                B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=45, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(19)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B19_EPISODIC_META_MEMORY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B19_EPISODIC_META_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b19_controller_profile, "episodic_meta_memory")
        self.assertEqual(decision.b19_decision, "episodic_consolidate_option")
        self.assertEqual(decision.b19_memory_state, "memory_consolidates_option")
        self.assertGreaterEqual(int(decision.b19_memory_lock), 1)
        self.assertGreater(float(decision.b19_episode_memory), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b20_working_memory_gate_uses_b19_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b19_episodic_meta_memory_source(tmpdir)
            config = build_b20_working_memory_gate_config(
                B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=46, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(20)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B20_WORKING_MEMORY_GATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B20_WORKING_MEMORY_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b20_controller_profile, "working_memory_gate")
        self.assertEqual(decision.b20_decision, "working_memory_gate_continue")
        self.assertEqual(decision.b20_buffer_state, "working_memory_holds_context")
        self.assertGreaterEqual(int(decision.b20_buffer_lock), 1)
        self.assertGreater(float(decision.b20_working_buffer), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b21_hippocampal_replay_uses_b20_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b20_working_memory_gate_source(tmpdir)
            config = build_b21_hippocampal_replay_config(
                B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=47, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(21)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B21_HIPPOCAMPAL_REPLAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b21_controller_profile, "hippocampal_replay")
        self.assertEqual(decision.b21_decision, "hippocampal_replay_continue")
        self.assertEqual(decision.b21_replay_state, "replay_rehearses_route")
        self.assertGreaterEqual(int(decision.b21_replay_lock), 1)
        self.assertGreater(float(decision.b21_sequence_memory), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b22_prospective_replay_uses_b21_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b21_hippocampal_replay_source(tmpdir)
            config = build_b22_prospective_replay_config(
                B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=48, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(22)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B22_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b22_controller_profile, "prospective_map_replay")
        self.assertEqual(decision.b22_decision, "prospective_replay_continue")
        self.assertEqual(decision.b22_sim_state, "prospective_sim_commits_route")
        self.assertGreaterEqual(int(decision.b22_sim_lock), 1)
        self.assertGreater(float(decision.b22_prospective_sim), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b23_conflict_monitor_uses_b22_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b22_prospective_replay_source(tmpdir)
            config = build_b23_conflict_monitor_config(
                B23_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=49, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(23)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B23_CONFLICT_MONITOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B23_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b23_controller_profile, "conflict_monitor")
        self.assertEqual(decision.b23_decision, "conflict_monitor_continue")
        self.assertEqual(decision.b23_conflict_state, "conflict_monitor_stabilizes_route")
        self.assertGreaterEqual(int(decision.b23_monitor_lock), 1)
        self.assertGreater(float(decision.b23_stability_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b24_precision_conflict_uses_b23_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b23_conflict_monitor_source(tmpdir)
            config = build_b24_precision_conflict_config(
                B24_PRECISION_CONFLICT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=50, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(24)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B24_PRECISION_CONFLICT_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B24_PRECISION_CONFLICT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b24_controller_profile, "precision_conflict")
        self.assertEqual(decision.b24_decision, "precision_conflict_continue")
        self.assertEqual(decision.b24_precision_state, "precision_conflict_stabilizes_route")
        self.assertGreaterEqual(int(decision.b24_precision_lock), 1)
        self.assertGreater(float(decision.b24_precision_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b25_metacognitive_confidence_uses_b24_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b24_precision_conflict_source(tmpdir)
            config = build_b25_metacognitive_confidence_config(
                B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=51, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(25)
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
            B25_METACOGNITIVE_CONFIDENCE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b25_controller_profile, "metacognitive_confidence")
        self.assertIn(
            decision.b25_decision,
            {"metacognitive_confidence_continue", "continue_meta_lock"},
        )
        self.assertNotEqual(decision.b25_metacognitive_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b25_meta_lock), 1)
        self.assertGreater(float(decision.b25_confidence_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b26_allostatic_prediction_uses_b25_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b25_metacognitive_confidence_source(tmpdir)
            config = build_b26_allostatic_prediction_config(
                B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=52, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(26)
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
            B26_ALLOSTATIC_PREDICTION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b26_controller_profile, "allostatic_prediction")
        self.assertIn(
            decision.b26_decision,
            {"allostatic_prediction_continue", "continue_allostatic_lock"},
        )
        self.assertNotEqual(decision.b26_allostatic_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b26_stability_lock), 1)
        self.assertGreater(float(decision.b26_control_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b27_arousal_gain_uses_b26_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b26_allostatic_prediction_source(tmpdir)
            config = build_b27_arousal_gain_config(
                B27_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=53, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(27)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B27_AROUSAL_GAIN_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B27_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b27_controller_profile, "arousal_gain")
        self.assertIn(
            decision.b27_decision,
            {"arousal_gain_continue", "continue_arousal_lock"},
        )
        self.assertNotEqual(decision.b27_arousal_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b27_arousal_lock), 1)
        self.assertGreater(float(decision.b27_gain_modulation), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b28_interoceptive_attention_uses_b27_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b27_arousal_gain_source(tmpdir)
            config = build_b28_interoceptive_attention_config(
                B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=54, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(28)
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
            B28_INTEROCEPTIVE_ATTENTION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b28_controller_profile, "interoceptive_attention")
        self.assertIn(
            decision.b28_decision,
            {"interoceptive_attention_continue", "continue_attention_lock"},
        )
        self.assertNotEqual(decision.b28_attention_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b28_attention_lock), 1)
        self.assertGreater(float(decision.b28_attention_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)
