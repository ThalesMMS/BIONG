from __future__ import annotations

from .shared import *



class BSeriesRuntimeTestPart3(unittest.TestCase):
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
