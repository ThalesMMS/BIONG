from __future__ import annotations

from .shared import *


class BSeriesRuntimeTestPart4(unittest.TestCase):
    def test_b78_vestibular_balance_uses_b77_error_context(self) -> None:
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
            brain = SpiderBrain(seed=104, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 0.0
        meta["shelter_role"] = "at_shelter"
        meta["recent_pain"] = 0.05
        meta["recent_contact"] = 0.05
        for tick in range(78, 104):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.57},
                    sleep={"health": 0.57, "sleep_debt": 0.34, "on_shelter": 1.0},
                    threat={
                        "predator_smell_strength": 0.05,
                        "predator_motion_salience": 0.08,
                    },
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, "B78-vestibular-balance")
        self.assertEqual(
            decision.semantic_action_source,
            "b78_vestibular_balance_controller",
        )
        self.assertEqual(decision.b78_controller_profile, "vestibular_balance")
        self.assertIn(
            decision.b78_decision,
            {"vestibular_balance_hold", "continue_balance_lock"},
        )
        self.assertIn(decision.semantic_action, {"SLEEP", "STAY"})
        self.assertGreater(float(decision.b78_balance_error), 0.0)
        self.assertGreater(float(decision.b78_head_stabilization), 0.0)
        self.assertGreater(float(decision.b78_locomotor_confidence), 0.0)
        self.assertGreater(float(decision.b78_slip_risk), 0.0)
        self.assertGreaterEqual(int(decision.b78_balance_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)
