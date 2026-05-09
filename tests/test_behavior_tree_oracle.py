import unittest

from spider_cortex_sim.behavior_tree_oracle import BehaviorTreeOraclePolicy
from spider_cortex_sim.world import SpiderWorld


class BehaviorTreeOraclePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(
            width=12,
            height=12,
            food_count=4,
            day_length=18,
            night_length=12,
            seed=7,
            reward_profile="ecological",
            map_template="central_burrow",
        )
        self.world.reset(seed=7)
        self.oracle = BehaviorTreeOraclePolicy(self.world)

    def test_rest_phase_stays_in_deep_shelter_when_recovery_is_pending(self) -> None:
        rest_target = self.oracle._best_rest_shelter_target()
        assert rest_target is not None
        self.world.state.x, self.world.state.y = rest_target
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.18
        self.world.state.fatigue = 0.20
        self.world.state.sleep_debt = 0.20
        self.world.state.sleep_events = 0

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "REST")
        self.assertEqual(decision.action, "STAY")

    def test_reactivate_phase_heads_toward_entrance_after_recovery(self) -> None:
        rest_target = self.oracle._best_rest_shelter_target()
        assert rest_target is not None
        self.world.state.x, self.world.state.y = rest_target
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.60
        self.world.state.fatigue = 0.01
        self.world.state.sleep_debt = 0.01
        self.world.state.sleep_events = 5

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "POST_REST_REACTIVATE")
        self.assertIn(decision.action, {"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"})

    def test_reactivate_phase_starts_after_daytime_recovery_even_without_high_hunger(self) -> None:
        rest_target = self.oracle._best_rest_shelter_target()
        assert rest_target is not None
        self.world.state.x, self.world.state.y = rest_target
        self.world.tick = 0
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.18
        self.world.state.fatigue = 0.01
        self.world.state.sleep_debt = 0.01
        self.world.state.sleep_events = 3
        self.oracle.current_phase = "RECOVERED_IN_SHELTER"
        self.oracle.phase_sleep_start = 0

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "POST_REST_REACTIVATE")
        self.assertIn(decision.action, {"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"})

    def test_reactivation_commitment_ignores_distant_shelter_adjacent_threat(self) -> None:
        self.world.state.x, self.world.state.y = 5, 6
        self.world.lizard.x, self.world.lizard.y = 8, 7
        self.world.tick = 0
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.20
        self.world.state.fatigue = 0.01
        self.world.state.sleep_debt = 0.01
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.0
        self.oracle.current_phase = "POST_REST_REACTIVATE"
        self.oracle.phase_sleep_start = 0

        self.assertTrue(self.oracle._should_commit_reactivation_probe())

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "POST_REST_REACTIVATE")
        self.assertEqual(decision.action, "MOVE_UP")
        self.assertEqual(decision.target, (5, 5))

    def test_reactivation_commitment_yields_to_close_predator(self) -> None:
        self.world.state.x, self.world.state.y = 6, 6
        self.world.lizard.x, self.world.lizard.y = 8, 7
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.0
        self.oracle.current_phase = "POST_REST_REACTIVATE"

        self.assertFalse(self.oracle._should_commit_reactivation_probe())

    def test_reactivation_commitment_ignores_distant_first_outside_probe(self) -> None:
        self.world.state.x, self.world.state.y = 4, 5
        self.world.lizard.x, self.world.lizard.y = 8, 8
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.22
        self.world.state.fatigue = 0.02
        self.world.state.sleep_debt = 0.05
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.0
        self.oracle.current_phase = "POST_REST_REACTIVATE"
        self.oracle.phase_food_start = 3

        self.assertTrue(self.oracle._should_commit_reactivation_probe())

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "LATE_FORAGE")

    def test_late_forage_outside_prefers_outside_only_food_path(self) -> None:
        self.world.state.x, self.world.state.y = 4, 5
        self.world.food_positions = [(11, 7)]
        self.world.lizard.x, self.world.lizard.y = 10, 0
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.22
        self.world.state.fatigue = 0.02
        self.world.state.sleep_debt = 0.05
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.0
        self.oracle.current_phase = "LATE_FORAGE"
        self.oracle.phase_food_start = 3

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "LATE_FORAGE")
        self.assertEqual(decision.reason, "follow_outside_food_target")
        self.assertEqual(decision.action, "MOVE_UP")

    def test_late_forage_outside_prefers_safer_food_target_when_predator_corridor_is_hot(self) -> None:
        self.world.state.x, self.world.state.y = 4, 4
        self.world.food_positions = [(11, 7), (4, 3), (6, 0)]
        self.world.lizard.x, self.world.lizard.y = 8, 8
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.22
        self.world.state.fatigue = 0.02
        self.world.state.sleep_debt = 0.05
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.0
        self.oracle.current_phase = "LATE_FORAGE"
        self.oracle.phase_food_start = 3

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "LATE_FORAGE")
        self.assertEqual(decision.reason, "follow_outside_food_target")
        self.assertNotEqual(decision.target, (11, 7))
        self.assertIn(decision.target, {(4, 3), (6, 0)})
        self.assertEqual(decision.action, "MOVE_UP")

    def test_escape_phase_can_commit_to_single_post_rest_food_when_target_is_close(self) -> None:
        self.world.state.x, self.world.state.y = 9, 6
        self.world.food_positions = [(11, 7)]
        self.world.lizard.x, self.world.lizard.y = 9, 7
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.35
        self.world.state.fatigue = 0.02
        self.world.state.sleep_debt = 0.04
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.12
        self.world.state.recent_pain = 0.08
        self.oracle.current_phase = "ESCAPE_OR_ACUTE_THREAT"
        self.oracle.phase_food_start = 3

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "LATE_FORAGE")
        self.assertIn(
            decision.reason,
            {"post_rest_food_commitment", "post_rest_elevated_food_approach"},
        )
        self.assertIn(decision.target, {(11, 7), (11, 6)})
        self.assertIn(decision.action, {"MOVE_DOWN", "MOVE_RIGHT"})

    def test_escape_phase_uses_elevated_lane_for_single_corridor_food_target(self) -> None:
        self.world.state.x, self.world.state.y = 9, 7
        self.world.food_positions = [(11, 7)]
        self.world.lizard.x, self.world.lizard.y = 9, 7
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.35
        self.world.state.fatigue = 0.02
        self.world.state.sleep_debt = 0.04
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.12
        self.world.state.recent_pain = 0.08
        self.oracle.current_phase = "ESCAPE_OR_ACUTE_THREAT"
        self.oracle.phase_food_start = 3

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "LATE_FORAGE")
        self.assertEqual(decision.reason, "post_rest_elevated_food_approach")
        self.assertEqual(decision.target, (9, 6))
        self.assertEqual(decision.action, "MOVE_UP")

    def test_return_phase_side_steps_up_when_lizard_blocks_corridor_left(self) -> None:
        self.world.state.x, self.world.state.y = 10, 7
        self.world.food_positions = [(9, 7)]
        self.world.lizard.x, self.world.lizard.y = 9, 7
        self.world.state.food_eaten = 5
        self.world.state.hunger = 0.08
        self.world.state.fatigue = 0.12
        self.world.state.sleep_debt = 0.08
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.10
        self.world.state.recent_pain = 0.07
        self.oracle.current_phase = "RETURN_AFTER_LATE_FORAGE"
        self.oracle.phase_food_start = 3

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "RETURN_AFTER_LATE_FORAGE")
        self.assertEqual(decision.reason, "post_rest_retreat_side_step")
        self.assertEqual(decision.target, (10, 6))
        self.assertEqual(decision.action, "MOVE_UP")

    def test_late_forage_returns_after_first_post_rest_corridor_meal(self) -> None:
        self.world.state.x, self.world.state.y = 11, 7
        self.world.food_positions = [(10, 7)]
        self.world.lizard.x, self.world.lizard.y = 9, 6
        self.world.state.food_eaten = 4
        self.world.state.hunger = 0.08
        self.world.state.fatigue = 0.11
        self.world.state.sleep_debt = 0.08
        self.world.state.sleep_events = 3
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.0
        self.oracle.current_phase = "LATE_FORAGE"
        self.oracle.phase_food_start = 3

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "RETURN_AFTER_LATE_FORAGE")
        self.assertEqual(decision.reason, "post_rest_corridor_meal_return")
        self.assertEqual(decision.target, (11, 6))
        self.assertEqual(decision.action, "MOVE_UP")

    def test_acute_threat_ignores_tiny_pain_residue_without_live_predator_signal(self) -> None:
        self.world.state.x, self.world.state.y = 4, 4
        self.world.lizard.x, self.world.lizard.y = 8, 8
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.002

        self.assertFalse(self.oracle._acute_threat())

    def test_deepen_phase_prefers_deep_cell_with_more_predator_clearance(self) -> None:
        self.world.state.x, self.world.state.y = 7, 7
        self.world.lizard.x, self.world.lizard.y = 8, 7
        self.world.state.food_eaten = 3
        self.world.state.hunger = 0.18
        self.world.state.fatigue = 0.33
        self.world.state.sleep_debt = 0.24
        self.world.state.sleep_events = 0
        self.world.state.recent_contact = 0.0
        self.world.state.recent_pain = 0.05

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "DEEPEN_IN_SHELTER")
        self.assertEqual(decision.action, "MOVE_LEFT")
        self.assertEqual(decision.target, (5, 7))

    def test_forage_phase_targets_live_food_when_outside(self) -> None:
        self.world.state.x = 8
        self.world.state.y = 7
        self.world.state.food_eaten = 0
        self.world.state.hunger = 0.90
        self.world.state.fatigue = 0.05
        self.world.state.sleep_debt = 0.05
        self.world.state.sleep_events = 0

        decision = self.oracle._decide()

        self.assertEqual(decision.phase, "INITIAL_FORAGE")
        self.assertIsNotNone(decision.target)
