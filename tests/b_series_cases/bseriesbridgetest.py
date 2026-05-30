from __future__ import annotations

from .shared import *


class BSeriesBridgeTest(unittest.TestCase):
    def test_move_to_food_selects_primitive_food_progress(self) -> None:
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", _bridge_observation())
        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")
        self.assertEqual(decision.reason, "food_progress")
        self.assertAlmostEqual(decision.food_delta_used, 1.0)

    def test_move_to_food_inside_shelter_first_favors_exit_geodesic(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
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
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)
        self.assertEqual(decision.primitive_action, "MOVE_UP")
        self.assertEqual(decision.reason, "food_exit_to_outside")

    def test_move_to_food_uses_memory_vector_after_shelter_exit(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["shelter_role"] = "outside"
        meta["memory_vectors"] = {
            "food": {"dx": 0.60, "dy": 0.10, "age": 0.0, "ttl": 30},
        }
        affordances = meta["local_affordances"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(affordances, dict)
        assert isinstance(transitions, dict)
        affordances["MOVE_DOWN"] = {"blocked": False, "next_role": "entrance"}
        transitions["MOVE_DOWN"] = {
            "food_dist_delta": 1.0,
            "shelter_dist_delta": 1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_RIGHT"] = {
            "food_dist_delta": 0.0,
            "shelter_dist_delta": -1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }

        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)

        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")
        self.assertEqual(decision.reason, "food_memory_vector")

    def test_move_to_food_does_not_reenter_shelter_without_food(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["shelter_role"] = "outside"
        affordances = meta["local_affordances"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(affordances, dict)
        assert isinstance(transitions, dict)
        affordances["MOVE_DOWN"] = {"blocked": False, "next_role": "entrance"}
        transitions["MOVE_DOWN"] = {
            "food_dist_delta": 1.0,
            "shelter_dist_delta": 1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_RIGHT"] = {
            "food_dist_delta": 0.25,
            "shelter_dist_delta": -1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_LEFT"] = {
            "food_dist_delta": 0.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }

        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)

        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")

    def test_move_to_shelter_selects_primitive_shelter_progress(self) -> None:
        decision = bridge_b_semantic_action("MOVE_TO_SHELTER", _bridge_observation())
        self.assertEqual(decision.primitive_action, "MOVE_LEFT")
        self.assertEqual(decision.reason, "shelter_progress")
        self.assertAlmostEqual(decision.shelter_delta_used, 1.5)

    def test_move_to_shelter_holds_when_already_deep(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        decision = bridge_b_semantic_action("MOVE_TO_SHELTER", observation)
        self.assertEqual(decision.primitive_action, "STAY")
        self.assertEqual(decision.reason, "already_deep_shelter")

    def test_blocked_winning_move_is_masked_without_semantic_rewrite(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        affordances = meta["local_affordances"]
        assert isinstance(affordances, dict)
        affordances["MOVE_RIGHT"] = {"blocked": True}
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.primitive_action, "MOVE_LEFT")
        self.assertTrue(decision.blocked_mask["MOVE_RIGHT"])
        self.assertEqual(decision.external_override_count, 0)

    def test_non_movement_semantic_actions_map_to_stay(self) -> None:
        for semantic_action in ("STAY", "EAT", "SLEEP"):
            with self.subTest(semantic_action=semantic_action):
                decision = bridge_b_semantic_action(
                    semantic_action,
                    _bridge_observation(),
                )
                self.assertEqual(decision.primitive_action, "STAY")

    def test_explore_selects_unblocked_primitive_move(self) -> None:
        decision = bridge_b_semantic_action("EXPLORE", _bridge_observation())
        self.assertIn(decision.primitive_action, {"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"})
