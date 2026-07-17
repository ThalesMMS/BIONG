from __future__ import annotations

import unittest
from unittest import mock

from spider_cortex_sim.direct_policy_affordances import (
    AFFORDANCE_SHELTER_COLUMN_TO_INDEX,
    AFFORDANCE_SHELTER_POSITION_TO_INDEX,
    AFFORDANCE_SHELTER_ROLE_TO_INDEX,
)
from spider_cortex_sim.interfaces import ACTION_TO_INDEX
from spider_cortex_sim.simulation import SpiderSimulation


class _TeacherTargetWorld:
    move_deltas = ((0, -1), (0, 1), (-1, 0), (1, 0))
    shelter_cells = {(1, 1), (2, 1)}

    @staticmethod
    def is_walkable(pos: tuple[int, int]) -> bool:
        return pos != (1, 0)

    @staticmethod
    def shelter_role_at(pos: tuple[int, int]) -> str:
        if pos == (1, 1):
            return "entrance"
        if pos == (2, 1):
            return "inside"
        return "outside"


class DirectPolicyTeacherTargetsTest(unittest.TestCase):
    def test_movement_actions_produce_spatial_teacher_targets(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=1)
        sim.world = _TeacherTargetWorld()
        state = {"x": 1, "y": 1}
        up_idx = ACTION_TO_INDEX["MOVE_UP"]
        right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]

        blocked, roles = sim._direct_policy_affordance_targets(
            current_state=state
        )
        geometry = sim._direct_policy_geometry_targets(current_state=state)
        columns = sim._direct_policy_shelter_column_targets(current_state=state)
        positions = sim._direct_policy_shelter_position_targets(current_state=state)

        self.assertEqual(blocked[up_idx], 1.0)
        self.assertEqual(roles[right_idx], AFFORDANCE_SHELTER_ROLE_TO_INDEX["inside"])
        self.assertEqual(geometry[right_idx * 3], 1.0)
        self.assertEqual(columns[right_idx], AFFORDANCE_SHELTER_COLUMN_TO_INDEX["right"])
        self.assertEqual(
            positions[right_idx],
            AFFORDANCE_SHELTER_POSITION_TO_INDEX["inside_right"],
        )

    def test_probe_cycle_continues_through_return_and_rerest_stages(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=1)
        sim._direct_policy_probe_cycle_teacher_state = {
            "stage": "forage_window",
            "food_count_before": 0,
            "sleep_events_before": 0,
            "forage_tick": 1,
        }
        state = {
            "x": 1,
            "y": 1,
            "shelter_role": "outside",
            "sleep_phase": "AWAKE",
            "rest_streak": 0,
            "food_eaten": 1,
            "sleep_events": 0,
        }
        expected_return = (ACTION_TO_INDEX["MOVE_LEFT"], "cycle_return")

        with mock.patch.object(
            sim,
            "_teacher_shelter_return_action",
            return_value=expected_return,
        ):
            return_action = sim._direct_policy_probe_cycle_redirect_action(
                observation={},
                current_state=state,
                baseline_action_name="STAY",
                food_direction_action=None,
                tick=2,
            )

        self.assertEqual(return_action, expected_return)
        self.assertEqual(
            sim._direct_policy_probe_cycle_teacher_state["stage"],
            "return_window",
        )

        state["shelter_role"] = "inside"
        rerest_action = sim._direct_policy_probe_cycle_redirect_action(
            observation={},
            current_state=state,
            baseline_action_name="STAY",
            food_direction_action=None,
            tick=3,
        )

        self.assertEqual(rerest_action, (ACTION_TO_INDEX["STAY"], "cycle_rerest"))
        self.assertEqual(
            sim._direct_policy_probe_cycle_teacher_state["stage"],
            "rerest_window",
        )

    def test_deep_shelter_action_is_none_when_already_at_target(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=1)
        current_pos = sim.world.spider_pos()

        with mock.patch.object(
            sim.world,
            "safest_shelter_target",
            return_value=current_pos,
        ):
            action = sim._teacher_deep_shelter_action()

        self.assertIsNone(action)

    def test_probe_trace_return_window_stops_at_timeout(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=1)
        sim._direct_policy_probe_trace_teacher_state = {
            "stage": "return_window",
            "return_tick": 0,
            "food_count_before": 0,
            "sleep_events_before": 0,
        }
        state = {
            "x": 1,
            "y": 1,
            "shelter_role": "outside",
            "sleep_phase": "AWAKE",
            "rest_streak": 0,
            "food_eaten": 1,
            "sleep_events": 0,
        }

        with mock.patch.object(
            sim,
            "_teacher_deep_shelter_action",
            return_value=(ACTION_TO_INDEX["MOVE_LEFT"], "trace_return"),
        ):
            action = sim._direct_policy_probe_trace_redirect_action(
                observation={},
                current_state=state,
                food_direction_action=None,
                tick=19,
            )

        self.assertEqual(action, (-1, None))
        self.assertEqual(
            sim._direct_policy_probe_trace_teacher_state["stage"],
            "done",
        )


if __name__ == "__main__":
    unittest.main()
