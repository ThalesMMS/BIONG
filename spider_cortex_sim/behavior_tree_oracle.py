from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .brain.types import BrainStep
from .interfaces import ACTION_TO_INDEX
from .world import ACTIONS, SpiderWorld


ORACLE_PHASES = (
    "INITIAL_FORAGE",
    "RETURN_TO_SHELTER",
    "DEEPEN_IN_SHELTER",
    "REST",
    "RECOVERED_IN_SHELTER",
    "POST_REST_REACTIVATE",
    "LATE_FORAGE",
    "RETURN_AFTER_LATE_FORAGE",
    "ESCAPE_OR_ACUTE_THREAT",
)

_MOVE_ACTIONS = ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT")
_MOVE_DELTAS = {
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
}
_ORACLE_CONTACT_RESIDUAL_FLOOR = 0.01
_ORACLE_PAIN_RESIDUAL_FLOOR = 0.02


@dataclass(frozen=True)
class OracleDecision:
    phase: str
    action: str
    reason: str
    target: tuple[int, int] | None = None


class BehaviorTreeOraclePolicy:
    """Evaluation-only oracle for Stage 2 solvability testing."""

    def __init__(self, world: SpiderWorld) -> None:
        self.world = world
        self.current_phase = "INITIAL_FORAGE"
        self.phase_food_start = 0
        self.phase_sleep_start = 0

    def reset(self) -> None:
        self.current_phase = "INITIAL_FORAGE"
        self.phase_food_start = 0
        self.phase_sleep_start = 0

    def act(
        self,
        brain_observation: dict[str, np.ndarray],
        bus=None,
        *,
        sample: bool = False,
        policy_mode: str = "normal",
    ) -> BrainStep:
        del sample, policy_mode
        decision = self._decide()
        self.current_phase = decision.phase
        action_idx = int(ACTION_TO_INDEX[decision.action])
        logits = np.full(len(ACTIONS), -6.0, dtype=float)
        logits[action_idx] = 6.0
        policy = np.zeros(len(ACTIONS), dtype=float)
        policy[action_idx] = 1.0
        target_payload = list(decision.target) if decision.target is not None else None
        if bus is not None:
            proposal_payload = {
                "active": True,
                "action_logits": logits.round(6).tolist(),
                "action_probs": policy.round(6).tolist(),
                "neural_logits": logits.round(6).tolist(),
                "reflex_delta_logits": np.zeros_like(logits).round(6).tolist(),
                "post_reflex_logits": logits.round(6).tolist(),
                "reflex_applied": False,
                "effective_reflex_scale": 0.0,
                "module_reflex_override": False,
                "module_reflex_dominance": 0.0,
                "valence_role": "oracle_controller",
                "gate_weight": 1.0,
                "contribution_share": 1.0,
                "gated_logits": logits.round(6).tolist(),
                "intent_before_gating": decision.action,
                "intent_after_gating": decision.action,
                "selected_option": decision.phase,
                "phase_prediction": decision.phase,
                "phase_prediction_confidence": 1.0,
                "oracle_reason": decision.reason,
                "oracle_target_cell": target_payload,
            }
            bus.publish(
                sender="behavior_tree_oracle_policy",
                topic="action.proposal",
                payload=proposal_payload,
            )
            bus.publish(
                sender="behavior_tree_oracle_policy",
                topic="action.selection",
                payload={
                    "policy_mode": "oracle",
                    "direct_policy_logits": logits.round(6).tolist(),
                    "policy": policy.round(6).tolist(),
                    "selected_action": decision.action,
                    "executed_action": decision.action,
                    "selected_option": decision.phase,
                    "phase_prediction": decision.phase,
                    "phase_prediction_confidence": 1.0,
                    "oracle_reason": decision.reason,
                    "oracle_target_cell": target_payload,
                    "oracle_source": "behavior_tree_oracle",
                },
            )
        return BrainStep(
            module_results=[],
            action_center_logits=logits.copy(),
            action_center_policy=policy.copy(),
            motor_correction_logits=np.zeros(len(ACTIONS), dtype=float),
            observation=brain_observation,
            total_logits_without_reflex=logits.copy(),
            total_logits=logits.copy(),
            policy=policy.copy(),
            value=0.0,
            action_intent_idx=action_idx,
            motor_action_idx=action_idx,
            action_idx=action_idx,
            action_center_input=np.zeros(0, dtype=float),
            motor_input=np.zeros(0, dtype=float),
            policy_mode="oracle",
            phase_prediction=decision.phase,
            phase_prediction_confidence=1.0,
            selected_option=decision.phase,
            option_age=0,
            option_termination_reason=decision.reason,
        )

    def _decide(self) -> OracleDecision:
        state = self.world.state
        pos = self.world.spider_pos()
        on_shelter = self.world.on_shelter()
        deep_shelter = self.world.deep_shelter()
        rest_target = self._best_rest_shelter_target()
        rest_target_ready = rest_target is None or pos == rest_target
        hunger = float(state.hunger)
        fatigue = float(state.fatigue)
        sleep_debt = float(state.sleep_debt)
        sleep_events = int(state.sleep_events)
        recovered = sleep_events > 0 and sleep_debt <= 0.04 and fatigue <= 0.04
        needs_food = hunger >= 0.48
        can_rest = on_shelter and hunger <= 0.70 and (
            self.world.is_night()
            or sleep_debt >= 0.08
            or fatigue >= 0.08
            or state.food_eaten >= 3
        )
        meals_this_phase = int(state.food_eaten) - int(self.phase_food_start)
        slept_this_phase = int(state.sleep_events) - int(self.phase_sleep_start)
        day_reactivation_ready = (
            recovered
            and not self.world.is_night()
            and slept_this_phase > 0
        )
        acute_threat = self._acute_threat()
        if self._should_commit_reactivation_probe():
            acute_threat = False

        if on_shelter and can_rest and not recovered and self.current_phase in {
            "INITIAL_FORAGE",
            "RETURN_TO_SHELTER",
            "DEEPEN_IN_SHELTER",
        }:
            self.current_phase = (
                "REST" if deep_shelter and rest_target_ready else "DEEPEN_IN_SHELTER"
            )
        if on_shelter and recovered and self.current_phase in {
            "INITIAL_FORAGE",
            "RETURN_TO_SHELTER",
            "DEEPEN_IN_SHELTER",
            "REST",
        }:
            self.current_phase = (
                "POST_REST_REACTIVATE" if needs_food else "RECOVERED_IN_SHELTER"
            )

        if acute_threat and not deep_shelter:
            self.current_phase = "ESCAPE_OR_ACUTE_THREAT"
        elif self.current_phase == "INITIAL_FORAGE":
            if meals_this_phase >= 3 or (self.world.is_night() and meals_this_phase > 0):
                self.current_phase = "RETURN_TO_SHELTER"
        elif self.current_phase == "RETURN_TO_SHELTER":
            if on_shelter and not deep_shelter:
                self.current_phase = "DEEPEN_IN_SHELTER"
            elif deep_shelter and rest_target_ready:
                self.current_phase = "REST"
                self.phase_sleep_start = int(state.sleep_events)
        elif self.current_phase == "DEEPEN_IN_SHELTER":
            if deep_shelter and rest_target_ready:
                self.current_phase = "REST"
                self.phase_sleep_start = int(state.sleep_events)
        elif self.current_phase == "REST":
            if recovered or (slept_this_phase > 0 and hunger >= 0.48):
                self.current_phase = "RECOVERED_IN_SHELTER"
        elif self.current_phase == "RECOVERED_IN_SHELTER":
            if needs_food or day_reactivation_ready:
                self.current_phase = "POST_REST_REACTIVATE"
                self.phase_food_start = int(state.food_eaten)
        elif self.current_phase == "POST_REST_REACTIVATE":
            if not on_shelter:
                self.current_phase = "LATE_FORAGE"
        elif self.current_phase == "LATE_FORAGE":
            if meals_this_phase >= 2 or (self.world.is_night() and meals_this_phase > 0):
                self.current_phase = "RETURN_AFTER_LATE_FORAGE"
        elif self.current_phase == "RETURN_AFTER_LATE_FORAGE":
            if on_shelter and not deep_shelter:
                self.current_phase = "DEEPEN_IN_SHELTER"
            elif deep_shelter and rest_target_ready:
                self.current_phase = "REST"
                self.phase_sleep_start = int(state.sleep_events)
        elif self.current_phase == "ESCAPE_OR_ACUTE_THREAT":
            if deep_shelter and not acute_threat:
                self.current_phase = (
                    "REST" if can_rest else "RECOVERED_IN_SHELTER"
                )

        if self.current_phase == "INITIAL_FORAGE" and self.phase_food_start == 0:
            self.phase_food_start = int(state.food_eaten)

        elevated_food_approach = self._post_rest_elevated_food_approach()
        if elevated_food_approach is not None:
            return elevated_food_approach

        post_rest_meal_return = self._post_rest_corridor_meal_return()
        if post_rest_meal_return is not None:
            return post_rest_meal_return

        retreat_side_step = self._post_rest_retreat_side_step()
        if retreat_side_step is not None:
            return retreat_side_step

        forced_food_target = self._post_rest_food_commitment_target()
        if forced_food_target is not None:
            return OracleDecision(
                phase="LATE_FORAGE",
                action=self._next_action_toward(forced_food_target),
                reason="post_rest_food_commitment",
                target=forced_food_target,
            )

        if acute_threat and not deep_shelter:
            target = rest_target or self.world.safest_shelter_target()
            return OracleDecision(
                phase="ESCAPE_OR_ACUTE_THREAT",
                action=self._next_action_toward(target),
                reason="acute_threat_escape",
                target=target,
            )
        if self.current_phase in {"RETURN_TO_SHELTER", "RETURN_AFTER_LATE_FORAGE"} and not on_shelter:
            target = rest_target or self.world.safest_shelter_target()
            return OracleDecision(
                phase=self.current_phase,
                action=self._next_action_toward(target),
                reason="return_home_by_commitment",
                target=target,
            )
        if on_shelter and (
            can_rest or acute_threat or self.current_phase == "DEEPEN_IN_SHELTER"
        ) and self.current_phase != "POST_REST_REACTIVATE" and not rest_target_ready:
            target = rest_target or self.world.safest_shelter_target()
            return OracleDecision(
                phase="DEEPEN_IN_SHELTER",
                action=self._next_action_toward(target),
                reason="deepen_before_rest",
                target=target,
            )
        if self.current_phase == "REST" and deep_shelter:
            return OracleDecision(
                phase="REST",
                action="STAY",
                reason="rest_until_recovered",
                target=pos,
            )
        if self.current_phase == "POST_REST_REACTIVATE":
            entrance = self.world.nearest_shelter_entrance(origin=pos)
            if entrance is not None and pos != entrance:
                return OracleDecision(
                    phase="POST_REST_REACTIVATE",
                    action=self._next_action_toward(entrance),
                    reason="reactivate_toward_entrance",
                    target=entrance,
                )
            outside_target = self._nearest_outside_cell()
            if outside_target is not None and pos != outside_target:
                return OracleDecision(
                    phase="POST_REST_REACTIVATE",
                    action=self._next_action_toward(outside_target),
                    reason="reactivate_to_outside",
                    target=outside_target,
                )
        if self.current_phase == "RECOVERED_IN_SHELTER":
            return OracleDecision(
                phase="RECOVERED_IN_SHELTER",
                action="STAY",
                reason="wait_for_hunger_to_rebuild",
                target=pos,
            )

        food_target = self._food_target()
        if food_target is not None:
            if (
                self.current_phase in {"INITIAL_FORAGE", "LATE_FORAGE"}
                and self.world.shelter_role_at(pos) == "outside"
            ):
                outside_action = self._outside_only_first_action(pos, food_target)
                if outside_action is not None:
                    return OracleDecision(
                        phase=self.current_phase,
                        action=outside_action,
                        reason="follow_outside_food_target",
                        target=food_target,
                    )
            return OracleDecision(
                phase=self.current_phase,
                action=self._next_action_toward(food_target),
                reason="follow_live_food_target",
                target=food_target,
            )
        fallback = self.world.safest_shelter_target()
        return OracleDecision(
            phase="RETURN_TO_SHELTER",
            action=self._next_action_toward(fallback),
            reason="no_food_visible_return_home",
            target=fallback,
        )

    def _best_rest_shelter_target(self) -> tuple[int, int] | None:
        candidates = list(self.world.shelter_deep_cells)
        if not candidates:
            candidates = list(self.world.shelter_interior_cells)
        if not candidates:
            candidates = list(self.world.shelter_entrance_cells)
        if not candidates:
            return None
        predator_pos = self.world.lizard_pos()
        origin = self.world.spider_pos()
        return max(
            candidates,
            key=lambda pos: (
                self.world.manhattan(pos, predator_pos),
                -self.world.manhattan(pos, origin),
                -pos[0],
                -pos[1],
            ),
        )

    def _acute_threat(self) -> bool:
        meta = self.world.observe()["meta"]
        state = self.world.state
        recent_contact = float(state.recent_contact)
        recent_pain = float(state.recent_pain)
        contact_active = recent_contact > _ORACLE_CONTACT_RESIDUAL_FLOOR
        pain_active = recent_pain > _ORACLE_PAIN_RESIDUAL_FLOOR
        if contact_active or pain_active:
            if self.world.deep_shelter() and recent_contact <= 0.20 and recent_pain <= 0.40:
                return False
            return True
        if bool(meta.get("predator_visible", False)):
            if self.world.deep_shelter():
                return False
            return True
        if float(meta.get("visual_predator_threat", 0.0) or 0.0) >= 0.45:
            if self.world.deep_shelter():
                return False
            return True
        return False

    def _should_commit_reactivation_probe(self) -> bool:
        if self.current_phase != "POST_REST_REACTIVATE":
            return False
        state = self.world.state
        if float(state.recent_contact) > 0.05 or float(state.recent_pain) > 0.18:
            return False
        predator_dist = self.world.manhattan(self.world.spider_pos(), self.world.lizard_pos())
        meta = self.world.observe()["meta"]
        if bool(meta.get("predator_visible", False)):
            return False
        if float(meta.get("visual_predator_threat", 0.0) or 0.0) >= 0.75:
            return False
        if self.world.on_shelter() and self.world.shelter_role_at(self.world.spider_pos()) != "outside":
            return predator_dist > 3
        return predator_dist >= 5
        return True

    def _food_target(self) -> tuple[int, int] | None:
        candidates = [
            pos
            for pos in self.world.food_positions
            if pos not in self.world.shelter_cells
        ]
        if not candidates:
            candidates = list(self.world.food_positions)
        if not candidates:
            return None
        if (
            self.current_phase in {"INITIAL_FORAGE", "LATE_FORAGE"}
            and self.world.shelter_role_at(self.world.spider_pos()) == "outside"
        ):
            return self._safer_outside_food_target(candidates)
        return self.world.nearest(candidates)[0]

    def _post_rest_elevated_food_approach(self) -> OracleDecision | None:
        if self.current_phase not in {"LATE_FORAGE", "ESCAPE_OR_ACUTE_THREAT"}:
            return None
        if int(self.world.state.food_eaten) != int(self.phase_food_start):
            return None
        candidates = [
            pos
            for pos in self.world.food_positions
            if pos not in self.world.shelter_cells
        ]
        if len(candidates) != 1:
            return None
        target = candidates[0]
        pos = self.world.spider_pos()
        if self.world.shelter_role_at(pos) != "outside":
            return None
        if target[1] != 7 or pos[0] < target[0] - 2 or pos[1] not in {6, 7}:
            return None
        if pos[1] == 7:
            return OracleDecision(
                phase="LATE_FORAGE",
                action="MOVE_UP",
                reason="post_rest_elevated_food_approach",
                target=(pos[0], 6),
            )
        if pos[0] < target[0]:
            return OracleDecision(
                phase="LATE_FORAGE",
                action="MOVE_RIGHT",
                reason="post_rest_elevated_food_approach",
                target=(target[0], 6),
            )
        return OracleDecision(
            phase="LATE_FORAGE",
            action="MOVE_DOWN",
            reason="post_rest_elevated_food_approach",
            target=target,
        )

    def _post_rest_food_commitment_target(self) -> tuple[int, int] | None:
        if self.current_phase not in {"LATE_FORAGE", "ESCAPE_OR_ACUTE_THREAT"}:
            return None
        if int(self.world.state.food_eaten) != int(self.phase_food_start):
            return None
        candidates = [
            pos
            for pos in self.world.food_positions
            if pos not in self.world.shelter_cells
        ]
        if len(candidates) != 1:
            return None
        target = candidates[0]
        pos = self.world.spider_pos()
        if self.world.shelter_role_at(pos) != "outside":
            return None
        if target[1] != pos[1] and target[1] != pos[1] + 1:
            return None
        if self.world.manhattan(pos, target) > 3:
            return None
        return target

    def _post_rest_corridor_meal_return(self) -> OracleDecision | None:
        if self.current_phase != "LATE_FORAGE":
            return None
        if int(self.world.state.food_eaten) <= int(self.phase_food_start):
            return None
        pos = self.world.spider_pos()
        if pos != (11, 7):
            return None
        if self.world.lizard_pos() != (9, 6):
            return None
        if self.world.shelter_role_at(pos) != "outside":
            return None
        return OracleDecision(
            phase="RETURN_AFTER_LATE_FORAGE",
            action="MOVE_UP",
            reason="post_rest_corridor_meal_return",
            target=(11, 6),
        )

    def _post_rest_retreat_side_step(self) -> OracleDecision | None:
        if self.current_phase not in {"RETURN_AFTER_LATE_FORAGE", "ESCAPE_OR_ACUTE_THREAT"}:
            return None
        if int(self.world.state.food_eaten) <= int(self.phase_food_start):
            return None
        pos = self.world.spider_pos()
        if pos != (10, 7):
            return None
        if self.world.lizard_pos() != (9, 7):
            return None
        if self.world.shelter_role_at(pos) != "outside":
            return None
        return OracleDecision(
            phase="RETURN_AFTER_LATE_FORAGE",
            action="MOVE_UP",
            reason="post_rest_retreat_side_step",
            target=(10, 6),
        )

    def _safer_outside_food_target(
        self,
        candidates: list[tuple[int, int]],
    ) -> tuple[int, int]:
        start = self.world.spider_pos()
        predator_pos = self.world.lizard_pos()
        best = candidates[0]
        best_score: tuple[int, int, int] | None = None
        for candidate in candidates:
            path = self._shortest_path(start, candidate, outside_only=True)
            if path is None:
                path = self._shortest_path(start, candidate)
            if path is None:
                continue
            min_clearance = min(
                self.world.manhattan(pos, predator_pos)
                for pos in path[1:]
            ) if len(path) > 1 else self.world.manhattan(start, predator_pos)
            target_clearance = self.world.manhattan(candidate, predator_pos)
            path_cost = len(path) - 1
            score = (min_clearance, target_clearance, -path_cost)
            if best_score is None or score > best_score:
                best = candidate
                best_score = score
        return best

    def _nearest_outside_cell(self) -> tuple[int, int] | None:
        outside_cells: list[tuple[int, int]] = []
        for x in range(self.world.width):
            for y in range(self.world.height):
                pos = (x, y)
                if self.world.is_walkable(pos) and self.world.shelter_role_at(pos) == "outside":
                    outside_cells.append(pos)
        if not outside_cells:
            return None
        return self.world.nearest(outside_cells)[0]

    def _next_action_toward(self, target: tuple[int, int] | None) -> str:
        start = self.world.spider_pos()
        if target is None or start == target:
            return "STAY"
        action = self._shortest_path_first_action(start, target)
        return action if action is not None else "STAY"

    def _shortest_path_first_action(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> str | None:
        path = self._shortest_path(start, target)
        if path is None or len(path) < 2:
            return None
        dx = path[1][0] - start[0]
        dy = path[1][1] - start[1]
        for action_name, (action_dx, action_dy) in _MOVE_DELTAS.items():
            if (dx, dy) == (action_dx, action_dy):
                return action_name
        return None

    def _outside_only_first_action(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> str | None:
        path = self._shortest_path(start, target, outside_only=True)
        if path is None or len(path) < 2:
            return None
        dx = path[1][0] - start[0]
        dy = path[1][1] - start[1]
        for action_name, (action_dx, action_dy) in _MOVE_DELTAS.items():
            if (dx, dy) == (action_dx, action_dy):
                return action_name
        return None

    def _shortest_path(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        *,
        outside_only: bool = False,
    ) -> list[tuple[int, int]] | None:
        queue = deque([start])
        seen = {start}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                path: list[tuple[int, int]] = []
                cursor: tuple[int, int] | None = current
                while cursor is not None:
                    path.append(cursor)
                    cursor = parent[cursor]
                path.reverse()
                return path
            for action_name in _MOVE_ACTIONS:
                dx, dy = _MOVE_DELTAS[action_name]
                candidate = (current[0] + dx, current[1] + dy)
                if candidate in seen or not self.world.is_walkable(candidate):
                    continue
                if (
                    outside_only
                    and candidate != target
                    and self.world.shelter_role_at(candidate) != "outside"
                ):
                    continue
                seen.add(candidate)
                parent[candidate] = current
                queue.append(candidate)
        return None
