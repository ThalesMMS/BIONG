from __future__ import annotations

from .simulation_episode_shared import *


class _SimulationEpisodeTeacherTargetsMixin:
    def _reset_direct_policy_handoff_teacher_state(self) -> None:
        self._direct_policy_handoff_teacher_state = {
            "stage": "idle",
            "release_tick": -1,
            "hold_tick": -1,
            "continuation_count": 0,
            "forage_tick": -1,
        }

    def _reset_direct_policy_probe_family_teacher_state(self) -> None:
        self._direct_policy_probe_family_teacher_state = {
            "stage": "idle",
            "release_tick": -1,
            "hold_tick": -1,
            "continuation_count": 0,
            "forage_tick": -1,
        }

    def _reset_direct_policy_probe_trajectory_teacher_state(self) -> None:
        self._direct_policy_probe_trajectory_teacher_state = {
            "stage": "idle",
            "release_tick": -1,
            "continuation_count": 0,
        }

    def _reset_direct_policy_probe_cycle_teacher_state(self) -> None:
        self._direct_policy_probe_cycle_teacher_state = {
            "stage": "idle",
            "release_tick": -1,
            "continuation_count": 0,
            "food_count_before": 0,
            "sleep_events_before": 0,
            "forage_tick": -1,
            "return_tick": -1,
        }

    def _reset_direct_policy_probe_trace_teacher_state(self) -> None:
        self._direct_policy_probe_trace_teacher_state = {
            "stage": "idle",
            "sequence_index": 0,
            "food_count_before": 0,
            "sleep_events_before": 0,
            "forage_tick": -1,
            "return_tick": -1,
        }

    def _reset_direct_policy_probe_rollout_teacher_state(self) -> None:
        self._direct_policy_probe_rollout_teacher_state = {
            "stage": "idle",
            "sequence_index": 0,
            "active": False,
        }

    def _reset_direct_policy_probe_replayable_teacher_state(self) -> None:
        self._direct_policy_probe_replayable_teacher_state = {
            "active": False,
            "released": False,
            "down_redirects": 0,
            "first_up_redirects": 0,
        }

    def _frontier_teacher_checkpoint_dir(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent
            / "artifacts"
            / "learning_discovery"
            / "2026-05-07"
            / "checkpoints"
            / "cycle05_large_ecological_late_finetuning"
            / "true_monolithic_policy"
            / "seed_7"
            / "best"
        )

    def _teacher_center_column(self) -> int:
        shelter_columns = sorted({int(cell[0]) for cell in self.world.shelter_cells})
        if not shelter_columns:
            return 0
        return int(shelter_columns[len(shelter_columns) // 2])

    def _teacher_predator_distance(self, current_state: Dict[str, object]) -> float:
        spider_x = int(current_state["x"])
        spider_y = int(current_state["y"])
        predator_positions = current_state.get("predator_positions", [])
        distances: list[float] = []
        if isinstance(predator_positions, list):
            for item in predator_positions:
                if (
                    isinstance(item, list)
                    and len(item) >= 2
                    and item[0] is not None
                    and item[1] is not None
                ):
                    distances.append(
                        abs(int(item[0]) - spider_x) + abs(int(item[1]) - spider_y)
                    )
        if distances:
            return float(min(distances))
        return float("inf")

    def _teacher_no_acute_threat(self, current_state: Dict[str, object]) -> bool:
        predator_trace = current_state.get("predator_trace", {})
        if not isinstance(predator_trace, dict):
            predator_trace = {}
        predator_motion_salience = float(
            current_state.get("predator_motion_salience", 0.0)
        )
        return bool(
            float(current_state.get("recent_contact", 0.0)) <= 0.0
            and float(current_state.get("recent_pain", 0.0)) <= 0.0
            and predator_motion_salience <= 0.0
            and float(predator_trace.get("strength", 0.0)) < 0.45
            and self._teacher_predator_distance(current_state) > 2.0
        )

    def _direct_policy_handoff_teacher_target(
        self,
        *,
        current_state: Dict[str, object],
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None]:
        teacher_state = getattr(self, "_direct_policy_handoff_teacher_state", None)
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_handoff_teacher_state()
            teacher_state = self._direct_policy_handoff_teacher_state
        stage = str(teacher_state.get("stage", "idle"))
        if stage == "done":
            return -1, None

        if stage == "idle":
            if not self._teacher_no_acute_threat(current_state):
                return -1, None
        elif (
            float(current_state.get("recent_contact", 0.0)) > 0.0
            or float(current_state.get("recent_pain", 0.0)) > 0.0
        ):
            teacher_state["stage"] = "done"
            return -1, None

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        hunger = float(current_state.get("hunger", 0.0))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        rest_streak = int(current_state.get("rest_streak", 0))
        food_memory = current_state.get("food_memory", {})
        if not isinstance(food_memory, dict):
            food_memory = {}
        food_memory_target = food_memory.get("target")
        food_memory_age = int(food_memory.get("age", 999))
        has_rightward_food_memory = bool(
            isinstance(food_memory_target, list)
            and len(food_memory_target) >= 2
            and food_memory_target[0] is not None
            and int(food_memory_target[0]) > current_x
        )
        if food_direction_action != "MOVE_RIGHT" or not has_rightward_food_memory:
            if stage in {"await_hold", "continuation"}:
                teacher_state["stage"] = "done"
            return -1, None

        if stage == "idle":
            if (
                current_role == "inside"
                and current_x == center_x
                and sleep_phase in {"RESTING", "DEEP_SLEEP"}
                and rest_streak >= 2
                and 0.12 <= hunger <= 0.30
                and food_memory_age <= 2
            ):
                teacher_state["stage"] = "await_hold"
                teacher_state["release_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "handoff_release"
            if (
                bool(
                    getattr(
                        self.brain.config,
                        "direct_policy_post_rest_release_sequence_teacher",
                        False,
                    )
                )
                and int(current_state.get("sleep_events", 0)) > 0
                and current_role in {"inside", "deep"}
                and sleep_phase in {"AWAKE", "SETTLING"}
                and has_rightward_food_memory
                and food_memory_age <= 4
            ):
                teacher_state["stage"] = "await_hold"
                teacher_state["release_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "handoff_release"
            if (
                bool(
                    getattr(
                        self.brain.config,
                        "direct_policy_post_rest_action_teacher",
                        False,
                    )
                )
                and int(current_state.get("sleep_events", 0)) > 0
                and current_role in {"entrance", "outside"}
                and food_memory_age <= 4
                and food_direction_action == "MOVE_RIGHT"
            ):
                return int(ACTION_TO_INDEX["MOVE_RIGHT"]), "handoff_post_rest_forage"
            return -1, None

        if stage == "await_hold":
            if int(tick) - int(teacher_state.get("release_tick", tick)) > 3:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role == "entrance" and current_x == center_x:
                teacher_state["stage"] = "continuation"
                teacher_state["hold_tick"] = int(tick)
                teacher_state["continuation_count"] = 0
                return int(ACTION_TO_INDEX["STAY"]), "handoff_hold"
            return -1, None

        if stage == "continuation":
            if int(tick) - int(teacher_state.get("hold_tick", tick)) > 4:
                teacher_state["stage"] = "done"
                return -1, None
            continuation_count = int(teacher_state.get("continuation_count", 0))
            if current_role == "outside":
                teacher_state["stage"] = "done"
                return -1, None
            if current_x != center_x or current_role not in {"entrance", "inside", "deep"}:
                teacher_state["stage"] = "done"
                return -1, None
            teacher_state["continuation_count"] = continuation_count + 1
            if teacher_state["continuation_count"] >= 3:
                teacher_state["stage"] = "forage_window"
                teacher_state["forage_tick"] = int(tick)
            return int(ACTION_TO_INDEX["MOVE_DOWN"]), "handoff_continue"

        if (
            bool(
                getattr(
                    self.brain.config,
                    "direct_policy_post_rest_action_teacher",
                    False,
                )
            )
            and stage == "forage_window"
        ):
            if int(tick) - int(teacher_state.get("forage_tick", tick)) > 4:
                teacher_state["stage"] = "done"
                return -1, None
            if (
                current_role in {"entrance", "outside"}
                and food_memory_age <= 4
                and food_direction_action == "MOVE_RIGHT"
            ):
                return int(ACTION_TO_INDEX["MOVE_RIGHT"]), "handoff_forage_window"
            teacher_state["stage"] = "done"
            return -1, None

        teacher_state["stage"] = "done"
        return -1, None

    def _direct_policy_handoff_option_teacher_target(
        self,
        *,
        current_state: Dict[str, object],
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None]:
        teacher_state = getattr(self, "_direct_policy_handoff_teacher_state", None)
        if teacher_state is None:
            self._reset_direct_policy_handoff_teacher_state()
            teacher_state = self._direct_policy_handoff_teacher_state
        current_role = str(current_state.get("shelter_role", "outside"))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        sleep_events = int(current_state.get("sleep_events", 0))
        food_memory = current_state.get("food_memory", {})
        if not isinstance(food_memory, dict):
            food_memory = {}
        food_memory_target = food_memory.get("target")
        food_memory_age = int(food_memory.get("age", 999))
        has_rightward_food_memory = bool(
            isinstance(food_memory_target, list)
            and len(food_memory_target) >= 2
            and food_memory_target[0] is not None
            and int(food_memory_target[0]) > current_x
        )
        no_acute_threat = self._teacher_no_acute_threat(current_state)
        stage = str(teacher_state.get("stage", "idle"))
        if stage in {"await_hold", "continuation"}:
            return int(OPTION_TO_INDEX["POST_REST_REACTIVATE"]), "option_reactivate"
        if stage == "forage_window":
            if int(tick) - int(teacher_state.get("forage_tick", tick)) > 4:
                teacher_state["stage"] = "done"
                return -1, None
            if (
                no_acute_threat
                and has_rightward_food_memory
                and food_memory_age <= 4
                and food_direction_action == "MOVE_RIGHT"
            ):
                return int(OPTION_TO_INDEX["FORAGE"]), "option_forage_window"
            teacher_state["stage"] = "done"
            return -1, None
        if (
            sleep_events > 0
            and no_acute_threat
            and has_rightward_food_memory
            and food_memory_age <= 4
        ):
            if sleep_phase in {"AWAKE", "SETTLING"} and current_role in {
                "deep",
                "inside",
            }:
                return int(OPTION_TO_INDEX["POST_REST_REACTIVATE"]), (
                    "option_post_rest_inside"
                )
            if (
                sleep_phase in {"AWAKE", "SETTLING"}
                and current_role in {"entrance", "outside"}
                and food_direction_action == "MOVE_RIGHT"
            ):
                return int(OPTION_TO_INDEX["FORAGE"]), "option_post_rest_forage"
        if not no_acute_threat and current_role in {"entrance", "outside"}:
            return int(OPTION_TO_INDEX["RETURN_TO_SHELTER"]), (
                "option_threatened_return"
            )
        return -1, None

    def _direct_policy_probe_family_teacher_target(
        self,
        *,
        current_state: Dict[str, object],
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None]:
        teacher_state = getattr(
            self,
            "_direct_policy_probe_family_teacher_state",
            None,
        )
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_probe_family_teacher_state()
            teacher_state = self._direct_policy_probe_family_teacher_state
        stage = str(teacher_state.get("stage", "idle"))
        if stage == "done":
            return -1, None

        if not self._teacher_no_acute_threat(current_state):
            if stage != "idle":
                teacher_state["stage"] = "done"
            return -1, None

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        sleep_events = int(current_state.get("sleep_events", 0))
        food_memory = current_state.get("food_memory", {})
        if not isinstance(food_memory, dict):
            food_memory = {}
        food_memory_target = food_memory.get("target")
        food_memory_age = int(food_memory.get("age", 999))
        has_rightward_food_memory = bool(
            isinstance(food_memory_target, list)
            and len(food_memory_target) >= 2
            and food_memory_target[0] is not None
            and int(food_memory_target[0]) > current_x
        )
        if food_direction_action != "MOVE_RIGHT" or not has_rightward_food_memory:
            if stage in {"await_hold", "continuation", "forage_window"}:
                teacher_state["stage"] = "done"
            return -1, None

        if stage == "idle":
            if (
                sleep_events > 0
                and current_role in {"inside", "deep"}
                and current_x == center_x
                and sleep_phase in {"AWAKE", "SETTLING", "RESTING", "DEEP_SLEEP"}
                and food_memory_age <= 4
            ):
                teacher_state["stage"] = "await_hold"
                teacher_state["release_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "family_release"
            if (
                sleep_events > 0
                and current_role == "entrance"
                and current_x == center_x
                and food_memory_age <= 4
            ):
                teacher_state["stage"] = "continuation"
                teacher_state["hold_tick"] = int(tick)
                teacher_state["continuation_count"] = 0
                return int(ACTION_TO_INDEX["STAY"]), "family_hold"
            return -1, None

        if stage == "await_hold":
            if int(tick) - int(teacher_state.get("release_tick", tick)) > 3:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role == "entrance" and current_x == center_x:
                teacher_state["stage"] = "continuation"
                teacher_state["hold_tick"] = int(tick)
                teacher_state["continuation_count"] = 0
                return int(ACTION_TO_INDEX["STAY"]), "family_hold"
            return -1, None

        if stage == "continuation":
            if int(tick) - int(teacher_state.get("hold_tick", tick)) > 4:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role == "outside":
                teacher_state["stage"] = "done"
                return -1, None
            if current_x != center_x or current_role not in {"entrance", "inside", "deep"}:
                teacher_state["stage"] = "done"
                return -1, None
            continuation_count = int(teacher_state.get("continuation_count", 0)) + 1
            teacher_state["continuation_count"] = continuation_count
            if continuation_count >= 3:
                teacher_state["stage"] = "forage_window"
                teacher_state["forage_tick"] = int(tick)
            return int(ACTION_TO_INDEX["MOVE_DOWN"]), "family_continue"

        if stage == "forage_window":
            if int(tick) - int(teacher_state.get("forage_tick", tick)) > 6:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role in {"entrance", "outside"}:
                return int(ACTION_TO_INDEX["MOVE_RIGHT"]), "family_forage"
            teacher_state["stage"] = "done"
            return -1, None

        teacher_state["stage"] = "done"
        return -1, None

    def _direct_policy_probe_handoff_teacher_target(
        self,
        *,
        current_state: Dict[str, object],
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None]:
        teacher_state = getattr(
            self,
            "_direct_policy_probe_family_teacher_state",
            None,
        )
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_probe_family_teacher_state()
            teacher_state = self._direct_policy_probe_family_teacher_state
        stage = str(teacher_state.get("stage", "idle"))
        if stage == "done":
            return -1, None

        if not self._teacher_no_acute_threat(current_state):
            if stage != "idle":
                teacher_state["stage"] = "done"
            return -1, None

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        hunger = float(current_state.get("hunger", 0.0))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        rest_streak = int(current_state.get("rest_streak", 0))
        sleep_events = int(current_state.get("sleep_events", 0))
        if food_direction_action != "MOVE_RIGHT":
            if stage in {"await_hold", "continuation", "forage_window"}:
                teacher_state["stage"] = "done"
            return -1, None

        if stage == "idle":
            if (
                current_role == "inside"
                and current_x == center_x
                and sleep_phase in {"RESTING", "DEEP_SLEEP"}
                and rest_streak >= 2
                and 0.12 <= hunger <= 0.32
            ):
                teacher_state["stage"] = "await_hold"
                teacher_state["release_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "handoff_family_release"
            if (
                sleep_events > 0
                and current_role in {"inside", "deep"}
                and current_x == center_x
                and sleep_phase in {"AWAKE", "SETTLING"}
                and hunger <= 0.36
            ):
                teacher_state["stage"] = "await_hold"
                teacher_state["release_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "handoff_family_release"
            return -1, None

        if stage == "await_hold":
            if int(tick) - int(teacher_state.get("release_tick", tick)) > 3:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role == "entrance" and current_x == center_x:
                teacher_state["stage"] = "continuation"
                teacher_state["hold_tick"] = int(tick)
                teacher_state["continuation_count"] = 0
                return int(ACTION_TO_INDEX["STAY"]), "handoff_family_hold"
            return -1, None

        if stage == "continuation":
            if int(tick) - int(teacher_state.get("hold_tick", tick)) > 5:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role == "outside":
                teacher_state["stage"] = "done"
                return -1, None
            if current_x != center_x or current_role not in {"entrance", "inside", "deep"}:
                teacher_state["stage"] = "done"
                return -1, None
            continuation_count = int(teacher_state.get("continuation_count", 0)) + 1
            teacher_state["continuation_count"] = continuation_count
            if continuation_count >= 3:
                teacher_state["stage"] = "forage_window"
                teacher_state["forage_tick"] = int(tick)
            return int(ACTION_TO_INDEX["MOVE_DOWN"]), "handoff_family_continue"

        if stage == "forage_window":
            if int(tick) - int(teacher_state.get("forage_tick", tick)) > 6:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role in {"entrance", "outside"}:
                return int(ACTION_TO_INDEX["MOVE_RIGHT"]), "handoff_family_forage"
            teacher_state["stage"] = "done"
            return -1, None

        teacher_state["stage"] = "done"
        return -1, None

    def _direct_policy_probe_trajectory_redirect_action(
        self,
        *,
        current_state: Dict[str, object],
        baseline_action_name: str,
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None]:
        teacher_state = getattr(
            self,
            "_direct_policy_probe_trajectory_teacher_state",
            None,
        )
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_probe_trajectory_teacher_state()
            teacher_state = self._direct_policy_probe_trajectory_teacher_state
        stage = str(teacher_state.get("stage", "idle"))
        if stage == "done":
            return -1, None

        if not self._teacher_no_acute_threat(current_state):
            if stage != "idle":
                teacher_state["stage"] = "done"
            return -1, None

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        hunger = float(current_state.get("hunger", 0.0))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        rest_streak = int(current_state.get("rest_streak", 0))

        if stage == "idle":
            if (
                food_direction_action == "MOVE_RIGHT"
                and current_role == "inside"
                and current_x == center_x
                and sleep_phase in {"RESTING", "DEEP_SLEEP"}
                and rest_streak >= 2
                and 0.12 <= hunger <= 0.32
            ):
                teacher_state["stage"] = "await_hold"
                teacher_state["release_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "trajectory_release"
            return -1, None

        if stage == "await_hold":
            if int(tick) - int(teacher_state.get("release_tick", tick)) > 3:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role == "entrance" and current_x == center_x:
                teacher_state["stage"] = "continuation"
                teacher_state["continuation_count"] = 0
                return int(ACTION_TO_INDEX["STAY"]), "trajectory_hold"
            return -1, None

        if stage == "continuation":
            if current_role == "outside":
                teacher_state["stage"] = "done"
                return -1, None
            if (
                food_direction_action == "MOVE_RIGHT"
                and current_x == center_x
                and current_role in {"entrance", "inside", "deep"}
                and sleep_phase in {"AWAKE", "SETTLING"}
            ):
                continuation_count = int(
                    teacher_state.get("continuation_count", 0)
                ) + 1
                teacher_state["continuation_count"] = continuation_count
                if continuation_count >= 3:
                    teacher_state["stage"] = "done"
                return int(ACTION_TO_INDEX["MOVE_DOWN"]), "trajectory_continue"
            if int(teacher_state.get("continuation_count", 0)) > 0:
                teacher_state["stage"] = "done"
            return -1, None

        teacher_state["stage"] = "done"
        return -1, None

    def _teacher_shelter_return_action(
        self,
        *,
        observation: Dict[str, np.ndarray],
        current_state: Dict[str, object],
    ) -> tuple[int, str] | None:
        sleep_obs = self.brain._bound_observation("sleep_center", observation)
        if sleep_obs["shelter_memory_age"] < 1.0:
            shelter_action = direction_action(
                sleep_obs["shelter_memory_dx"],
                sleep_obs["shelter_memory_dy"],
            )
            if shelter_action != "STAY":
                return int(ACTION_TO_INDEX[shelter_action]), "cycle_return"
        current_role = str(current_state.get("shelter_role", "outside"))
        if current_role == "entrance":
            return int(ACTION_TO_INDEX["STAY"]), "cycle_settle"
        return None

    def _teacher_food_seek_action(self) -> tuple[int, str] | None:
        if not self.world.food_positions:
            return None
        if self.world.on_food():
            return int(ACTION_TO_INDEX["STAY"]), "trace_forage"
        target, _ = self.world.nearest(
            self.world.food_positions,
            origin=self.world.spider_pos(),
        )
        dx = int(target[0]) - int(self.world.spider_pos()[0])
        dy = int(target[1]) - int(self.world.spider_pos()[1])
        action_name = direction_action(dx, dy)
        if action_name == "STAY":
            return int(ACTION_TO_INDEX["STAY"]), "trace_forage"
        return int(ACTION_TO_INDEX[action_name]), "trace_forage"

    def _teacher_deep_shelter_action(self) -> tuple[int, str] | None:
        target = self.world.safest_shelter_target()
        dx = int(target[0]) - int(self.world.spider_pos()[0])
        dy = int(target[1]) - int(self.world.spider_pos()[1])
        action_name = direction_action(dx, dy)
        if action_name == "STAY":
            return int(ACTION_TO_INDEX["STAY"]), "trace_rerest"
        return int(ACTION_TO_INDEX[action_name]), "trace_return"

    def _direct_policy_probe_cycle_redirect_action(
        self,
        *,
        observation: Dict[str, np.ndarray],
        current_state: Dict[str, object],
        baseline_action_name: str,
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None]:
        teacher_state = getattr(
            self,
            "_direct_policy_probe_cycle_teacher_state",
            None,
        )
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_probe_cycle_teacher_state()
            teacher_state = self._direct_policy_probe_cycle_teacher_state
        stage = str(teacher_state.get("stage", "idle"))
        if stage == "done":
            return -1, None

        if not self._teacher_no_acute_threat(current_state):
            if stage != "idle":
                teacher_state["stage"] = "done"
            return -1, None

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        hunger = float(current_state.get("hunger", 0.0))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        rest_streak = int(current_state.get("rest_streak", 0))
        food_eaten = int(current_state.get("food_eaten", 0))
        sleep_events = int(current_state.get("sleep_events", 0))

        if stage == "idle":
            if (
                food_direction_action == "MOVE_RIGHT"
                and current_role == "inside"
                and current_x == center_x
                and sleep_phase in {"RESTING", "DEEP_SLEEP"}
                and rest_streak >= 2
                and 0.12 <= hunger <= 0.32
            ):
                teacher_state["stage"] = "await_hold"
                teacher_state["release_tick"] = int(tick)
                teacher_state["food_count_before"] = int(food_eaten)
                teacher_state["sleep_events_before"] = int(sleep_events)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "cycle_release"
            return -1, None

        if stage == "await_hold":
            if int(tick) - int(teacher_state.get("release_tick", tick)) > 3:
                teacher_state["stage"] = "done"
                return -1, None
            if current_role == "entrance" and current_x == center_x:
                teacher_state["stage"] = "continuation"
                teacher_state["continuation_count"] = 0
                return int(ACTION_TO_INDEX["STAY"]), "cycle_hold"
            return -1, None

        if stage == "continuation":
            if current_role == "outside":
                teacher_state["stage"] = "forage_window"
                teacher_state["forage_tick"] = int(tick)
                return -1, None
            if (
                food_direction_action == "MOVE_RIGHT"
                and current_x == center_x
                and current_role in {"entrance", "inside", "deep"}
                and sleep_phase in {"AWAKE", "SETTLING"}
            ):
                continuation_count = int(
                    teacher_state.get("continuation_count", 0)
                ) + 1
                teacher_state["continuation_count"] = continuation_count
                if continuation_count >= 3:
                    teacher_state["stage"] = "forage_window"
                    teacher_state["forage_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_DOWN"]), "cycle_continue"
            if int(teacher_state.get("continuation_count", 0)) > 0:
                teacher_state["stage"] = "forage_window"
                teacher_state["forage_tick"] = int(tick)
            return -1, None

        if stage == "forage_window":
            if food_eaten > int(teacher_state.get("food_count_before", 0)):
                teacher_state["stage"] = "return_window"
                teacher_state["return_tick"] = int(tick)
            elif (
                int(tick) - int(teacher_state.get("forage_tick", tick)) <= 12
                and current_role in {"entrance", "outside"}
                and food_direction_action == "MOVE_RIGHT"
            ):
                return int(ACTION_TO_INDEX["MOVE_RIGHT"]), "cycle_forage"
            else:
                teacher_state["stage"] = "return_window"
                teacher_state["return_tick"] = int(tick)

        if stage == "return_window":
            if current_role in {"inside", "deep"}:
                teacher_state["stage"] = "rerest_window"
            else:
                shelter_return = self._teacher_shelter_return_action(
                    observation=observation,
                    current_state=current_state,
                )
                if shelter_return is not None:
                    return shelter_return
                if int(tick) - int(teacher_state.get("return_tick", tick)) > 12:
                    teacher_state["stage"] = "done"
                return -1, None

        if stage == "rerest_window":
            if sleep_events > int(teacher_state.get("sleep_events_before", 0)):
                teacher_state["stage"] = "done"
                return int(ACTION_TO_INDEX["STAY"]), "cycle_rerest"
            if current_role in {"inside", "deep"}:
                if sleep_phase in {"RESTING", "DEEP_SLEEP"} or rest_streak >= 2:
                    teacher_state["stage"] = "done"
                return int(ACTION_TO_INDEX["STAY"]), "cycle_rerest"
            shelter_return = self._teacher_shelter_return_action(
                observation=observation,
                current_state=current_state,
            )
            if shelter_return is not None:
                return shelter_return
            teacher_state["stage"] = "done"
            return -1, None

        teacher_state["stage"] = "done"
        return -1, None

    def _direct_policy_probe_trace_redirect_action(
        self,
        *,
        observation: Dict[str, np.ndarray],
        current_state: Dict[str, object],
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None]:
        teacher_state = getattr(
            self,
            "_direct_policy_probe_trace_teacher_state",
            None,
        )
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_probe_trace_teacher_state()
            teacher_state = self._direct_policy_probe_trace_teacher_state
        stage = str(teacher_state.get("stage", "idle"))
        if stage == "done":
            return -1, None

        if not self._teacher_no_acute_threat(current_state):
            if stage != "idle":
                teacher_state["stage"] = "done"
            return -1, None

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        hunger = float(current_state.get("hunger", 0.0))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        rest_streak = int(current_state.get("rest_streak", 0))
        food_eaten = int(current_state.get("food_eaten", 0))
        sleep_events = int(current_state.get("sleep_events", 0))
        trace_sequence = (
            ("MOVE_UP", "trace_release"),
            ("STAY", "trace_hold"),
            ("MOVE_DOWN", "trace_continue"),
            ("MOVE_DOWN", "trace_continue"),
            ("MOVE_DOWN", "trace_continue"),
            ("MOVE_LEFT", "trace_slide_left"),
            ("MOVE_LEFT", "trace_slide_left"),
            ("MOVE_RIGHT", "trace_realign"),
        )

        if stage == "idle":
            if (
                food_direction_action == "MOVE_RIGHT"
                and current_role == "inside"
                and current_x == center_x
                and sleep_phase in {"RESTING", "DEEP_SLEEP"}
                and rest_streak >= 2
                and 0.12 <= hunger <= 0.32
            ):
                teacher_state["stage"] = "trace_sequence"
                teacher_state["sequence_index"] = 1
                teacher_state["food_count_before"] = int(food_eaten)
                teacher_state["sleep_events_before"] = int(sleep_events)
                teacher_state["forage_tick"] = int(tick)
                return int(ACTION_TO_INDEX["MOVE_UP"]), "trace_release"
            else:
                return -1, None

        if stage == "trace_sequence":
            sequence_index = int(teacher_state.get("sequence_index", 0))
            if sequence_index >= len(trace_sequence):
                teacher_state["stage"] = "forage_window"
                teacher_state["forage_tick"] = int(tick)
                stage = "forage_window"
            else:
                teacher_state["sequence_index"] = sequence_index + 1
                action_name, action_stage = trace_sequence[sequence_index]
                return int(ACTION_TO_INDEX[action_name]), action_stage

        if stage == "forage_window":
            if food_eaten > int(teacher_state.get("food_count_before", 0)):
                teacher_state["stage"] = "return_window"
                teacher_state["return_tick"] = int(tick)
                stage = "return_window"
            else:
                forage_action = self._teacher_food_seek_action()
                if forage_action is not None:
                    return forage_action
                if int(tick) - int(teacher_state.get("forage_tick", tick)) > 18:
                    teacher_state["stage"] = "done"
                return -1, None

        if stage == "return_window":
            if current_role == "deep":
                teacher_state["stage"] = "rerest_window"
                stage = "rerest_window"
            else:
                return_action = self._teacher_deep_shelter_action()
                if return_action is not None:
                    return return_action
                if int(tick) - int(teacher_state.get("return_tick", tick)) > 18:
                    teacher_state["stage"] = "done"
                return -1, None

        if stage == "rerest_window":
            if sleep_events > int(teacher_state.get("sleep_events_before", 0)):
                teacher_state["stage"] = "done"
                return int(ACTION_TO_INDEX["STAY"]), "trace_rerest"
            if current_role in {"entrance", "outside"}:
                return_action = self._teacher_deep_shelter_action()
                if return_action is not None:
                    return return_action
                teacher_state["stage"] = "done"
                return -1, None
            if sleep_phase in {"RESTING", "DEEP_SLEEP"} or rest_streak >= 2:
                teacher_state["stage"] = "done"
            return int(ACTION_TO_INDEX["STAY"]), "trace_rerest"

        teacher_state["stage"] = "done"
        return -1, None

    def _direct_policy_probe_rollout_redirect_action(
        self,
        *,
        current_state: Dict[str, object],
        food_direction_action: str | None,
        tick: int,
    ) -> tuple[int, str | None, bool]:
        teacher_state = getattr(
            self,
            "_direct_policy_probe_rollout_teacher_state",
            None,
        )
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_probe_rollout_teacher_state()
            teacher_state = self._direct_policy_probe_rollout_teacher_state
        stage = str(teacher_state.get("stage", "idle"))
        active = bool(teacher_state.get("active", False))
        if stage == "done":
            return -1, None, active

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        hunger = float(current_state.get("hunger", 0.0))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        rest_streak = int(current_state.get("rest_streak", 0))
        rollout_sequence = (
            ("MOVE_UP", "rollout_release"),
            ("STAY", "rollout_hold"),
            ("MOVE_DOWN", "rollout_continue"),
            ("MOVE_DOWN", "rollout_continue"),
            ("MOVE_DOWN", "rollout_continue"),
            ("MOVE_LEFT", "rollout_slide_left"),
            ("MOVE_LEFT", "rollout_slide_left"),
            ("MOVE_RIGHT", "rollout_realign"),
        )

        if stage == "idle":
            if not self._teacher_no_acute_threat(current_state):
                return -1, None, active
            if (
                food_direction_action == "MOVE_RIGHT"
                and current_role == "inside"
                and current_x == center_x
                and sleep_phase in {"RESTING", "DEEP_SLEEP"}
                and rest_streak >= 2
                and 0.12 <= hunger <= 0.32
            ):
                teacher_state["stage"] = "prefix"
                teacher_state["sequence_index"] = 1
                teacher_state["active"] = True
                return int(ACTION_TO_INDEX["MOVE_UP"]), "rollout_release", True
            return -1, None, active

        if stage == "prefix":
            sequence_index = int(teacher_state.get("sequence_index", 0))
            if sequence_index >= len(rollout_sequence):
                teacher_state["stage"] = "live_rollout"
                return -1, "rollout_live", True
            teacher_state["sequence_index"] = sequence_index + 1
            action_name, action_stage = rollout_sequence[sequence_index]
            return int(ACTION_TO_INDEX[action_name]), action_stage, True

        if stage == "live_rollout":
            return -1, "rollout_live", True

        teacher_state["stage"] = "done"
        return -1, None, active

    def _direct_policy_probe_replayable_teacher_action(
        self,
        *,
        current_state: Dict[str, object],
        food_direction_action: str | None,
    ) -> tuple[int, str | None, bool]:
        teacher_state = getattr(
            self,
            "_direct_policy_probe_replayable_teacher_state",
            None,
        )
        if not isinstance(teacher_state, dict):
            self._reset_direct_policy_probe_replayable_teacher_state()
            teacher_state = self._direct_policy_probe_replayable_teacher_state

        if not self._teacher_no_acute_threat(current_state):
            return -1, None, bool(teacher_state.get("active", False))

        current_role = str(current_state.get("shelter_role", "outside"))
        current_x = int(current_state.get("x", 0))
        center_x = self._teacher_center_column()
        hunger = float(current_state.get("hunger", 0.0))
        sleep_phase = str(current_state.get("sleep_phase", "AWAKE"))
        rest_streak = int(current_state.get("rest_streak", 0))

        if (
            not bool(teacher_state.get("released", False))
            and food_direction_action == "MOVE_RIGHT"
            and current_role == "inside"
            and current_x == center_x
            and sleep_phase in {"RESTING", "DEEP_SLEEP"}
            and rest_streak >= 2
            and 0.12 <= hunger <= 0.32
        ):
            teacher_state["active"] = True
            teacher_state["released"] = True
            teacher_state["first_up_redirects"] = int(
                teacher_state.get("first_up_redirects", 0)
            ) + 1
            return int(ACTION_TO_INDEX["MOVE_UP"]), "probe_release", True

        if (
            bool(teacher_state.get("released", False))
            and food_direction_action == "MOVE_RIGHT"
            and current_role in {"inside", "entrance"}
            and sleep_phase == "AWAKE"
            and int(teacher_state.get("down_redirects", 0)) < 3
        ):
            teacher_state["active"] = True
            teacher_state["down_redirects"] = int(
                teacher_state.get("down_redirects", 0)
            ) + 1
            return int(ACTION_TO_INDEX["MOVE_DOWN"]), "probe_down_continue", True

        return -1, None, bool(teacher_state.get("active", False))

    def _direct_policy_affordance_targets(
        self,
        *,
        current_state: Dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
        current_pos = (int(current_state["x"]), int(current_state["y"]))
        current_role = self.world.shelter_role_at(current_pos)
        blocked_targets = np.zeros(len(ACTIONS), dtype=float)
        role_targets = np.zeros(len(ACTIONS), dtype=int)
        current_role_idx = int(AFFORDANCE_SHELTER_ROLE_TO_INDEX[current_role])
        for action_idx, action_name in enumerate(ACTIONS):
            if action_name not in self.world.move_deltas:
                role_targets[action_idx] = current_role_idx
                continue
            dx, dy = self.world.move_deltas[action_name]
            next_pos = (current_pos[0] + int(dx), current_pos[1] + int(dy))
            if not self.world.is_walkable(next_pos):
                blocked_targets[action_idx] = 1.0
                role_targets[action_idx] = current_role_idx
                continue
            role_targets[action_idx] = int(
                AFFORDANCE_SHELTER_ROLE_TO_INDEX[
                    self.world.shelter_role_at(next_pos)
                ]
            )
        return blocked_targets, role_targets

    def _direct_policy_geometry_targets(
        self,
        *,
        current_state: Dict[str, object],
    ) -> np.ndarray:
        current_pos = (int(current_state["x"]), int(current_state["y"]))
        current_role = self.world.shelter_role_at(current_pos)
        current_role_idx = int(AFFORDANCE_SHELTER_ROLE_TO_INDEX[current_role])
        geometry_targets = np.zeros(
            len(ACTIONS) * len(AFFORDANCE_GEOMETRY_TARGET_NAMES),
            dtype=float,
        )
        for action_idx, action_name in enumerate(ACTIONS):
            offset = action_idx * len(AFFORDANCE_GEOMETRY_TARGET_NAMES)
            if action_name not in self.world.move_deltas:
                continue
            dx, dy = self.world.move_deltas[action_name]
            next_pos = (current_pos[0] + int(dx), current_pos[1] + int(dy))
            if not self.world.is_walkable(next_pos):
                continue
            next_role = self.world.shelter_role_at(next_pos)
            next_role_idx = int(AFFORDANCE_SHELTER_ROLE_TO_INDEX[next_role])
            geometry_targets[offset + 0] = float(next_role_idx > current_role_idx)
            geometry_targets[offset + 1] = float(next_role == "entrance")
            geometry_targets[offset + 2] = float(next_role == "outside")
        return geometry_targets

    def _shelter_column_label(self, pos: tuple[int, int]) -> str:
        if pos not in self.world.shelter_cells:
            return "outside"
        shelter_columns = sorted({int(cell[0]) for cell in self.world.shelter_cells})
        if not shelter_columns:
            return "outside"
        x = int(pos[0])
        if len(shelter_columns) == 1:
            return "center"
        if len(shelter_columns) == 2:
            return "left" if x == shelter_columns[0] else "right"
        if x == shelter_columns[0]:
            return "left"
        if x == shelter_columns[-1]:
            return "right"
        return "center"

    def _direct_policy_shelter_column_targets(
        self,
        *,
        current_state: Dict[str, object],
    ) -> np.ndarray:
        current_pos = (int(current_state["x"]), int(current_state["y"]))
        current_column_idx = int(
            AFFORDANCE_SHELTER_COLUMN_TO_INDEX[
                self._shelter_column_label(current_pos)
            ]
        )
        column_targets = np.full(len(ACTIONS), current_column_idx, dtype=int)
        for action_idx, action_name in enumerate(ACTIONS):
            if action_name not in self.world.move_deltas:
                continue
            dx, dy = self.world.move_deltas[action_name]
            next_pos = (current_pos[0] + int(dx), current_pos[1] + int(dy))
            if not self.world.is_walkable(next_pos):
                continue
            column_targets[action_idx] = int(
                AFFORDANCE_SHELTER_COLUMN_TO_INDEX[
                    self._shelter_column_label(next_pos)
                ]
            )
        return column_targets

    def _shelter_position_label(self, pos: tuple[int, int]) -> str:
        role = self.world.shelter_role_at(pos)
        if role == "outside":
            return "outside"
        column = self._shelter_column_label(pos)
        return f"{role}_{column}"

    def _direct_policy_shelter_position_targets(
        self,
        *,
        current_state: Dict[str, object],
    ) -> np.ndarray:
        current_pos = (int(current_state["x"]), int(current_state["y"]))
        current_position_idx = int(
            AFFORDANCE_SHELTER_POSITION_TO_INDEX[
                self._shelter_position_label(current_pos)
            ]
        )
        position_targets = np.full(len(ACTIONS), current_position_idx, dtype=int)
        for action_idx, action_name in enumerate(ACTIONS):
            if action_name not in self.world.move_deltas:
                continue
            dx, dy = self.world.move_deltas[action_name]
            next_pos = (current_pos[0] + int(dx), current_pos[1] + int(dy))
            if not self.world.is_walkable(next_pos):
                continue
            position_targets[action_idx] = int(
                AFFORDANCE_SHELTER_POSITION_TO_INDEX[
                    self._shelter_position_label(next_pos)
                ]
            )
        return position_targets

    @staticmethod
    def _direct_policy_transition_prediction_targets(
        *,
        observation_meta: Dict[str, object],
    ) -> np.ndarray:
        targets = np.zeros(
            len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4,
            dtype=float,
        )
        transitions = observation_meta.get("local_transition_consequences")
        if not isinstance(transitions, dict):
            return targets
        offset = 0
        for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES:
            consequence = transitions.get(action_name, {})
            if not isinstance(consequence, dict):
                consequence = {}
            targets[offset + 0] = float(
                np.clip(consequence.get("food_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 1] = float(
                np.clip(consequence.get("shelter_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 2] = float(
                np.clip(consequence.get("predator_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 3] = float(
                bool(consequence.get("next_cell_has_food", False))
            )
            offset += 4
        return targets
