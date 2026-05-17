"""Simulation runtime facade.

Multi-seed comparison workflows and comparison statistics live in
`spider_cortex_sim.comparison`; claim-test evaluation lives in
`spider_cortex_sim.claim_evaluation`.

``default_behavior_evaluation`` and ``ensure_behavior_evaluation`` are the
public homes for the default behavior-evaluation helpers previously kept in
the CLI entrypoint.
"""

from __future__ import annotations

import math
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence

import numpy as np

from .ablations import (
    BrainAblationConfig,
    PROPOSAL_SOURCE_NAMES,
    default_brain_config,
    resolve_ablation_configs,
    resolve_ablation_scenario_group,
)
from .agent import BrainStep, SpiderBrain
from .bus import MessageBus
from .direct_policy_affordances import (
    AFFORDANCE_GEOMETRY_TARGET_NAMES,
    AFFORDANCE_SHELTER_COLUMN_NAMES,
    AFFORDANCE_SHELTER_COLUMN_TO_INDEX,
    AFFORDANCE_SHELTER_POSITION_TO_INDEX,
    AFFORDANCE_SHELTER_ROLE_TO_INDEX,
    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
)
from .direct_policy_options import OPTION_NAMES, OPTION_TO_INDEX
from .interfaces import ACTION_TO_INDEX, MODULE_INTERFACES
from .maps import MAP_TEMPLATE_NAMES
from .metrics import (
    BehavioralEpisodeScore,
    EpisodeMetricAccumulator,
    EpisodeStats,
    PREDATOR_TYPE_NAMES,
    aggregate_behavior_scores,
    aggregate_episode_stats,
    competence_label_from_eval_reflex_scale,
    flatten_behavior_rows,
    normalize_competence_label,
    summarize_behavior_suite,
)
from .checkpointing import (
    CHECKPOINT_METRIC_ORDER,
    CHECKPOINT_PENALTY_MODE_NAMES,
    CheckpointPenaltyMode,
    CheckpointSelectionConfig,
    checkpoint_candidate_composite_score,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    file_sha256,
    jsonify_observation,
    mean_reward_from_behavior_payload,
    persist_checkpoint_pair,
    resolve_checkpoint_load_dir,
)
from .curriculum import (
    CURRICULUM_COLUMNS,
    CURRICULUM_FOCUS_SCENARIOS,
    CURRICULUM_PROFILE_NAMES,
    SUBSKILL_CHECK_MAPPINGS,
    CurriculumPhaseDefinition,
    PromotionCheckCriteria,
    empty_curriculum_summary,
    evaluate_promotion_check_specs,
    promotion_check_spec_records,
    regime_row_metadata_from_summary,
    resolve_curriculum_phase_budgets,
    resolve_curriculum_profile,
    validate_curriculum_profile,
)
from .distillation import DistillationDataset
from .distillation.teacher import load_teacher_checkpoint
from .export import (
    compact_aggregate,
    compact_behavior_payload,
    jsonify,
    save_behavior_csv,
    save_summary,
    save_trace,
)
from .noise import (
    NoiseConfig,
    resolve_noise_profile,
)
from .observation_adapters import (
    adapt_observation_contracts,
    adapter_trace_summary,
    observation_vectors_from_adapters,
)
from .operational_profiles import OperationalProfile, runtime_operational_profile
from .phase import PHASE_TO_INDEX, derive_phase_target
from .predator import PREDATOR_STATES
from .reflexes import _direction_action as direction_action
from .scenarios import (
    CAPABILITY_PROBE_SCENARIOS as SCENARIO_CAPABILITY_PROBE_SCENARIOS,
    BenchmarkTier,
    ProbeType,
    SCENARIO_NAMES,
    get_scenario,
)
from .training_regimes import (
    AnnealingSchedule,
    TrainingRegimeSpec,
    resolve_training_regime,
)
from .world import ACTIONS, REWARD_COMPONENT_NAMES, SpiderWorld

from .simulation_helpers import EXPERIMENT_OF_RECORD_REGIME, CAPABILITY_PROBE_SCENARIOS, default_behavior_evaluation, ensure_behavior_evaluation, is_capability_probe, _probe_metadata_for_scenario


class SimulationEpisodeMixin:
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

    @staticmethod
    def _direct_policy_transition_rollout_prediction_targets(
        *,
        observation_meta: Dict[str, object],
    ) -> np.ndarray:
        targets = np.zeros(
            len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4,
            dtype=float,
        )
        rollouts = observation_meta.get("local_transition_rollouts")
        if not isinstance(rollouts, dict):
            return targets
        offset = 0
        for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES:
            rollout = rollouts.get(action_name, {})
            if not isinstance(rollout, dict):
                rollout = {}
            targets[offset + 0] = float(
                np.clip(rollout.get("best_food_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 1] = float(
                np.clip(rollout.get("best_shelter_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 2] = float(
                np.clip(rollout.get("best_predator_dist_delta", 0.0), -1.0, 1.0)
            )
            targets[offset + 3] = float(
                bool(rollout.get("food_reachable_within_two_steps", False))
            )
            offset += 4
        return targets

    def collect_direct_policy_probe_distillation_rollout(
        self,
        *,
        scenario_name: str,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        if scenario_name not in {
            "continuous_survival_post_rest_inside_v1",
            "continuous_survival_post_rest_entrance_v1",
        }:
            raise ValueError(
                "Direct-policy probe distillation only supports post-rest continuation scenarios."
            )
        scenario = get_scenario(scenario_name)
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe",
                "scenario_name": str(scenario_name),
                "episodes": int(max(0, episodes)),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            episode_index = int(episode_start + episode_offset)
            episode_seed = self.seed + 997 * (episode_index + 1)
            episode_action_stages: list[str] = []
            episode_option_stages: list[str] = []
            if self.world.map_template_name != scenario.map_template:
                self.world.configure_map_template(scenario.map_template)
            self.world.reset(seed=episode_seed)
            scenario.setup(self.world)
            observation = self.world.observe()
            self._reset_direct_policy_handoff_teacher_state()
            for step in range(int(scenario.max_steps)):
                observation_adapters = adapt_observation_contracts(
                    observation,
                    tick=step,
                )
                brain_observation = observation_vectors_from_adapters(
                    observation_adapters
                )
                current_state = self.world.state_dict()
                food_memory = current_state.get("food_memory")
                if isinstance(food_memory, dict) and isinstance(
                    food_memory.get("target"),
                    tuple,
                ):
                    current_state["food_memory"] = dict(food_memory)
                    current_state["food_memory"]["target"] = list(
                        food_memory["target"]
                    )
                food_direction_action = self.brain._food_direction_bias_action(
                    brain_observation
                )
                teacher_action_idx, teacher_action_stage = (
                    self._direct_policy_handoff_teacher_target(
                        current_state=current_state,
                        food_direction_action=food_direction_action,
                        tick=step,
                    )
                )
                if teacher_action_idx < 0:
                    break
                _, teacher_option_stage = (
                    self._direct_policy_handoff_option_teacher_target(
                        current_state=current_state,
                        food_direction_action=food_direction_action,
                        tick=step,
                    )
                )
                if teacher_action_stage is not None:
                    episode_action_stages.append(str(teacher_action_stage))
                if teacher_option_stage is not None:
                    episode_option_stages.append(str(teacher_option_stage))
                teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                teacher_policy[int(teacher_action_idx)] = 1.0
                teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                teacher_logits[int(teacher_action_idx)] = 6.0
                dataset.add_sample(
                    episode=episode_index,
                    step=step,
                    observation=brain_observation,
                    teacher_policy=teacher_policy,
                    teacher_total_logits=teacher_logits,
                    teacher_action_center_policy=teacher_policy,
                    teacher_action_center_logits=teacher_logits,
                    teacher_action_intent_idx=int(teacher_action_idx),
                    teacher_module_policies={},
                )
                next_observation, _, done, _ = self.world.step(int(teacher_action_idx))
                observation = next_observation
                if done:
                    break
            dataset.teacher_metadata.setdefault("stages", [])
            dataset.teacher_metadata["stages"].extend(episode_action_stages)
            dataset.teacher_metadata.setdefault("option_stages", [])
            dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        return dataset

    def collect_direct_policy_probe_sequence_distillation_rollout(
        self,
        *,
        scenario_name: str,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        scripted_sequences: dict[str, tuple[str, ...]] = {
            "continuous_survival_post_rest_inside_v1": (
                "MOVE_UP",
                "MOVE_UP",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
            ),
            "continuous_survival_post_rest_entrance_v1": (
                "MOVE_UP",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
            ),
            "continuous_survival_return_after_late_forage_v1": (
                "MOVE_LEFT",
                "MOVE_LEFT",
                "MOVE_LEFT",
                "MOVE_LEFT",
                "STAY",
                "STAY",
            ),
            "continuous_survival_re_rest_after_return_v1": (
                "MOVE_DOWN",
                "STAY",
                "STAY",
            ),
        }
        if scenario_name not in scripted_sequences:
            raise ValueError(
                "Direct-policy probe sequence distillation only supports scripted continuation scenarios."
            )
        scenario = get_scenario(scenario_name)
        sequence = scripted_sequences[str(scenario_name)]
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe_sequence",
                "scenario_name": str(scenario_name),
                "episodes": int(max(0, episodes)),
                "scripted_sequence": list(sequence),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            episode_index = int(episode_start + episode_offset)
            episode_seed = self.seed + 997 * (episode_index + 1)
            if self.world.map_template_name != scenario.map_template:
                self.world.configure_map_template(scenario.map_template)
            self.world.reset(seed=episode_seed)
            scenario.setup(self.world)
            observation = self.world.observe()
            for step, action_name in enumerate(sequence):
                action_idx = int(ACTION_TO_INDEX[action_name])
                observation_adapters = adapt_observation_contracts(
                    observation,
                    tick=step,
                )
                brain_observation = observation_vectors_from_adapters(
                    observation_adapters
                )
                teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                teacher_policy[action_idx] = 1.0
                teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                teacher_logits[action_idx] = 6.0
                dataset.add_sample(
                    episode=episode_index,
                    step=step,
                    observation=brain_observation,
                    teacher_policy=teacher_policy,
                    teacher_total_logits=teacher_logits,
                    teacher_action_center_policy=teacher_policy,
                    teacher_action_center_logits=teacher_logits,
                    teacher_action_intent_idx=action_idx,
                    teacher_module_policies={},
                )
                next_observation, _, done, _ = self.world.step(action_idx)
                observation = next_observation
                if done:
                    break
        return dataset

    def _direct_policy_probe_family_fallback_action(
        self,
        *,
        observation: Dict[str, np.ndarray],
        current_state: Dict[str, object],
        food_direction_action: str | None,
    ) -> tuple[int, str]:
        threat_escape_action = self.brain._threat_escape_bias_action(observation)
        if threat_escape_action is not None:
            return int(ACTION_TO_INDEX[threat_escape_action]), "fallback_escape"
        sleep_rest_action = self.brain._sleep_rest_bias_action(observation)
        if sleep_rest_action == "STAY":
            return int(ACTION_TO_INDEX["STAY"]), "fallback_rest"
        current_role = str(current_state.get("shelter_role", "outside"))
        hunger = float(current_state.get("hunger", 0.0))
        if food_direction_action is not None and (
            current_role in {"entrance", "outside"} or hunger >= 0.18
        ):
            return int(ACTION_TO_INDEX[food_direction_action]), "fallback_food"
        sleep_obs = self.brain._bound_observation("sleep_center", observation)
        if (
            sleep_obs["shelter_memory_age"] < 1.0
            and (
                sleep_obs["night"] > 0.0
                or float(current_state.get("fatigue", 0.0)) >= 0.12
                or float(current_state.get("sleep_debt", 0.0)) >= 0.08
                or current_role == "outside"
            )
        ):
            shelter_action = direction_action(
                sleep_obs["shelter_memory_dx"],
                sleep_obs["shelter_memory_dy"],
            )
            if shelter_action != "STAY":
                return int(ACTION_TO_INDEX[shelter_action]), "fallback_return"
        if current_role == "entrance":
            return int(ACTION_TO_INDEX["STAY"]), "fallback_hold"
        return int(ACTION_TO_INDEX["STAY"]), "fallback_idle"

    def collect_direct_policy_probe_family_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe_family",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self._reset_direct_policy_handoff_teacher_state()
                self._reset_direct_policy_probe_family_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(
                            food_memory.get("target"),
                            tuple,
                        )
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_family_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx, teacher_action_stage = (
                            self._direct_policy_probe_family_fallback_action(
                                observation=brain_observation,
                                current_state=current_state,
                                food_direction_action=food_direction_action,
                            )
                        )
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        return dataset

    def collect_direct_policy_probe_handoff_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "scripted_direct_policy_post_rest_probe_handoff",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self._reset_direct_policy_handoff_teacher_state()
                self._reset_direct_policy_probe_family_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if isinstance(food_memory, dict) and isinstance(
                        food_memory.get("target"),
                        tuple,
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_handoff_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx, teacher_action_stage = (
                            self._direct_policy_probe_family_fallback_action(
                                observation=brain_observation,
                                current_state=current_state,
                                food_direction_action=food_direction_action,
                            )
                        )
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        return dataset

    def collect_direct_policy_probe_trajectory_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_trajectory",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_trajectory_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(
                            food_memory.get("target"),
                            tuple,
                        )
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    baseline_action_idx = int(teacher_decision.action_idx)
                    baseline_action_name = ACTIONS[baseline_action_idx]
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_trajectory_redirect_action(
                            current_state=current_state,
                            baseline_action_name=baseline_action_name,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = baseline_action_idx
                        teacher_action_stage = "live_policy"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_cycle_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_cycle",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_cycle_teacher_state()
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    baseline_action_idx = int(teacher_decision.action_idx)
                    baseline_action_name = ACTIONS[baseline_action_idx]
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_cycle_redirect_action(
                            observation=brain_observation,
                            current_state=current_state,
                            baseline_action_name=baseline_action_name,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = baseline_action_idx
                        teacher_action_stage = "live_policy"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_trace_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
            "continuous_survival_return_after_late_forage_v1",
            "continuous_survival_re_rest_after_return_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_trace",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_trace_teacher_state()
                if scenario_name == "continuous_survival_return_after_late_forage_v1":
                    self._direct_policy_probe_trace_teacher_state["stage"] = "return_window"
                    self._direct_policy_probe_trace_teacher_state["return_tick"] = 0
                    self._direct_policy_probe_trace_teacher_state["sleep_events_before"] = int(
                        self.world.state.sleep_events
                    )
                elif scenario_name == "continuous_survival_re_rest_after_return_v1":
                    self._direct_policy_probe_trace_teacher_state["stage"] = "rerest_window"
                    self._direct_policy_probe_trace_teacher_state["sleep_events_before"] = int(
                        self.world.state.sleep_events
                    )
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage = (
                        self._direct_policy_probe_trace_redirect_action(
                            observation=brain_observation,
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if teacher_action_idx < 0:
                        teacher_decision = self.brain.act(
                            brain_observation,
                            bus=None,
                            sample=False,
                            training=False,
                        )
                        teacher_action_idx = int(teacher_decision.action_idx)
                        teacher_action_stage = "live_policy"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    episode_action_stages.append(str(teacher_action_stage))
                    if teacher_option_stage is not None:
                        episode_option_stages.append(str(teacher_option_stage))
                    teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                    teacher_policy[int(teacher_action_idx)] = 1.0
                    teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                    teacher_logits[int(teacher_action_idx)] = 6.0
                    dataset.add_sample(
                        episode=episode_index,
                        step=step,
                        observation=brain_observation,
                        teacher_policy=teacher_policy,
                        teacher_total_logits=teacher_logits,
                        teacher_action_center_policy=teacher_policy,
                        teacher_action_center_logits=teacher_logits,
                        teacher_action_intent_idx=int(teacher_action_idx),
                        teacher_module_policies={},
                    )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                dataset.teacher_metadata.setdefault("action_stages", [])
                dataset.teacher_metadata["action_stages"].extend(episode_action_stages)
                dataset.teacher_metadata.setdefault("option_stages", [])
                dataset.teacher_metadata["option_stages"].extend(episode_option_stages)
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_rollout_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "redirected_direct_policy_post_rest_probe_rollout",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_rollout_teacher_state()
                triggered = False
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage, active = (
                        self._direct_policy_probe_rollout_redirect_action(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = int(teacher_decision.action_idx)
                    if teacher_action_stage is None and active:
                        teacher_action_stage = "rollout_live"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if active:
                        triggered = True
                        episode_action_stages.append(str(teacher_action_stage))
                        if teacher_option_stage is not None:
                            episode_option_stages.append(str(teacher_option_stage))
                        teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                        teacher_policy[int(teacher_action_idx)] = 1.0
                        teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                        teacher_logits[int(teacher_action_idx)] = 6.0
                        dataset.add_sample(
                            episode=episode_index,
                            step=step,
                            observation=brain_observation,
                            teacher_policy=teacher_policy,
                            teacher_total_logits=teacher_logits,
                            teacher_action_center_policy=teacher_policy,
                            teacher_action_center_logits=teacher_logits,
                            teacher_action_intent_idx=int(teacher_action_idx),
                            teacher_module_policies={},
                        )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                if triggered:
                    dataset.teacher_metadata.setdefault("action_stages", [])
                    dataset.teacher_metadata["action_stages"].extend(
                        episode_action_stages
                    )
                    dataset.teacher_metadata.setdefault("option_stages", [])
                    dataset.teacher_metadata["option_stages"].extend(
                        episode_option_stages
                    )
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_frontier_teacher_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = (
            "continuous_survival_canonical",
            "continuous_survival_easy_v1",
        )
        checkpoint_dir = self._frontier_teacher_checkpoint_dir()
        teacher_checkpoint_error: str | None = None
        try:
            teacher_brain, teacher_checkpoint_metadata = load_teacher_checkpoint(
                checkpoint_dir
            )
        except (FileNotFoundError, ValueError) as exc:
            teacher_brain = self.brain
            teacher_checkpoint_metadata = {}
            teacher_checkpoint_error = str(exc)
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "checkpoint_direct_policy_post_rest_probe_frontier_teacher",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
                "teacher_checkpoint": str(checkpoint_dir),
                "teacher_ablation": str(
                    teacher_checkpoint_metadata.get("ablation_config", {}).get(
                        "name",
                        "",
                    )
                ),
                "teacher_checkpoint_error": teacher_checkpoint_error,
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                teacher_brain.reset_hidden_states()
                self._reset_direct_policy_probe_rollout_teacher_state()
                triggered = False
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = teacher_brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage, active = (
                        self._direct_policy_probe_rollout_redirect_action(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    teacher_decision = teacher_brain.act_inference(
                        brain_observation,
                        bus=None,
                        sample=False,
                        policy_mode="normal",
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = int(teacher_decision.action_idx)
                    if teacher_action_stage is None and active:
                        teacher_action_stage = "frontier_live"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if active:
                        triggered = True
                        episode_action_stages.append(str(teacher_action_stage))
                        if teacher_option_stage is not None:
                            episode_option_stages.append(str(teacher_option_stage))
                        if str(teacher_action_stage).startswith("rollout_"):
                            teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                            teacher_policy[int(teacher_action_idx)] = 1.0
                            teacher_total_logits = np.zeros(len(ACTIONS), dtype=float)
                            teacher_total_logits[int(teacher_action_idx)] = 6.0
                            teacher_action_center_policy = teacher_policy
                            teacher_action_center_logits = teacher_total_logits
                            teacher_action_intent_idx = int(teacher_action_idx)
                        else:
                            teacher_policy = teacher_decision.policy
                            teacher_total_logits = teacher_decision.total_logits
                            teacher_action_center_policy = (
                                teacher_decision.action_center_policy
                            )
                            teacher_action_center_logits = (
                                teacher_decision.action_center_logits
                            )
                            teacher_action_intent_idx = int(
                                teacher_decision.action_intent_idx
                            )
                        dataset.add_sample(
                            episode=episode_index,
                            step=step,
                            observation=brain_observation,
                            teacher_policy=teacher_policy,
                            teacher_total_logits=teacher_total_logits,
                            teacher_action_center_policy=teacher_action_center_policy,
                            teacher_action_center_logits=teacher_action_center_logits,
                            teacher_action_intent_idx=teacher_action_intent_idx,
                            teacher_module_policies={},
                        )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                if triggered:
                    dataset.teacher_metadata.setdefault("action_stages", [])
                    dataset.teacher_metadata["action_stages"].extend(
                        episode_action_stages
                    )
                    dataset.teacher_metadata.setdefault("option_stages", [])
                    dataset.teacher_metadata["option_stages"].extend(
                        episode_option_stages
                    )
        teacher_brain.reset_hidden_states()
        self.brain.reset_hidden_states()
        return dataset

    def collect_direct_policy_probe_replayable_teacher_distillation_rollout(
        self,
        *,
        episodes: int = 1,
        episode_start: int = 0,
    ) -> DistillationDataset:
        teacher_scenarios = ("continuous_survival_canonical",)
        dataset = DistillationDataset(
            teacher_metadata={
                "source": "stateful_direct_policy_post_rest_probe_replayable_teacher",
                "episodes": int(max(0, episodes)),
                "teacher_scenarios": list(teacher_scenarios),
            }
        )
        total_episodes = max(0, int(episodes))
        for episode_offset in range(total_episodes):
            for scenario_offset, scenario_name in enumerate(teacher_scenarios):
                scenario = get_scenario(scenario_name)
                episode_index = int(
                    episode_start
                    + (episode_offset * len(teacher_scenarios))
                    + scenario_offset
                )
                episode_seed = self.seed + 997 * (episode_index + 1)
                episode_action_stages: list[str] = []
                episode_option_stages: list[str] = []
                if self.world.map_template_name != scenario.map_template:
                    self.world.configure_map_template(scenario.map_template)
                self.world.reset(seed=episode_seed)
                scenario.setup(self.world)
                observation = self.world.observe()
                self.brain.reset_hidden_states()
                self._reset_direct_policy_probe_replayable_teacher_state()
                triggered = False
                for step in range(int(scenario.max_steps)):
                    observation_adapters = adapt_observation_contracts(
                        observation,
                        tick=step,
                    )
                    brain_observation = observation_vectors_from_adapters(
                        observation_adapters
                    )
                    current_state = self.world.state_dict()
                    food_memory = current_state.get("food_memory")
                    if (
                        isinstance(food_memory, dict)
                        and food_memory.get("target") is not None
                        and isinstance(food_memory.get("target"), tuple)
                    ):
                        current_state["food_memory"] = dict(food_memory)
                        current_state["food_memory"]["target"] = list(
                            food_memory["target"]
                        )
                    food_direction_action = self.brain._food_direction_bias_action(
                        brain_observation
                    )
                    teacher_action_idx, teacher_action_stage, active = (
                        self._direct_policy_probe_replayable_teacher_action(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                        )
                    )
                    teacher_decision = self.brain.act(
                        brain_observation,
                        bus=None,
                        sample=False,
                        training=False,
                    )
                    if teacher_action_idx < 0:
                        teacher_action_idx = int(teacher_decision.action_idx)
                    if teacher_action_stage is None and active:
                        teacher_action_stage = "probe_live"
                    _, teacher_option_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    if active:
                        triggered = True
                        episode_action_stages.append(str(teacher_action_stage))
                        if teacher_option_stage is not None:
                            episode_option_stages.append(str(teacher_option_stage))
                        teacher_policy = np.zeros(len(ACTIONS), dtype=float)
                        teacher_policy[int(teacher_action_idx)] = 1.0
                        teacher_logits = np.zeros(len(ACTIONS), dtype=float)
                        teacher_logits[int(teacher_action_idx)] = 6.0
                        dataset.add_sample(
                            episode=episode_index,
                            step=step,
                            observation=brain_observation,
                            teacher_policy=teacher_policy,
                            teacher_total_logits=teacher_logits,
                            teacher_action_center_policy=teacher_policy,
                            teacher_action_center_logits=teacher_logits,
                            teacher_action_intent_idx=int(teacher_action_idx),
                            teacher_module_policies={},
                        )
                    next_observation, _, done, _ = self.world.step(
                        int(teacher_action_idx)
                    )
                    observation = next_observation
                    if done:
                        break
                if triggered:
                    dataset.teacher_metadata.setdefault("action_stages", [])
                    dataset.teacher_metadata["action_stages"].extend(
                        episode_action_stages
                    )
                    dataset.teacher_metadata.setdefault("option_stages", [])
                    dataset.teacher_metadata["option_stages"].extend(
                        episode_option_stages
                    )
                    dataset.teacher_metadata.setdefault(
                        "first_up_redirects",
                        0,
                    )
                    dataset.teacher_metadata["first_up_redirects"] += int(
                        self._direct_policy_probe_replayable_teacher_state.get(
                            "first_up_redirects",
                            0,
                        )
                    )
                    dataset.teacher_metadata.setdefault(
                        "down_redirects",
                        0,
                    )
                    dataset.teacher_metadata["down_redirects"] += int(
                        self._direct_policy_probe_replayable_teacher_state.get(
                            "down_redirects",
                            0,
                        )
                    )
        self.brain.reset_hidden_states()
        return dataset

    def collect_teacher_rollout(
        self,
        teacher_brain: SpiderBrain,
        *,
        episodes: int,
        teacher_metadata: Dict[str, object] | None = None,
        episode_start: int = 0,
        policy_mode: str = "normal",
        sample: bool = False,
        scenario_name: str | None = None,
    ) -> DistillationDataset:
        """
        Collect rollouts from a teacher policy and assemble them into a DistillationDataset for policy distillation.
        
        This runs the teacher brain in inference mode across a sequence of episodes (optionally using a named scenario), records per-step observation vectors and teacher outputs, and returns a dataset suitable for supervised distillation of student policies.
        
        Parameters:
        	teacher_brain (SpiderBrain): Teacher policy/brain providing `act_inference` and `VALENCE_ORDER`.
        	episodes (int): Number of episodes to collect.
        	teacher_metadata (Dict[str, object] | None): Optional metadata stored on the returned dataset.
        	episode_start (int): Base index added to sequential episode indices written into samples.
        	policy_mode (str): Decision routing mode to use; must be `"normal"` or `"reflex_only"`.
        	sample (bool): If true, the teacher will sample stochastic actions when available.
        	scenario_name (str | None): Optional scenario identifier to load per-episode environment setup.
        
        Returns:
        	DistillationDataset: A dataset populated with per-step samples containing episode and step indices, vectorized observations, teacher action selections, logits, per-module policy outputs, and optional valence scores.
        
        Raises:
        	ValueError: If `policy_mode` is not `"normal"` or `"reflex_only"`.
        """
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "Invalid policy_mode. Use 'normal' or 'reflex_only'."
            )
        dataset = DistillationDataset(teacher_metadata=teacher_metadata)
        total_episodes = max(0, int(episodes))
        scenario = get_scenario(scenario_name) if scenario_name is not None else None
        for episode_offset in range(total_episodes):
            episode_index = int(episode_start + episode_offset)
            episode_seed = self.seed + 997 * (episode_index + 1)
            episode_map_template = (
                scenario.map_template if scenario is not None else self.default_map_template
            )
            if self.world.map_template_name != episode_map_template:
                self.world.configure_map_template(episode_map_template)
            observation = self.world.reset(seed=episode_seed)
            teacher_brain.reset_hidden_states()
            episode_max_steps = self.max_steps
            if scenario is not None:
                scenario.setup(self.world)
                observation = self.world.observe()
                episode_max_steps = scenario.max_steps
            for step in range(episode_max_steps):
                observation_adapters = adapt_observation_contracts(
                    observation,
                    tick=step,
                )
                brain_observation = observation_vectors_from_adapters(
                    observation_adapters
                )
                decision = teacher_brain.act_inference(
                    brain_observation,
                    bus=None,
                    sample=sample,
                    policy_mode=policy_mode,
                )
                arbitration = decision.arbitration_decision
                dataset.add_sample(
                    episode=episode_index,
                    step=step,
                    observation=brain_observation,
                    teacher_policy=decision.policy,
                    teacher_total_logits=decision.total_logits,
                    teacher_action_center_policy=decision.action_center_policy,
                    teacher_action_center_logits=decision.action_center_logits,
                    teacher_action_intent_idx=decision.action_intent_idx,
                    teacher_valence_probs=(
                        [
                            float(arbitration.valence_scores.get(name, 0.0))
                            for name in teacher_brain.VALENCE_ORDER
                        ]
                        if arbitration is not None
                        else None
                    ),
                    teacher_valence_logits=(
                        [
                            float(arbitration.valence_logits.get(name, 0.0))
                            for name in teacher_brain.VALENCE_ORDER
                        ]
                        if arbitration is not None
                        else None
                    ),
                    teacher_module_policies={
                        result.name: result.probs
                        for result in decision.module_results
                        if result.active
                    },
                )
                next_observation, _, done, _ = self.world.step(decision.action_idx)
                observation = next_observation
                if done:
                    break
        return dataset

    @staticmethod
    def _episode_stats_behavior_metrics(stats: EpisodeStats) -> Dict[str, object]:
        """
        Produce a flat mapping of behavior-metric fields derived from an EpisodeStats object.
        
        The returned dictionary contains named metric entries (floats, ints, and dicts) such as module dominance/share, routing/router statistics, owner alignment/rank/suppression, motor slip and orientation metrics, per-proposal-source contribution shares (keys prefixed with `module_contribution_`), and per-terrain slip rates (keys prefixed with `terrain_slip_rate_`). These fields are intended for inclusion in scenario scorecards and downstream reporting.
        
        Returns:
            Dict[str, object]: A dictionary mapping metric names to primitive values suitable for reporting.
        """
        metrics: Dict[str, object] = {
            "dominant_module": stats.dominant_module,
            "dominant_module_share": float(stats.dominant_module_share),
            "effective_module_count": float(stats.effective_module_count),
            "gate_entropy": float(stats.gate_entropy),
            "dominance_rate": float(stats.dominance_rate),
            "effective_proposer_count": float(stats.effective_proposer_count),
            "module_agreement_rate": float(stats.module_agreement_rate),
            "module_disagreement_rate": float(stats.module_disagreement_rate),
            "route_active_modules": dict(stats.route_active_modules),
            "router_health": dict(stats.router_health),
            "owner_alignment": float(stats.owner_alignment),
            "owner_rank": int(stats.owner_rank),
            "owner_suppressed_rate": float(stats.owner_suppressed_rate),
            "motor_slip_rate": float(stats.motor_slip_rate),
            "mean_orientation_alignment": float(stats.mean_orientation_alignment),
            "mean_terrain_difficulty": float(stats.mean_terrain_difficulty),
        }
        for module_name in PROPOSAL_SOURCE_NAMES:
            metrics[f"module_contribution_{module_name}"] = float(
                stats.module_contribution_share.get(module_name, 0.0)
            )
        for terrain, rate in sorted(stats.terrain_slip_rates.items()):
            metrics[f"terrain_slip_rate_{terrain}"] = float(rate)
        return metrics

    @staticmethod
    def _attach_motor_execution_info(decision: BrainStep, info: Dict[str, object]) -> None:
        """Attach authoritative post-world motor execution diagnostics to a BrainStep-like object."""
        slip_info = info.get("motor_slip", {})
        if not isinstance(slip_info, dict):
            slip_info = {}
        components = info.get("motor_execution_components", {})
        if not isinstance(components, dict):
            components = slip_info.get("components", {})
        if not isinstance(components, dict):
            components = {}

        slip_occurred = bool(
            slip_info.get("occurred", info.get("motor_noise_applied", False))
        )
        decision.execution_slip_occurred = slip_occurred
        decision.motor_slip_occurred = slip_occurred
        decision.motor_noise_applied = slip_occurred
        decision.slip_reason = str(
            info.get(
                "slip_reason",
                info.get("motor_slip_reason", slip_info.get("reason", "none")),
            )
        )
        decision.orientation_alignment = float(
            components.get(
                "orientation_alignment",
                getattr(decision, "orientation_alignment", 1.0),
            )
        )
        decision.terrain_difficulty = float(
            components.get(
                "terrain_difficulty",
                getattr(decision, "terrain_difficulty", 0.0),
            )
        )
        decision.execution_difficulty = float(
            info.get(
                "motor_execution_difficulty",
                slip_info.get(
                    "execution_difficulty",
                    getattr(decision, "execution_difficulty", 0.0),
                ),
            )
        )

    @staticmethod
    def _motor_stage_summary(history: List[EpisodeStats]) -> Dict[str, object]:
        """Aggregate motor execution observability metrics for summary output."""
        if not history:
            return {
                "episodes": 0,
                "mean_motor_slip_rate": 0.0,
                "mean_orientation_alignment": 0.0,
                "mean_terrain_difficulty": 0.0,
                "mean_terrain_slip_rates": {},
            }
        terrain_names = sorted(
            {
                terrain
                for stats in history
                for terrain in stats.terrain_slip_rates
            }
        )
        return {
            "episodes": len(history),
            "mean_motor_slip_rate": float(
                sum(stats.motor_slip_rate for stats in history) / len(history)
            ),
            "mean_orientation_alignment": float(
                sum(stats.mean_orientation_alignment for stats in history) / len(history)
            ),
            "mean_terrain_difficulty": float(
                sum(stats.mean_terrain_difficulty for stats in history) / len(history)
            ),
            "mean_terrain_slip_rates": {
                terrain: float(
                    sum(
                        stats.terrain_slip_rates.get(terrain, 0.0)
                        for stats in history
                    )
                    / len(history)
                )
                for terrain in terrain_names
            },
        }

    @staticmethod
    def _direct_policy_event_features(
        *,
        meta: Dict[str, object],
        strength: float = 1.0,
    ) -> list[float]:
        diagnostic = meta.get("diagnostic", {})
        if not isinstance(diagnostic, dict):
            diagnostic = {}
        threat = max(
            float(meta.get("visual_predator_threat", 0.0) or 0.0),
            float(meta.get("olfactory_predator_threat", 0.0) or 0.0),
            float(meta.get("predator_visible", 0.0) or 0.0),
        )
        return [
            float(strength),
            float(diagnostic.get("hunger", 0.0) or 0.0),
            float(meta.get("sleep_debt", 0.0) or 0.0),
            1.0 if bool(meta.get("on_shelter", False)) else 0.0,
            float(threat),
        ]

    def _record_direct_policy_events(
        self,
        *,
        step: int,
        decision: BrainStep,
        observation: Dict[str, object],
        next_observation: Dict[str, object],
        current_state: Dict[str, object],
        next_state: Dict[str, object],
        info: Dict[str, object],
    ) -> None:
        self.brain.set_direct_policy_event_clock(step + 1)
        if not self.brain.config.direct_policy_event_attention:
            return
        current_meta = observation["meta"]
        next_meta = next_observation["meta"]
        current_on_shelter = bool(current_meta.get("on_shelter", False))
        next_on_shelter = bool(next_meta.get("on_shelter", False))
        current_sleep_phase = str(current_meta.get("sleep_phase", ""))
        next_sleep_phase = str(next_meta.get("sleep_phase", ""))
        if bool(info.get("ate", False)):
            self.brain.record_direct_policy_event(
                "FOOD_EATEN",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta, strength=1.0),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if current_on_shelter and not next_on_shelter:
            self.brain.record_direct_policy_event(
                "SHELTER_EXIT",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if not current_on_shelter and next_on_shelter:
            self.brain.record_direct_policy_event(
                "SHELTER_RETURN",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if bool(info.get("slept", False)) and current_sleep_phase == "AWAKE":
            self.brain.record_direct_policy_event(
                "REST_STARTED",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if current_sleep_phase != "DEEP_SLEEP" and next_sleep_phase == "DEEP_SLEEP":
            self.brain.record_direct_policy_event(
                "DEEP_SLEEP_REACHED",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            bool(next_meta.get("day", False))
            and next_on_shelter
            and int(next_state.get("sleep_events", 0) or 0) > 0
            and next_sleep_phase == "AWAKE"
        ):
            self.brain.record_direct_policy_event(
                "RECOVERY_COMPLETED",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            bool(current_meta.get("day", False))
            and current_on_shelter
            and int(current_state.get("sleep_events", 0) or 0) > 0
            and current_sleep_phase == "AWAKE"
            and ACTIONS[decision.action_idx] != "STAY"
        ):
            self.brain.record_direct_policy_event(
                "POST_REST_RELEASE_ATTEMPT",
                features=np.asarray(
                    self._direct_policy_event_features(meta=current_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            str(info.get("intended_action", "")).startswith("MOVE_")
            and current_state.get("x") == next_state.get("x")
            and current_state.get("y") == next_state.get("y")
        ):
            self.brain.record_direct_policy_event(
                "BLOCKED_MOVE",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            bool(info.get("predator_contact", False))
            or bool(next_meta.get("predator_visible", False))
            or float(next_state.get("recent_contact", 0.0) or 0.0) > 0.0
            or float(next_state.get("recent_pain", 0.0) or 0.0) > 0.0
        ):
            self.brain.record_direct_policy_event(
                "ACUTE_PREDATOR_THREAT",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta, strength=1.2),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            not bool(next_meta.get("predator_visible", False))
            and float(next_meta.get("predator_memory_age", 1.0) or 1.0) < 1.0
        ):
            self.brain.record_direct_policy_event(
                "RESIDUAL_PREDATOR_MEMORY",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta, strength=0.7),
                    dtype=float,
                ),
                tick=step + 1,
            )

    def run_episode(
        self,
        episode_index: int,
        *,
        training: bool,
        sample: bool,
        render: bool = False,
        capture_trace: bool = False,
        scenario_name: str | None = None,
        debug_trace: bool = False,
        policy_mode: str = "normal",
    ) -> tuple[EpisodeStats, List[Dict[str, object]]]:
        """
        Execute one simulation episode and collect aggregated episode metrics and an optional per-tick trace.
        
        Parameters:
            episode_index (int): Index used to derive the episode RNG seed and to tag trace items.
            training (bool): If True, enables learning updates during the episode; only allowed with policy_mode="normal".
            sample (bool): If True, allow stochastic action sampling; otherwise use deterministic selection.
            render (bool): If True, render the world each tick.
            capture_trace (bool): If True, collect a list of per-tick dictionaries describing state, actions, rewards, and messages.
            scenario_name (str | None): Optional scenario identifier that may override the map template and max steps when provided.
            debug_trace (bool): If True and capture_trace is True, include expanded diagnostic fields in each trace item.
            policy_mode (str): Inference mode for the brain; valid values are "normal" or "reflex_only". "reflex_only" requires a modular brain with reflexes enabled.
        
        Returns:
            tuple[EpisodeStats, List[Dict[str, object]]]: EpisodeStats with aggregated episode metrics, and a list of per-tick trace dictionaries (empty when capture_trace is False).
        
        Raises:
            ValueError: If policy_mode is invalid or incompatible with the brain configuration, or if training=True is used with a non-"normal" policy_mode.
        """
        # Validate policy_mode and brain compatibility before any state mutations
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "Invalid policy_mode. Use 'normal' or 'reflex_only'."
            )
        if policy_mode == "reflex_only" and not self.brain.config.is_modular:
            raise ValueError(
                "policy_mode='reflex_only' requires the modular architecture."
            )
        if policy_mode == "reflex_only" and not self.brain.config.enable_reflexes:
            raise ValueError(
                "policy_mode='reflex_only' requires reflexes to be enabled."
            )
        if training and policy_mode != "normal":
            raise ValueError(
                "run_episode() only supports training=True with policy_mode='normal'."
            )
        episode_seed = self.seed + 997 * (episode_index + 1)
        normalized_scenario = str(scenario_name) if scenario_name is not None else None
        scenario = get_scenario(normalized_scenario) if normalized_scenario is not None else None
        episode_map_template = scenario.map_template if scenario is not None else self.default_map_template
        if self.world.map_template_name != episode_map_template:
            self.world.configure_map_template(episode_map_template)
        observation = self.world.reset(seed=episode_seed)
        self.brain.reset_hidden_states()
        episode_max_steps = self.max_steps
        if scenario is not None:
            scenario.setup(self.world)
            observation = self.world.observe()
            episode_max_steps = scenario.max_steps

        trace: List[Dict[str, object]] = []
        total_reward = 0.0
        metrics = EpisodeMetricAccumulator(
            reward_component_names=REWARD_COMPONENT_NAMES,
            predator_states=PREDATOR_STATES,
        )
        self._reset_direct_policy_handoff_teacher_state()

        for step in range(episode_max_steps):
            self.bus.set_tick(step)
            self.brain.set_direct_policy_event_clock(step)
            current_state_snapshot = self.world.state_dict()
            self.bus.publish(
                sender="environment",
                topic="observation",
                payload={
                    "state": current_state_snapshot,
                    "meta": observation["meta"],
                },
            )
            observation_adapters = adapt_observation_contracts(
                observation,
                tick=step,
            )
            brain_observation = observation_vectors_from_adapters(
                observation_adapters
            )
            if self.brain.config.is_b_series:
                brain_observation["meta"] = observation["meta"]

            if training:
                decision = self.brain.act_train(
                    brain_observation,
                    self.bus,
                    sample=sample,
                    policy_mode=policy_mode,
                )
            elif sample:
                decision = self.brain.act_exploration(
                    brain_observation,
                    self.bus,
                    policy_mode=policy_mode,
                )
            else:
                decision = self.brain.act_inference(
                    brain_observation,
                    self.bus,
                    sample=False,
                    policy_mode=policy_mode,
                )
            if self.brain.config.direct_policy_phase_head:
                phase_target = derive_phase_target(
                    state=self.world.state_dict(),
                    observation_meta=observation["meta"],
                )
                decision.phase_target = phase_target
                decision.phase_target_idx = int(PHASE_TO_INDEX[phase_target])
                self.bus.publish(
                    sender=self.brain.TRUE_MONOLITHIC_POLICY_NAME,
                    topic="phase.prediction",
                    payload={
                        "phase_target": phase_target,
                        "phase_prediction": decision.phase_prediction,
                        "phase_prediction_confidence": round(
                            float(decision.phase_prediction_confidence),
                            6,
                        ),
                        "selected_action": ACTIONS[decision.action_idx],
                    },
                )
            decision.scenario_name = normalized_scenario
            if self.brain.config.direct_policy_affordance_head:
                (
                    decision.affordance_blocked_targets,
                    decision.affordance_role_targets,
                ) = self._direct_policy_affordance_targets(
                    current_state=current_state_snapshot,
                )
            if self.brain.config.direct_policy_geometry_head:
                decision.geometry_targets = self._direct_policy_geometry_targets(
                    current_state=current_state_snapshot,
                )
            if self.brain.config.direct_policy_shelter_column_head:
                decision.shelter_column_targets = (
                    self._direct_policy_shelter_column_targets(
                        current_state=current_state_snapshot,
                    )
                )
            if self.brain.config.direct_policy_shelter_position_head:
                decision.shelter_position_targets = (
                    self._direct_policy_shelter_position_targets(
                        current_state=current_state_snapshot,
                    )
                )
            if self.brain.config.direct_policy_transition_prediction_head:
                decision.transition_prediction_targets = (
                    self._direct_policy_transition_prediction_targets(
                        observation_meta=observation["meta"],
                    )
                )
            if self.brain.config.direct_policy_transition_rollout_prediction_head:
                decision.transition_rollout_prediction_targets = (
                    self._direct_policy_transition_rollout_prediction_targets(
                        observation_meta=observation["meta"],
                    )
                )
            if self.brain.config.direct_policy_handoff_teacher:
                food_direction_action = self.brain._food_direction_bias_action(
                    brain_observation
                )
                teacher_action_target_idx, teacher_action_target_stage = (
                    self._direct_policy_handoff_teacher_target(
                        current_state=current_state_snapshot,
                        food_direction_action=food_direction_action,
                        tick=step,
                    )
                )
                decision.teacher_action_target_idx = int(teacher_action_target_idx)
                decision.teacher_action_target_stage = teacher_action_target_stage
                decision.teacher_action_target_name = (
                    None
                    if teacher_action_target_idx < 0
                    else ACTIONS[int(teacher_action_target_idx)]
                )
                if self.brain.config.direct_policy_handoff_option_teacher:
                    teacher_option_target_idx, teacher_option_target_stage = (
                        self._direct_policy_handoff_option_teacher_target(
                            current_state=current_state_snapshot,
                            food_direction_action=food_direction_action,
                            tick=step,
                        )
                    )
                    decision.teacher_option_target_idx = int(
                        teacher_option_target_idx
                    )
                    decision.teacher_option_target_stage = (
                        teacher_option_target_stage
                    )
                    decision.teacher_option_target_name = (
                        None
                        if teacher_option_target_idx < 0
                        else OPTION_NAMES[int(teacher_option_target_idx)]
                    )
            metrics.record_decision(decision)
            predator_state_before = self.world.lizard.mode
            next_observation, reward, done, info = self.world.step(decision.action_idx)
            next_state_snapshot = self.world.state_dict()
            next_observation_adapters = adapt_observation_contracts(
                next_observation,
                tick=step + 1,
            )
            next_brain_observation = observation_vectors_from_adapters(
                next_observation_adapters
            )
            if self.brain.config.is_b_series:
                next_brain_observation["meta"] = next_observation["meta"]
            self._attach_motor_execution_info(decision, info)
            self._record_direct_policy_events(
                step=step,
                decision=decision,
                observation=observation,
                next_observation=next_observation,
                current_state=current_state_snapshot,
                next_state=next_state_snapshot,
                info=info,
            )
            learn_stats: Dict[str, object] = {}
            if training:
                learn_stats = self.brain.learn(
                    decision,
                    reward,
                    next_brain_observation,
                    done,
                )
                if decision.arbitration_decision is not None:
                    decision.arbitration_decision = replace(
                        decision.arbitration_decision,
                        route_active_modules=list(
                            learn_stats.get("route_active_modules", [])
                        ),
                        route_credit_weights=dict(
                            learn_stats.get("route_credit_weights", {})
                        ),
                    )
                metrics.record_learning(learn_stats)
                self.bus.publish(
                    sender="learning",
                    topic="td_update",
                    payload=dict(learn_stats),
                )

            self.bus.publish(
                sender="environment",
                topic="transition",
                payload={
                    "selected_action": ACTIONS[decision.action_idx],
                    "executed_action": info.get("executed_action", ACTIONS[decision.action_idx]),
                    "motor_noise_applied": bool(info.get("motor_noise_applied", False)),
                    "reward": round(float(reward), 6),
                    "reward_components": info["reward_components"],
                    "done": bool(done),
                    "info": info,
                },
            )

            metrics.record_transition(
                step=step,
                observation_meta=observation["meta"],
                next_meta=next_observation["meta"],
                info=info,
                state=self.world.state,
                predator_state_before=predator_state_before,
                predator_state=self.world.lizard.mode,
            )

            if capture_trace:
                state_snapshot = self.world.state_dict()
                threat_snapshot = self.brain._bound_observation(
                    "threat_center",
                    next_brain_observation,
                )
                state_snapshot["predator_smell_strength"] = float(
                    threat_snapshot["predator_smell_strength"]
                )
                item: Dict[str, object] = {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "tick": step,
                    "training": training,
                    "policy_mode": policy_mode,
                    "scenario": normalized_scenario,
                    "action": info.get("action"),
                    "intended_action": info.get("intended_action"),
                    "executed_action": info.get("executed_action"),
                    "motor_noise_applied": bool(info.get("motor_noise_applied", False)),
                    "slip_reason": str(info.get("slip_reason", "none")),
                    "ate": bool(info.get("ate")),
                    "slept": bool(info.get("slept")),
                    "predator_contact": bool(info.get("predator_contact")),
                    "predator_escape": bool(info.get("predator_escape")),
                    "state": state_snapshot,
                    "reward": float(reward),
                    "reward_components": info["reward_components"],
                    "predator_transition": info.get("predator_transition"),
                    "distance_deltas": info.get("distance_deltas", {}),
                    "event_log": info["event_log"],
                    "done": bool(done),
                    "messages": self.bus.serialize_current_tick(),
                }
                if self.brain.config.is_b_series:
                    item["b_level"] = int(decision.b_level)
                    item["b_effective_level"] = decision.b_effective_level
                    item["b_mode"] = decision.b_mode
                    item["b_parent_level"] = decision.b_parent_level
                    item["b_transfer_source_checkpoint"] = (
                        decision.b_transfer_source_checkpoint
                    )
                    item["b_transfer_coverage"] = decision.b_transfer_coverage
                    item["b_current_threat_pressure"] = (
                        decision.b_current_threat_pressure
                    )
                    item["b_temporal_threat_pressure"] = (
                        decision.b_temporal_threat_pressure
                    )
                    item["b_predator_memory_pressure"] = (
                        decision.b_predator_memory_pressure
                    )
                    item["b_predator_trace_pressure"] = (
                        decision.b_predator_trace_pressure
                    )
                    item["b3_contact_cooldown"] = decision.b3_contact_cooldown
                    item["b3_post_food_cooldown"] = (
                        decision.b3_post_food_cooldown
                    )
                    item["b3_hunger_drop"] = decision.b3_hunger_drop
                    item["b3_controller_profile"] = decision.b3_controller_profile
                    item["b4_controller_profile"] = decision.b4_controller_profile
                    item["b4_recovery_pressure"] = decision.b4_recovery_pressure
                    item["b4_sleep_hold"] = decision.b4_sleep_hold
                    item["b4_exit_blocked"] = decision.b4_exit_blocked
                    item["b4_hunger_release"] = decision.b4_hunger_release
                    item["b4_genetic_generation"] = decision.b4_genetic_generation
                    item["b4_genetic_candidate"] = decision.b4_genetic_candidate
                    item["b5_controller_profile"] = decision.b5_controller_profile
                    item["b5_hunger_urgency"] = decision.b5_hunger_urgency
                    item["b5_sleep_pressure"] = decision.b5_sleep_pressure
                    item["b5_recovery_debt"] = decision.b5_recovery_debt
                    item["b5_threat_gate"] = decision.b5_threat_gate
                    item["b5_sleep_bout_lock"] = decision.b5_sleep_bout_lock
                    item["b5_forage_commitment_lock"] = (
                        decision.b5_forage_commitment_lock
                    )
                    item["b5_homeostatic_decision"] = (
                        decision.b5_homeostatic_decision
                    )
                    item["b5_genetic_generation"] = decision.b5_genetic_generation
                    item["b5_genetic_candidate"] = decision.b5_genetic_candidate
                    item["b6_controller_family"] = decision.b6_controller_family
                    item["b6_controller_profile"] = decision.b6_controller_profile
                    item["b6_risk_pressure"] = decision.b6_risk_pressure
                    item["b6_threat_priority"] = decision.b6_threat_priority
                    item["b6_forage_suppressed"] = decision.b6_forage_suppressed
                    item["b6_corridor_commitment"] = (
                        decision.b6_corridor_commitment
                    )
                    item["b6_corridor_progress_memory"] = (
                        decision.b6_corridor_progress_memory
                    )
                    item["b6_recurrent_state"] = decision.b6_recurrent_state
                    item["b6_return_lock"] = decision.b6_return_lock
                    item["b6_decision"] = decision.b6_decision
                    item["b6_genetic_generation"] = decision.b6_genetic_generation
                    item["b6_genetic_candidate"] = decision.b6_genetic_candidate
                    item["b7_controller_profile"] = decision.b7_controller_profile
                    item["b7_affordance_state"] = decision.b7_affordance_state
                    item["b7_energy_budget"] = decision.b7_energy_budget
                    item["b7_budget_margin"] = decision.b7_budget_margin
                    item["b7_food_steps_estimate"] = (
                        decision.b7_food_steps_estimate
                    )
                    item["b7_return_steps_estimate"] = (
                        decision.b7_return_steps_estimate
                    )
                    item["b7_corridor_viability"] = (
                        decision.b7_corridor_viability
                    )
                    item["b7_abort_return"] = decision.b7_abort_return
                    item["b7_commitment_lock"] = decision.b7_commitment_lock
                    item["b7_decision"] = decision.b7_decision
                    item["b7_genetic_generation"] = decision.b7_genetic_generation
                    item["b7_genetic_candidate"] = decision.b7_genetic_candidate
                    item["b8_controller_profile"] = decision.b8_controller_profile
                    item["b8_spatial_map_state"] = decision.b8_spatial_map_state
                    item["b8_local_affordance_score"] = (
                        decision.b8_local_affordance_score
                    )
                    item["b8_return_vector_strength"] = (
                        decision.b8_return_vector_strength
                    )
                    item["b8_corridor_dead_end_risk"] = (
                        decision.b8_corridor_dead_end_risk
                    )
                    item["b8_abort_executed"] = decision.b8_abort_executed
                    item["b8_place_memory"] = decision.b8_place_memory
                    item["b8_decision"] = decision.b8_decision
                    item["b8_genetic_generation"] = decision.b8_genetic_generation
                    item["b8_genetic_candidate"] = decision.b8_genetic_candidate
                    item["b9_controller_profile"] = decision.b9_controller_profile
                    item["b9_route_state"] = decision.b9_route_state
                    item["b9_route_confidence"] = decision.b9_route_confidence
                    item["b9_waypoint_lock"] = decision.b9_waypoint_lock
                    item["b9_path_integrator"] = decision.b9_path_integrator
                    item["b9_replan_signal"] = decision.b9_replan_signal
                    item["b9_decision"] = decision.b9_decision
                    item["b9_genetic_generation"] = decision.b9_genetic_generation
                    item["b9_genetic_candidate"] = decision.b9_genetic_candidate
                    item["b10_controller_profile"] = decision.b10_controller_profile
                    item["b10_replay_state"] = decision.b10_replay_state
                    item["b10_prospective_value"] = decision.b10_prospective_value
                    item["b10_rollout_depth"] = decision.b10_rollout_depth
                    item["b10_replay_memory"] = decision.b10_replay_memory
                    item["b10_plan_commitment"] = decision.b10_plan_commitment
                    item["b10_abort_signal"] = decision.b10_abort_signal
                    item["b10_decision"] = decision.b10_decision
                    item["b10_genetic_generation"] = decision.b10_genetic_generation
                    item["b10_genetic_candidate"] = decision.b10_genetic_candidate
                    item["b11_controller_profile"] = decision.b11_controller_profile
                    item["b11_confidence_state"] = decision.b11_confidence_state
                    item["b11_plan_confidence"] = decision.b11_plan_confidence
                    item["b11_uncertainty"] = decision.b11_uncertainty
                    item["b11_neuromod_signal"] = decision.b11_neuromod_signal
                    item["b11_confidence_lock"] = decision.b11_confidence_lock
                    item["b11_decision"] = decision.b11_decision
                    item["b11_genetic_generation"] = decision.b11_genetic_generation
                    item["b11_genetic_candidate"] = decision.b11_genetic_candidate
                    item["b12_controller_profile"] = decision.b12_controller_profile
                    item["b12_attention_state"] = decision.b12_attention_state
                    item["b12_prediction_error"] = decision.b12_prediction_error
                    item["b12_attention_gain"] = decision.b12_attention_gain
                    item["b12_expected_progress"] = decision.b12_expected_progress
                    item["b12_search_lock"] = decision.b12_search_lock
                    item["b12_decision"] = decision.b12_decision
                    item["b12_genetic_generation"] = decision.b12_genetic_generation
                    item["b12_genetic_candidate"] = decision.b12_genetic_candidate
                    item["b13_controller_profile"] = decision.b13_controller_profile
                    item["b13_search_state"] = decision.b13_search_state
                    item["b13_local_route_score"] = decision.b13_local_route_score
                    item["b13_affordance_samples"] = decision.b13_affordance_samples
                    item["b13_search_memory"] = decision.b13_search_memory
                    item["b13_dead_end_score"] = decision.b13_dead_end_score
                    item["b13_search_lock"] = decision.b13_search_lock
                    item["b13_decision"] = decision.b13_decision
                    item["b13_genetic_generation"] = decision.b13_genetic_generation
                    item["b13_genetic_candidate"] = decision.b13_genetic_candidate
                    item["b14_controller_profile"] = decision.b14_controller_profile
                    item["b14_uncertainty_state"] = decision.b14_uncertainty_state
                    item["b14_affordance_confidence"] = decision.b14_affordance_confidence
                    item["b14_uncertainty"] = decision.b14_uncertainty
                    item["b14_risk_adjusted_score"] = decision.b14_risk_adjusted_score
                    item["b14_commitment_lock"] = decision.b14_commitment_lock
                    item["b14_decision"] = decision.b14_decision
                    item["b14_genetic_generation"] = decision.b14_genetic_generation
                    item["b14_genetic_candidate"] = decision.b14_genetic_candidate
                    item["b15_controller_profile"] = decision.b15_controller_profile
                    item["b15_option_state"] = decision.b15_option_state
                    item["b15_option_value"] = decision.b15_option_value
                    item["b15_termination_pressure"] = decision.b15_termination_pressure
                    item["b15_persistence_score"] = decision.b15_persistence_score
                    item["b15_option_lock"] = decision.b15_option_lock
                    item["b15_decision"] = decision.b15_decision
                    item["b15_genetic_generation"] = decision.b15_genetic_generation
                    item["b15_genetic_candidate"] = decision.b15_genetic_candidate
                    item["b16_controller_profile"] = decision.b16_controller_profile
                    item["b16_ensemble_state"] = decision.b16_ensemble_state
                    item["b16_continue_vote"] = decision.b16_continue_vote
                    item["b16_return_vote"] = decision.b16_return_vote
                    item["b16_option_votes"] = decision.b16_option_votes
                    item["b16_consensus_score"] = decision.b16_consensus_score
                    item["b16_conflict_score"] = decision.b16_conflict_score
                    item["b16_ensemble_lock"] = decision.b16_ensemble_lock
                    item["b16_decision"] = decision.b16_decision
                    item["b16_genetic_generation"] = decision.b16_genetic_generation
                    item["b16_genetic_candidate"] = decision.b16_genetic_candidate
                    item["b17_controller_profile"] = decision.b17_controller_profile
                    item["b17_modulator_state"] = decision.b17_modulator_state
                    item["b17_arousal_signal"] = decision.b17_arousal_signal
                    item["b17_homeostatic_gain"] = decision.b17_homeostatic_gain
                    item["b17_option_gain"] = decision.b17_option_gain
                    item["b17_conflict_release"] = decision.b17_conflict_release
                    item["b17_modulation_lock"] = decision.b17_modulation_lock
                    item["b17_decision"] = decision.b17_decision
                    item["b17_genetic_generation"] = decision.b17_genetic_generation
                    item["b17_genetic_candidate"] = decision.b17_genetic_candidate
                    item["b18_controller_profile"] = decision.b18_controller_profile
                    item["b18_trace_state"] = decision.b18_trace_state
                    item["b18_eligibility_trace"] = decision.b18_eligibility_trace
                    item["b18_reward_prediction_proxy"] = (
                        decision.b18_reward_prediction_proxy
                    )
                    item["b18_stability_bias"] = decision.b18_stability_bias
                    item["b18_switch_pressure"] = decision.b18_switch_pressure
                    item["b18_trace_lock"] = decision.b18_trace_lock
                    item["b18_decision"] = decision.b18_decision
                    item["b18_genetic_generation"] = decision.b18_genetic_generation
                    item["b18_genetic_candidate"] = decision.b18_genetic_candidate
                    item["b19_controller_profile"] = decision.b19_controller_profile
                    item["b19_memory_state"] = decision.b19_memory_state
                    item["b19_episode_memory"] = decision.b19_episode_memory
                    item["b19_consolidation_score"] = decision.b19_consolidation_score
                    item["b19_stability_vote"] = decision.b19_stability_vote
                    item["b19_switch_suppression"] = decision.b19_switch_suppression
                    item["b19_memory_lock"] = decision.b19_memory_lock
                    item["b19_decision"] = decision.b19_decision
                    item["b19_genetic_generation"] = decision.b19_genetic_generation
                    item["b19_genetic_candidate"] = decision.b19_genetic_candidate
                    item["b20_controller_profile"] = decision.b20_controller_profile
                    item["b20_buffer_state"] = decision.b20_buffer_state
                    item["b20_working_buffer"] = decision.b20_working_buffer
                    item["b20_context_binding"] = decision.b20_context_binding
                    item["b20_gate_vote"] = decision.b20_gate_vote
                    item["b20_release_vote"] = decision.b20_release_vote
                    item["b20_buffer_lock"] = decision.b20_buffer_lock
                    item["b20_decision"] = decision.b20_decision
                    item["b20_genetic_generation"] = decision.b20_genetic_generation
                    item["b20_genetic_candidate"] = decision.b20_genetic_candidate
                    item["b21_controller_profile"] = decision.b21_controller_profile
                    item["b21_replay_state"] = decision.b21_replay_state
                    item["b21_sequence_memory"] = decision.b21_sequence_memory
                    item["b21_replay_score"] = decision.b21_replay_score
                    item["b21_route_commitment"] = decision.b21_route_commitment
                    item["b21_abort_prediction"] = decision.b21_abort_prediction
                    item["b21_replay_lock"] = decision.b21_replay_lock
                    item["b21_decision"] = decision.b21_decision
                    item["b21_genetic_generation"] = decision.b21_genetic_generation
                    item["b21_genetic_candidate"] = decision.b21_genetic_candidate
                    item["b22_controller_profile"] = decision.b22_controller_profile
                    item["b22_sim_state"] = decision.b22_sim_state
                    item["b22_prospective_sim"] = decision.b22_prospective_sim
                    item["b22_forward_model_score"] = (
                        decision.b22_forward_model_score
                    )
                    item["b22_viability_projection"] = (
                        decision.b22_viability_projection
                    )
                    item["b22_abort_projection"] = decision.b22_abort_projection
                    item["b22_sim_lock"] = decision.b22_sim_lock
                    item["b22_decision"] = decision.b22_decision
                    item["b22_genetic_generation"] = decision.b22_genetic_generation
                    item["b22_genetic_candidate"] = decision.b22_genetic_candidate
                    item["b23_controller_profile"] = decision.b23_controller_profile
                    item["b23_conflict_state"] = decision.b23_conflict_state
                    item["b23_prediction_error"] = decision.b23_prediction_error
                    item["b23_conflict_memory"] = decision.b23_conflict_memory
                    item["b23_stability_vote"] = decision.b23_stability_vote
                    item["b23_abort_bias"] = decision.b23_abort_bias
                    item["b23_monitor_lock"] = decision.b23_monitor_lock
                    item["b23_decision"] = decision.b23_decision
                    item["b23_genetic_generation"] = decision.b23_genetic_generation
                    item["b23_genetic_candidate"] = decision.b23_genetic_candidate
                    item["b24_controller_profile"] = decision.b24_controller_profile
                    item["b24_precision_state"] = decision.b24_precision_state
                    item["b24_precision_memory"] = decision.b24_precision_memory
                    item["b24_precision_vote"] = decision.b24_precision_vote
                    item["b24_uncertainty_pressure"] = (
                        decision.b24_uncertainty_pressure
                    )
                    item["b24_abort_precision"] = decision.b24_abort_precision
                    item["b24_precision_lock"] = decision.b24_precision_lock
                    item["b24_decision"] = decision.b24_decision
                    item["b24_genetic_generation"] = decision.b24_genetic_generation
                    item["b24_genetic_candidate"] = decision.b24_genetic_candidate
                    item["b25_controller_profile"] = decision.b25_controller_profile
                    item["b25_metacognitive_state"] = decision.b25_metacognitive_state
                    item["b25_confidence_memory"] = decision.b25_confidence_memory
                    item["b25_confidence_vote"] = decision.b25_confidence_vote
                    item["b25_doubt_pressure"] = decision.b25_doubt_pressure
                    item["b25_control_gain"] = decision.b25_control_gain
                    item["b25_meta_lock"] = decision.b25_meta_lock
                    item["b25_decision"] = decision.b25_decision
                    item["b25_genetic_generation"] = decision.b25_genetic_generation
                    item["b25_genetic_candidate"] = decision.b25_genetic_candidate
                    item["b26_controller_profile"] = decision.b26_controller_profile
                    item["b26_allostatic_state"] = decision.b26_allostatic_state
                    item["b26_prediction_error"] = decision.b26_prediction_error
                    item["b26_setpoint_pressure"] = decision.b26_setpoint_pressure
                    item["b26_control_vote"] = decision.b26_control_vote
                    item["b26_stability_lock"] = decision.b26_stability_lock
                    item["b26_decision"] = decision.b26_decision
                    item["b26_genetic_generation"] = decision.b26_genetic_generation
                    item["b26_genetic_candidate"] = decision.b26_genetic_candidate
                    item["b27_controller_profile"] = decision.b27_controller_profile
                    item["b27_arousal_state"] = decision.b27_arousal_state
                    item["b27_arousal_level"] = decision.b27_arousal_level
                    item["b27_gain_modulation"] = decision.b27_gain_modulation
                    item["b27_stress_pressure"] = decision.b27_stress_pressure
                    item["b27_arousal_lock"] = decision.b27_arousal_lock
                    item["b27_decision"] = decision.b27_decision
                    item["b27_genetic_generation"] = decision.b27_genetic_generation
                    item["b27_genetic_candidate"] = decision.b27_genetic_candidate
                    item["b28_controller_profile"] = decision.b28_controller_profile
                    item["b28_attention_state"] = decision.b28_attention_state
                    item["b28_interoceptive_focus"] = decision.b28_interoceptive_focus
                    item["b28_attention_gain"] = decision.b28_attention_gain
                    item["b28_distractor_pressure"] = decision.b28_distractor_pressure
                    item["b28_attention_lock"] = decision.b28_attention_lock
                    item["b28_decision"] = decision.b28_decision
                    item["b28_genetic_generation"] = decision.b28_genetic_generation
                    item["b28_genetic_candidate"] = decision.b28_genetic_candidate
                    item["b29_controller_profile"] = decision.b29_controller_profile
                    item["b29_salience_state"] = decision.b29_salience_state
                    item["b29_threat_salience"] = decision.b29_threat_salience
                    item["b29_homeostatic_salience"] = (
                        decision.b29_homeostatic_salience
                    )
                    item["b29_corridor_salience"] = decision.b29_corridor_salience
                    item["b29_winner_channel"] = decision.b29_winner_channel
                    item["b29_salience_lock"] = decision.b29_salience_lock
                    item["b29_decision"] = decision.b29_decision
                    item["b29_genetic_generation"] = decision.b29_genetic_generation
                    item["b29_genetic_candidate"] = decision.b29_genetic_candidate
                    item["b30_controller_profile"] = decision.b30_controller_profile
                    item["b30_gate_state"] = decision.b30_gate_state
                    item["b30_go_signal"] = decision.b30_go_signal
                    item["b30_no_go_signal"] = decision.b30_no_go_signal
                    item["b30_action_gate"] = decision.b30_action_gate
                    item["b30_gate_lock"] = decision.b30_gate_lock
                    item["b30_decision"] = decision.b30_decision
                    item["b30_genetic_generation"] = decision.b30_genetic_generation
                    item["b30_genetic_candidate"] = decision.b30_genetic_candidate
                    item["b31_controller_profile"] = decision.b31_controller_profile
                    item["b31_dopamine_state"] = decision.b31_dopamine_state
                    item["b31_reward_prediction_error"] = (
                        decision.b31_reward_prediction_error
                    )
                    item["b31_tonic_dopamine"] = decision.b31_tonic_dopamine
                    item["b31_phasic_dopamine"] = decision.b31_phasic_dopamine
                    item["b31_gate_bias"] = decision.b31_gate_bias
                    item["b31_dopamine_lock"] = decision.b31_dopamine_lock
                    item["b31_decision"] = decision.b31_decision
                    item["b31_genetic_generation"] = decision.b31_genetic_generation
                    item["b31_genetic_candidate"] = decision.b31_genetic_candidate
                    item["b32_controller_profile"] = decision.b32_controller_profile
                    item["b32_critic_value"] = decision.b32_critic_value
                    item["b32_actor_advantage"] = decision.b32_actor_advantage
                    item["b32_value_error"] = decision.b32_value_error
                    item["b32_policy_bias"] = decision.b32_policy_bias
                    item["b32_value_lock"] = decision.b32_value_lock
                    item["b32_decision"] = decision.b32_decision
                    item["b32_genetic_generation"] = decision.b32_genetic_generation
                    item["b32_genetic_candidate"] = decision.b32_genetic_candidate
                    item["b33_controller_profile"] = decision.b33_controller_profile
                    item["b33_td_error"] = decision.b33_td_error
                    item["b33_bootstrap_value"] = decision.b33_bootstrap_value
                    item["b33_reward_trace"] = decision.b33_reward_trace
                    item["b33_actor_update"] = decision.b33_actor_update
                    item["b33_td_lock"] = decision.b33_td_lock
                    item["b33_decision"] = decision.b33_decision
                    item["b33_genetic_generation"] = decision.b33_genetic_generation
                    item["b33_genetic_candidate"] = decision.b33_genetic_candidate
                    item["b34_controller_profile"] = decision.b34_controller_profile
                    item["b34_eligibility_trace"] = decision.b34_eligibility_trace
                    item["b34_credit_assignment"] = decision.b34_credit_assignment
                    item["b34_synaptic_tag"] = decision.b34_synaptic_tag
                    item["b34_decay_memory"] = decision.b34_decay_memory
                    item["b34_credit_lock"] = decision.b34_credit_lock
                    item["b34_decision"] = decision.b34_decision
                    item["b34_genetic_generation"] = decision.b34_genetic_generation
                    item["b34_genetic_candidate"] = decision.b34_genetic_candidate
                    item["b35_controller_profile"] = decision.b35_controller_profile
                    item["b35_forward_value"] = decision.b35_forward_value
                    item["b35_transition_error"] = decision.b35_transition_error
                    item["b35_model_confidence"] = decision.b35_model_confidence
                    item["b35_prediction_memory"] = decision.b35_prediction_memory
                    item["b35_model_lock"] = decision.b35_model_lock
                    item["b35_decision"] = decision.b35_decision
                    item["b35_genetic_generation"] = decision.b35_genetic_generation
                    item["b35_genetic_candidate"] = decision.b35_genetic_candidate
                    item["b36_controller_profile"] = decision.b36_controller_profile
                    item["b36_latent_state"] = decision.b36_latent_state
                    item["b36_belief_error"] = decision.b36_belief_error
                    item["b36_state_confidence"] = decision.b36_state_confidence
                    item["b36_context_memory"] = decision.b36_context_memory
                    item["b36_belief_lock"] = decision.b36_belief_lock
                    item["b36_decision"] = decision.b36_decision
                    item["b36_genetic_generation"] = decision.b36_genetic_generation
                    item["b36_genetic_candidate"] = decision.b36_genetic_candidate
                    item["b37_controller_profile"] = decision.b37_controller_profile
                    item["b37_external_state_factor"] = (
                        decision.b37_external_state_factor
                    )
                    item["b37_internal_state_factor"] = (
                        decision.b37_internal_state_factor
                    )
                    item["b37_factor_alignment"] = decision.b37_factor_alignment
                    item["b37_factor_confidence"] = decision.b37_factor_confidence
                    item["b37_factor_lock"] = decision.b37_factor_lock
                    item["b37_decision"] = decision.b37_decision
                    item["b37_genetic_generation"] = decision.b37_genetic_generation
                    item["b37_genetic_candidate"] = decision.b37_genetic_candidate
                    item["b38_controller_profile"] = decision.b38_controller_profile
                    item["b38_external_attention"] = decision.b38_external_attention
                    item["b38_internal_attention"] = decision.b38_internal_attention
                    item["b38_attention_balance"] = decision.b38_attention_balance
                    item["b38_attention_gain"] = decision.b38_attention_gain
                    item["b38_attention_lock"] = decision.b38_attention_lock
                    item["b38_decision"] = decision.b38_decision
                    item["b38_genetic_generation"] = decision.b38_genetic_generation
                    item["b38_genetic_candidate"] = decision.b38_genetic_candidate
                    item["b39_controller_profile"] = decision.b39_controller_profile
                    item["b39_binding_strength"] = decision.b39_binding_strength
                    item["b39_cross_factor_coherence"] = (
                        decision.b39_cross_factor_coherence
                    )
                    item["b39_bound_context"] = decision.b39_bound_context
                    item["b39_binding_gain"] = decision.b39_binding_gain
                    item["b39_binding_lock"] = decision.b39_binding_lock
                    item["b39_decision"] = decision.b39_decision
                    item["b39_genetic_generation"] = decision.b39_genetic_generation
                    item["b39_genetic_candidate"] = decision.b39_genetic_candidate
                    item["b40_controller_profile"] = decision.b40_controller_profile
                    item["b40_workspace_activation"] = decision.b40_workspace_activation
                    item["b40_broadcast_gain"] = decision.b40_broadcast_gain
                    item["b40_context_availability"] = (
                        decision.b40_context_availability
                    )
                    item["b40_workspace_stability"] = decision.b40_workspace_stability
                    item["b40_workspace_lock"] = decision.b40_workspace_lock
                    item["b40_decision"] = decision.b40_decision
                    item["b40_genetic_generation"] = decision.b40_genetic_generation
                    item["b40_genetic_candidate"] = decision.b40_genetic_candidate
                    item["b41_controller_profile"] = decision.b41_controller_profile
                    item["b41_executive_selection"] = decision.b41_executive_selection
                    item["b41_inhibitory_pressure"] = decision.b41_inhibitory_pressure
                    item["b41_goal_context"] = decision.b41_goal_context
                    item["b41_executive_stability"] = decision.b41_executive_stability
                    item["b41_executive_lock"] = decision.b41_executive_lock
                    item["b41_decision"] = decision.b41_decision
                    item["b41_genetic_generation"] = decision.b41_genetic_generation
                    item["b41_genetic_candidate"] = decision.b41_genetic_candidate
                    item["b42_controller_profile"] = decision.b42_controller_profile
                    item["b42_error_signal"] = decision.b42_error_signal
                    item["b42_conflict_signal"] = decision.b42_conflict_signal
                    item["b42_performance_context"] = decision.b42_performance_context
                    item["b42_monitor_stability"] = decision.b42_monitor_stability
                    item["b42_monitor_lock"] = decision.b42_monitor_lock
                    item["b42_decision"] = decision.b42_decision
                    item["b42_genetic_generation"] = decision.b42_genetic_generation
                    item["b42_genetic_candidate"] = decision.b42_genetic_candidate
                    item["b43_controller_profile"] = decision.b43_controller_profile
                    item["b43_precision_signal"] = decision.b43_precision_signal
                    item["b43_adaptive_threshold"] = decision.b43_adaptive_threshold
                    item["b43_arousal_context"] = decision.b43_arousal_context
                    item["b43_control_stability"] = decision.b43_control_stability
                    item["b43_precision_lock"] = decision.b43_precision_lock
                    item["b43_decision"] = decision.b43_decision
                    item["b43_genetic_generation"] = decision.b43_genetic_generation
                    item["b43_genetic_candidate"] = decision.b43_genetic_candidate
                    item["b44_controller_profile"] = decision.b44_controller_profile
                    item["b44_relay_gate"] = decision.b44_relay_gate
                    item["b44_sensory_precision"] = decision.b44_sensory_precision
                    item["b44_context_relay"] = decision.b44_context_relay
                    item["b44_gate_stability"] = decision.b44_gate_stability
                    item["b44_relay_lock"] = decision.b44_relay_lock
                    item["b44_decision"] = decision.b44_decision
                    item["b44_genetic_generation"] = decision.b44_genetic_generation
                    item["b44_genetic_candidate"] = decision.b44_genetic_candidate
                    item["b45_controller_profile"] = decision.b45_controller_profile
                    item["b45_inhibitory_gate"] = decision.b45_inhibitory_gate
                    item["b45_sensory_filter"] = decision.b45_sensory_filter
                    item["b45_context_suppression"] = decision.b45_context_suppression
                    item["b45_loop_stability"] = decision.b45_loop_stability
                    item["b45_inhibition_lock"] = decision.b45_inhibition_lock
                    item["b45_decision"] = decision.b45_decision
                    item["b45_genetic_generation"] = decision.b45_genetic_generation
                    item["b45_genetic_candidate"] = decision.b45_genetic_candidate
                    item["b46_controller_profile"] = decision.b46_controller_profile
                    item["b46_feedback_gain"] = decision.b46_feedback_gain
                    item["b46_topdown_context"] = decision.b46_topdown_context
                    item["b46_prediction_match"] = decision.b46_prediction_match
                    item["b46_feedback_stability"] = decision.b46_feedback_stability
                    item["b46_feedback_lock"] = decision.b46_feedback_lock
                    item["b46_decision"] = decision.b46_decision
                    item["b46_genetic_generation"] = decision.b46_genetic_generation
                    item["b46_genetic_candidate"] = decision.b46_genetic_candidate
                    item["b47_controller_profile"] = decision.b47_controller_profile
                    item["b47_phase_alignment"] = decision.b47_phase_alignment
                    item["b47_synchrony_gain"] = decision.b47_synchrony_gain
                    item["b47_cross_loop_coherence"] = (
                        decision.b47_cross_loop_coherence
                    )
                    item["b47_phase_lock"] = decision.b47_phase_lock
                    item["b47_decision"] = decision.b47_decision
                    item["b47_genetic_generation"] = decision.b47_genetic_generation
                    item["b47_genetic_candidate"] = decision.b47_genetic_candidate
                    item["b48_controller_profile"] = decision.b48_controller_profile
                    item["b48_timing_error"] = decision.b48_timing_error
                    item["b48_predictive_timing"] = decision.b48_predictive_timing
                    item["b48_corrective_gain"] = decision.b48_corrective_gain
                    item["b48_calibration_lock"] = decision.b48_calibration_lock
                    item["b48_decision"] = decision.b48_decision
                    item["b48_genetic_generation"] = decision.b48_genetic_generation
                    item["b48_genetic_candidate"] = decision.b48_genetic_candidate
                    item["b49_controller_profile"] = decision.b49_controller_profile
                    item["b49_go_signal"] = decision.b49_go_signal
                    item["b49_no_go_signal"] = decision.b49_no_go_signal
                    item["b49_action_gate_balance"] = decision.b49_action_gate_balance
                    item["b49_selection_lock"] = decision.b49_selection_lock
                    item["b49_decision"] = decision.b49_decision
                    item["b49_genetic_generation"] = decision.b49_genetic_generation
                    item["b49_genetic_candidate"] = decision.b49_genetic_candidate
                    item["b50_controller_profile"] = decision.b50_controller_profile
                    item["b50_habit_strength"] = decision.b50_habit_strength
                    item["b50_chunk_value"] = decision.b50_chunk_value
                    item["b50_habit_stability"] = decision.b50_habit_stability
                    item["b50_chunk_lock"] = decision.b50_chunk_lock
                    item["b50_decision"] = decision.b50_decision
                    item["b50_genetic_generation"] = decision.b50_genetic_generation
                    item["b50_genetic_candidate"] = decision.b50_genetic_candidate
                    item["b51_controller_profile"] = decision.b51_controller_profile
                    item["b51_prediction_error"] = decision.b51_prediction_error
                    item["b51_dopamine_gain"] = decision.b51_dopamine_gain
                    item["b51_habit_modulation"] = decision.b51_habit_modulation
                    item["b51_modulation_lock"] = decision.b51_modulation_lock
                    item["b51_decision"] = decision.b51_decision
                    item["b51_genetic_generation"] = decision.b51_genetic_generation
                    item["b51_genetic_candidate"] = decision.b51_genetic_candidate
                    item["b52_controller_profile"] = decision.b52_controller_profile
                    item["b52_acetylcholine_level"] = decision.b52_acetylcholine_level
                    item["b52_precision_gain"] = decision.b52_precision_gain
                    item["b52_uncertainty_signal"] = decision.b52_uncertainty_signal
                    item["b52_attention_lock"] = decision.b52_attention_lock
                    item["b52_decision"] = decision.b52_decision
                    item["b52_genetic_generation"] = decision.b52_genetic_generation
                    item["b52_genetic_candidate"] = decision.b52_genetic_candidate
                    item["b53_controller_profile"] = decision.b53_controller_profile
                    item["b53_norepinephrine_level"] = decision.b53_norepinephrine_level
                    item["b53_arousal_gain"] = decision.b53_arousal_gain
                    item["b53_surprise_signal"] = decision.b53_surprise_signal
                    item["b53_gain_lock"] = decision.b53_gain_lock
                    item["b53_decision"] = decision.b53_decision
                    item["b53_genetic_generation"] = decision.b53_genetic_generation
                    item["b53_genetic_candidate"] = decision.b53_genetic_candidate
                    item["b54_controller_profile"] = decision.b54_controller_profile
                    item["b54_serotonin_level"] = decision.b54_serotonin_level
                    item["b54_patience_signal"] = decision.b54_patience_signal
                    item["b54_impulse_suppression"] = decision.b54_impulse_suppression
                    item["b54_patience_lock"] = decision.b54_patience_lock
                    item["b54_decision"] = decision.b54_decision
                    item["b54_genetic_generation"] = decision.b54_genetic_generation
                    item["b54_genetic_candidate"] = decision.b54_genetic_candidate
                    item["b55_controller_profile"] = decision.b55_controller_profile
                    item["b55_hypothalamic_drive"] = decision.b55_hypothalamic_drive
                    item["b55_satiety_signal"] = decision.b55_satiety_signal
                    item["b55_recovery_bias"] = decision.b55_recovery_bias
                    item["b55_drive_balance"] = decision.b55_drive_balance
                    item["b55_drive_lock"] = decision.b55_drive_lock
                    item["b55_decision"] = decision.b55_decision
                    item["b55_genetic_generation"] = decision.b55_genetic_generation
                    item["b55_genetic_candidate"] = decision.b55_genetic_candidate
                    item["b56_controller_profile"] = decision.b56_controller_profile
                    item["b56_cortisol_level"] = decision.b56_cortisol_level
                    item["b56_stress_load"] = decision.b56_stress_load
                    item["b56_recovery_signal"] = decision.b56_recovery_signal
                    item["b56_endocrine_balance"] = decision.b56_endocrine_balance
                    item["b56_stress_lock"] = decision.b56_stress_lock
                    item["b56_decision"] = decision.b56_decision
                    item["b56_genetic_generation"] = decision.b56_genetic_generation
                    item["b56_genetic_candidate"] = decision.b56_genetic_candidate
                    item["b57_controller_profile"] = decision.b57_controller_profile
                    item["b57_interoceptive_awareness"] = (
                        decision.b57_interoceptive_awareness
                    )
                    item["b57_visceral_salience"] = decision.b57_visceral_salience
                    item["b57_body_state_confidence"] = (
                        decision.b57_body_state_confidence
                    )
                    item["b57_awareness_balance"] = decision.b57_awareness_balance
                    item["b57_awareness_lock"] = decision.b57_awareness_lock
                    item["b57_decision"] = decision.b57_decision
                    item["b57_genetic_generation"] = decision.b57_genetic_generation
                    item["b57_genetic_candidate"] = decision.b57_genetic_candidate
                    item["b58_controller_profile"] = decision.b58_controller_profile
                    item["b58_conflict_signal"] = decision.b58_conflict_signal
                    item["b58_error_likelihood"] = decision.b58_error_likelihood
                    item["b58_control_allocation"] = decision.b58_control_allocation
                    item["b58_resolution_balance"] = decision.b58_resolution_balance
                    item["b58_conflict_lock"] = decision.b58_conflict_lock
                    item["b58_decision"] = decision.b58_decision
                    item["b58_genetic_generation"] = decision.b58_genetic_generation
                    item["b58_genetic_candidate"] = decision.b58_genetic_candidate
                    item["b59_controller_profile"] = decision.b59_controller_profile
                    item["b59_goal_context"] = decision.b59_goal_context
                    item["b59_working_set_stability"] = (
                        decision.b59_working_set_stability
                    )
                    item["b59_task_set_confidence"] = decision.b59_task_set_confidence
                    item["b59_executive_balance"] = decision.b59_executive_balance
                    item["b59_executive_lock"] = decision.b59_executive_lock
                    item["b59_decision"] = decision.b59_decision
                    item["b59_genetic_generation"] = decision.b59_genetic_generation
                    item["b59_genetic_candidate"] = decision.b59_genetic_candidate
                    item["b60_controller_profile"] = decision.b60_controller_profile
                    item["b60_outcome_value"] = decision.b60_outcome_value
                    item["b60_reversal_signal"] = decision.b60_reversal_signal
                    item["b60_goal_value_confidence"] = (
                        decision.b60_goal_value_confidence
                    )
                    item["b60_value_balance"] = decision.b60_value_balance
                    item["b60_value_lock"] = decision.b60_value_lock
                    item["b60_decision"] = decision.b60_decision
                    item["b60_genetic_generation"] = decision.b60_genetic_generation
                    item["b60_genetic_candidate"] = decision.b60_genetic_candidate
                    item["b61_controller_profile"] = decision.b61_controller_profile
                    item["b61_safety_value"] = decision.b61_safety_value
                    item["b61_threat_value"] = decision.b61_threat_value
                    item["b61_safety_confidence"] = decision.b61_safety_confidence
                    item["b61_affective_balance"] = decision.b61_affective_balance
                    item["b61_safety_lock"] = decision.b61_safety_lock
                    item["b61_decision"] = decision.b61_decision
                    item["b61_genetic_generation"] = decision.b61_genetic_generation
                    item["b61_genetic_candidate"] = decision.b61_genetic_candidate
                    item["b62_controller_profile"] = decision.b62_controller_profile
                    item["b62_defensive_mode"] = decision.b62_defensive_mode
                    item["b62_freeze_pressure"] = decision.b62_freeze_pressure
                    item["b62_flee_pressure"] = decision.b62_flee_pressure
                    item["b62_shelter_bias"] = decision.b62_shelter_bias
                    item["b62_defense_balance"] = decision.b62_defense_balance
                    item["b62_defense_lock"] = decision.b62_defense_lock
                    item["b62_decision"] = decision.b62_decision
                    item["b62_genetic_generation"] = decision.b62_genetic_generation
                    item["b62_genetic_candidate"] = decision.b62_genetic_candidate
                    item["semantic_action"] = decision.semantic_action
                    item["learned_semantic_action"] = decision.learned_semantic_action
                    item["semantic_action_source"] = decision.semantic_action_source
                    item["semantic_action_reason"] = decision.semantic_action_reason
                    item["semantic_override_count"] = int(
                        decision.semantic_override_count
                    )
                    item["semantic_logits"] = decision.semantic_logits.round(6).tolist()
                    item["bridge_primitive_action"] = decision.bridge_primitive_action
                    item["bridge_reason"] = decision.bridge_reason
                    item["blocked_mask"] = dict(decision.blocked_mask)
                    item["food_delta_used"] = round(float(decision.food_delta_used), 6)
                    item["shelter_delta_used"] = round(
                        float(decision.shelter_delta_used),
                        6,
                    )
                    item["external_override_count"] = int(
                        decision.external_override_count
                    )
                if debug_trace:
                    arbitration_payload = (
                        decision.arbitration_decision.to_payload()
                        if decision.arbitration_decision is not None
                        else {}
                    )
                    item["observation"] = jsonify_observation(observation)
                    item["next_observation"] = jsonify_observation(next_observation)
                    item["debug"] = {
                        "observation_contracts": adapter_trace_summary(
                            observation_adapters
                        ),
                        "next_observation_contracts": adapter_trace_summary(
                            next_observation_adapters
                        ),
                        "memory_vectors": next_observation["meta"].get("memory_vectors", {}),
                        "reflexes": {
                            result.name: {
                                "active": bool(result.active),
                                "best_action": ACTIONS[int(result.probs.argmax())],
                                "neural_logits": result.neural_logits.round(6).tolist() if result.neural_logits is not None else None,
                                "reflex_delta_logits": result.reflex_delta_logits.round(6).tolist() if result.reflex_delta_logits is not None else None,
                                "post_reflex_logits": result.post_reflex_logits.round(6).tolist() if result.post_reflex_logits is not None else None,
                                "reflex_applied": bool(result.reflex_applied),
                                "effective_reflex_scale": round(float(result.effective_reflex_scale), 6),
                                "module_reflex_override": bool(result.module_reflex_override),
                                "module_reflex_dominance": round(float(result.module_reflex_dominance), 6),
                                "contribution_share": round(float(result.contribution_share), 6),
                                "reflex": result.reflex.to_payload() if result.reflex is not None else None,
                            }
                            for result in decision.module_results
                        },
                        "action_center": {
                            "policy_mode": decision.policy_mode,
                            "input": decision.action_center_input.round(6).tolist(),
                            "logits": decision.action_center_logits.round(6).tolist(),
                            "policy": decision.action_center_policy.round(6).tolist(),
                            "selected_intent": ACTIONS[decision.action_intent_idx],
                            **dict(arbitration_payload),
                        },
                        "arbitration": dict(arbitration_payload),
                        "motor_cortex": {
                            "policy_mode": decision.policy_mode,
                            "input": decision.motor_input.round(6).tolist(),
                            "correction_logits": decision.motor_correction_logits.round(6).tolist(),
                            "motor_override": bool(decision.motor_override),
                            "selected_intent": ACTIONS[decision.action_intent_idx],
                            "arbitrated_intent": ACTIONS[decision.action_intent_idx],
                            "selected_action": ACTIONS[decision.motor_action_idx],
                            "motor_selected_action": ACTIONS[decision.motor_action_idx],
                            "sampled_action": ACTIONS[decision.action_idx],
                            "intended_action": info.get("intended_action", ACTIONS[decision.action_idx]),
                            "executed_action": info.get("executed_action", ACTIONS[decision.action_idx]),
                            "execution_slip_occurred": bool(decision.execution_slip_occurred),
                            "motor_noise_applied": bool(decision.motor_noise_applied),
                            "slip_reason": str(decision.slip_reason),
                            "orientation_alignment": round(float(decision.orientation_alignment), 6),
                            "terrain_difficulty": round(float(decision.terrain_difficulty), 6),
                            "execution_difficulty": round(float(decision.execution_difficulty), 6),
                        },
                        "phase": {
                            "target": decision.phase_target,
                            "prediction": decision.phase_prediction,
                            "prediction_confidence": round(
                                float(decision.phase_prediction_confidence),
                                6,
                            ),
                            "logits": decision.phase_logits.round(6).tolist(),
                            "selected_action": ACTIONS[decision.action_idx],
                        },
                        "event_attention": {
                            "top_type": decision.event_attention_top_type,
                            "top_age": int(decision.event_attention_top_age),
                            "entropy": round(
                                float(decision.event_attention_entropy),
                                6,
                            ),
                        },
                        "option": {
                            "selected_option": decision.selected_option,
                            "age": int(decision.option_age),
                            "termination_reason": decision.option_termination_reason,
                            "logits": decision.option_logits.round(6).tolist(),
                            "outside_or_corridor": bool(
                                (not bool(next_observation["meta"].get("on_shelter", False)))
                                or str(next_observation["meta"].get("shelter_role", "")) == "outside"
                            ),
                        },
                        "affordance": {
                            "blocked_logits": decision.affordance_blocked_logits.round(6).tolist(),
                            "blocked_targets": decision.affordance_blocked_targets.round(6).tolist(),
                            "role_logits": decision.affordance_role_logits.round(6).tolist(),
                            "role_targets": decision.affordance_role_targets.astype(int).tolist(),
                            "geometry_logits": decision.geometry_logits.round(6).tolist(),
                            "geometry_targets": decision.geometry_targets.round(6).tolist(),
                            "shelter_column_logits": decision.shelter_column_logits.round(6).tolist(),
                            "shelter_column_targets": decision.shelter_column_targets.astype(int).tolist(),
                            "shelter_position_logits": decision.shelter_position_logits.round(6).tolist(),
                            "shelter_position_targets": decision.shelter_position_targets.astype(int).tolist(),
                            "transition_prediction_logits": decision.transition_prediction_logits.round(6).tolist(),
                            "transition_prediction_targets": decision.transition_prediction_targets.round(6).tolist(),
                            "transition_rollout_prediction_logits": decision.transition_rollout_prediction_logits.round(6).tolist(),
                            "transition_rollout_prediction_targets": decision.transition_rollout_prediction_targets.round(6).tolist(),
                        },
                        "teacher": {
                            "action_target": decision.teacher_action_target_name,
                            "action_target_idx": int(decision.teacher_action_target_idx),
                            "stage": decision.teacher_action_target_stage,
                            "option_target": decision.teacher_option_target_name,
                            "option_target_idx": int(decision.teacher_option_target_idx),
                            "option_stage": decision.teacher_option_target_stage,
                        },
                        **(
                            {
                                "b_series": {
                                    "b_level": int(decision.b_level),
                                    "b_effective_level": decision.b_effective_level,
                                    "b_mode": decision.b_mode,
                                    "b_parent_level": decision.b_parent_level,
                                    "b_transfer_source_checkpoint": (
                                        decision.b_transfer_source_checkpoint
                                    ),
                                    "b_transfer_coverage": (
                                        decision.b_transfer_coverage
                                    ),
                                    "b_current_threat_pressure": (
                                        decision.b_current_threat_pressure
                                    ),
                                    "b_temporal_threat_pressure": (
                                        decision.b_temporal_threat_pressure
                                    ),
                                    "b_predator_memory_pressure": (
                                        decision.b_predator_memory_pressure
                                    ),
                                    "b_predator_trace_pressure": (
                                        decision.b_predator_trace_pressure
                                    ),
                                    "b3_contact_cooldown": (
                                        decision.b3_contact_cooldown
                                    ),
                                    "b3_post_food_cooldown": (
                                        decision.b3_post_food_cooldown
                                    ),
                                    "b3_hunger_drop": decision.b3_hunger_drop,
                                    "b3_controller_profile": (
                                        decision.b3_controller_profile
                                    ),
                                    "b4_controller_profile": (
                                        decision.b4_controller_profile
                                    ),
                                    "b4_recovery_pressure": (
                                        decision.b4_recovery_pressure
                                    ),
                                    "b4_sleep_hold": decision.b4_sleep_hold,
                                    "b4_exit_blocked": decision.b4_exit_blocked,
                                    "b4_hunger_release": (
                                        decision.b4_hunger_release
                                    ),
                                    "b4_genetic_generation": (
                                        decision.b4_genetic_generation
                                    ),
                                    "b4_genetic_candidate": (
                                        decision.b4_genetic_candidate
                                    ),
                                    "b5_controller_profile": (
                                        decision.b5_controller_profile
                                    ),
                                    "b5_hunger_urgency": decision.b5_hunger_urgency,
                                    "b5_sleep_pressure": decision.b5_sleep_pressure,
                                    "b5_recovery_debt": decision.b5_recovery_debt,
                                    "b5_threat_gate": decision.b5_threat_gate,
                                    "b5_sleep_bout_lock": (
                                        decision.b5_sleep_bout_lock
                                    ),
                                    "b5_forage_commitment_lock": (
                                        decision.b5_forage_commitment_lock
                                    ),
                                    "b5_homeostatic_decision": (
                                        decision.b5_homeostatic_decision
                                    ),
                                    "b5_genetic_generation": (
                                        decision.b5_genetic_generation
                                    ),
                                    "b5_genetic_candidate": (
                                        decision.b5_genetic_candidate
                                    ),
                                    "b6_controller_family": (
                                        decision.b6_controller_family
                                    ),
                                    "b6_controller_profile": (
                                        decision.b6_controller_profile
                                    ),
                                    "b6_risk_pressure": (
                                        decision.b6_risk_pressure
                                    ),
                                    "b6_threat_priority": (
                                        decision.b6_threat_priority
                                    ),
                                    "b6_forage_suppressed": (
                                        decision.b6_forage_suppressed
                                    ),
                                    "b6_corridor_commitment": (
                                        decision.b6_corridor_commitment
                                    ),
                                    "b6_corridor_progress_memory": (
                                        decision.b6_corridor_progress_memory
                                    ),
                                    "b6_recurrent_state": (
                                        decision.b6_recurrent_state
                                    ),
                                    "b6_return_lock": decision.b6_return_lock,
                                    "b6_decision": decision.b6_decision,
                                    "b6_genetic_generation": (
                                        decision.b6_genetic_generation
                                    ),
                                    "b6_genetic_candidate": (
                                        decision.b6_genetic_candidate
                                    ),
                                    "b7_controller_profile": (
                                        decision.b7_controller_profile
                                    ),
                                    "b7_affordance_state": (
                                        decision.b7_affordance_state
                                    ),
                                    "b7_energy_budget": decision.b7_energy_budget,
                                    "b7_budget_margin": decision.b7_budget_margin,
                                    "b7_food_steps_estimate": (
                                        decision.b7_food_steps_estimate
                                    ),
                                    "b7_return_steps_estimate": (
                                        decision.b7_return_steps_estimate
                                    ),
                                    "b7_corridor_viability": (
                                        decision.b7_corridor_viability
                                    ),
                                    "b7_abort_return": decision.b7_abort_return,
                                    "b7_commitment_lock": (
                                        decision.b7_commitment_lock
                                    ),
                                    "b7_decision": decision.b7_decision,
                                    "b7_genetic_generation": (
                                        decision.b7_genetic_generation
                                    ),
                                    "b7_genetic_candidate": (
                                        decision.b7_genetic_candidate
                                    ),
                                    "b8_controller_profile": (
                                        decision.b8_controller_profile
                                    ),
                                    "b8_spatial_map_state": (
                                        decision.b8_spatial_map_state
                                    ),
                                    "b8_local_affordance_score": (
                                        decision.b8_local_affordance_score
                                    ),
                                    "b8_return_vector_strength": (
                                        decision.b8_return_vector_strength
                                    ),
                                    "b8_corridor_dead_end_risk": (
                                        decision.b8_corridor_dead_end_risk
                                    ),
                                    "b8_abort_executed": decision.b8_abort_executed,
                                    "b8_place_memory": decision.b8_place_memory,
                                    "b8_decision": decision.b8_decision,
                                    "b8_genetic_generation": (
                                        decision.b8_genetic_generation
                                    ),
                                    "b8_genetic_candidate": (
                                        decision.b8_genetic_candidate
                                    ),
                                    "b9_controller_profile": (
                                        decision.b9_controller_profile
                                    ),
                                    "b9_route_state": decision.b9_route_state,
                                    "b9_route_confidence": (
                                        decision.b9_route_confidence
                                    ),
                                    "b9_waypoint_lock": decision.b9_waypoint_lock,
                                    "b9_path_integrator": (
                                        decision.b9_path_integrator
                                    ),
                                    "b9_replan_signal": decision.b9_replan_signal,
                                    "b9_decision": decision.b9_decision,
                                    "b9_genetic_generation": (
                                        decision.b9_genetic_generation
                                    ),
                                    "b9_genetic_candidate": (
                                        decision.b9_genetic_candidate
                                    ),
                                    "b10_controller_profile": (
                                        decision.b10_controller_profile
                                    ),
                                    "b10_replay_state": decision.b10_replay_state,
                                    "b10_prospective_value": (
                                        decision.b10_prospective_value
                                    ),
                                    "b10_rollout_depth": decision.b10_rollout_depth,
                                    "b10_replay_memory": decision.b10_replay_memory,
                                    "b10_plan_commitment": (
                                        decision.b10_plan_commitment
                                    ),
                                    "b10_abort_signal": decision.b10_abort_signal,
                                    "b10_decision": decision.b10_decision,
                                    "b10_genetic_generation": (
                                        decision.b10_genetic_generation
                                    ),
                                    "b10_genetic_candidate": (
                                        decision.b10_genetic_candidate
                                    ),
                                    "b11_controller_profile": (
                                        decision.b11_controller_profile
                                    ),
                                    "b11_confidence_state": (
                                        decision.b11_confidence_state
                                    ),
                                    "b11_plan_confidence": (
                                        decision.b11_plan_confidence
                                    ),
                                    "b11_uncertainty": decision.b11_uncertainty,
                                    "b11_neuromod_signal": (
                                        decision.b11_neuromod_signal
                                    ),
                                    "b11_confidence_lock": (
                                        decision.b11_confidence_lock
                                    ),
                                    "b11_decision": decision.b11_decision,
                                    "b11_genetic_generation": (
                                        decision.b11_genetic_generation
                                    ),
                                    "b11_genetic_candidate": (
                                        decision.b11_genetic_candidate
                                    ),
                                    "b12_controller_profile": (
                                        decision.b12_controller_profile
                                    ),
                                    "b12_attention_state": (
                                        decision.b12_attention_state
                                    ),
                                    "b12_prediction_error": (
                                        decision.b12_prediction_error
                                    ),
                                    "b12_attention_gain": (
                                        decision.b12_attention_gain
                                    ),
                                    "b12_expected_progress": (
                                        decision.b12_expected_progress
                                    ),
                                    "b12_search_lock": decision.b12_search_lock,
                                    "b12_decision": decision.b12_decision,
                                    "b12_genetic_generation": (
                                        decision.b12_genetic_generation
                                    ),
                                    "b12_genetic_candidate": (
                                        decision.b12_genetic_candidate
                                    ),
                                    "b13_controller_profile": (
                                        decision.b13_controller_profile
                                    ),
                                    "b13_search_state": decision.b13_search_state,
                                    "b13_local_route_score": (
                                        decision.b13_local_route_score
                                    ),
                                    "b13_affordance_samples": (
                                        decision.b13_affordance_samples
                                    ),
                                    "b13_search_memory": decision.b13_search_memory,
                                    "b13_dead_end_score": decision.b13_dead_end_score,
                                    "b13_search_lock": decision.b13_search_lock,
                                    "b13_decision": decision.b13_decision,
                                    "b13_genetic_generation": (
                                        decision.b13_genetic_generation
                                    ),
                                    "b13_genetic_candidate": (
                                        decision.b13_genetic_candidate
                                    ),
                                    "b14_controller_profile": (
                                        decision.b14_controller_profile
                                    ),
                                    "b14_uncertainty_state": (
                                        decision.b14_uncertainty_state
                                    ),
                                    "b14_affordance_confidence": (
                                        decision.b14_affordance_confidence
                                    ),
                                    "b14_uncertainty": decision.b14_uncertainty,
                                    "b14_risk_adjusted_score": (
                                        decision.b14_risk_adjusted_score
                                    ),
                                    "b14_commitment_lock": (
                                        decision.b14_commitment_lock
                                    ),
                                    "b14_decision": decision.b14_decision,
                                    "b14_genetic_generation": (
                                        decision.b14_genetic_generation
                                    ),
                                    "b14_genetic_candidate": (
                                        decision.b14_genetic_candidate
                                    ),
                                    "b15_controller_profile": (
                                        decision.b15_controller_profile
                                    ),
                                    "b15_option_state": decision.b15_option_state,
                                    "b15_option_value": decision.b15_option_value,
                                    "b15_termination_pressure": (
                                        decision.b15_termination_pressure
                                    ),
                                    "b15_persistence_score": (
                                        decision.b15_persistence_score
                                    ),
                                    "b15_option_lock": decision.b15_option_lock,
                                    "b15_decision": decision.b15_decision,
                                    "b15_genetic_generation": (
                                        decision.b15_genetic_generation
                                    ),
                                    "b15_genetic_candidate": (
                                        decision.b15_genetic_candidate
                                    ),
                                    "b16_controller_profile": (
                                        decision.b16_controller_profile
                                    ),
                                    "b16_ensemble_state": (
                                        decision.b16_ensemble_state
                                    ),
                                    "b16_continue_vote": decision.b16_continue_vote,
                                    "b16_return_vote": decision.b16_return_vote,
                                    "b16_option_votes": decision.b16_option_votes,
                                    "b16_consensus_score": (
                                        decision.b16_consensus_score
                                    ),
                                    "b16_conflict_score": (
                                        decision.b16_conflict_score
                                    ),
                                    "b16_ensemble_lock": decision.b16_ensemble_lock,
                                    "b16_decision": decision.b16_decision,
                                    "b16_genetic_generation": (
                                        decision.b16_genetic_generation
                                    ),
                                    "b16_genetic_candidate": (
                                        decision.b16_genetic_candidate
                                    ),
                                    "b17_controller_profile": (
                                        decision.b17_controller_profile
                                    ),
                                    "b17_modulator_state": (
                                        decision.b17_modulator_state
                                    ),
                                    "b17_arousal_signal": decision.b17_arousal_signal,
                                    "b17_homeostatic_gain": (
                                        decision.b17_homeostatic_gain
                                    ),
                                    "b17_option_gain": decision.b17_option_gain,
                                    "b17_conflict_release": (
                                        decision.b17_conflict_release
                                    ),
                                    "b17_modulation_lock": (
                                        decision.b17_modulation_lock
                                    ),
                                    "b17_decision": decision.b17_decision,
                                    "b17_genetic_generation": (
                                        decision.b17_genetic_generation
                                    ),
                                    "b17_genetic_candidate": (
                                        decision.b17_genetic_candidate
                                    ),
                                    "b18_controller_profile": (
                                        decision.b18_controller_profile
                                    ),
                                    "b18_trace_state": decision.b18_trace_state,
                                    "b18_eligibility_trace": (
                                        decision.b18_eligibility_trace
                                    ),
                                    "b18_reward_prediction_proxy": (
                                        decision.b18_reward_prediction_proxy
                                    ),
                                    "b18_stability_bias": (
                                        decision.b18_stability_bias
                                    ),
                                    "b18_switch_pressure": (
                                        decision.b18_switch_pressure
                                    ),
                                    "b18_trace_lock": decision.b18_trace_lock,
                                    "b18_decision": decision.b18_decision,
                                    "b18_genetic_generation": (
                                        decision.b18_genetic_generation
                                    ),
                                    "b18_genetic_candidate": (
                                        decision.b18_genetic_candidate
                                    ),
                                    "b19_controller_profile": (
                                        decision.b19_controller_profile
                                    ),
                                    "b19_memory_state": decision.b19_memory_state,
                                    "b19_episode_memory": (
                                        decision.b19_episode_memory
                                    ),
                                    "b19_consolidation_score": (
                                        decision.b19_consolidation_score
                                    ),
                                    "b19_stability_vote": (
                                        decision.b19_stability_vote
                                    ),
                                    "b19_switch_suppression": (
                                        decision.b19_switch_suppression
                                    ),
                                    "b19_memory_lock": decision.b19_memory_lock,
                                    "b19_decision": decision.b19_decision,
                                    "b19_genetic_generation": (
                                        decision.b19_genetic_generation
                                    ),
                                    "b19_genetic_candidate": (
                                        decision.b19_genetic_candidate
                                    ),
                                    "b20_controller_profile": (
                                        decision.b20_controller_profile
                                    ),
                                    "b20_buffer_state": decision.b20_buffer_state,
                                    "b20_working_buffer": (
                                        decision.b20_working_buffer
                                    ),
                                    "b20_context_binding": (
                                        decision.b20_context_binding
                                    ),
                                    "b20_gate_vote": decision.b20_gate_vote,
                                    "b20_release_vote": decision.b20_release_vote,
                                    "b20_buffer_lock": decision.b20_buffer_lock,
                                    "b20_decision": decision.b20_decision,
                                    "b20_genetic_generation": (
                                        decision.b20_genetic_generation
                                    ),
                                    "b20_genetic_candidate": (
                                        decision.b20_genetic_candidate
                                    ),
                                    "b21_controller_profile": (
                                        decision.b21_controller_profile
                                    ),
                                    "b21_replay_state": decision.b21_replay_state,
                                    "b21_sequence_memory": (
                                        decision.b21_sequence_memory
                                    ),
                                    "b21_replay_score": decision.b21_replay_score,
                                    "b21_route_commitment": (
                                        decision.b21_route_commitment
                                    ),
                                    "b21_abort_prediction": (
                                        decision.b21_abort_prediction
                                    ),
                                    "b21_replay_lock": decision.b21_replay_lock,
                                    "b21_decision": decision.b21_decision,
                                    "b21_genetic_generation": (
                                        decision.b21_genetic_generation
                                    ),
                                    "b21_genetic_candidate": (
                                        decision.b21_genetic_candidate
                                    ),
                                    "b22_controller_profile": (
                                        decision.b22_controller_profile
                                    ),
                                    "b22_sim_state": decision.b22_sim_state,
                                    "b22_prospective_sim": (
                                        decision.b22_prospective_sim
                                    ),
                                    "b22_forward_model_score": (
                                        decision.b22_forward_model_score
                                    ),
                                    "b22_viability_projection": (
                                        decision.b22_viability_projection
                                    ),
                                    "b22_abort_projection": (
                                        decision.b22_abort_projection
                                    ),
                                    "b22_sim_lock": decision.b22_sim_lock,
                                    "b22_decision": decision.b22_decision,
                                    "b22_genetic_generation": (
                                        decision.b22_genetic_generation
                                    ),
                                    "b22_genetic_candidate": (
                                        decision.b22_genetic_candidate
                                    ),
                                    "b23_controller_profile": (
                                        decision.b23_controller_profile
                                    ),
                                    "b23_conflict_state": (
                                        decision.b23_conflict_state
                                    ),
                                    "b23_prediction_error": (
                                        decision.b23_prediction_error
                                    ),
                                    "b23_conflict_memory": (
                                        decision.b23_conflict_memory
                                    ),
                                    "b23_stability_vote": decision.b23_stability_vote,
                                    "b23_abort_bias": decision.b23_abort_bias,
                                    "b23_monitor_lock": decision.b23_monitor_lock,
                                    "b23_decision": decision.b23_decision,
                                    "b23_genetic_generation": (
                                        decision.b23_genetic_generation
                                    ),
                                    "b23_genetic_candidate": (
                                        decision.b23_genetic_candidate
                                    ),
                                    "b24_controller_profile": (
                                        decision.b24_controller_profile
                                    ),
                                    "b24_precision_state": (
                                        decision.b24_precision_state
                                    ),
                                    "b24_precision_memory": (
                                        decision.b24_precision_memory
                                    ),
                                    "b24_precision_vote": decision.b24_precision_vote,
                                    "b24_uncertainty_pressure": (
                                        decision.b24_uncertainty_pressure
                                    ),
                                    "b24_abort_precision": (
                                        decision.b24_abort_precision
                                    ),
                                    "b24_precision_lock": decision.b24_precision_lock,
                                    "b24_decision": decision.b24_decision,
                                    "b24_genetic_generation": (
                                        decision.b24_genetic_generation
                                    ),
                                    "b24_genetic_candidate": (
                                        decision.b24_genetic_candidate
                                    ),
                                    "b25_controller_profile": (
                                        decision.b25_controller_profile
                                    ),
                                    "b25_metacognitive_state": (
                                        decision.b25_metacognitive_state
                                    ),
                                    "b25_confidence_memory": (
                                        decision.b25_confidence_memory
                                    ),
                                    "b25_confidence_vote": (
                                        decision.b25_confidence_vote
                                    ),
                                    "b25_doubt_pressure": (
                                        decision.b25_doubt_pressure
                                    ),
                                    "b25_control_gain": decision.b25_control_gain,
                                    "b25_meta_lock": decision.b25_meta_lock,
                                    "b25_decision": decision.b25_decision,
                                    "b25_genetic_generation": (
                                        decision.b25_genetic_generation
                                    ),
                                    "b25_genetic_candidate": (
                                        decision.b25_genetic_candidate
                                    ),
                                    "b26_controller_profile": (
                                        decision.b26_controller_profile
                                    ),
                                    "b26_allostatic_state": (
                                        decision.b26_allostatic_state
                                    ),
                                    "b26_prediction_error": (
                                        decision.b26_prediction_error
                                    ),
                                    "b26_setpoint_pressure": (
                                        decision.b26_setpoint_pressure
                                    ),
                                    "b26_control_vote": decision.b26_control_vote,
                                    "b26_stability_lock": (
                                        decision.b26_stability_lock
                                    ),
                                    "b26_decision": decision.b26_decision,
                                    "b26_genetic_generation": (
                                        decision.b26_genetic_generation
                                    ),
                                    "b26_genetic_candidate": (
                                        decision.b26_genetic_candidate
                                    ),
                                    "b27_controller_profile": (
                                        decision.b27_controller_profile
                                    ),
                                    "b27_arousal_state": decision.b27_arousal_state,
                                    "b27_arousal_level": decision.b27_arousal_level,
                                    "b27_gain_modulation": (
                                        decision.b27_gain_modulation
                                    ),
                                    "b27_stress_pressure": (
                                        decision.b27_stress_pressure
                                    ),
                                    "b27_arousal_lock": decision.b27_arousal_lock,
                                    "b27_decision": decision.b27_decision,
                                    "b27_genetic_generation": (
                                        decision.b27_genetic_generation
                                    ),
                                    "b27_genetic_candidate": (
                                        decision.b27_genetic_candidate
                                    ),
                                    "b28_controller_profile": (
                                        decision.b28_controller_profile
                                    ),
                                    "b28_attention_state": (
                                        decision.b28_attention_state
                                    ),
                                    "b28_interoceptive_focus": (
                                        decision.b28_interoceptive_focus
                                    ),
                                    "b28_attention_gain": decision.b28_attention_gain,
                                    "b28_distractor_pressure": (
                                        decision.b28_distractor_pressure
                                    ),
                                    "b28_attention_lock": decision.b28_attention_lock,
                                    "b28_decision": decision.b28_decision,
                                    "b28_genetic_generation": (
                                        decision.b28_genetic_generation
                                    ),
                                    "b28_genetic_candidate": (
                                        decision.b28_genetic_candidate
                                    ),
                                    "b29_controller_profile": (
                                        decision.b29_controller_profile
                                    ),
                                    "b29_salience_state": (
                                        decision.b29_salience_state
                                    ),
                                    "b29_threat_salience": (
                                        decision.b29_threat_salience
                                    ),
                                    "b29_homeostatic_salience": (
                                        decision.b29_homeostatic_salience
                                    ),
                                    "b29_corridor_salience": (
                                        decision.b29_corridor_salience
                                    ),
                                    "b29_winner_channel": decision.b29_winner_channel,
                                    "b29_salience_lock": decision.b29_salience_lock,
                                    "b29_decision": decision.b29_decision,
                                    "b29_genetic_generation": (
                                        decision.b29_genetic_generation
                                    ),
                                    "b29_genetic_candidate": (
                                        decision.b29_genetic_candidate
                                    ),
                                    "b30_controller_profile": (
                                        decision.b30_controller_profile
                                    ),
                                    "b30_gate_state": decision.b30_gate_state,
                                    "b30_go_signal": decision.b30_go_signal,
                                    "b30_no_go_signal": decision.b30_no_go_signal,
                                    "b30_action_gate": decision.b30_action_gate,
                                    "b30_gate_lock": decision.b30_gate_lock,
                                    "b30_decision": decision.b30_decision,
                                    "b30_genetic_generation": (
                                        decision.b30_genetic_generation
                                    ),
                                    "b30_genetic_candidate": (
                                        decision.b30_genetic_candidate
                                    ),
                                    "b31_controller_profile": (
                                        decision.b31_controller_profile
                                    ),
                                    "b31_dopamine_state": (
                                        decision.b31_dopamine_state
                                    ),
                                    "b31_reward_prediction_error": (
                                        decision.b31_reward_prediction_error
                                    ),
                                    "b31_tonic_dopamine": (
                                        decision.b31_tonic_dopamine
                                    ),
                                    "b31_phasic_dopamine": (
                                        decision.b31_phasic_dopamine
                                    ),
                                    "b31_gate_bias": decision.b31_gate_bias,
                                    "b31_dopamine_lock": decision.b31_dopamine_lock,
                                    "b31_decision": decision.b31_decision,
                                    "b31_genetic_generation": (
                                        decision.b31_genetic_generation
                                    ),
                                    "b31_genetic_candidate": (
                                        decision.b31_genetic_candidate
                                    ),
                                    "b32_controller_profile": (
                                        decision.b32_controller_profile
                                    ),
                                    "b32_critic_value": decision.b32_critic_value,
                                    "b32_actor_advantage": (
                                        decision.b32_actor_advantage
                                    ),
                                    "b32_value_error": decision.b32_value_error,
                                    "b32_policy_bias": decision.b32_policy_bias,
                                    "b32_value_lock": decision.b32_value_lock,
                                    "b32_decision": decision.b32_decision,
                                    "b32_genetic_generation": (
                                        decision.b32_genetic_generation
                                    ),
                                    "b32_genetic_candidate": (
                                        decision.b32_genetic_candidate
                                    ),
                                    "b33_controller_profile": (
                                        decision.b33_controller_profile
                                    ),
                                    "b33_td_error": decision.b33_td_error,
                                    "b33_bootstrap_value": decision.b33_bootstrap_value,
                                    "b33_reward_trace": decision.b33_reward_trace,
                                    "b33_actor_update": decision.b33_actor_update,
                                    "b33_td_lock": decision.b33_td_lock,
                                    "b33_decision": decision.b33_decision,
                                    "b33_genetic_generation": (
                                        decision.b33_genetic_generation
                                    ),
                                    "b33_genetic_candidate": (
                                        decision.b33_genetic_candidate
                                    ),
                                    "b34_controller_profile": (
                                        decision.b34_controller_profile
                                    ),
                                    "b34_eligibility_trace": (
                                        decision.b34_eligibility_trace
                                    ),
                                    "b34_credit_assignment": (
                                        decision.b34_credit_assignment
                                    ),
                                    "b34_synaptic_tag": decision.b34_synaptic_tag,
                                    "b34_decay_memory": decision.b34_decay_memory,
                                    "b34_credit_lock": decision.b34_credit_lock,
                                    "b34_decision": decision.b34_decision,
                                    "b34_genetic_generation": (
                                        decision.b34_genetic_generation
                                    ),
                                    "b34_genetic_candidate": (
                                        decision.b34_genetic_candidate
                                    ),
                                    "b35_controller_profile": (
                                        decision.b35_controller_profile
                                    ),
                                    "b35_forward_value": decision.b35_forward_value,
                                    "b35_transition_error": (
                                        decision.b35_transition_error
                                    ),
                                    "b35_model_confidence": (
                                        decision.b35_model_confidence
                                    ),
                                    "b35_prediction_memory": (
                                        decision.b35_prediction_memory
                                    ),
                                    "b35_model_lock": decision.b35_model_lock,
                                    "b35_decision": decision.b35_decision,
                                    "b35_genetic_generation": (
                                        decision.b35_genetic_generation
                                    ),
                                    "b35_genetic_candidate": (
                                        decision.b35_genetic_candidate
                                    ),
                                    "b36_controller_profile": (
                                        decision.b36_controller_profile
                                    ),
                                    "b36_latent_state": decision.b36_latent_state,
                                    "b36_belief_error": decision.b36_belief_error,
                                    "b36_state_confidence": (
                                        decision.b36_state_confidence
                                    ),
                                    "b36_context_memory": decision.b36_context_memory,
                                    "b36_belief_lock": decision.b36_belief_lock,
                                    "b36_decision": decision.b36_decision,
                                    "b36_genetic_generation": (
                                        decision.b36_genetic_generation
                                    ),
                                    "b36_genetic_candidate": (
                                        decision.b36_genetic_candidate
                                    ),
                                    "b37_controller_profile": (
                                        decision.b37_controller_profile
                                    ),
                                    "b37_external_state_factor": (
                                        decision.b37_external_state_factor
                                    ),
                                    "b37_internal_state_factor": (
                                        decision.b37_internal_state_factor
                                    ),
                                    "b37_factor_alignment": (
                                        decision.b37_factor_alignment
                                    ),
                                    "b37_factor_confidence": (
                                        decision.b37_factor_confidence
                                    ),
                                    "b37_factor_lock": decision.b37_factor_lock,
                                    "b37_decision": decision.b37_decision,
                                    "b37_genetic_generation": (
                                        decision.b37_genetic_generation
                                    ),
                                    "b37_genetic_candidate": (
                                        decision.b37_genetic_candidate
                                    ),
                                    "b38_controller_profile": (
                                        decision.b38_controller_profile
                                    ),
                                    "b38_external_attention": (
                                        decision.b38_external_attention
                                    ),
                                    "b38_internal_attention": (
                                        decision.b38_internal_attention
                                    ),
                                    "b38_attention_balance": (
                                        decision.b38_attention_balance
                                    ),
                                    "b38_attention_gain": decision.b38_attention_gain,
                                    "b38_attention_lock": decision.b38_attention_lock,
                                    "b38_decision": decision.b38_decision,
                                    "b38_genetic_generation": (
                                        decision.b38_genetic_generation
                                    ),
                                    "b38_genetic_candidate": (
                                        decision.b38_genetic_candidate
                                    ),
                                    "b39_controller_profile": (
                                        decision.b39_controller_profile
                                    ),
                                    "b39_binding_strength": (
                                        decision.b39_binding_strength
                                    ),
                                    "b39_cross_factor_coherence": (
                                        decision.b39_cross_factor_coherence
                                    ),
                                    "b39_bound_context": decision.b39_bound_context,
                                    "b39_binding_gain": decision.b39_binding_gain,
                                    "b39_binding_lock": decision.b39_binding_lock,
                                    "b39_decision": decision.b39_decision,
                                    "b39_genetic_generation": (
                                        decision.b39_genetic_generation
                                    ),
                                    "b39_genetic_candidate": (
                                        decision.b39_genetic_candidate
                                    ),
                                    "b40_controller_profile": (
                                        decision.b40_controller_profile
                                    ),
                                    "b40_workspace_activation": (
                                        decision.b40_workspace_activation
                                    ),
                                    "b40_broadcast_gain": (
                                        decision.b40_broadcast_gain
                                    ),
                                    "b40_context_availability": (
                                        decision.b40_context_availability
                                    ),
                                    "b40_workspace_stability": (
                                        decision.b40_workspace_stability
                                    ),
                                    "b40_workspace_lock": decision.b40_workspace_lock,
                                    "b40_decision": decision.b40_decision,
                                    "b40_genetic_generation": (
                                        decision.b40_genetic_generation
                                    ),
                                    "b40_genetic_candidate": (
                                        decision.b40_genetic_candidate
                                    ),
                                    "b41_controller_profile": (
                                        decision.b41_controller_profile
                                    ),
                                    "b41_executive_selection": (
                                        decision.b41_executive_selection
                                    ),
                                    "b41_inhibitory_pressure": (
                                        decision.b41_inhibitory_pressure
                                    ),
                                    "b41_goal_context": decision.b41_goal_context,
                                    "b41_executive_stability": (
                                        decision.b41_executive_stability
                                    ),
                                    "b41_executive_lock": decision.b41_executive_lock,
                                    "b41_decision": decision.b41_decision,
                                    "b41_genetic_generation": (
                                        decision.b41_genetic_generation
                                    ),
                                    "b41_genetic_candidate": (
                                        decision.b41_genetic_candidate
                                    ),
                                    "b42_controller_profile": (
                                        decision.b42_controller_profile
                                    ),
                                    "b42_error_signal": decision.b42_error_signal,
                                    "b42_conflict_signal": decision.b42_conflict_signal,
                                    "b42_performance_context": (
                                        decision.b42_performance_context
                                    ),
                                    "b42_monitor_stability": (
                                        decision.b42_monitor_stability
                                    ),
                                    "b42_monitor_lock": decision.b42_monitor_lock,
                                    "b42_decision": decision.b42_decision,
                                    "b42_genetic_generation": (
                                        decision.b42_genetic_generation
                                    ),
                                    "b42_genetic_candidate": (
                                        decision.b42_genetic_candidate
                                    ),
                                    "b43_controller_profile": (
                                        decision.b43_controller_profile
                                    ),
                                    "b43_precision_signal": (
                                        decision.b43_precision_signal
                                    ),
                                    "b43_adaptive_threshold": (
                                        decision.b43_adaptive_threshold
                                    ),
                                    "b43_arousal_context": (
                                        decision.b43_arousal_context
                                    ),
                                    "b43_control_stability": (
                                        decision.b43_control_stability
                                    ),
                                    "b43_precision_lock": decision.b43_precision_lock,
                                    "b43_decision": decision.b43_decision,
                                    "b43_genetic_generation": (
                                        decision.b43_genetic_generation
                                    ),
                                    "b43_genetic_candidate": (
                                        decision.b43_genetic_candidate
                                    ),
                                    "b44_controller_profile": (
                                        decision.b44_controller_profile
                                    ),
                                    "b44_relay_gate": decision.b44_relay_gate,
                                    "b44_sensory_precision": (
                                        decision.b44_sensory_precision
                                    ),
                                    "b44_context_relay": decision.b44_context_relay,
                                    "b44_gate_stability": decision.b44_gate_stability,
                                    "b44_relay_lock": decision.b44_relay_lock,
                                    "b44_decision": decision.b44_decision,
                                    "b44_genetic_generation": (
                                        decision.b44_genetic_generation
                                    ),
                                    "b44_genetic_candidate": (
                                        decision.b44_genetic_candidate
                                    ),
                                    "b45_controller_profile": (
                                        decision.b45_controller_profile
                                    ),
                                    "b45_inhibitory_gate": (
                                        decision.b45_inhibitory_gate
                                    ),
                                    "b45_sensory_filter": decision.b45_sensory_filter,
                                    "b45_context_suppression": (
                                        decision.b45_context_suppression
                                    ),
                                    "b45_loop_stability": decision.b45_loop_stability,
                                    "b45_inhibition_lock": decision.b45_inhibition_lock,
                                    "b45_decision": decision.b45_decision,
                                    "b45_genetic_generation": (
                                        decision.b45_genetic_generation
                                    ),
                                    "b45_genetic_candidate": (
                                        decision.b45_genetic_candidate
                                    ),
                                    "b46_controller_profile": (
                                        decision.b46_controller_profile
                                    ),
                                    "b46_feedback_gain": decision.b46_feedback_gain,
                                    "b46_topdown_context": (
                                        decision.b46_topdown_context
                                    ),
                                    "b46_prediction_match": (
                                        decision.b46_prediction_match
                                    ),
                                    "b46_feedback_stability": (
                                        decision.b46_feedback_stability
                                    ),
                                    "b46_feedback_lock": decision.b46_feedback_lock,
                                    "b46_decision": decision.b46_decision,
                                    "b46_genetic_generation": (
                                        decision.b46_genetic_generation
                                    ),
                                    "b46_genetic_candidate": (
                                        decision.b46_genetic_candidate
                                    ),
                                    "b47_controller_profile": (
                                        decision.b47_controller_profile
                                    ),
                                    "b47_phase_alignment": (
                                        decision.b47_phase_alignment
                                    ),
                                    "b47_synchrony_gain": (
                                        decision.b47_synchrony_gain
                                    ),
                                    "b47_cross_loop_coherence": (
                                        decision.b47_cross_loop_coherence
                                    ),
                                    "b47_phase_lock": decision.b47_phase_lock,
                                    "b47_decision": decision.b47_decision,
                                    "b47_genetic_generation": (
                                        decision.b47_genetic_generation
                                    ),
                                    "b47_genetic_candidate": (
                                        decision.b47_genetic_candidate
                                    ),
                                    "b48_controller_profile": (
                                        decision.b48_controller_profile
                                    ),
                                    "b48_timing_error": decision.b48_timing_error,
                                    "b48_predictive_timing": (
                                        decision.b48_predictive_timing
                                    ),
                                    "b48_corrective_gain": (
                                        decision.b48_corrective_gain
                                    ),
                                    "b48_calibration_lock": (
                                        decision.b48_calibration_lock
                                    ),
                                    "b48_decision": decision.b48_decision,
                                    "b48_genetic_generation": (
                                        decision.b48_genetic_generation
                                    ),
                                    "b48_genetic_candidate": (
                                        decision.b48_genetic_candidate
                                    ),
                                    "b49_controller_profile": (
                                        decision.b49_controller_profile
                                    ),
                                    "b49_go_signal": decision.b49_go_signal,
                                    "b49_no_go_signal": decision.b49_no_go_signal,
                                    "b49_action_gate_balance": (
                                        decision.b49_action_gate_balance
                                    ),
                                    "b49_selection_lock": decision.b49_selection_lock,
                                    "b49_decision": decision.b49_decision,
                                    "b49_genetic_generation": (
                                        decision.b49_genetic_generation
                                    ),
                                    "b49_genetic_candidate": (
                                        decision.b49_genetic_candidate
                                    ),
                                    "b50_controller_profile": (
                                        decision.b50_controller_profile
                                    ),
                                    "b50_habit_strength": (
                                        decision.b50_habit_strength
                                    ),
                                    "b50_chunk_value": decision.b50_chunk_value,
                                    "b50_habit_stability": (
                                        decision.b50_habit_stability
                                    ),
                                    "b50_chunk_lock": decision.b50_chunk_lock,
                                    "b50_decision": decision.b50_decision,
                                    "b50_genetic_generation": (
                                        decision.b50_genetic_generation
                                    ),
                                    "b50_genetic_candidate": (
                                        decision.b50_genetic_candidate
                                    ),
                                    "b51_controller_profile": (
                                        decision.b51_controller_profile
                                    ),
                                    "b51_prediction_error": (
                                        decision.b51_prediction_error
                                    ),
                                    "b51_dopamine_gain": decision.b51_dopamine_gain,
                                    "b51_habit_modulation": (
                                        decision.b51_habit_modulation
                                    ),
                                    "b51_modulation_lock": (
                                        decision.b51_modulation_lock
                                    ),
                                    "b51_decision": decision.b51_decision,
                                    "b51_genetic_generation": (
                                        decision.b51_genetic_generation
                                    ),
                                    "b51_genetic_candidate": (
                                        decision.b51_genetic_candidate
                                    ),
                                    "b52_controller_profile": (
                                        decision.b52_controller_profile
                                    ),
                                    "b52_acetylcholine_level": (
                                        decision.b52_acetylcholine_level
                                    ),
                                    "b52_precision_gain": decision.b52_precision_gain,
                                    "b52_uncertainty_signal": (
                                        decision.b52_uncertainty_signal
                                    ),
                                    "b52_attention_lock": (
                                        decision.b52_attention_lock
                                    ),
                                    "b52_decision": decision.b52_decision,
                                    "b52_genetic_generation": (
                                        decision.b52_genetic_generation
                                    ),
                                    "b52_genetic_candidate": (
                                        decision.b52_genetic_candidate
                                    ),
                                    "b53_controller_profile": (
                                        decision.b53_controller_profile
                                    ),
                                    "b53_norepinephrine_level": (
                                        decision.b53_norepinephrine_level
                                    ),
                                    "b53_arousal_gain": decision.b53_arousal_gain,
                                    "b53_surprise_signal": (
                                        decision.b53_surprise_signal
                                    ),
                                    "b53_gain_lock": decision.b53_gain_lock,
                                    "b53_decision": decision.b53_decision,
                                    "b53_genetic_generation": (
                                        decision.b53_genetic_generation
                                    ),
                                    "b53_genetic_candidate": (
                                        decision.b53_genetic_candidate
                                    ),
                                    "b54_controller_profile": (
                                        decision.b54_controller_profile
                                    ),
                                    "b54_serotonin_level": (
                                        decision.b54_serotonin_level
                                    ),
                                    "b54_patience_signal": (
                                        decision.b54_patience_signal
                                    ),
                                    "b54_impulse_suppression": (
                                        decision.b54_impulse_suppression
                                    ),
                                    "b54_patience_lock": decision.b54_patience_lock,
                                    "b54_decision": decision.b54_decision,
                                    "b54_genetic_generation": (
                                        decision.b54_genetic_generation
                                    ),
                                    "b54_genetic_candidate": (
                                        decision.b54_genetic_candidate
                                    ),
                                    "b55_controller_profile": (
                                        decision.b55_controller_profile
                                    ),
                                    "b55_hypothalamic_drive": (
                                        decision.b55_hypothalamic_drive
                                    ),
                                    "b55_satiety_signal": decision.b55_satiety_signal,
                                    "b55_recovery_bias": decision.b55_recovery_bias,
                                    "b55_drive_balance": decision.b55_drive_balance,
                                    "b55_drive_lock": decision.b55_drive_lock,
                                    "b55_decision": decision.b55_decision,
                                    "b55_genetic_generation": (
                                        decision.b55_genetic_generation
                                    ),
                                    "b55_genetic_candidate": (
                                        decision.b55_genetic_candidate
                                    ),
                                    "b56_controller_profile": (
                                        decision.b56_controller_profile
                                    ),
                                    "b56_cortisol_level": decision.b56_cortisol_level,
                                    "b56_stress_load": decision.b56_stress_load,
                                    "b56_recovery_signal": (
                                        decision.b56_recovery_signal
                                    ),
                                    "b56_endocrine_balance": (
                                        decision.b56_endocrine_balance
                                    ),
                                    "b56_stress_lock": decision.b56_stress_lock,
                                    "b56_decision": decision.b56_decision,
                                    "b56_genetic_generation": (
                                        decision.b56_genetic_generation
                                    ),
                                    "b56_genetic_candidate": (
                                        decision.b56_genetic_candidate
                                    ),
                                    "b57_controller_profile": (
                                        decision.b57_controller_profile
                                    ),
                                    "b57_interoceptive_awareness": (
                                        decision.b57_interoceptive_awareness
                                    ),
                                    "b57_visceral_salience": (
                                        decision.b57_visceral_salience
                                    ),
                                    "b57_body_state_confidence": (
                                        decision.b57_body_state_confidence
                                    ),
                                    "b57_awareness_balance": (
                                        decision.b57_awareness_balance
                                    ),
                                    "b57_awareness_lock": decision.b57_awareness_lock,
                                    "b57_decision": decision.b57_decision,
                                    "b57_genetic_generation": (
                                        decision.b57_genetic_generation
                                    ),
                                    "b57_genetic_candidate": (
                                        decision.b57_genetic_candidate
                                    ),
                                    "b58_controller_profile": (
                                        decision.b58_controller_profile
                                    ),
                                    "b58_conflict_signal": decision.b58_conflict_signal,
                                    "b58_error_likelihood": (
                                        decision.b58_error_likelihood
                                    ),
                                    "b58_control_allocation": (
                                        decision.b58_control_allocation
                                    ),
                                    "b58_resolution_balance": (
                                        decision.b58_resolution_balance
                                    ),
                                    "b58_conflict_lock": decision.b58_conflict_lock,
                                    "b58_decision": decision.b58_decision,
                                    "b58_genetic_generation": (
                                        decision.b58_genetic_generation
                                    ),
                                    "b58_genetic_candidate": (
                                        decision.b58_genetic_candidate
                                    ),
                                    "b59_controller_profile": (
                                        decision.b59_controller_profile
                                    ),
                                    "b59_goal_context": decision.b59_goal_context,
                                    "b59_working_set_stability": (
                                        decision.b59_working_set_stability
                                    ),
                                    "b59_task_set_confidence": (
                                        decision.b59_task_set_confidence
                                    ),
                                    "b59_executive_balance": (
                                        decision.b59_executive_balance
                                    ),
                                    "b59_executive_lock": decision.b59_executive_lock,
                                    "b59_decision": decision.b59_decision,
                                    "b59_genetic_generation": (
                                        decision.b59_genetic_generation
                                    ),
                                    "b59_genetic_candidate": (
                                        decision.b59_genetic_candidate
                                    ),
                                    "b60_controller_profile": (
                                        decision.b60_controller_profile
                                    ),
                                    "b60_outcome_value": decision.b60_outcome_value,
                                    "b60_reversal_signal": decision.b60_reversal_signal,
                                    "b60_goal_value_confidence": (
                                        decision.b60_goal_value_confidence
                                    ),
                                    "b60_value_balance": decision.b60_value_balance,
                                    "b60_value_lock": decision.b60_value_lock,
                                    "b60_decision": decision.b60_decision,
                                    "b60_genetic_generation": (
                                        decision.b60_genetic_generation
                                    ),
                                    "b60_genetic_candidate": (
                                        decision.b60_genetic_candidate
                                    ),
                                    "b61_controller_profile": (
                                        decision.b61_controller_profile
                                    ),
                                    "b61_safety_value": decision.b61_safety_value,
                                    "b61_threat_value": decision.b61_threat_value,
                                    "b61_safety_confidence": (
                                        decision.b61_safety_confidence
                                    ),
                                    "b61_affective_balance": (
                                        decision.b61_affective_balance
                                    ),
                                    "b61_safety_lock": decision.b61_safety_lock,
                                    "b61_decision": decision.b61_decision,
                                    "b61_genetic_generation": (
                                        decision.b61_genetic_generation
                                    ),
                                    "b61_genetic_candidate": (
                                        decision.b61_genetic_candidate
                                    ),
                                    "b62_controller_profile": (
                                        decision.b62_controller_profile
                                    ),
                                    "b62_defensive_mode": decision.b62_defensive_mode,
                                    "b62_freeze_pressure": (
                                        decision.b62_freeze_pressure
                                    ),
                                    "b62_flee_pressure": decision.b62_flee_pressure,
                                    "b62_shelter_bias": decision.b62_shelter_bias,
                                    "b62_defense_balance": (
                                        decision.b62_defense_balance
                                    ),
                                    "b62_defense_lock": decision.b62_defense_lock,
                                    "b62_decision": decision.b62_decision,
                                    "b62_genetic_generation": (
                                        decision.b62_genetic_generation
                                    ),
                                    "b62_genetic_candidate": (
                                        decision.b62_genetic_candidate
                                    ),
                                    "semantic_action": decision.semantic_action,
                                    "semantic_action_idx": int(
                                        decision.semantic_action_idx
                                    ),
                                    "learned_semantic_action": (
                                        decision.learned_semantic_action
                                    ),
                                    "learned_semantic_action_idx": int(
                                        decision.learned_semantic_action_idx
                                    ),
                                    "semantic_action_source": (
                                        decision.semantic_action_source
                                    ),
                                    "semantic_action_reason": (
                                        decision.semantic_action_reason
                                    ),
                                    "semantic_override_count": int(
                                        decision.semantic_override_count
                                    ),
                                    "semantic_logits": decision.semantic_logits.round(
                                        6
                                    ).tolist(),
                                    "semantic_policy": decision.semantic_policy.round(
                                        6
                                    ).tolist(),
                                    "bridge_primitive_action": (
                                        decision.bridge_primitive_action
                                    ),
                                    "bridge_reason": decision.bridge_reason,
                                    "blocked_mask": dict(decision.blocked_mask),
                                    "food_delta_used": round(
                                        float(decision.food_delta_used),
                                        6,
                                    ),
                                    "shelter_delta_used": round(
                                        float(decision.shelter_delta_used),
                                        6,
                                    ),
                                    "external_override_count": int(
                                        decision.external_override_count
                                    ),
                                },
                            }
                            if self.brain.config.is_b_series
                            else {}
                        ),
                        "final_reflex_override": bool(decision.final_reflex_override),
                        "total_logits_without_reflex": decision.total_logits_without_reflex.round(6).tolist(),
                        "total_logits": decision.total_logits.round(6).tolist(),
                        "vision": next_observation["meta"].get("vision", {}),
                        "predator": {
                            "mode": self.world.lizard.mode,
                            "mode_ticks": self.world.lizard.mode_ticks,
                            "patrol_target": self.world.lizard.patrol_target,
                            "last_known_spider": self.world.lizard.last_known_spider,
                            "investigate_ticks": self.world.lizard.investigate_ticks,
                            "investigate_target": self.world.lizard.investigate_target,
                            "recover_ticks": self.world.lizard.recover_ticks,
                            "wait_target": self.world.lizard.wait_target,
                            "ambush_ticks": self.world.lizard.ambush_ticks,
                            "chase_streak": self.world.lizard.chase_streak,
                            "failed_chases": self.world.lizard.failed_chases,
                        },
                    }
                trace.append(item)

            total_reward += reward
            observation = next_observation

            if render:
                print(self.world.render())
                print("-" * self.world.width)

            if done:
                break

        state = self.world.state
        stats = metrics.finalize(
            episode=episode_index,
            seed=episode_seed,
            training=training,
            scenario=normalized_scenario,
            total_reward=float(total_reward),
            state=state,
        )
        return stats, trace
