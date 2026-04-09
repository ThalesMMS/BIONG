from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple


if TYPE_CHECKING:
    from .world import SpiderWorld


PREDATOR_STATES: Sequence[str] = (
    "PATROL",
    "ORIENT",
    "INVESTIGATE",
    "CHASE",
    "WAIT",
    "RECOVER",
)


@dataclass
class LizardState:
    x: int
    y: int
    mode: str = "PATROL"
    mode_ticks: int = 0
    patrol_target: Tuple[int, int] | None = None
    last_known_spider: Tuple[int, int] | None = None
    investigate_ticks: int = 0
    investigate_target: Tuple[int, int] | None = None
    recover_ticks: int = 0
    wait_target: Tuple[int, int] | None = None
    ambush_ticks: int = 0
    chase_streak: int = 0
    failed_chases: int = 0


class PredatorController:
    """Explicit lizard FSM that keeps the logic outside `world.py`."""

    ORIENT_TICKS = 1
    INVESTIGATE_MOVES = 4
    WAIT_TICKS = 3
    RECOVER_MOVES = 4
    MAX_FAILED_CHASES = 3

    def update(self, world: "SpiderWorld") -> bool:
        """
        Advance the predator's finite-state machine for a single game tick, handling detection-driven transitions, target priming, movement scheduling, and post-move bookkeeping.
        
        This updates the lizard's mode, targeting fields, and counters (e.g., `chase_streak`, `failed_chases`, `ambush_ticks`, `recover_ticks`, `mode_ticks`) based on whether the spider is detected and the current mode, may change mode via the controller's helper methods, and attempts a scheduled movement when allowed. It also applies mode-specific post-move rules for `WAIT` and `RECOVER`.
        
        Parameters:
            world (SpiderWorld): The game world used to read and modify the lizard state, query detection/positions, and perform movement.
        
        Returns:
            bool: `True` if the lizard moved during this tick, `False` otherwise.
        """
        lizard = world.lizard
        detected = world.lizard_detects_spider()

        if detected:
            self._prime_targets(world, world.spider_pos())
            if lizard.mode in {"PATROL", "WAIT", "RECOVER"}:
                self._set_mode(lizard, "ORIENT")
            elif lizard.mode in {"ORIENT", "INVESTIGATE"}:
                self._set_mode(lizard, "CHASE")
                lizard.chase_streak = max(1, lizard.chase_streak)
            elif lizard.mode == "CHASE":
                lizard.chase_streak += 1
        else:
            if lizard.mode == "ORIENT" and lizard.mode_ticks >= self.ORIENT_TICKS:
                self._enter_investigation(world)
            elif lizard.mode == "CHASE":
                self._handle_lost_chase(world)
            elif (
                lizard.mode == "INVESTIGATE"
                and lizard.investigate_ticks >= self._investigate_budget(lizard)
            ):
                self._enter_wait_or_recover(world)

        moved = False
        if self._can_move_this_tick(world):
            moved = self._step_mode(world)

        if lizard.mode == "WAIT":
            if lizard.wait_target is not None and world.manhattan(world.lizard_pos(), lizard.wait_target) <= 1:
                if moved:
                    lizard.mode_ticks = 0
                    lizard.ambush_ticks = 0
                else:
                    lizard.ambush_ticks += 1
                if lizard.ambush_ticks >= self.WAIT_TICKS + min(2, lizard.failed_chases):
                    if lizard.recover_ticks > 0:
                        self._set_mode(lizard, "RECOVER")
                    else:
                        lizard.failed_chases = max(0, lizard.failed_chases - 1)
                        lizard.chase_streak = 0
                        self._set_mode(lizard, "PATROL")
        elif lizard.mode == "RECOVER" and lizard.recover_ticks <= 0:
            lizard.failed_chases = max(0, lizard.failed_chases - 1)
            lizard.chase_streak = 0
            self._set_mode(lizard, "PATROL")

        lizard.mode_ticks += 1
        return moved

    def _set_mode(self, lizard: LizardState, mode: str) -> None:
        """
        Set the lizard's finite-state mode and reset or clear state fields that depend on the new mode.
        
        If `mode` is different from the current mode, updates `lizard.mode` and resets `lizard.mode_ticks` to 0, then:
        - clears `lizard.patrol_target` when the new mode is not "PATROL";
        - clears `lizard.wait_target` when the new mode is not "WAIT" or "RECOVER";
        - resets `lizard.ambush_ticks` to 0 when the new mode is not "WAIT";
        - resets `lizard.investigate_ticks` to 0 and clears `lizard.investigate_target` when the new mode is not "INVESTIGATE".
        
        Parameters:
            lizard (LizardState): The lizard state to modify.
            mode (str): Target FSM mode; must be one of PREDATOR_STATES.
        
        Raises:
            ValueError: If `mode` is not a valid predator state.
        """
        if mode not in PREDATOR_STATES:
            raise ValueError(f"Invalid predator mode: {mode}")
        if lizard.mode != mode:
            lizard.mode = mode
            lizard.mode_ticks = 0
            if mode != "PATROL":
                lizard.patrol_target = None
            if mode not in {"WAIT", "RECOVER"}:
                lizard.wait_target = None
            if mode != "WAIT":
                lizard.ambush_ticks = 0
            if mode != "INVESTIGATE":
                lizard.investigate_ticks = 0
                lizard.investigate_target = None

    def _can_move_this_tick(self, world: "SpiderWorld") -> bool:
        """
        Decides whether the lizard is allowed to move on the current world tick.
        
        Returns:
            `true` if the lizard may move this tick according to the world's move interval (the interval is doubled while the lizard's mode is `RECOVER`), `false` otherwise.
        """
        interval = world.lizard_move_interval
        if world.lizard.mode == "RECOVER":
            interval *= 2
        return world.tick % max(1, interval) == 0

    def _candidate_moves(self, world: "SpiderWorld") -> List[Tuple[int, int]]:
        seen: set[Tuple[int, int]] = set()
        candidates: List[Tuple[int, int]] = []
        for dx, dy in world.move_deltas:
            nx = max(0, min(world.width - 1, world.lizard.x + dx))
            ny = max(0, min(world.height - 1, world.lizard.y + dy))
            candidate = (nx, ny)
            if candidate == world.lizard_pos() or candidate in seen:
                continue
            if not world.is_walkable(candidate):
                continue
            if not world.is_lizard_walkable(candidate):
                continue
            seen.add(candidate)
            candidates.append(candidate)
        return candidates

    def _step_towards(self, world: "SpiderWorld", target: Tuple[int, int]) -> bool:
        """
        Move the lizard one grid step toward the given target coordinate.
        
        Selects valid neighboring moves, ranks them by Manhattan distance to the target (closer preferred), prefers cells with terrain `"OPEN"`, and uses the position as a final tie-break. If multiple moves share the best rank, optionally choose randomly among them according to world.noise_profile.predator["random_choice_prob"] using world.predator_rng. Updates world.lizard.x and world.lizard.y when a move is made.
        
        Parameters:
            world (SpiderWorld): Game world used for move validation, ranking, and RNG; the lizard position is updated on success.
            target (Tuple[int, int]): (x, y) coordinate to approach.
        
        Returns:
            bool: `True` if the lizard moved to a new position, `False` otherwise.
        """
        candidates = self._candidate_moves(world)
        if not candidates:
            return False
        ranked = sorted(
            candidates,
            key=lambda pos: (
                world.manhattan(pos, target),
                0 if world.terrain_at(pos) == "OPEN" else 1,
                pos,
            ),
        )
        best_key = (
            world.manhattan(ranked[0], target),
            0 if world.terrain_at(ranked[0]) == "OPEN" else 1,
        )
        best_candidates = [
            pos
            for pos in ranked
            if (
                world.manhattan(pos, target),
                0 if world.terrain_at(pos) == "OPEN" else 1,
            )
            == best_key
        ]
        random_choice_prob = min(
            1.0,
            max(0.0, float(world.noise_profile.predator["random_choice_prob"])),
        )
        if (
            len(best_candidates) > 1
            and float(world.predator_rng.random()) < random_choice_prob
        ):
            choice_idx = int(world.predator_rng.integers(0, len(best_candidates)))
            choice = best_candidates[choice_idx]
        else:
            choice = ranked[0]
        world.lizard.x, world.lizard.y = choice
        return True

    def _pick_patrol_target(
        self,
        world: "SpiderWorld",
        *,
        retreat_from: Tuple[int, int] | None = None,
    ) -> Tuple[int, int] | None:
        """
        Choose a patrol target from available lizard spawn cells, preferring open terrain and applying distance-based ordering.
        
        If `retreat_from` is provided, candidates that increase Manhattan distance from that anchor are preferred; otherwise candidates closer to the current spider position are preferred. Ties among equally-best candidates may be broken randomly according to the predator noise profile.
        
        Parameters:
            retreat_from (Tuple[int, int] | None): If provided, prefer targets that increase distance from this position.
        
        Returns:
            Tuple[int, int] | None: The selected patrol target coordinate, or `None` if no valid spawn cell exists.
        """
        current = world.lizard_pos()
        candidates = [
            pos
            for pos in world.map_template.lizard_spawn_cells
            if pos != current and world.is_lizard_walkable(pos)
        ]
        if not candidates:
            return None
        if retreat_from is not None:
            candidates.sort(
                key=lambda pos: (
                    -world.manhattan(pos, retreat_from),
                    0 if world.terrain_at(pos) == "OPEN" else 1,
                    pos,
                )
            )
        else:
            candidates.sort(
                key=lambda pos: (
                    0 if world.terrain_at(pos) == "OPEN" else 1,
                    -world.manhattan(pos, world.spider_pos()),
                    pos,
                )
            )
        if retreat_from is not None:
            best_key = (
                -world.manhattan(candidates[0], retreat_from),
                0 if world.terrain_at(candidates[0]) == "OPEN" else 1,
            )
            best_candidates = [
                pos
                for pos in candidates
                if (
                    -world.manhattan(pos, retreat_from),
                    0 if world.terrain_at(pos) == "OPEN" else 1,
                )
                == best_key
            ]
        else:
            best_key = (
                0 if world.terrain_at(candidates[0]) == "OPEN" else 1,
                -world.manhattan(candidates[0], world.spider_pos()),
            )
            best_candidates = [
                pos
                for pos in candidates
                if (
                    0 if world.terrain_at(pos) == "OPEN" else 1,
                    -world.manhattan(pos, world.spider_pos()),
                )
                == best_key
            ]
        random_choice_prob = min(
            1.0,
            max(0.0, float(world.noise_profile.predator["random_choice_prob"])),
        )
        if (
            len(best_candidates) > 1
            and float(world.predator_rng.random()) < random_choice_prob
        ):
            idx = int(world.predator_rng.integers(0, len(best_candidates)))
            return best_candidates[idx]
        return candidates[0]

    def _prime_targets(self, world: "SpiderWorld", target: Tuple[int, int]) -> None:
        """
        Set the lizard's targeting fields from a detected spider position.
        
        Updates the lizard's `last_known_spider` to `target`, sets `wait_target` to the nearest shelter entrance from `target`, and selects an `investigate_target` that is reachable from `target`.
        Parameters:
            target (Tuple[int, int]): Coordinates (x, y) of the detected spider.
        """
        lizard = world.lizard
        lizard.last_known_spider = target
        lizard.wait_target = world.nearest_shelter_entrance(origin=target)
        lizard.investigate_target = self._reachable_probe_target(world, target)

    def _reachable_probe_target(
        self,
        world: "SpiderWorld",
        anchor: Tuple[int, int],
    ) -> Tuple[int, int] | None:
        """
        Selects a reachable probe position near an anchor to use as an investigation target.
        
        Parameters:
            world (SpiderWorld): The simulation world providing walkability, terrain, distances, and shelter queries.
            anchor (Tuple[int,int]): The reference position (typically the last-known spider location) to prioritize proximity.
        
        Returns:
            Tuple[int,int] | None: A walkable grid cell chosen as the probe target, preferring cells that are closer to `anchor`, then closer to the nearest shelter entrance from `anchor`, and preferring `OPEN` terrain; returns `None` if no walkable cells exist.
        """
        wait_target = world.nearest_shelter_entrance(origin=anchor)
        candidates = [
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if world.is_lizard_walkable((x, y))
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda pos: (
                world.manhattan(pos, anchor),
                world.manhattan(pos, wait_target) if wait_target is not None else 0,
                0 if world.terrain_at(pos) == "OPEN" else 1,
            )
        )
        return candidates[0]

    def _investigate_budget(self, lizard: LizardState) -> int:
        """
        Compute the maximum number of ticks/moves the lizard may spend investigating.
        
        Parameters:
            lizard (LizardState): The lizard's state whose `failed_chases` and `chase_streak` influence the extra investigation allowance.
        
        Returns:
            int: Maximum investigate duration (INVESTIGATE_MOVES plus up to 3 additional ticks derived from `lizard.failed_chases` and half of `lizard.chase_streak`).
        """
        return self.INVESTIGATE_MOVES + min(3, lizard.failed_chases + lizard.chase_streak // 2)

    def _enter_investigation(self, world: "SpiderWorld") -> None:
        """
        Transition the lizard into an investigation-related mode and prime targets based on the last known spider position.
        
        If the lizard has no last known spider position, switches to PATROL. Otherwise primes targeting fields from that position and selects the next mode in priority order: INVESTIGATE (if a reachable investigate target exists), WAIT (if a shelter wait target exists), RECOVER (if recover time remains), or PATROL.
        
        Parameters:
            world (SpiderWorld): The simulation world providing the lizard state and target computation helpers.
        """
        lizard = world.lizard
        if lizard.last_known_spider is None:
            self._set_mode(lizard, "PATROL")
            return
        self._prime_targets(world, lizard.last_known_spider)
        if lizard.investigate_target is not None:
            self._set_mode(lizard, "INVESTIGATE")
        elif lizard.wait_target is not None:
            self._set_mode(lizard, "WAIT")
        elif lizard.recover_ticks > 0:
            self._set_mode(lizard, "RECOVER")
        else:
            self._set_mode(lizard, "PATROL")

    def _handle_lost_chase(self, world: "SpiderWorld") -> None:
        """
        Handle transition when the lizard loses sight of the spider by updating recovery counters and selecting the next mode.
        
        Increments the lizard's `failed_chases`, ensures `recover_ticks` is sufficiently long based on recent chase history, and re-primes targets if a last-known spider position exists. Chooses the next FSM mode in the following priority: set to `WAIT` when the last-known position maps to a shelter and a `wait_target` is available; otherwise `INVESTIGATE` if an `investigate_target` exists; otherwise `RECOVER` if `recover_ticks > 0`; otherwise `PATROL`.
        """
        lizard = world.lizard
        lizard.failed_chases = min(self.MAX_FAILED_CHASES, lizard.failed_chases + 1)
        lizard.recover_ticks = max(
            lizard.recover_ticks,
            self.RECOVER_MOVES + min(3, max(0, lizard.chase_streak - 1) + lizard.failed_chases),
        )
        if lizard.last_known_spider is not None:
            self._prime_targets(world, lizard.last_known_spider)
        target_role = (
            world.shelter_role_at(lizard.last_known_spider)
            if lizard.last_known_spider is not None
            else "outside"
        )
        if target_role != "outside" and lizard.wait_target is not None:
            self._set_mode(lizard, "WAIT")
        elif lizard.investigate_target is not None:
            self._set_mode(lizard, "INVESTIGATE")
        elif lizard.recover_ticks > 0:
            self._set_mode(lizard, "RECOVER")
        else:
            self._set_mode(lizard, "PATROL")

    def _enter_wait_or_recover(self, world: "SpiderWorld") -> None:
        """
        Choose the next mode after an investigation ends, priming targets if a last-known spider position exists.
        
        Primes targeting fields from the lizard's last known spider (if present) and then sets the lizard's mode to:
        - "WAIT" if a wait target is available,
        - otherwise "RECOVER" if recover ticks remain,
        - otherwise "PATROL".
        
        Parameters:
            world (SpiderWorld): Simulation world providing the lizard state and target computations.
        """
        lizard = world.lizard
        if lizard.last_known_spider is not None:
            self._prime_targets(world, lizard.last_known_spider)
        if lizard.wait_target is not None:
            self._set_mode(lizard, "WAIT")
        elif lizard.recover_ticks > 0:
            self._set_mode(lizard, "RECOVER")
        else:
            self._set_mode(lizard, "PATROL")

    def _step_mode(self, world: "SpiderWorld") -> bool:
        """
        Advance the lizard one step according to its current finite-state-machine mode.
        
        Per the active mode, this will attempt to move the lizard toward an appropriate target (patrol destination, spider position, investigation probe, wait shelter, or recovery patrol point), update or select targets and mode-specific counters, and trigger mode transitions when conditions are met.
        
        Parameters:
            world (SpiderWorld): Simulation state used to evaluate walkability, targets, and to perform the move.
        
        Returns:
            bool: `True` if the lizard moved during this call, `False` otherwise.
        """
        lizard = world.lizard
        if lizard.mode == "PATROL":
            if lizard.patrol_target is None or world.lizard_pos() == lizard.patrol_target:
                lizard.patrol_target = self._pick_patrol_target(world)
            if lizard.patrol_target is None:
                return False
            moved = self._step_towards(world, lizard.patrol_target)
            if moved and world.lizard_pos() == lizard.patrol_target:
                lizard.patrol_target = self._pick_patrol_target(world)
            return moved

        if lizard.mode == "CHASE":
            return self._step_towards(world, world.spider_pos())

        if lizard.mode == "INVESTIGATE":
            if lizard.last_known_spider is None:
                self._set_mode(lizard, "PATROL")
                return False
            if lizard.investigate_target is None:
                lizard.investigate_target = self._reachable_probe_target(world, lizard.last_known_spider)
            if lizard.investigate_target is None:
                self._enter_wait_or_recover(world)
                return False
            moved = self._step_towards(world, lizard.investigate_target)
            lizard.investigate_ticks += 1
            if world.lizard_pos() == lizard.investigate_target:
                self._enter_wait_or_recover(world)
            return moved

        if lizard.mode == "WAIT":
            if lizard.wait_target is None:
                return False
            if world.manhattan(world.lizard_pos(), lizard.wait_target) <= 1:
                return False
            moved = self._step_towards(world, lizard.wait_target)
            if moved and world.manhattan(world.lizard_pos(), lizard.wait_target) <= 1:
                lizard.mode_ticks = 0
            return moved

        if lizard.mode == "RECOVER":
            lizard.recover_ticks = max(0, lizard.recover_ticks - 1)
            if lizard.patrol_target is None or world.lizard_pos() == lizard.patrol_target:
                lizard.patrol_target = self._pick_patrol_target(
                    world,
                    retreat_from=lizard.wait_target or lizard.last_known_spider,
                )
            if lizard.patrol_target is None:
                return False
            return self._step_towards(world, lizard.patrol_target)

        return False
