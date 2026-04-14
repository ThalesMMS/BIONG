from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple

from .perception import predator_detects_spider


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


@dataclass(frozen=True)
class PredatorProfile:
    name: str
    vision_range: int
    smell_range: int
    detection_style: str
    move_interval: int
    detection_threshold: float

    def __post_init__(self) -> None:
        """
        Validate that the profile's detection_style is either 'visual' or 'olfactory'.
        
        Raises:
            ValueError: If `detection_style` is not 'visual' or 'olfactory'.
        """
        if self.detection_style not in {"visual", "olfactory"}:
            raise ValueError(
                "PredatorProfile.detection_style must be 'visual' or 'olfactory'."
            )


DEFAULT_LIZARD_PROFILE = PredatorProfile(
    name="lizard",
    vision_range=3,
    smell_range=7,
    detection_style="visual",
    move_interval=2,
    detection_threshold=0.45,
)

VISUAL_HUNTER_PROFILE = PredatorProfile(
    name="visual_hunter",
    vision_range=6,
    smell_range=2,
    detection_style="visual",
    move_interval=2,
    detection_threshold=0.45,
)

OLFACTORY_HUNTER_PROFILE = PredatorProfile(
    name="olfactory_hunter",
    vision_range=3,
    smell_range=6,
    detection_style="olfactory",
    move_interval=2,
    detection_threshold=0.45,
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
    profile: PredatorProfile | None = None


class PredatorController:
    """Explicit lizard FSM that keeps the logic outside `world.py`."""

    ORIENT_TICKS = 1
    INVESTIGATE_MOVES = 4
    WAIT_TICKS = 3
    RECOVER_MOVES = 4
    MAX_FAILED_CHASES = 3

    def __init__(self, predator_index: int = 0) -> None:
        """
        Initialize the PredatorController with the index of the predator it will control.
        
        Parameters:
            predator_index (int): Index of the predator entity in the world (converted to int).
        """
        self.predator_index = int(predator_index)

    def _predator(self, world: "SpiderWorld") -> LizardState:
        """
        Retrieve the LizardState for this controller's predator from the given world.
        
        Returns:
            The LizardState instance corresponding to this controller's predator_index.
        """
        return world.get_predator(self.predator_index)

    def _predator_pos(self, world: "SpiderWorld") -> tuple[int, int]:
        """
        Get the controlled predator's current grid coordinates.
        
        Returns:
            tuple[int, int]: (x, y) coordinates of the predator within the world.
        """
        predator = self._predator(world)
        return predator.x, predator.y

    def update(self, world: "SpiderWorld") -> bool:
        """
        Advance the predator's finite-state machine by one simulation tick.
        
        Updates the predator's mode, targeting fields, and internal counters (for example: chase_streak, failed_chases, ambush_ticks, recover_ticks, mode_ticks); primes targets on detection, performs mode-driven transitions, schedules and attempts movement when allowed, and applies mode-specific post-move rules for WAIT and RECOVER.
        
        Parameters:
            world (SpiderWorld): The game world used to read and modify the predator state and query positions/detection.
        
        Returns:
            bool: True if the predator moved during this tick, False otherwise.
        """
        lizard = self._predator(world)
        detected = self._detect_spider(world)

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
            if lizard.wait_target is not None and world.manhattan(self._predator_pos(world), lizard.wait_target) <= 1:
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
        Determine whether this predator may move on the current world tick.
        
        Uses the resolved predator profile's move_interval; when the predator's mode is "RECOVER" the interval is doubled.
        
        Returns:
            True if the predator may move this tick, False otherwise.
        """
        interval = self._resolved_profile(world).move_interval
        if self._predator(world).mode == "RECOVER":
            interval *= 2
        return world.tick % max(1, interval) == 0

    def _world_profile(self, world: "SpiderWorld") -> PredatorProfile:
        """
        Constructs a PredatorProfile for the lizard using the world's operational perception and lizard-specific configuration.
        
        The returned profile uses:
        - name "lizard"
        - vision and smell ranges from world.lizard_vision_range and world.predator_smell_range (cast to int)
        - detection_style set to "visual"
        - move_interval taken from world.lizard_move_interval and clamped to at least 1
        - detection_threshold read from world.operational_profile.perception["lizard_detection_threshold"] (cast to float)
        
        Returns:
            PredatorProfile: A profile populated from the world's lizard configuration.
        """
        cfg = world.operational_profile.perception
        return PredatorProfile(
            name="lizard",
            vision_range=int(world.lizard_vision_range),
            smell_range=int(world.predator_smell_range),
            detection_style="visual",
            move_interval=max(1, int(world.lizard_move_interval)),
            detection_threshold=float(cfg["lizard_detection_threshold"]),
        )

    def _resolved_profile(self, world: "SpiderWorld") -> PredatorProfile:
        """
        Resolve the effective predator profile for the controlled lizard.
        
        Parameters:
            world (SpiderWorld): Simulation world used to build a world-derived profile when the predator has no explicit profile.
        
        Returns:
            PredatorProfile: The predator's explicit profile when one is set; otherwise a profile constructed from world configuration.
        """
        profile = self._predator(world).profile
        if profile is None:
            return self._world_profile(world)
        return profile

    def _explicit_profile(self, world: "SpiderWorld") -> PredatorProfile | None:
        """
        Return the predator's explicit profile, if one is set.
        
        Parameters:
            world (SpiderWorld): Simulation world containing the predator.
        
        Returns:
            PredatorProfile | None: The predator's explicit profile, or `None` if the predator has no profile.
        """
        profile = self._predator(world).profile
        if profile is None:
            return None
        return profile

    def _targeting_radius(self, world: "SpiderWorld") -> int | None:
        """
        Determine the explicit targeting radius derived from the predator's profile.
        
        If the predator has an explicit profile, returns the larger of its vision and smell ranges, coerced to an integer and clamped to at least 1. If no explicit profile is set, returns None.
        
        Returns:
            int | None: Radius in cells (at least 1) when an explicit profile exists, `None` otherwise.
        """
        profile = self._explicit_profile(world)
        if profile is None:
            return None
        return max(1, int(max(profile.vision_range, profile.smell_range)))

    def _detect_spider(self, world: "SpiderWorld") -> bool:
        """
        Determine whether this controller's predator currently detects the spider in the given world.
        
        Returns:
            True if the predator detects the spider, False otherwise.
        """
        return predator_detects_spider(world, self._predator(world))

    def _candidate_moves(self, world: "SpiderWorld") -> List[Tuple[int, int]]:
        """
        Produce one-step candidate destination coordinates for the controlled predator.
        
        Clamps moves to map bounds and returns positions reachable in a single step from the predator's current location while excluding: the predator's current cell, duplicate positions, cells occupied by other predators, non-walkable cells, and cells not suitable for lizards.
        
        Parameters:
            world (SpiderWorld): The game world containing map, occupancy, and movement deltas.
        
        Returns:
            List[Tuple[int, int]]: A list of candidate (x, y) coordinates, in the order discovered from world.move_deltas.
        """
        seen: set[Tuple[int, int]] = set()
        candidates: List[Tuple[int, int]] = []
        predator = self._predator(world)
        other_predator_positions = set(world.predator_positions())
        other_predator_positions.discard(self._predator_pos(world))
        for dx, dy in world.move_deltas:
            nx = max(0, min(world.width - 1, predator.x + dx))
            ny = max(0, min(world.height - 1, predator.y + dy))
            candidate = (nx, ny)
            if candidate == self._predator_pos(world) or candidate in seen:
                continue
            if candidate in other_predator_positions:
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
        Move the controlled predator one step toward the given target coordinate.
        
        Attempts to choose a valid neighboring cell that reduces Manhattan distance to target, preferring cells on terrain labeled "OPEN" and breaking ties deterministically except when the world's predator noise profile permits a random choice. On success updates the predator's stored (x, y) position.
        
        Parameters:
            world (SpiderWorld): World context used for move validation, terrain checks, and RNG.
            target (Tuple[int, int]): (x, y) coordinate to approach.
        
        Returns:
            bool: `True` if the predator moved to a new position, `False` otherwise.
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
        predator = self._predator(world)
        predator.x, predator.y = choice
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
        current = self._predator_pos(world)
        occupied_spawn_cells = set(world.predator_positions())
        candidates = [
            pos
            for pos in world.map_template.lizard_spawn_cells
            if pos != current
            and pos not in occupied_spawn_cells
            and world.is_lizard_walkable(pos)
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
        lizard = self._predator(world)
        target_radius = self._targeting_radius(world)
        lizard.last_known_spider = target
        lizard.wait_target = self._nearest_wait_target(world, target, radius=target_radius)
        lizard.investigate_target = self._reachable_probe_target(
            world,
            target,
            radius=target_radius,
        )

    def _nearest_wait_target(
        self,
        world: "SpiderWorld",
        anchor: Tuple[int, int],
        *,
        radius: int | None = None,
    ) -> Tuple[int, int] | None:
        """
        Select the nearest shelter entrance to an anchor coordinate, optionally restricting candidates by radius.
        
        Parameters:
            world (SpiderWorld): Simulation world providing shelter entrance locations and distance metric.
            anchor (Tuple[int, int]): Reference coordinate to measure Manhattan distance from.
            radius (int | None, optional): If provided, only consider entrances with Manhattan distance <= max(1, radius).
        
        Returns:
            Tuple[int, int] | None: The nearest entrance coordinate (ties broken by position order), or `None` if no entrance qualifies.
        """
        entrances = sorted(world.shelter_entrance_cells)
        if radius is not None:
            entrances = [
                pos for pos in entrances if world.manhattan(pos, anchor) <= max(1, radius)
            ]
        if not entrances:
            return None
        entrances.sort(key=lambda pos: (world.manhattan(pos, anchor), pos))
        return entrances[0]

    def _reachable_probe_target(
        self,
        world: "SpiderWorld",
        anchor: Tuple[int, int],
        *,
        radius: int | None = None,
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
        if radius is not None:
            candidates = [
                pos for pos in candidates if world.manhattan(pos, anchor) <= max(1, radius)
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
        lizard = self._predator(world)
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
        lizard = self._predator(world)
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
        lizard = self._predator(world)
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
        lizard = self._predator(world)
        if lizard.mode == "PATROL":
            if lizard.patrol_target is None or self._predator_pos(world) == lizard.patrol_target:
                lizard.patrol_target = self._pick_patrol_target(world)
            if lizard.patrol_target is None:
                return False
            moved = self._step_towards(world, lizard.patrol_target)
            if moved and self._predator_pos(world) == lizard.patrol_target:
                lizard.patrol_target = self._pick_patrol_target(world)
            return moved

        if lizard.mode == "CHASE":
            return self._step_towards(world, world.spider_pos())

        if lizard.mode == "INVESTIGATE":
            if lizard.last_known_spider is None:
                self._set_mode(lizard, "PATROL")
                return False
            if lizard.investigate_target is None:
                lizard.investigate_target = self._reachable_probe_target(
                    world,
                    lizard.last_known_spider,
                    radius=self._targeting_radius(world),
                )
            if lizard.investigate_target is None:
                self._enter_wait_or_recover(world)
                return False
            moved = self._step_towards(world, lizard.investigate_target)
            lizard.investigate_ticks += 1
            if self._predator_pos(world) == lizard.investigate_target:
                self._enter_wait_or_recover(world)
            return moved

        if lizard.mode == "WAIT":
            if lizard.wait_target is None:
                return False
            if world.manhattan(self._predator_pos(world), lizard.wait_target) <= 1:
                return False
            moved = self._step_towards(world, lizard.wait_target)
            if moved and world.manhattan(self._predator_pos(world), lizard.wait_target) <= 1:
                lizard.mode_ticks = 0
            return moved

        if lizard.mode == "RECOVER":
            lizard.recover_ticks = max(0, lizard.recover_ticks - 1)
            if lizard.patrol_target is None or self._predator_pos(world) == lizard.patrol_target:
                lizard.patrol_target = self._pick_patrol_target(
                    world,
                    retreat_from=lizard.wait_target or lizard.last_known_spider,
                )
            if lizard.patrol_target is None:
                return False
            return self._step_towards(world, lizard.patrol_target)

        return False
