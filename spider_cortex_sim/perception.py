from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import atan2, degrees
from typing import TYPE_CHECKING, Final, Iterable, Literal, Mapping, Protocol

import numpy as np

from .interfaces import (
    OBSERVATION_DIMS,
    OBSERVATION_INTERFACE_BY_KEY,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    MotorContextObservation,
    ObservationView,
    SensoryObservation,
    SleepObservation,
    VisualObservation,
)
from .maps import BLOCKED, CLUTTER, NARROW, OPEN
from .world_types import PerceptTrace

if TYPE_CHECKING:
    from .world import SpiderWorld


VisibilityZone = Literal["foveal", "peripheral", "outside"]


class HasPosition(Protocol):
    x: float
    y: float


@dataclass(frozen=True)
class PerceivedTarget:
    """Perception summary for a single candidate target.

    Visible percepts must carry a concrete grid `position`. Non-visible or
    no-target percepts may keep `position=None`.
    """

    visible: float
    certainty: float
    occluded: float
    dx: float
    dy: float
    dist: int
    position: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if float(self.visible) > 0.0 and self.position is None:
            raise ValueError("Visible PerceivedTarget requires position.")


NO_TARGET_DISTANCE = 10**9
DOMINANT_PREDATOR_TYPE_NONE = 0.0
DOMINANT_PREDATOR_TYPE_VISUAL = 0.5
DOMINANT_PREDATOR_TYPE_OLFACTORY = 1.0


TERRAIN_DIFFICULTY: Final[Mapping[str, float]] = {
    OPEN: 0.0,
    CLUTTER: 0.4,
    NARROW: 0.7,
}


def predator_motion_salience(
    world: "SpiderWorld",
    *,
    predator_view: PerceivedTarget | None = None,
) -> float:
    """
    Compute the motion salience contributed by a predator.
    
    Returns the world's configured predator motion bonus when the predator is perceived (the provided view indicates visibility or occlusion) and the predator's mode is "CHASE" or "INVESTIGATE"; otherwise returns 0.0. If no predator_view is provided or it has no position, the function falls back to the first available predator candidate.
    
    Parameters:
        predator_view (PerceivedTarget | None): Optional perceived view of a predator; used to determine whether the predator is currently perceived and to locate the corresponding predator entity.
    
    Returns:
        float: The motion salience value (the configured predator motion bonus) when conditions are met, `0.0` otherwise.
    """
    perceived = (
        predator_view is not None
        and (predator_view.visible > 0.0 or predator_view.occluded > 0.0)
    )
    predator = (
        _predator_for_position(world, predator_view.position)
        if predator_view is not None and predator_view.position is not None
        else None
    )
    if predator is None:
        predators = _predator_candidates(world)
        predator = predators[0] if predators else None
    return (
        float(world.operational_profile.perception["predator_motion_bonus"])
        if perceived and predator is not None and predator.mode in {"CHASE", "INVESTIGATE"}
        else 0.0
    )


OBSERVATION_LEAKAGE_AUDIT = {
    "predator_visible_and_direction": {
        "classification": "direct_perception",
        "risk": "low",
        "modules": ["visual", "alert", "action_context", "motor_context"],
        "source": "spider_cortex_sim.perception.predator_visible_to_spider",
        "notes": "Derived from line of sight, certainty, and relative direction, so it is a direct ecological signal.",
    },
    "food_and_predator_smell_gradients": {
        "classification": "direct_perception",
        "risk": "low",
        "modules": ["sensory", "hunger", "alert"],
        "source": "spider_cortex_sim.perception.smell_gradient",
        "notes": "These are local sensory signals with explicit noise and no direct access to the optimal path.",
    },
    "predator_dist": {
        "classification": "privileged_world_signal",
        "risk": "resolved",
        "status": "removed_from_observations",
        "modules": ["alert", "action_context", "motor_context"],
        "source": "spider_cortex_sim.perception.observe_world",
        "evidence": "obs['meta']['diagnostic']['diagnostic_predator_dist']",
        "notes": "Removed from network-facing observations and retained only as diagnostic metadata for leakage audits and offline analysis.",
    },
    "home_vector": {
        "classification": "world_derived_navigation_hint",
        "risk": "resolved",
        "status": "removed_from_observations",
        "modules": ["sleep", "alert"],
        "source": "spider_cortex_sim.perception.observe_world",
        "evidence": "obs['meta']['diagnostic'] contains diagnostic_home_dx, diagnostic_home_dy, and diagnostic_home_dist",
        "notes": "Removed from network-facing observations and retained only as diagnostic metadata for leakage audits and offline analysis.",
    },
    "predator_memory_vector": {
        "classification": "plausible_memory",
        "risk": "low",
        "modules": ["alert"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Derived from predator memory that is written only from visual perception or local contact events.",
    },
    "shelter_memory_vector": {
        "classification": "plausible_memory",
        "risk": "low",
        "modules": ["sleep"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Derived from shelter memory that stores perceived shelter cells rather than a world-selected best shelter.",
    },
    "escape_memory_vector": {
        "classification": "plausible_memory",
        "risk": "low",
        "modules": ["alert"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Derived from movement history and world-boundary clamping without walkability or shelter-policy queries.",
    },
    "foveal_scan_age": {
        "classification": "self_knowledge",
        "risk": "low",
        "modules": ["visual"],
        "source": "spider_cortex_sim.world._scan_age_for_heading",
        "notes": "Derived from the spider's own heading-change history and does not expose hidden target locations.",
    },
}


def observation_leakage_audit() -> dict[str, dict[str, object]]:
    """
    Return a deep-copied audit catalog describing perception-side observation leakage candidates.
    
    The catalog maps leakage identifiers to metadata describing the leakage class, risk level, implicated modules, source identifiers, and evidence or notes; the returned mapping is a deep copy to prevent callers from mutating the module-level catalog.
    
    Returns:
        audit (dict[str, dict[str, object]]): A mapping from leakage keys to metadata dictionaries. Each metadata dictionary includes fields such as classification, risk, modules, sources, and notes.
    """
    return deepcopy(OBSERVATION_LEAKAGE_AUDIT)


def _clip_signed(value: float) -> float:
    """
    Clip a numeric value to the range [-1.0, 1.0].
    
    Parameters:
        value (float): Input value to be limited to the signed unit interval.
    
    Returns:
        float: The input constrained to be between -1.0 and 1.0.
    """
    return float(np.clip(value, -1.0, 1.0))


def _fov_thresholds(world: "SpiderWorld") -> tuple[float, float]:
    """
    Compute configured field-of-view half-angle thresholds.
    
    Reads `fov_half_angle` (default 45.0) and `peripheral_half_angle` (default 70.0) from the world's perception configuration, clips each to the range [0.0, 180.0], and returns the foveal half-angle and the peripheral half-angle (the latter is at least the foveal value).
    
    Returns:
        tuple[float, float]: (foveal_half_angle, peripheral_half_angle) in degrees.
    """
    cfg = world.operational_profile.perception
    foveal_half_angle = float(np.clip(cfg.get("fov_half_angle", 45.0), 0.0, 180.0))
    peripheral_half_angle = float(np.clip(cfg.get("peripheral_half_angle", 70.0), 0.0, 180.0))
    return foveal_half_angle, max(foveal_half_angle, peripheral_half_angle)


def _max_scan_age(world: "SpiderWorld") -> float:
    """Return the scan-age normalization horizon in ticks."""
    return max(1.0, float(world.operational_profile.perception.get("max_scan_age", 10.0)))


def _normalized_scan_age(
    world: "SpiderWorld",
    heading_dx: float,
    heading_dy: float,
) -> float:
    """Return normalized age for the current foveal heading."""
    scan_age = world._scan_age_for_heading(int(heading_dx), int(heading_dy))
    return float(np.clip(float(scan_age) / _max_scan_age(world), 0.0, 1.0))


def _compute_target_visibility_zone(
    world: "SpiderWorld",
    source: tuple[int, int],
    target: tuple[int, int],
) -> VisibilityZone:
    """
    Classify a target into the spider's foveal, peripheral, or outside visual zone.
    """
    source = tuple(source)
    target = tuple(target)
    if source != world.spider_pos():
        return "foveal"
    heading_x = float(world.state.heading_dx)
    heading_y = float(world.state.heading_dy)
    if heading_x == 0.0 and heading_y == 0.0:
        return "foveal"
    rel_x = float(target[0] - source[0])
    rel_y = float(target[1] - source[1])
    if rel_x == 0 and rel_y == 0:
        return "foveal"
    cross = heading_x * rel_y - heading_y * rel_x
    dot = heading_x * rel_x + heading_y * rel_y
    angle = abs(degrees(atan2(cross, dot)))
    foveal_half_angle, peripheral_half_angle = _fov_thresholds(world)
    if angle <= foveal_half_angle:
        return "foveal"
    if angle <= peripheral_half_angle:
        return "peripheral"
    return "outside"


def _heading_allows_target(
    world: "SpiderWorld",
    source: tuple[int, int],
    target: tuple[int, int],
) -> bool:
    """
    Determine whether the target lies within the spider's configured visual field (foveal or peripheral).
    
    Returns:
        bool: True if the target is classified as inside the visual field, False if classified as "outside".
    """
    return _compute_target_visibility_zone(world, source, target) != "outside"


def empty_percept_trace() -> PerceptTrace:
    """
    Create a new empty short-lived percept trace slot.
    """
    return PerceptTrace(target=None, age=0, certainty=0.0, heading_dx=0, heading_dy=0)


def _percept_trace_ttl(world: "SpiderWorld") -> int:
    """
    Return the configured TTL for short percept traces.
    """
    return max(1, round(world.operational_profile.perception["percept_trace_ttl"]))


def _percept_trace_decay(world: "SpiderWorld") -> float:
    """
    Return the configured multiplicative decay for short percept traces.
    """
    return float(np.clip(world.operational_profile.perception["percept_trace_decay"], 0.0, 1.0))


def trace_strength(world: "SpiderWorld", trace: PerceptTrace) -> float:
    """
    Compute the current decayed strength of a short percept trace.
    """
    if trace.target is None or trace.age >= _percept_trace_ttl(world):
        return 0.0
    return float(np.clip(trace.certainty * (_percept_trace_decay(world) ** trace.age), 0.0, 1.0))


def trace_view(world: "SpiderWorld", trace: PerceptTrace) -> dict[str, object]:
    """
    Serialize a percept trace with derived direction and strength metadata.
    """
    strength = trace_strength(world, trace)
    if trace.target is None or strength <= 0.0:
        dx = 0.0
        dy = 0.0
        heading_dx = 0
        heading_dy = 0
    else:
        dx, dy, _ = world._relative(trace.target)
        heading_dx = int(trace.heading_dx)
        heading_dy = int(trace.heading_dy)
    return {
        "target": (
            [int(trace.target[0]), int(trace.target[1])]
            if trace.target is not None and strength > 0.0
            else None
        ),
        "age": int(trace.age),
        "certainty": float(trace.certainty),
        "strength": float(strength),
        "dx": float(dx),
        "dy": float(dy),
        "heading_dx": int(heading_dx),
        "heading_dy": int(heading_dy),
        "ttl": _percept_trace_ttl(world),
        "decay": _percept_trace_decay(world),
    }


PERCEPTION_CATEGORY_CERTAINTY_THRESHOLD = 0.5


def _perception_category(
    certainty: float,
    trace_strength: float,
    scan_age: float,
    is_delayed: bool,
) -> str:
    """
    Classify a perception sample for debug and audit output.

    The helper is intentionally not part of any observation interface. It
    returns:
    - ``"direct"`` when ``certainty`` is above the debug threshold and the
      foveal scan is fresh (``scan_age < 2`` ticks).
    - ``"trace"`` when a short percept trace is active.
    - ``"delayed"`` when the caller reports that perceptual delay is active and
      the current payload was not refreshed by an active scan.
    - ``"uncertain"`` when none of those conditions applies.

    ``is_delayed`` should be passed as the already-resolved condition
    ``perceptual_delay_ticks > 0 and not refreshed``.
    """
    if float(certainty) > PERCEPTION_CATEGORY_CERTAINTY_THRESHOLD and float(scan_age) < 2.0:
        return "direct"
    if float(trace_strength) > 0.0:
        return "trace"
    if bool(is_delayed):
        return "delayed"
    return "uncertain"


def _apply_visibility_zone_certainty(
    world: "SpiderWorld",
    certainty: float,
    zone: VisibilityZone,
) -> float:
    """
    Adjusts a visual certainty value based on the target's visibility zone.
    
    Parameters:
        certainty (float): Input certainty value (expected in [0.0, 1.0]) to be adjusted.
        zone (VisibilityZone): Visibility zone of the target; one of "foveal", "peripheral", or "outside".
    
    Returns:
        float: The adjusted certainty clipped to the range [0.0, 1.0]. Returns 0.0 when `zone` is "outside"; applies a configurable penalty when `zone` is "peripheral".
    """
    if zone == "outside":
        return 0.0
    if zone == "peripheral":
        penalty = float(
            np.clip(
                world.operational_profile.perception.get("peripheral_certainty_penalty", 0.35),
                0.0,
                1.0,
            )
        )
        certainty -= penalty
    return float(np.clip(certainty, 0.0, 1.0))


def _apply_visual_noise(world: "SpiderWorld", target: PerceivedTarget) -> PerceivedTarget:
    """
    Apply configured visual noise to a perceived target, returning a new PerceivedTarget with modified certainty, visibility, and direction components.
    
    Parameters:
        world (SpiderWorld): Source of the visual noise profile and RNG.
        target (PerceivedTarget): Original perceived target to perturb; if `target.certainty <= 0.0` it is returned unchanged.
    
    Returns:
        PerceivedTarget: A new perception object where:
            - `certainty` is jittered and clipped to [0.0, 1.0];
            - a probabilistic dropout may halve `certainty` and set `visible`, `dx`, and `dy` to 0.0;
            - when `visible > 0.0` and `certainty > 0.0`, nonzero `dx`/`dy` are jittered and clipped to [-1.0, 1.0];
            - otherwise `dx` and `dy` are set to 0.0;
            - `occluded` is preserved and `dist` is returned as an int.
    """
    if target.certainty <= 0.0:
        return target
    cfg = world.noise_profile.visual
    certainty_jitter = max(0.0, float(cfg["certainty_jitter"]))
    direction_jitter = max(0.0, float(cfg["direction_jitter"]))
    dropout_prob = min(1.0, max(0.0, float(cfg["dropout_prob"])))
    visibility_threshold = float(
        world.operational_profile.perception["visibility_binary_threshold"]
    )

    certainty = float(
        np.clip(
            target.certainty
            + float(world.visual_rng.uniform(-certainty_jitter, certainty_jitter)),
            0.0,
            1.0,
        )
    )
    occluded = float(target.occluded)
    dx = float(target.dx)
    dy = float(target.dy)

    if dropout_prob > 0.0 and float(world.visual_rng.random()) < dropout_prob:
        certainty *= 0.5
    visible = (
        1.0
        if target.visible > 0.0 and certainty >= visibility_threshold
        else 0.0
    )

    if visible > 0.0 and certainty > 0.0:
        if dx != 0.0:
            dx = _clip_signed(dx + float(world.visual_rng.uniform(-direction_jitter, direction_jitter)))
        if dy != 0.0:
            dy = _clip_signed(dy + float(world.visual_rng.uniform(-direction_jitter, direction_jitter)))
    else:
        dx = 0.0
        dy = 0.0

    return PerceivedTarget(
        visible=visible,
        certainty=certainty,
        occluded=occluded,
        dx=dx,
        dy=dy,
        dist=int(target.dist),
        position=target.position,
    )


def _apply_olfactory_noise(
    world: "SpiderWorld",
    *,
    strength: float,
    grad_x: float,
    grad_y: float,
    best_dist: int,
    apply_noise: bool = True,
) -> tuple[float, float, float, int]:
    """
    Apply configured olfactory noise to a smell strength and its gradient direction.
    
    This uses the world's olfactory noise profile to jitter the scalar strength and, if nonzero, the
    gradient components; strength is clipped to the range [0.0, 1.0] and gradient components are
    kept within [-1.0, 1.0]. If the input or jittered strength is less than or equal to 0.0, the
    function returns zero strength and zero gradients while preserving `best_dist`.
    
    Parameters:
        strength (float): Measured smell strength to perturb.
        grad_x (float): X component of the smell gradient (may be 0.0).
        grad_y (float): Y component of the smell gradient (may be 0.0).
        best_dist (int): Distance to the nearest source; returned unchanged.
    
    Returns:
        tuple[float, float, float, int]: `(noisy_strength, noisy_grad_x, noisy_grad_y, best_dist)`
        where `noisy_strength` is in [0.0, 1.0], `noisy_grad_x` and `noisy_grad_y` are in [-1.0, 1.0];
        if `noisy_strength <= 0.0` then both gradients are `0.0`.
    """
    if strength <= 0.0:
        return 0.0, 0.0, 0.0, best_dist
    if not apply_noise:
        return float(strength), float(grad_x), float(grad_y), int(best_dist)
    cfg = world.noise_profile.olfactory
    strength_jitter = max(0.0, float(cfg["strength_jitter"]))
    direction_jitter = max(0.0, float(cfg["direction_jitter"]))
    noisy_strength = float(np.clip(
        strength + float(world.olfactory_rng.uniform(-strength_jitter, strength_jitter)),
        0.0,
        1.0,
    ))
    if noisy_strength <= 0.0:
        return 0.0, 0.0, 0.0, best_dist
    noisy_grad_x = float(grad_x)
    noisy_grad_y = float(grad_y)
    if noisy_grad_x != 0.0:
        noisy_grad_x = _clip_signed(
            noisy_grad_x + float(world.olfactory_rng.uniform(-direction_jitter, direction_jitter))
        )
    if noisy_grad_y != 0.0:
        noisy_grad_y = _clip_signed(
            noisy_grad_y + float(world.olfactory_rng.uniform(-direction_jitter, direction_jitter))
        )
    return noisy_strength, noisy_grad_x, noisy_grad_y, best_dist


def visible_range(world: "SpiderWorld") -> int:
    """
    Determine the spider's effective vision radius accounting for night adjustments.
    
    During daytime this returns world.vision_range. During nighttime this returns
    the greater of perception['night_vision_min_range'] and
    world.vision_range minus perception['night_vision_range_penalty'].
    
    Returns:
        int: Effective vision radius.
    """
    cfg = world.operational_profile.perception
    if not world.is_night():
        return world.vision_range
    return max(
        int(cfg["night_vision_min_range"]),
        world.vision_range - int(cfg["night_vision_range_penalty"]),
    )


def visibility_confidence(
    world: "SpiderWorld",
    *,
    source: tuple[int, int],
    target: tuple[int, int],
    dist: int,
    radius: int,
    motion_bonus: float = 0.0,
    visibility_zone: VisibilityZone = "foveal",
    zone: VisibilityZone | None = None,
) -> float:
    """
    Compute a visibility certainty score for a target from a source location.

    The score starts from a distance-based falloff, then is adjusted by a motion bonus and penalties for target/source terrain, night, and visual zone; final value is clipped to the range [0.0, 1.0].

    Parameters:
        world (SpiderWorld): World context used to query terrain and time-of-day.
        source (tuple[int, int]): Grid coordinates of the observer.
        target (tuple[int, int]): Grid coordinates of the observed object.
        dist (int): Manhattan distance between source and target.
        radius (int): Effective sensing radius used to scale distance falloff.
        motion_bonus (float): Additive bonus for source or target motion (default 0.0).
        visibility_zone: Foveal/peripheral/outside visual zone for the target.
        zone: Alias for visibility_zone. When provided, this takes precedence.

    Returns:
        float: Certainty in [0.0, 1.0] that the target is visible from the source.
    """
    cfg = world.operational_profile.perception
    certainty = 1.0 - (dist / float(max(1, radius + 1)))
    certainty += motion_bonus
    target_terrain = world.terrain_at(target)
    source_terrain = world.terrain_at(source)
    if target_terrain == CLUTTER:
        certainty -= cfg["visibility_clutter_penalty"]
    elif target_terrain == NARROW:
        certainty -= cfg["visibility_narrow_penalty"]
    if source_terrain == NARROW:
        certainty -= cfg["visibility_source_narrow_penalty"]
    if world.is_night():
        certainty -= cfg["visibility_night_penalty"]
    effective_zone = zone if zone is not None else visibility_zone
    return _apply_visibility_zone_certainty(world, certainty, effective_zone)


def line_cells(origin: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Compute the grid cells along a straight line from origin toward target, excluding the origin and target cells.
    
    Parameters:
        origin (tuple[int, int]): Starting grid coordinate as (x, y).
        target (tuple[int, int]): Ending grid coordinate as (x, y).
    
    Returns:
        list[tuple[int, int]]: Ordered list of intermediate grid cell coordinates between origin and target (closest to origin first); empty if no intermediate cells exist.
    """
    x0, y0 = origin
    x1, y1 = target
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        if (x, y) != (x1, y1):
            points.append((x, y))
    return points


def has_line_of_sight(
    world: "SpiderWorld",
    origin: tuple[int, int],
    target: tuple[int, int],
    *,
    block_clutter: bool = True,
) -> bool:
    """
    Determine whether an unobstructed line of sight exists from `origin` to `target`, taking shelter roles and terrain occlusion into account.
    
    Checks intermediate grid cells between the two positions for disqualifying conditions:
    - If the origin is in the "outside" shelter role and the target is "deep", visibility is blocked.
    - If the origin is "outside" and the target is "inside", visibility requires at least one intermediate cell to be a shelter entrance.
    - Any intermediate cell with `BLOCKED` terrain blocks visibility.
    - If `block_clutter` is true, any intermediate cell with `CLUTTER` terrain also blocks visibility.
    
    Parameters:
        world: The simulation world providing terrain and shelter-role queries.
        origin: (x, y) grid coordinates of the observer.
        target: (x, y) grid coordinates of the observed cell.
        block_clutter: If `True`, treat `CLUTTER` terrain as occluding; if `False`, clutter does not block sight.
    
    Returns:
        `True` if `origin` can see `target` under the shelter-role and terrain rules described above, `False` otherwise.
    """
    origin_role = world.shelter_role_at(origin)
    target_role = world.shelter_role_at(target)
    cells = line_cells(origin, target)
    if origin_role == "outside" and target_role == "deep":
        return False
    if origin_role == "outside" and target_role == "inside":
        if not any(cell in world.shelter_entrance_cells for cell in cells):
            return False
    for cell in cells:
        terrain = world.terrain_at(cell)
        if terrain == BLOCKED:
            return False
        if block_clutter and terrain == CLUTTER:
            return False
    return True


def advance_percept_trace(
    world: "SpiderWorld",
    trace: PerceptTrace,
    percept: PerceivedTarget,
    positions: Iterable[tuple[int, int]],
) -> PerceptTrace:
    """
    Refresh or age a short-lived percept trace using the latest raw percept.

    Visible, non-occluded percepts reset the trace only when their position is
    one of the candidate positions, still matches the reported distance from
    the spider, lies inside the current visual field, and has line of sight.
    Otherwise the existing trace ages and expires through configured trace TTL
    and decay.
    """
    if percept.visible > 0.0 and percept.occluded <= 0.0 and percept.position is not None:
        source = world.spider_pos()
        candidate_set = {tuple(pos) for pos in positions}
        percept_dist = int(percept.dist)
        target = tuple(percept.position)
        if target in candidate_set:
            visibility_zone = _compute_target_visibility_zone(world, source, target)
            if (
                world.manhattan(source, target) == percept_dist
                and visibility_zone != "outside"
                and has_line_of_sight(world, source, target)
            ):
                return PerceptTrace(
                    target=(int(target[0]), int(target[1])),
                    age=0,
                    certainty=float(np.clip(percept.certainty, 0.0, 1.0)),
                    heading_dx=int(world.state.heading_dx),
                    heading_dy=int(world.state.heading_dy),
                )

    if trace.target is None:
        return empty_percept_trace()
    aged = PerceptTrace(
        target=trace.target,
        age=int(trace.age) + 1,
        certainty=float(np.clip(trace.certainty, 0.0, 1.0)),
        heading_dx=int(trace.heading_dx),
        heading_dy=int(trace.heading_dy),
    )
    if trace_strength(world, aged) <= 0.0:
        return empty_percept_trace()
    return aged


def visible_object(
    world: "SpiderWorld",
    positions: Iterable[tuple[int, int]],
    *,
    radius: int,
    origin: tuple[int, int] | None = None,
    motion_bonus: float = 0.0,
    apply_noise: bool = True,
) -> PerceivedTarget:
    """
    Select the nearest candidate within Manhattan `radius` and return its perceived metrics from `origin`'s perspective.
    
    Examines `positions` within `radius` of `origin` (or the spider position when `origin` is None). Targets classified as `"outside"` by the field-of-view model are ignored. If any candidate with line of sight exists the nearest visible candidate is chosen; otherwise the nearest occluded candidate is chosen; if none are in range a "no target" perception is returned. The returned perception may be modified by visual noise when `apply_noise` is True.
    
    Parameters:
        origin (tuple[int, int] | None): Observer position; when None use the spider position.
        motion_bonus (float): Additive bonus to visibility confidence reflecting recent motion.
        apply_noise (bool): When True, apply visual noise model to the produced PerceivedTarget.
    
    Returns:
        PerceivedTarget: Summary of the selected candidate with fields:
            visible: `1.0` if certainty meets or exceeds the configured visibility threshold, `0.0` otherwise.
            certainty: Confidence in visibility (clipped to [0.0, 1.0]); for occluded targets a heuristic occlusion certainty.
            occluded: `1.0` when the returned target was occluded, `0.0` otherwise.
            dx, dy: Relative direction components from `origin` to the selected target (set to `0.0` when not visible).
            dist: Manhattan distance to the selected target, or `NO_TARGET_DISTANCE` when no candidate is found.
            position: Grid coordinates of the selected target, or `None` when no candidate is found.
    """
    source = tuple(origin) if origin is not None else world.spider_pos()
    best_visible = None
    best_visible_zone: VisibilityZone | None = None
    best_visible_dist = radius + 1
    best_occluded = None
    best_occluded_zone: VisibilityZone | None = None
    best_occluded_dist = radius + 1
    for pos in positions:
        pos = tuple(pos)
        dist = world.manhattan(source, pos)
        if dist > radius:
            continue
        visibility_zone = _compute_target_visibility_zone(world, source, pos)
        if visibility_zone == "outside":
            continue
        if has_line_of_sight(world, source, pos):
            if dist < best_visible_dist or (dist == best_visible_dist and tuple(pos) < tuple(best_visible)):
                best_visible = pos
                best_visible_zone = visibility_zone
                best_visible_dist = dist
        elif dist < best_occluded_dist or (dist == best_occluded_dist and tuple(pos) < tuple(best_occluded)):
            best_occluded = pos
            best_occluded_zone = visibility_zone
            best_occluded_dist = dist

    cfg = world.operational_profile.perception
    if best_visible is not None:
        certainty = visibility_confidence(
            world,
            source=source,
            target=best_visible,
            dist=best_visible_dist,
            radius=radius,
            motion_bonus=motion_bonus,
            visibility_zone=best_visible_zone or "foveal",
        )
        visible = 1.0 if certainty >= cfg["visibility_binary_threshold"] else 0.0
        dx, dy, _ = world._relative(best_visible, origin=source)
        if visible <= 0.0:
            dx = 0.0
            dy = 0.0
        target = PerceivedTarget(
            visible=float(visible),
            certainty=float(certainty),
            occluded=0.0,
            dx=float(dx),
            dy=float(dy),
            dist=int(best_visible_dist),
            position=tuple(best_visible),
        )
        return _apply_visual_noise(world, target) if apply_noise else target

    if best_occluded is not None:
        certainty = max(
            cfg["occluded_certainty_min"],
            cfg["occluded_certainty_base"]
            - cfg["occluded_certainty_decay_per_step"] * max(0, best_occluded_dist - 1),
        )
        certainty = _apply_visibility_zone_certainty(
            world,
            certainty,
            best_occluded_zone or "foveal",
        )
        target = PerceivedTarget(
            visible=0.0,
            certainty=float(certainty),
            occluded=1.0,
            dx=0.0,
            dy=0.0,
            dist=int(best_occluded_dist),
            position=tuple(best_occluded),
        )
        return _apply_visual_noise(world, target) if apply_noise else target

    return PerceivedTarget(
        visible=0.0,
        certainty=0.0,
        occluded=0.0,
        dx=0.0,
        dy=0.0,
        dist=NO_TARGET_DISTANCE,
        position=None,
    )


def smell_gradient(
    world: "SpiderWorld",
    positions: Iterable[tuple[int, int]],
    *,
    radius: int,
    origin: tuple[int, int] | None = None,
    apply_noise: bool = True,
) -> tuple[float, float, float, int]:
    """
    Compute a smell-field summary over candidate positions within a Manhattan radius.
    
    Processes each candidate position within `radius` of `origin` (or the spider position if `origin` is None) and accumulates a weighted smell strength and a weighted average gradient direction. The gradient components are expressed as offsets normalized by (width - 1) and (height - 1) so they represent fractional map-space directions. If no positions fall inside the radius, returns zero strength and a sentinel large distance.
    
    Parameters:
        world: The simulation world providing geometry and distance helpers.
        positions: Iterable of (x, y) candidate positions that may contribute to the smell field.
        radius: Manhattan radius (inclusive) within which positions contribute.
        origin: Optional (x, y) origin to measure distances from; defaults to the spider's position.
    
    Returns:
        strength (float): Total smell strength clipped to a maximum of 1.0 (0.0 if no contributors).
        grad_x (float): Weighted-average x gradient component, normalized by (world.width - 1).
        grad_y (float): Weighted-average y gradient component, normalized by (world.height - 1).
        best_dist (int): Minimum Manhattan distance from `origin` to any contributing position, or `NO_TARGET_DISTANCE` if none.
    """
    source = origin if origin is not None else world.spider_pos()
    total_strength = 0.0
    grad_x = 0.0
    grad_y = 0.0
    best_dist = NO_TARGET_DISTANCE
    for pos in positions:
        dist = world.manhattan(source, pos)
        if dist > radius:
            continue
        _, _, dist_norm = world._relative(pos, origin=source)
        weight = max(0.0, 1.0 - dist / float(max(1, radius)))
        dx = 0.0 if dist_norm == 0.0 else (pos[0] - source[0]) / max(1, world.width - 1)
        dy = 0.0 if dist_norm == 0.0 else (pos[1] - source[1]) / max(1, world.height - 1)
        total_strength += weight
        grad_x += weight * dx
        grad_y += weight * dy
        best_dist = min(best_dist, dist)
    if total_strength <= 0.0:
        return 0.0, 0.0, 0.0, NO_TARGET_DISTANCE
    strength = min(1.0, total_strength)
    return _apply_olfactory_noise(
        world,
        strength=float(strength),
        grad_x=float(grad_x / total_strength),
        grad_y=float(grad_y / total_strength),
        best_dist=int(best_dist),
        apply_noise=apply_noise,
    )


def _predator_candidates(
    world: "SpiderWorld",
    predators: Iterable[object] | None = None,
) -> list[object]:
    """
    Return an explicit list of predator objects to consider for perception and threat calculations.

    Parameters:
        world: The SpiderWorld instance used as the fallback source of predators.
        predators: Optional iterable of predator objects to use instead of the world's configured predators.

    Returns:
        A list of predator objects. If `predators` is provided, a list copy of its items is returned. Otherwise, the function returns `world.predators` if non-empty; if that is empty or missing, it returns a single-element list containing `world.lizard` when available; returns an empty list if no predators can be obtained.
    """
    if predators is not None:
        return list(predators)
    predator_list = list(getattr(world, "predators", []))
    if predator_list:
        return predator_list
    try:
        return [world.lizard]
    except AttributeError:
        return []


def _predator_position(predator: HasPosition) -> tuple[int, int]:
    """
    Extract the predator's grid coordinates as integers.
    
    Parameters:
        predator (HasPosition): An object exposing numeric `x` and `y` attributes.
    
    Returns:
        tuple[int, int]: The `(x, y)` grid coordinates cast to `int`.
    """
    return int(predator.x), int(predator.y)


def _uses_world_level_profile(predator: object) -> bool:
    """
    Determine whether a predator should use the world-level fallback profile fields.
    
    Parameters:
        predator (object): Object that may provide a `profile` attribute.
    
    Returns:
        True when the predator has no explicit `profile`, False otherwise.
    """
    return getattr(predator, "profile", None) is None


def _predator_profile_fields(world: "SpiderWorld", predator: object) -> dict[str, object]:
    """
    Builds a perception profile mapping for a predator, suitable for downstream detection and threat computations.
    
    Parameters:
        world (SpiderWorld): World object providing operational perception defaults and lizard-based fallback values.
        predator (object): Predator object which may have a `profile` attribute; if absent, world-level lizard defaults are used.
    
    Returns:
        dict[str, object]: Mapping with the following keys:
            - "name" (str): Predator name identifier.
            - "vision_range" (int): Vision range in grid units.
            - "smell_range" (int): Smell range in grid units.
            - "detection_style" (str): Either "visual" or "olfactory".
            - "move_interval" (int): Minimum move interval (at least 1).
            - "detection_threshold" (float): Threshold used to determine detection.
    """
    cfg = world.operational_profile.perception
    if _uses_world_level_profile(predator):
        return {
            "name": "lizard",
            "vision_range": int(world.lizard_vision_range),
            "smell_range": int(world.predator_smell_range),
            "detection_style": "visual",
            "move_interval": max(1, int(world.lizard_move_interval)),
            "detection_threshold": float(cfg["lizard_detection_threshold"]),
        }
    profile = getattr(predator, "profile")
    return {
        "name": str(getattr(profile, "name", "predator")),
        "vision_range": int(getattr(profile, "vision_range")),
        "smell_range": int(getattr(profile, "smell_range")),
        "detection_style": str(getattr(profile, "detection_style", "visual")),
        "move_interval": max(1, int(getattr(profile, "move_interval", 1))),
        "detection_threshold": float(getattr(profile, "detection_threshold", 0.0)),
    }


def _predator_detection_style(world: "SpiderWorld", predator: object) -> str:
    """
    Return the predator's detection style used for grouping (for example, "visual" or "olfactory").
    
    The value is read from the predator's profile when present; if the predator lacks a specific profile, world-level predator defaults are used.
    
    Parameters:
        world: The world instance providing defaults and context.
        predator: A predator object (or mapping) expected to contain a `profile` with a `detection_style` field.
    
    Returns:
        detection_style (str): The detection style string (e.g., "visual" or "olfactory").
    """
    return str(_predator_profile_fields(world, predator)["detection_style"])


def _predator_for_position(
    world: "SpiderWorld",
    position: tuple[int, int] | None,
) -> object | None:
    """
    Return the predator object whose integer (x, y) position matches the given grid position.
    
    Parameters:
        world (SpiderWorld): World context used to enumerate predator candidates.
        position (tuple[int, int] | None): Grid (x, y) coordinates to match; if None, no lookup is performed.
    
    Returns:
        object | None: The predator whose (x, y) equals `position`, or `None` if no match is found or `position` is None.
    """
    if position is None:
        return None
    for predator in _predator_candidates(world):
        if _predator_position(predator) == tuple(position):
            return predator
    return None


def _predator_visual_view(
    world: "SpiderWorld",
    predator: object,
    *,
    apply_noise: bool,
) -> PerceivedTarget:
    """
    Obtain the visual perception of the given predator from its grid position.
    
    Parameters:
        predator (object): Predator-like object exposing numeric `x` and `y` attributes used as the candidate position.
        apply_noise (bool): If true, apply the world's visual noise model to the resulting percept.
    
    Returns:
        PerceivedTarget: A perception record for the predator at its position as observed from the spider (may indicate no target if out of range or fully occluded).
    """
    return visible_object(
        world,
        [_predator_position(predator)],
        radius=visible_range(world),
        apply_noise=apply_noise,
    )


def _no_target_perceived_target() -> PerceivedTarget:
    """Return a canonical no-target predator percept."""
    return PerceivedTarget(
        visible=0.0,
        certainty=0.0,
        occluded=0.0,
        dx=0.0,
        dy=0.0,
        dist=NO_TARGET_DISTANCE,
        position=None,
    )


def _normalized_proximity(dist: int, radius: int) -> float:
    """
    Compute a proximity score from a Manhattan distance relative to a sensing radius.
    
    Parameters:
        dist (int): Manhattan distance to the target. A value >= NO_TARGET_DISTANCE indicates no target and yields 0.0.
        radius (int): Sensing radius used to normalize proximity; treated as radius+1 in the denominator to avoid division by zero.
    
    Returns:
        float: Proximity in the range [0.0, 1.0], where 1.0 corresponds to distance 0 and values decrease linearly to 0.0 as distance approaches or exceeds radius+1.
    """
    if dist >= NO_TARGET_DISTANCE:
        return 0.0
    return float(np.clip(1.0 - dist / float(max(1, radius + 1)), 0.0, 1.0))


def _visual_threat_score(world: "SpiderWorld", view: PerceivedTarget) -> float:
    """
    Compute a normalized visual threat score for a perceived predator view.
    
    The score combines the view's certainty, proximity (relative to the world's visible range), and whether the predator is visible or occluded, using fixed weights, and is clipped to the range [0, 1].
    
    Parameters:
        world (SpiderWorld): World context used to determine visible range for proximity normalization.
        view (PerceivedTarget): Perceived predator view whose certainty, visibility/occlusion, and distance are used.
    
    Returns:
        float: Threat score in [0.0, 1.0], where larger values indicate greater visual threat.
    """
    if view.position is None:
        return 0.0
    proximity = _normalized_proximity(view.dist, visible_range(world))
    return float(
        np.clip(
            0.6 * float(view.certainty)
            + 0.25 * proximity
            + 0.15 * max(float(view.visible), float(view.occluded)),
            0.0,
            1.0,
        )
    )


def _select_best_predator_view(
    world: "SpiderWorld",
    candidates: list[object],
    *,
    apply_noise: bool,
) -> PerceivedTarget | None:
    """
    Select the most threatening predator from a list and return its visual percept.

    Evaluates each candidate by computing the same visual percept that will be
    returned to the caller and scoring it by (threat_score, certainty, -dist,
    position). Returns the winner's already-computed view, or None if the
    candidate list is empty.
    """
    best_predator = None
    best_view = None
    best_tie_break: tuple[float, float, float, tuple[int, int]] | None = None
    for predator in candidates:
        base_view = _predator_visual_view(world, predator, apply_noise=apply_noise)
        tie_break = _predator_view_tie_break(world, predator, base_view)
        if best_tie_break is None or tie_break > best_tie_break:
            best_predator = predator
            best_view = base_view
            best_tie_break = tie_break
    if best_predator is None:
        return None
    return best_view


def _predator_view_tie_break(
    world: "SpiderWorld",
    predator: object,
    view: PerceivedTarget,
) -> tuple[float, float, float, tuple[int, int]]:
    """Return the canonical tie-break tuple for predator visual views."""
    return (
        _visual_threat_score(world, view),
        float(view.certainty),
        -float(view.dist),
        _predator_position(predator),
    )


def _sample_predator_views(
    world: "SpiderWorld",
    predators: Iterable[object] | None = None,
    *,
    apply_noise: bool,
) -> dict[int, PerceivedTarget]:
    """Sample each candidate predator's visual view exactly once."""
    return {
        id(predator): _predator_visual_view(world, predator, apply_noise=apply_noise)
        for predator in _predator_candidates(world, predators)
    }


def _predator_views_by_type_from_sampled_views(
    world: "SpiderWorld",
    sampled_predator_views: Mapping[int, PerceivedTarget],
    predators: Iterable[object] | None = None,
) -> dict[str, PerceivedTarget]:
    """Select the strongest sampled predator view for each detection style."""
    best_views: dict[str, PerceivedTarget] = {}
    best_tie_breaks: dict[str, tuple[float, float, float, tuple[int, int]]] = {}
    for predator in _predator_candidates(world, predators):
        view = sampled_predator_views.get(id(predator))
        if view is None:
            continue
        detection_style = _predator_detection_style(world, predator)
        tie_break = _predator_view_tie_break(world, predator, view)
        if detection_style not in best_tie_breaks or tie_break > best_tie_breaks[detection_style]:
            best_views[detection_style] = view
            best_tie_breaks[detection_style] = tie_break
    return best_views


def predators_visible_to_spider(
    world: "SpiderWorld",
    predators: Iterable[object] | None = None,
    *,
    apply_noise: bool = True,
) -> dict[str, PerceivedTarget]:
    """
    Selects the single most threatening predator percept for each detection style and returns them keyed by detection style.

    For each detection style (e.g., "visual", "olfactory"), the function evaluates candidate predators and selects the predator with the highest visual threat score; ties are resolved by higher certainty, then by shorter Manhattan distance, then by predator position. The returned PerceivedTarget for each selected predator is produced with visual noise applied according to the `apply_noise` flag.

    Parameters:
        world (SpiderWorld): The world context used to evaluate perception and threat.
        predators (Iterable[object] | None): Optional iterable of predator objects to consider. If None, the world's predator candidates are used.
        apply_noise (bool): If True, apply the world's visual noise model to the returned PerceivedTarget objects; if False, return deterministic base views.

    Returns:
        dict[str, PerceivedTarget]: Mapping from detection style string to the selected PerceivedTarget for that style. Styles with no candidates are omitted.
    """
    sampled_predator_views = _sample_predator_views(
        world,
        predators,
        apply_noise=apply_noise,
    )
    return _predator_views_by_type_from_sampled_views(
        world,
        sampled_predator_views,
        predators,
    )


def _predator_view_from_views(
    world: "SpiderWorld",
    predator_views_by_type: Mapping[str, PerceivedTarget] | None,
) -> PerceivedTarget:
    """
    Select the most threatening predator view from an already sampled per-type mapping.

    This lets callers reuse the same noisy sampled views for both the single
    predator percept and the per-type threat features without advancing visual
    noise twice in the same tick.
    """
    if not predator_views_by_type:
        return _no_target_perceived_target()

    best_view: PerceivedTarget | None = None
    best_tie_break: tuple[float, float, float, tuple[int, int]] | None = None
    for view in predator_views_by_type.values():
        position = view.position if view.position is not None else (-1, -1)
        tie_break = (
            _visual_threat_score(world, view),
            float(view.certainty),
            -float(view.dist),
            position,
        )
        if best_tie_break is None or tie_break > best_tie_break:
            best_view = view
            best_tie_break = tie_break
    return best_view if best_view is not None else _no_target_perceived_target()


def predator_detects_spider(world: "SpiderWorld", predator: object) -> bool:
    """
    Determine whether a specific predator currently detects the spider.
    """
    cfg = world.operational_profile.perception
    shelter_role = world.shelter_role_at(world.spider_pos())
    if shelter_role == "deep":
        return False
    profile = _predator_profile_fields(world, predator)
    source = _predator_position(predator)
    if profile["detection_style"] == "olfactory":
        strength, _, _, _ = smell_gradient(
            world,
            [world.spider_pos()],
            radius=max(1, int(profile["smell_range"])),
            origin=source,
            apply_noise=False,
        )
        if shelter_role == "inside":
            strength -= float(cfg["lizard_detection_inside_penalty"])
        return bool(float(strength) >= float(profile["detection_threshold"]))

    dist = world.manhattan(world.spider_pos(), source)
    effective_range = int(profile["vision_range"])
    spider_is_moving = world.state.last_move_dx != 0 or world.state.last_move_dy != 0
    if spider_is_moving:
        effective_range += round(cfg["lizard_detection_range_motion_bonus"])
    if world.terrain_at(world.spider_pos()) == CLUTTER:
        effective_range -= round(cfg["lizard_detection_range_clutter_penalty"])
    if world.terrain_at(world.spider_pos()) == NARROW:
        effective_range -= round(cfg["lizard_detection_range_narrow_penalty"])
    if shelter_role == "inside":
        effective_range -= round(cfg["lizard_detection_range_inside_penalty"])
    if dist > max(1, effective_range):
        return False
    if not has_line_of_sight(world, source, world.spider_pos()):
        return False
    certainty = visibility_confidence(
        world,
        source=source,
        target=world.spider_pos(),
        dist=dist,
        radius=max(1, effective_range),
        motion_bonus=cfg["lizard_detection_motion_bonus"] if spider_is_moving else 0.0,
    )
    if shelter_role == "inside":
        certainty -= cfg["lizard_detection_inside_penalty"]
    return bool(certainty >= float(profile["detection_threshold"]))


def lizard_detects_spider(world: "SpiderWorld") -> bool:
    """
    Run the world's primary predator detection check for the spider.
    
    This is a backward-compatible wrapper that calls the generic predator detection logic
    for the world's primary predator (`world.lizard`).
    
    Returns:
        True if the primary predator detects the spider, False otherwise.
    """
    return predator_detects_spider(world, world.lizard)


def predator_visible_to_spider(
    world: "SpiderWorld",
    predators: Iterable[object] | None = None,
    *,
    apply_noise: bool = True,
) -> PerceivedTarget:
    """
    Select the single most threatening visible predator and return its visual percept.

    Evaluates candidate predators (from the provided iterable or the world) by computing a base, noise-free visual view for each and scoring them by visual threat, certainty, distance, and position. The selected predator's visual view is returned with visual noise applied when requested.

    Parameters:
        world (SpiderWorld): The simulation world used to compute views and scoring.
        predators (Iterable[object] | None): Optional iterable of predator objects to consider; if None, world predators are used.
        apply_noise (bool): If True, apply the world's visual noise model to the returned percept.

    Returns:
        PerceivedTarget: The visual percept for the most threatening visible predator. If no predator is detected, returns a "no target" PerceivedTarget with zeros for perception fields and position set to None.
    """
    best = _select_best_predator_view(
        world, _predator_candidates(world, predators), apply_noise=apply_noise
    )
    if best is None:
        return _no_target_perceived_target()
    return best


def compute_per_type_threats(
    world: "SpiderWorld",
    *,
    sampled_predator_views: Mapping[int, PerceivedTarget] | None = None,
    predator_views_by_type: Mapping[str, PerceivedTarget] | None = None,
) -> dict[str, float]:
    """
    Compute aggregated threat intensities for visual and olfactory predators and determine the dominant predator type.

    The policy-facing threat values use the same noisy visual and olfactory
    percepts as the observation view, so occlusion/noise affect these features
    consistently with predator visibility and smell inputs.
    
    Returns:
        dict[str, float]: Mapping with keys:
            - "visual_predator_threat": float in [0,1] representing the maximum visual-threat score among visual predators.
            - "olfactory_predator_threat": float in [0,1] representing the maximum olfactory-threat score among olfactory predators.
            - "dominant_predator_type": numeric code identifying the dominant predator type; one of DOMINANT_PREDATOR_TYPE_NONE, DOMINANT_PREDATOR_TYPE_OLFACTORY, or DOMINANT_PREDATOR_TYPE_VISUAL.
    """
    threats = {
        "visual_predator_threat": 0.0,
        "olfactory_predator_threat": 0.0,
        "dominant_predator_type": DOMINANT_PREDATOR_TYPE_NONE,
    }
    predators = _predator_candidates(world)
    if sampled_predator_views is None:
        sampled_predator_views = _sample_predator_views(
            world,
            predators,
            apply_noise=True,
        )
    if predator_views_by_type is None:
        predator_views_by_type = _predator_views_by_type_from_sampled_views(
            world,
            sampled_predator_views,
            predators,
        )
    for predator in predators:
        profile = _predator_profile_fields(world, predator)
        position = _predator_position(predator)
        detection_style = str(profile["detection_style"])
        distance = world.manhattan(world.spider_pos(), position)
        visual_view = sampled_predator_views.get(id(predator))
        if visual_view is None:
            visual_view = (
                predator_views_by_type.get(detection_style)
                if predator_views_by_type is not None
                else None
            )
        if visual_view is None:
            visual_view = _predator_visual_view(world, predator, apply_noise=True)
        if detection_style == "olfactory":
            smell_strength, _, _, _ = smell_gradient(
                world,
                [position],
                radius=max(1, int(profile["smell_range"])),
            )
            proximity = _normalized_proximity(distance, int(profile["smell_range"]))
            confidence = max(float(smell_strength), 0.5 * float(visual_view.certainty))
            threat_value = float(np.clip(0.55 * confidence + 0.45 * proximity, 0.0, 1.0))
            threats["olfactory_predator_threat"] = max(
                threats["olfactory_predator_threat"],
                threat_value,
            )
        else:
            proximity = _normalized_proximity(distance, int(profile["vision_range"]))
            confidence = max(float(visual_view.certainty), float(visual_view.occluded))
            threat_value = float(np.clip(0.55 * confidence + 0.45 * proximity, 0.0, 1.0))
            threats["visual_predator_threat"] = max(
                threats["visual_predator_threat"],
                threat_value,
            )

    visual_threat = threats["visual_predator_threat"]
    olfactory_threat = threats["olfactory_predator_threat"]
    if visual_threat <= 0.0 and olfactory_threat <= 0.0:
        threats["dominant_predator_type"] = DOMINANT_PREDATOR_TYPE_NONE
    elif olfactory_threat > visual_threat:
        threats["dominant_predator_type"] = DOMINANT_PREDATOR_TYPE_OLFACTORY
    else:
        threats["dominant_predator_type"] = DOMINANT_PREDATOR_TYPE_VISUAL
    return threats


def _unpack_trace_view(trace_view: Mapping[str, object]) -> tuple[float, float, float, float, float]:
    """
    Return the direction, strength, and scan-heading components from a trace-view mapping.
    
    Parameters:
        trace_view (Mapping[str, object]): Mapping containing keys "dx", "dy", "strength",
            "heading_dx", and "heading_dy".
    
    Returns:
        tuple[float, float, float, float, float]: A tuple
        (dx, dy, strength, heading_dx, heading_dy) converted to floats.
    """
    return (
        float(trace_view["dx"]),
        float(trace_view["dy"]),
        float(trace_view["strength"]),
        float(trace_view["heading_dx"]),
        float(trace_view["heading_dy"]),
    )


def serialize_observation_view(observation_key: str, view: ObservationView) -> np.ndarray:
    """
    Serialize an ObservationView into a fixed-size numpy vector using the interface registered for the given observation key.
    
    Parameters:
        observation_key (str): Key identifying which observation interface to use.
        view (ObservationView): The observation view to serialize; its mapping form will be converted to a vector.
    
    Returns:
        np.ndarray: One-dimensional numpy array containing the serialized observation vector.
    """
    interface = OBSERVATION_INTERFACE_BY_KEY[observation_key]
    return interface.vector_from_mapping(view.as_mapping())


def build_visual_observation(
    *,
    food_view: PerceivedTarget,
    shelter_view: PerceivedTarget,
    predator_view: PerceivedTarget,
    world: "SpiderWorld" | None = None,
    heading_dx: float,
    heading_dy: float,
    foveal_scan_age: float | None = None,
    food_trace_strength: float,
    food_trace_heading: tuple[float, float] = (0.0, 0.0),
    shelter_trace_strength: float,
    shelter_trace_heading: tuple[float, float] = (0.0, 0.0),
    predator_trace_strength: float,
    predator_trace_heading: tuple[float, float] = (0.0, 0.0),
    predator_motion_salience_value: float,
    visual_predator_threat: float,
    olfactory_predator_threat: float,
    day: float,
    night: float,
) -> VisualObservation:
    """
    Constructs a VisualObservation from perceived food, shelter, and predator targets plus heading, scan recency, trace strengths, predator salience/threats, and day/night flags.
    
    Parameters:
        food_view (PerceivedTarget): Perception for the nearest food target; populates food_* fields.
        shelter_view (PerceivedTarget): Perception for the nearest shelter target; populates shelter_* fields.
        predator_view (PerceivedTarget): Perception for the predator; populates predator_* fields.
        world (SpiderWorld | None): Optional world used to derive foveal_scan_age when it is not provided.
        heading_dx (float): Heading x component included as heading_dx.
        heading_dy (float): Heading y component included as heading_dy.
        foveal_scan_age (float | None): Normalized scan age in [0, 1], or None to derive it from world.
        food_trace_strength (float): Trace strength value for food_trace_strength.
        food_trace_heading (tuple[float, float]): Heading active when the food trace was refreshed.
        shelter_trace_strength (float): Trace strength value for shelter_trace_strength.
        shelter_trace_heading (tuple[float, float]): Heading active when the shelter trace was refreshed.
        predator_trace_strength (float): Trace strength value for predator_trace_strength.
        predator_trace_heading (tuple[float, float]): Heading active when the predator trace was refreshed.
        predator_motion_salience_value (float): Motion salience value included as predator_motion_salience.
        visual_predator_threat (float): Visual predator threat score included as visual_predator_threat.
        olfactory_predator_threat (float): Olfactory predator threat score included as olfactory_predator_threat.
        day (float): Day indicator value included as day.
        night (float): Night indicator value included as night.
    
    Returns:
        VisualObservation: Observation whose food_*, shelter_*, and predator_* perception fields are taken from the corresponding PerceivedTarget inputs, and which includes heading, scan recency, trace strengths, predator salience/threats, and day/night values.
    """
    if foveal_scan_age is None:
        foveal_scan_age = _normalized_scan_age(world, heading_dx, heading_dy) if world is not None else 1.0
    food_trace_heading_dx, food_trace_heading_dy = food_trace_heading
    shelter_trace_heading_dx, shelter_trace_heading_dy = shelter_trace_heading
    predator_trace_heading_dx, predator_trace_heading_dy = predator_trace_heading
    return VisualObservation(
        food_visible=food_view.visible,
        food_certainty=food_view.certainty,
        food_occluded=food_view.occluded,
        food_dx=food_view.dx,
        food_dy=food_view.dy,
        shelter_visible=shelter_view.visible,
        shelter_certainty=shelter_view.certainty,
        shelter_occluded=shelter_view.occluded,
        shelter_dx=shelter_view.dx,
        shelter_dy=shelter_view.dy,
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        predator_occluded=predator_view.occluded,
        predator_dx=predator_view.dx,
        predator_dy=predator_view.dy,
        heading_dx=heading_dx,
        heading_dy=heading_dy,
        foveal_scan_age=float(np.clip(foveal_scan_age, 0.0, 1.0)),
        food_trace_strength=food_trace_strength,
        food_trace_heading_dx=food_trace_heading_dx,
        food_trace_heading_dy=food_trace_heading_dy,
        shelter_trace_strength=shelter_trace_strength,
        shelter_trace_heading_dx=shelter_trace_heading_dx,
        shelter_trace_heading_dy=shelter_trace_heading_dy,
        predator_trace_strength=predator_trace_strength,
        predator_trace_heading_dx=predator_trace_heading_dx,
        predator_trace_heading_dy=predator_trace_heading_dy,
        predator_motion_salience=predator_motion_salience_value,
        visual_predator_threat=visual_predator_threat,
        olfactory_predator_threat=olfactory_predator_threat,
        day=day,
        night=night,
    )


def build_sensory_observation(
    world: "SpiderWorld",
    *,
    food_smell_strength: float,
    food_smell_dx: float,
    food_smell_dy: float,
    predator_smell_strength: float,
    predator_smell_dx: float,
    predator_smell_dy: float,
    light: float,
) -> SensoryObservation:
    """
    Builds a SensoryObservation combining the spider's current internal state with external smell and light inputs.
    
    Parameters:
        world (SpiderWorld): World object used to read the spider's recent_pain, recent_contact, health, hunger, and fatigue state.
        food_smell_strength (float): Food smell total strength (0.0-1.0) within sensing radius.
        food_smell_dx (float): Food smell horizontal gradient component (signed, normalized).
        food_smell_dy (float): Food smell vertical gradient component (signed, normalized).
        predator_smell_strength (float): Predator smell total strength (0.0-1.0) within sensing radius.
        predator_smell_dx (float): Predator smell horizontal gradient component (signed, normalized).
        predator_smell_dy (float): Predator smell vertical gradient component (signed, normalized).
        light (float): Ambient light level.
    
    Returns:
        SensoryObservation: Observation populated with recent_pain, recent_contact, health, hunger, fatigue,
        the provided food and predator smell gradient components, and the ambient light value.
    """
    return SensoryObservation(
        recent_pain=world.state.recent_pain,
        recent_contact=world.state.recent_contact,
        health=world.state.health,
        hunger=world.state.hunger,
        fatigue=world.state.fatigue,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        predator_smell_strength=predator_smell_strength,
        predator_smell_dx=predator_smell_dx,
        predator_smell_dy=predator_smell_dy,
        light=light,
    )


def build_hunger_observation(
    world: "SpiderWorld",
    *,
    on_food: float,
    food_view: PerceivedTarget,
    food_smell_strength: float,
    food_smell_dx: float,
    food_smell_dy: float,
    food_trace: tuple[float, float, float, float, float],
    food_memory: tuple[float, float, float],
) -> HungerObservation:
    """Build the hunger observation from food perception, smell, trace, and memory signals."""
    (
        food_trace_dx,
        food_trace_dy,
        food_trace_strength,
        food_trace_heading_dx,
        food_trace_heading_dy,
    ) = food_trace
    food_mem_dx, food_mem_dy, food_mem_age = food_memory
    return HungerObservation(
        hunger=world.state.hunger,
        on_food=on_food,
        food_visible=food_view.visible,
        food_certainty=food_view.certainty,
        food_occluded=food_view.occluded,
        food_dx=food_view.dx,
        food_dy=food_view.dy,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        food_trace_dx=food_trace_dx,
        food_trace_dy=food_trace_dy,
        food_trace_strength=food_trace_strength,
        food_trace_heading_dx=food_trace_heading_dx,
        food_trace_heading_dy=food_trace_heading_dy,
        food_memory_dx=food_mem_dx,
        food_memory_dy=food_mem_dy,
        food_memory_age=food_mem_age,
    )


def build_sleep_observation(
    world: "SpiderWorld",
    *,
    on_shelter: float,
    night: float,
    sleep_phase_level: float,
    rest_streak_norm: float,
    shelter_role_level: float,
    shelter_trace: tuple[float, float, float, float, float],
    shelter_memory: tuple[float, float, float],
) -> SleepObservation:
    """Build the sleep observation from shelter state, circadian state, trace, and memory."""
    (
        shelter_trace_dx,
        shelter_trace_dy,
        shelter_trace_strength,
        shelter_trace_heading_dx,
        shelter_trace_heading_dy,
    ) = shelter_trace
    shelter_mem_dx, shelter_mem_dy, shelter_mem_age = shelter_memory
    return SleepObservation(
        fatigue=world.state.fatigue,
        hunger=world.state.hunger,
        on_shelter=on_shelter,
        night=night,
        health=world.state.health,
        recent_pain=world.state.recent_pain,
        sleep_phase_level=sleep_phase_level,
        rest_streak_norm=rest_streak_norm,
        sleep_debt=world.state.sleep_debt,
        shelter_role_level=shelter_role_level,
        shelter_trace_dx=shelter_trace_dx,
        shelter_trace_dy=shelter_trace_dy,
        shelter_trace_strength=shelter_trace_strength,
        shelter_trace_heading_dx=shelter_trace_heading_dx,
        shelter_trace_heading_dy=shelter_trace_heading_dy,
        shelter_memory_dx=shelter_mem_dx,
        shelter_memory_dy=shelter_mem_dy,
        shelter_memory_age=shelter_mem_age,
    )


def build_alert_observation(
    world: "SpiderWorld",
    *,
    predator_view: PerceivedTarget,
    predator_smell_strength: float,
    predator_motion_salience_value: float,
    visual_predator_threat: float,
    olfactory_predator_threat: float,
    dominant_predator_type: float,
    on_shelter: float,
    night: float,
    predator_trace: tuple[float, float, float, float, float],
    predator_memory: tuple[float, float, float],
    escape_memory: tuple[float, float, float],
) -> AlertObservation:
    """
    Constructs an AlertObservation combining current predator perception, threat metrics, internal state, and trace/memory vectors.
    
    Parameters:
        predator_view (PerceivedTarget): Perceptual summary of the most relevant predator (visible/certainty/occluded/dx/dy).
        predator_smell_strength (float): Olfactory signal strength for the predator at the spider (0-1).
        predator_motion_salience_value (float): Motion-based salience bonus contributed by predator movement.
        visual_predator_threat (float): Aggregated visual-threat score (0-1).
        olfactory_predator_threat (float): Aggregated olfactory-threat score (0-1).
        dominant_predator_type (float): Numeric encoding of the dominant predator detection type (none/visual/olfactory).
        on_shelter (float): Indicator that the spider is on shelter (0 or 1).
        night (float): Night indicator (0 or 1).
        predator_trace (tuple[float, float, float, float, float]): Trace vector from predator sensing as
            (dx, dy, strength, heading_dx, heading_dy).
        predator_memory (tuple[float, float, float]): Stored predator memory as (dx, dy, age).
        escape_memory (tuple[float, float, float]): Stored escape memory as (dx, dy, age).
    
    Returns:
        AlertObservation: A populated AlertObservation containing predator perception fields, threat values, internal pain/contact state, shelter/night flags, predator trace and memory, and escape memory.
    """
    (
        predator_trace_dx,
        predator_trace_dy,
        predator_trace_strength,
        predator_trace_heading_dx,
        predator_trace_heading_dy,
    ) = predator_trace
    predator_mem_dx, predator_mem_dy, predator_mem_age = predator_memory
    escape_mem_dx, escape_mem_dy, escape_mem_age = escape_memory
    dominant_predator_none = 1.0 if dominant_predator_type == DOMINANT_PREDATOR_TYPE_NONE else 0.0
    dominant_predator_visual = 1.0 if dominant_predator_type == DOMINANT_PREDATOR_TYPE_VISUAL else 0.0
    dominant_predator_olfactory = 1.0 if dominant_predator_type == DOMINANT_PREDATOR_TYPE_OLFACTORY else 0.0
    return AlertObservation(
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        predator_occluded=predator_view.occluded,
        predator_dx=predator_view.dx,
        predator_dy=predator_view.dy,
        predator_smell_strength=predator_smell_strength,
        predator_motion_salience=predator_motion_salience_value,
        visual_predator_threat=visual_predator_threat,
        olfactory_predator_threat=olfactory_predator_threat,
        dominant_predator_none=dominant_predator_none,
        dominant_predator_visual=dominant_predator_visual,
        dominant_predator_olfactory=dominant_predator_olfactory,
        recent_pain=world.state.recent_pain,
        recent_contact=world.state.recent_contact,
        on_shelter=on_shelter,
        night=night,
        predator_trace_dx=predator_trace_dx,
        predator_trace_dy=predator_trace_dy,
        predator_trace_strength=predator_trace_strength,
        predator_trace_heading_dx=predator_trace_heading_dx,
        predator_trace_heading_dy=predator_trace_heading_dy,
        predator_memory_dx=predator_mem_dx,
        predator_memory_dy=predator_mem_dy,
        predator_memory_age=predator_mem_age,
        escape_memory_dx=escape_mem_dx,
        escape_memory_dy=escape_mem_dy,
        escape_memory_age=escape_mem_age,
    )


def build_action_context_observation(
    world: "SpiderWorld",
    *,
    on_food: float,
    on_shelter: float,
    predator_view: PerceivedTarget,
    day: float,
    night: float,
    shelter_role_level: float,
) -> ActionContextObservation:
    """Build the action-center arbitration context from local state and perceived threat."""
    return ActionContextObservation(
        hunger=world.state.hunger,
        fatigue=world.state.fatigue,
        health=world.state.health,
        recent_pain=world.state.recent_pain,
        recent_contact=world.state.recent_contact,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        day=day,
        night=night,
        last_move_dx=float(world.state.last_move_dx),
        last_move_dy=float(world.state.last_move_dy),
        sleep_debt=world.state.sleep_debt,
        shelter_role_level=shelter_role_level,
    )


def build_motor_context_observation(
    world: "SpiderWorld",
    *,
    on_food: float,
    on_shelter: float,
    predator_view: PerceivedTarget,
    day: float,
    night: float,
    shelter_role_level: float,
) -> MotorContextObservation:
    """Build the motor execution context used to execute the action-center intent.

    The context includes local embodiment signals so the motor stage can see
    body orientation, terrain movement cost, fatigue, and momentum.
    """
    terrain_difficulty = TERRAIN_DIFFICULTY.get(
        world.terrain_at(world.spider_pos()),
        0.0,
    )
    return MotorContextObservation(
        on_food=on_food,
        on_shelter=on_shelter,
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        day=day,
        night=night,
        last_move_dx=float(world.state.last_move_dx),
        last_move_dy=float(world.state.last_move_dy),
        shelter_role_level=shelter_role_level,
        heading_dx=float(world.state.heading_dx),
        heading_dy=float(world.state.heading_dy),
        terrain_difficulty=terrain_difficulty,
        fatigue=world.state.fatigue,
        momentum=float(np.clip(world.state.momentum, 0.0, 1.0)),
    )


def observe_world(world: "SpiderWorld") -> dict[str, object]:
    """
    Assembles and serializes the spider's observation vectors and diagnostic metadata from the current world state.
    
    Builds all perception, smell, memory and internal-state views, converts each typed observation into a flat numpy vector, validates expected shapes, and returns those vectors together with a comprehensive "meta" diagnostics mapping.
    
    Parameters:
        world (SpiderWorld): Simulation state and environment used to compute perceptions, gradients, memory vectors, traces, and internal-state features.
    
    Returns:
        dict[str, object]: A dictionary containing:
            - Serialized observation vectors keyed by view name: "visual", "sensory", "hunger", "sleep", "alert", "action_context", "motor_context". "motor_extra" is provided as a copy of "motor_context" for compatibility.
            - "meta": a dict of diagnostic fields including nearest distances, diagnostic-only privileged values under "diagnostic", day/night flags, on_shelter/on_food booleans, predator visibility flag, lizard position and mode, shelter role and terrain at the spider, phase and sleep metrics, heading, configured profiles, predator motion salience, per-vision PerceivedTarget entries under "vision", perceptual trace views under "percept_traces", memory vectors with TTLs under "memory_vectors", and individual memory ages (or None when absent).
    """
    from .memory import MEMORY_TTLS, memory_vector

    _, food_dist = world.nearest(world.food_positions)
    _, shelter_dist = world.nearest(world.shelter_cells)
    phase_sin, phase_cos = world.phase_features()
    day = 0.0 if world.is_night() else 1.0
    night = 1.0 - day
    light = world.light_level()
    on_shelter = float(world.on_shelter())
    on_food = float(world.on_food())
    open_exposed = float(world.shelter_role_at(world.spider_pos()) == "outside")
    shelter_role_level = world.shelter_role_level()
    sleep_phase = world.sleep_phase_level()
    rest_norm = world.rest_streak_norm()
    heading_dx = float(world.state.heading_dx)
    heading_dy = float(world.state.heading_dy)
    food_trace_view = trace_view(world, world.state.food_trace)
    shelter_trace_view = trace_view(world, world.state.shelter_trace)
    predator_trace_view = trace_view(world, world.state.predator_trace)
    (
        food_trace_dx,
        food_trace_dy,
        food_trace_strength,
        food_trace_heading_dx,
        food_trace_heading_dy,
    ) = _unpack_trace_view(food_trace_view)
    (
        shelter_trace_dx,
        shelter_trace_dy,
        shelter_trace_strength,
        shelter_trace_heading_dx,
        shelter_trace_heading_dy,
    ) = _unpack_trace_view(shelter_trace_view)
    (
        predator_trace_dx,
        predator_trace_dy,
        predator_trace_strength,
        predator_trace_heading_dx,
        predator_trace_heading_dy,
    ) = _unpack_trace_view(predator_trace_view)

    vision_radius = visible_range(world)
    food_view = visible_object(world, world.food_positions, radius=vision_radius)
    shelter_view = visible_object(world, world.shelter_cells, radius=vision_radius)
    sampled_predator_views = _sample_predator_views(world, apply_noise=True)
    predator_views_by_type = _predator_views_by_type_from_sampled_views(
        world,
        sampled_predator_views,
    )
    predator_view = _predator_view_from_views(world, predator_views_by_type)
    per_type_threats = compute_per_type_threats(
        world,
        sampled_predator_views=sampled_predator_views,
        predator_views_by_type=predator_views_by_type,
    )
    motion_salience = predator_motion_salience(world, predator_view=predator_view)

    food_smell_strength, food_smell_dx, food_smell_dy, _ = smell_gradient(
        world,
        world.food_positions,
        radius=world.food_smell_range,
    )
    predator_smell_strength, predator_smell_dx, predator_smell_dy, _ = smell_gradient(
        world,
        world.predator_positions(),
        radius=world.predator_smell_range,
    )

    food_mem_dx, food_mem_dy, food_mem_age = memory_vector(world, world.state.food_memory, ttl_name="food")
    shelter_mem_dx, shelter_mem_dy, shelter_mem_age = memory_vector(world, world.state.shelter_memory, ttl_name="shelter")
    predator_mem_dx, predator_mem_dy, predator_mem_age = memory_vector(world, world.state.predator_memory, ttl_name="predator")
    escape_mem_dx, escape_mem_dy, escape_mem_age = memory_vector(world, world.state.escape_memory, ttl_name="escape")

    visual_observation = build_visual_observation(
        food_view=food_view,
        shelter_view=shelter_view,
        predator_view=predator_view,
        world=world,
        heading_dx=heading_dx,
        heading_dy=heading_dy,
        food_trace_strength=food_trace_strength,
        food_trace_heading=(food_trace_heading_dx, food_trace_heading_dy),
        shelter_trace_strength=shelter_trace_strength,
        shelter_trace_heading=(shelter_trace_heading_dx, shelter_trace_heading_dy),
        predator_trace_strength=predator_trace_strength,
        predator_trace_heading=(predator_trace_heading_dx, predator_trace_heading_dy),
        predator_motion_salience_value=motion_salience,
        visual_predator_threat=per_type_threats["visual_predator_threat"],
        olfactory_predator_threat=per_type_threats["olfactory_predator_threat"],
        day=day,
        night=night,
    )
    sensory_observation = build_sensory_observation(
        world,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        predator_smell_strength=predator_smell_strength,
        predator_smell_dx=predator_smell_dx,
        predator_smell_dy=predator_smell_dy,
        light=light,
    )
    hunger_observation = build_hunger_observation(
        world,
        on_food=on_food,
        food_view=food_view,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        food_trace=(
            food_trace_dx,
            food_trace_dy,
            food_trace_strength,
            food_trace_heading_dx,
            food_trace_heading_dy,
        ),
        food_memory=(food_mem_dx, food_mem_dy, food_mem_age),
    )
    sleep_observation = build_sleep_observation(
        world,
        on_shelter=on_shelter,
        night=night,
        sleep_phase_level=sleep_phase,
        rest_streak_norm=rest_norm,
        shelter_role_level=shelter_role_level,
        shelter_trace=(
            shelter_trace_dx,
            shelter_trace_dy,
            shelter_trace_strength,
            shelter_trace_heading_dx,
            shelter_trace_heading_dy,
        ),
        shelter_memory=(shelter_mem_dx, shelter_mem_dy, shelter_mem_age),
    )
    alert_observation = build_alert_observation(
        world,
        predator_view=predator_view,
        predator_smell_strength=predator_smell_strength,
        predator_motion_salience_value=motion_salience,
        visual_predator_threat=per_type_threats["visual_predator_threat"],
        olfactory_predator_threat=per_type_threats["olfactory_predator_threat"],
        dominant_predator_type=per_type_threats["dominant_predator_type"],
        on_shelter=on_shelter,
        night=night,
        predator_trace=(
            predator_trace_dx,
            predator_trace_dy,
            predator_trace_strength,
            predator_trace_heading_dx,
            predator_trace_heading_dy,
        ),
        predator_memory=(predator_mem_dx, predator_mem_dy, predator_mem_age),
        escape_memory=(escape_mem_dx, escape_mem_dy, escape_mem_age),
    )
    action_context_observation = build_action_context_observation(
        world,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_view=predator_view,
        day=day,
        night=night,
        shelter_role_level=shelter_role_level,
    )
    motor_context_observation = build_motor_context_observation(
        world,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_view=predator_view,
        day=day,
        night=night,
        shelter_role_level=shelter_role_level,
    )

    obs = {
        "visual": serialize_observation_view("visual", visual_observation),
        "sensory": serialize_observation_view("sensory", sensory_observation),
        "hunger": serialize_observation_view("hunger", hunger_observation),
        "sleep": serialize_observation_view("sleep", sleep_observation),
        "alert": serialize_observation_view("alert", alert_observation),
        "action_context": serialize_observation_view("action_context", action_context_observation),
        "motor_context": serialize_observation_view("motor_context", motor_context_observation),
    }
    obs["motor_extra"] = obs["motor_context"].copy()

    for key, dim in OBSERVATION_DIMS.items():
        if key not in obs:
            continue
        if obs[key].shape != (dim,):
            raise ValueError(f"Observation '{key}' expected shape {(dim,)}, received {obs[key].shape}")

    # Diagnostic-only privileged values are kept out of network-facing observations
    # but remain available here for audits and offline inspection.
    predator_positions = world.predator_positions()
    diagnostic_predator_dist = min(
        (world.manhattan(world.spider_pos(), position) for position in predator_positions),
        default=NO_TARGET_DISTANCE,
    )
    diagnostic_home_dx, diagnostic_home_dy, diagnostic_home_dist = world._relative(
        world.safest_shelter_target()
    )
    dominant_type_value = float(per_type_threats["dominant_predator_type"])
    if dominant_type_value == DOMINANT_PREDATOR_TYPE_VISUAL:
        dominant_type_label = "visual"
    elif dominant_type_value == DOMINANT_PREDATOR_TYPE_OLFACTORY:
        dominant_type_label = "olfactory"
    else:
        dominant_type_label = "none"

    obs["meta"] = {
        "food_dist": food_dist,
        "shelter_dist": shelter_dist,
        "diagnostic": {
            "diagnostic_predator_dist": diagnostic_predator_dist,
            "diagnostic_home_dx": diagnostic_home_dx,
            "diagnostic_home_dy": diagnostic_home_dy,
            "diagnostic_home_dist": diagnostic_home_dist,
        },
        "predator_dist_visible": predator_view.dist,
        "night": bool(night),
        "day": bool(day),
        "on_shelter": bool(on_shelter),
        "on_food": bool(on_food),
        "predator_visible": bool(predator_view.visible > 0.5),
        "lizard_x": world.lizard.x,
        "lizard_y": world.lizard.y,
        "lizard_mode": world.lizard.mode,
        "open_exposed": bool(open_exposed),
        "phase_sin": phase_sin,
        "phase_cos": phase_cos,
        "sleep_phase": world.state.sleep_phase,
        "sleep_phase_level": sleep_phase,
        "rest_streak": world.state.rest_streak,
        "sleep_debt": world.state.sleep_debt,
        "shelter_role": world.shelter_role_at(world.spider_pos()),
        "shelter_role_level": shelter_role_level,
        "terrain": world.terrain_at(world.spider_pos()),
        "map_template": world.map_template_name,
        "reward_profile": world.reward_profile,
        "noise_profile": world.noise_profile.name,
        "heading": {"dx": int(world.state.heading_dx), "dy": int(world.state.heading_dy)},
        "active_sensing": {
            "foveal_scan_age": float(visual_observation.foveal_scan_age),
            "raw_foveal_scan_age": int(
                world._scan_age_for_heading(world.state.heading_dx, world.state.heading_dy)
            ),
            "max_scan_age": _max_scan_age(world),
        },
        "predator_motion_salience": motion_salience,
        "visual_predator_threat": per_type_threats["visual_predator_threat"],
        "olfactory_predator_threat": per_type_threats["olfactory_predator_threat"],
        "dominant_predator_type": dominant_type_value,
        "dominant_predator_type_label": dominant_type_label,
        "vision": {
            "food": asdict(food_view),
            "shelter": asdict(shelter_view),
            "predator": asdict(predator_view),
            "predators_by_type": {
                style: asdict(view)
                for style, view in predator_views_by_type.items()
            },
        },
        "predators": [
            {
                "index": idx,
                **asdict(predator),
            }
            for idx, predator in enumerate(_predator_candidates(world))
        ],
        "percept_traces": {
            "food": food_trace_view,
            "shelter": shelter_trace_view,
            "predator": predator_trace_view,
        },
        "memory_vectors": {
            "food": {"dx": food_mem_dx, "dy": food_mem_dy, "age": food_mem_age, "ttl": MEMORY_TTLS["food"]},
            "predator": {"dx": predator_mem_dx, "dy": predator_mem_dy, "age": predator_mem_age, "ttl": MEMORY_TTLS["predator"]},
            "shelter": {"dx": shelter_mem_dx, "dy": shelter_mem_dy, "age": shelter_mem_age, "ttl": MEMORY_TTLS["shelter"]},
            "escape": {"dx": escape_mem_dx, "dy": escape_mem_dy, "age": escape_mem_age, "ttl": MEMORY_TTLS["escape"]},
        },
        "food_memory_age": world.state.food_memory.age if world.state.food_memory.target is not None else None,
        "predator_memory_age": world.state.predator_memory.age if world.state.predator_memory.target is not None else None,
        "shelter_memory_age": world.state.shelter_memory.age if world.state.shelter_memory.target is not None else None,
        "escape_memory_age": world.state.escape_memory.age if world.state.escape_memory.target is not None else None,
    }
    return obs
