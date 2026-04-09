from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable

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
from .maps import BLOCKED, CLUTTER, NARROW

if TYPE_CHECKING:
    from .world import SpiderWorld


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


def predator_motion_salience(
    world: "SpiderWorld",
    *,
    predator_view: PerceivedTarget | None = None,
) -> float:
    """
    Return the explicit motion salience currently contributed by the predator.
    """
    perceived = (
        predator_view is not None
        and (predator_view.visible > 0.0 or predator_view.occluded > 0.0)
    )
    return (
        float(world.operational_profile.perception["predator_motion_bonus"])
        if perceived and world.lizard.mode in {"CHASE", "INVESTIGATE"}
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
        "risk": "high",
        "modules": ["alert", "action_context", "motor_context"],
        "source": "spider_cortex_sim.perception.observe_world",
        "evidence": "predator_dist_real = world.manhattan(world.spider_pos(), world.lizard_pos())",
        "notes": "Uses the lizard's real position even when useful detection should depend on vision, smell, or memory.",
    },
    "home_vector": {
        "classification": "world_derived_navigation_hint",
        "risk": "high",
        "modules": ["sleep", "alert"],
        "source": "spider_cortex_sim.perception.observe_world",
        "evidence": "shelter_target = world.safest_shelter_target(); home_dx/home_dy/home_dist = world._relative(shelter_target)",
        "notes": "Provides a ready-made direction toward safe shelter, which greatly shortens the navigation problem.",
    },
    "predator_memory_vector": {
        "classification": "world_owned_memory",
        "risk": "medium",
        "modules": ["alert"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Plausible as explicit memory, but the update remains world-owned rather than learned.",
    },
    "shelter_memory_vector": {
        "classification": "world_owned_memory",
        "risk": "high",
        "modules": ["sleep"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Inherits the target from safest_shelter_target and therefore can act as a persistent privileged waypoint.",
    },
    "escape_memory_vector": {
        "classification": "world_owned_memory",
        "risk": "medium",
        "modules": ["alert"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "The escape route is derived from the world's last movement, without uncertainty or agent-driven selection.",
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


def _heading_allows_target(
    world: "SpiderWorld",
    source: tuple[int, int],
    target: tuple[int, int],
) -> bool:
    """
    Check whether a target falls inside the spider's forward-facing 180 degree FOV.
    """
    if source != world.spider_pos():
        return True
    heading = (int(world.state.heading_dx), int(world.state.heading_dy))
    if heading == (0, 0):
        return True
    rel_x = int(target[0] - source[0])
    rel_y = int(target[1] - source[1])
    if rel_x == 0 and rel_y == 0:
        return True
    return (heading[0] * rel_x + heading[1] * rel_y) >= 0


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
) -> float:
    """
    Compute a visibility certainty score for a target from a source location.
    
    The score starts from a distance-based falloff, then is adjusted by a motion bonus and penalties for target/source terrain and night; final value is clipped to the range [0.0, 1.0].
    
    Parameters:
        world (SpiderWorld): World context used to query terrain and time-of-day.
        source (tuple[int, int]): Grid coordinates of the observer.
        target (tuple[int, int]): Grid coordinates of the observed object.
        dist (int): Manhattan distance between source and target.
        radius (int): Effective sensing radius used to scale distance falloff.
        motion_bonus (float): Additive bonus for source or target motion (default 0.0).
    
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
    return float(np.clip(certainty, 0.0, 1.0))


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
    Selects the nearest candidate within the Manhattan `radius` and returns its perceived metrics from the `origin` perspective.
    
    Evaluates candidates in `positions` that lie within `radius` of `origin` (or the spider position when `origin` is None). If any candidates have line of sight, the nearest visible candidate is returned; otherwise the nearest occluded candidate is returned; if no candidates are within range, a "no target" perception is returned.
    
    Parameters:
        origin (tuple[int, int] | None): Source position for perception; when None, the spider position is used.
        motion_bonus (float): Additive bonus applied to visibility confidence to reflect recent motion.
    
    Returns:
        PerceivedTarget: Perception summary for the selected candidate:
            visible: `1.0` when the target's visibility certainty meets or exceeds the configured visibility threshold, `0.0` otherwise.
            certainty: Confidence in the target's visibility (clipped to [0.0, 1.0]); for occluded targets this is a heuristic occlusion certainty.
            occluded: `1.0` if the returned target was occluded, `0.0` otherwise.
            dx, dy: Relative direction components from `origin` to the selected target; set to `0.0` when no visible target is reported.
            dist: Manhattan distance to the selected target, or `NO_TARGET_DISTANCE` when no candidate is found.
            position: Exact selected grid cell, or `None` when no candidate is found.
    """
    source = origin if origin is not None else world.spider_pos()
    best_visible = None
    best_visible_dist = radius + 1
    best_occluded = None
    best_occluded_dist = radius + 1
    for pos in positions:
        dist = world.manhattan(source, pos)
        if dist > radius:
            continue
        if not _heading_allows_target(world, source, pos):
            continue
        if has_line_of_sight(world, source, pos):
            if dist < best_visible_dist or (dist == best_visible_dist and tuple(pos) < tuple(best_visible)):
                best_visible = pos
                best_visible_dist = dist
        elif dist < best_occluded_dist or (dist == best_occluded_dist and tuple(pos) < tuple(best_occluded)):
            best_occluded = pos
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
        target = PerceivedTarget(
            visible=0.0,
            certainty=float(np.clip(certainty, 0.0, 1.0)),
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


def lizard_detects_spider(world: "SpiderWorld") -> bool:
    """
    Determine whether the lizard detects the spider.
    
    Detection is false if the spider's shelter role is "deep", if the spider is beyond the lizard's effective detection range (the lizard's base range adjusted for recent spider motion, terrain penalties, and shelter-role penalties), or if there is no line of sight from the lizard to the spider. When in range and visible, a visibility certainty is computed (including an optional motion bonus) and reduced by an additional inside penalty when the spider is inside; detection occurs if that final certainty meets or exceeds the configured lizard detection threshold.
    
    Returns:
        `True` if the lizard's computed visibility certainty is at or above the configured detection threshold, `False` otherwise.
    """
    cfg = world.operational_profile.perception
    shelter_role = world.shelter_role_at(world.spider_pos())
    if shelter_role == "deep":
        return False
    dist = world.manhattan(world.spider_pos(), world.lizard_pos())
    effective_range = world.lizard_vision_range
    if world.state.last_move_dx != 0 or world.state.last_move_dy != 0:
        effective_range += round(cfg["lizard_detection_range_motion_bonus"])
    if world.terrain_at(world.spider_pos()) == CLUTTER:
        effective_range -= round(cfg["lizard_detection_range_clutter_penalty"])
    if world.terrain_at(world.spider_pos()) == NARROW:
        effective_range -= round(cfg["lizard_detection_range_narrow_penalty"])
    if shelter_role == "inside":
        effective_range -= round(cfg["lizard_detection_range_inside_penalty"])
    if dist > max(1, effective_range):
        return False
    if not has_line_of_sight(world, world.lizard_pos(), world.spider_pos()):
        return False
    certainty = visibility_confidence(
        world,
        source=world.lizard_pos(),
        target=world.spider_pos(),
        dist=dist,
        radius=max(1, effective_range),
        motion_bonus=cfg["lizard_detection_motion_bonus"]
        if world.state.last_move_dx != 0 or world.state.last_move_dy != 0
        else 0.0,
    )
    if shelter_role == "inside":
        certainty -= cfg["lizard_detection_inside_penalty"]
    return bool(certainty >= cfg["lizard_detection_threshold"])


def predator_visible_to_spider(
    world: "SpiderWorld",
    *,
    apply_noise: bool = True,
) -> PerceivedTarget:
    """
    Compute the spider's perception of the predator (lizard).
    
    Returns:
        PerceivedTarget: Describes the predator with fields:
            - `visible`: `1.0` if the predator is considered visible (certainty >= configured visibility threshold), `0.0` otherwise.
            - `certainty`: visibility confidence score in the range `[0.0, 1.0]`.
            - `occluded`: `1.0` when the predator is only occluded, `0.0` otherwise.
            - `dx`, `dy`: relative vector from the spider to the predator; set to `0.0` when not visible.
            - `dist`: Manhattan distance to the predator.
    """
    return visible_object(
        world,
        [world.lizard_pos()],
        radius=visible_range(world),
        apply_noise=apply_noise,
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
    heading_dx: float,
    heading_dy: float,
    food_trace_strength: float,
    shelter_trace_strength: float,
    predator_trace_strength: float,
    predator_motion_salience_value: float,
    day: float,
    night: float,
) -> VisualObservation:
    """
    Create a VisualObservation by mapping perceived target metrics for food, shelter, and predator, and including day/night indicators.
    
    Parameters:
        food_view (PerceivedTarget): Perception tuple for the nearest food target; its fields populate the food_* observation fields.
        shelter_view (PerceivedTarget): Perception tuple for the nearest shelter target; its fields populate the shelter_* observation fields.
        predator_view (PerceivedTarget): Perception tuple for the predator; its fields populate the predator_* observation fields.
        day (float): Day indicator value included in the observation.
        night (float): Night indicator value included in the observation.
    
    Returns:
        VisualObservation: An observation whose `*_visible`, `*_certainty`, `*_occluded`, `*_dx`, and `*_dy` fields are taken from the corresponding PerceivedTarget inputs, and whose `day` and `night` fields are set from the provided values.
    """
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
        food_trace_strength=food_trace_strength,
        shelter_trace_strength=shelter_trace_strength,
        predator_trace_strength=predator_trace_strength,
        predator_motion_salience=predator_motion_salience_value,
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
    food_trace: tuple[float, float, float],
    food_memory: tuple[float, float, float],
) -> HungerObservation:
    """
    Builds a HungerObservation aggregating the agent's hunger state, immediate food perception, smell gradient, and remembered food memory.
    
    Parameters:
        world (SpiderWorld): The world state used to read the current hunger value.
        on_food (float): Indicator whether the agent is currently on a food cell (typically 0.0 or 1.0).
        food_view (PerceivedTarget): Perceived target metrics for the nearest food (visibility, certainty, occlusion, relative dx/dy, distance).
        food_smell_strength (float): Normalized smell strength for food (0.0-1.0).
        food_smell_dx (float): Normalized x component of the food smell gradient.
        food_smell_dy (float): Normalized y component of the food smell gradient.
        food_memory (tuple[float, float, float]): Memory triple (dx, dy, age) representing the stored direction to remembered food and its age.
    
    Returns:
        HungerObservation: Observation populated with hunger, on-food flag, food perception fields, smell gradient, and food memory fields.
    """
    food_trace_dx, food_trace_dy, food_trace_strength = food_trace
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
        food_memory_dx=food_mem_dx,
        food_memory_dy=food_mem_dy,
        food_memory_age=food_mem_age,
    )


def build_sleep_observation(
    world: "SpiderWorld",
    *,
    on_shelter: float,
    night: float,
    home_dx: float,
    home_dy: float,
    home_dist: float,
    sleep_phase_level: float,
    rest_streak_norm: float,
    shelter_role_level: float,
    shelter_trace: tuple[float, float, float],
    shelter_memory: tuple[float, float, float],
) -> SleepObservation:
    """
    Builds a SleepObservation populated from the world state and the provided shelter/home/sleep context.
    
    Parameters:
        home_dx (float): x component of the vector from the agent to its home/shelter.
        home_dy (float): y component of the vector from the agent to its home/shelter.
        home_dist (float): distance from the agent to its home/shelter.
        sleep_phase_level (float): normalized level of the current sleep phase.
        rest_streak_norm (float): normalized streak of recent restful periods.
        shelter_role_level (float): numeric level describing the agent's shelter role (e.g., open/exposed vs protected).
        shelter_memory (tuple[float, float, float]): `(dx, dy, age)` memory tuple for the remembered shelter location.
    
    Returns:
        SleepObservation: dataclass containing fatigue, hunger, on_shelter, night, home vector and distance, health, recent_pain, sleep phase/rest metrics, sleep_debt, shelter role level, and shelter memory fields populated from `world` and the provided arguments.
    """
    shelter_trace_dx, shelter_trace_dy, shelter_trace_strength = shelter_trace
    shelter_mem_dx, shelter_mem_dy, shelter_mem_age = shelter_memory
    return SleepObservation(
        fatigue=world.state.fatigue,
        hunger=world.state.hunger,
        on_shelter=on_shelter,
        night=night,
        home_dx=home_dx,
        home_dy=home_dy,
        home_dist=home_dist,
        health=world.state.health,
        recent_pain=world.state.recent_pain,
        sleep_phase_level=sleep_phase_level,
        rest_streak_norm=rest_streak_norm,
        sleep_debt=world.state.sleep_debt,
        shelter_role_level=shelter_role_level,
        shelter_trace_dx=shelter_trace_dx,
        shelter_trace_dy=shelter_trace_dy,
        shelter_trace_strength=shelter_trace_strength,
        shelter_memory_dx=shelter_mem_dx,
        shelter_memory_dy=shelter_mem_dy,
        shelter_memory_age=shelter_mem_age,
    )


def build_alert_observation(
    world: "SpiderWorld",
    *,
    predator_view: PerceivedTarget,
    predator_dist_norm: float,
    predator_smell_strength: float,
    predator_motion_salience_value: float,
    home_dx: float,
    home_dy: float,
    on_shelter: float,
    night: float,
    predator_trace: tuple[float, float, float],
    predator_memory: tuple[float, float, float],
    escape_memory: tuple[float, float, float],
) -> AlertObservation:
    """
    Builds an AlertObservation containing predator perception, smell and home vectors, recent sensory state, shelter/night flags, and memory entries.
    
    Parameters:
        world (SpiderWorld): World state used only to read recent_pain and recent_contact.
        predator_view (PerceivedTarget): Perceived predator metrics (visible, certainty, occluded, dx, dy, dist).
        predator_dist_norm (float): Predator distance normalized for the observation vector.
        predator_smell_strength (float): Smell gradient strength for the predator.
        home_dx (float): Normalized x component of the home (shelter) direction.
        home_dy (float): Normalized y component of the home (shelter) direction.
        on_shelter (float): Indicator (0/1) whether the agent is on shelter.
        night (float): Indicator (0/1) whether it is night.
        predator_memory (tuple[float, float, float]): (dx, dy, age) memory triple for predator memory.
        escape_memory (tuple[float, float, float]): (dx, dy, age) memory triple for escape memory.
    
    Returns:
        AlertObservation: Observation populated with predator perception fields, smell strength, home vector,
        recent pain/contact values, shelter/night flags, and predator/escape memory triples.
    """
    predator_trace_dx, predator_trace_dy, predator_trace_strength = predator_trace
    predator_mem_dx, predator_mem_dy, predator_mem_age = predator_memory
    escape_mem_dx, escape_mem_dy, escape_mem_age = escape_memory
    return AlertObservation(
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        predator_occluded=predator_view.occluded,
        predator_dx=predator_view.dx,
        predator_dy=predator_view.dy,
        predator_dist=predator_dist_norm,
        predator_smell_strength=predator_smell_strength,
        predator_motion_salience=predator_motion_salience_value,
        home_dx=home_dx,
        home_dy=home_dy,
        recent_pain=world.state.recent_pain,
        recent_contact=world.state.recent_contact,
        on_shelter=on_shelter,
        night=night,
        predator_trace_dx=predator_trace_dx,
        predator_trace_dy=predator_trace_dy,
        predator_trace_strength=predator_trace_strength,
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
    predator_dist_norm: float,
    day: float,
    night: float,
    shelter_role_level: float,
) -> ActionContextObservation:
    """
    Builds the action-center arbitration observation containing internal state, recent sensations, local affordances, predator signals, circadian flags, recent movement, and shelter/sleep indicators.
    
    Parameters:
        on_food (float): Indicator (0.0-1.0) whether the agent is currently on food.
        on_shelter (float): Indicator (0.0-1.0) whether the agent is currently on shelter.
        predator_view (PerceivedTarget): Perceived predator data used for visibility and certainty.
        predator_dist_norm (float): Normalized predator distance (0.0-1.0).
        day (float): Day indicator (1.0 for day, 0.0 for night).
        night (float): Night indicator (1.0 for night, 0.0 for day).
        shelter_role_level (float): Numeric encoding of the agent's current shelter role.
    
    Returns:
        ActionContextObservation: Observation populated with the arbitration context consumed by the action center.
    """
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
        predator_dist=predator_dist_norm,
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
    predator_dist_norm: float,
    day: float,
    night: float,
    shelter_role_level: float,
) -> MotorContextObservation:
    """
    Assembles the motor execution context consumed by the motor cortex correction stage.
    
    Returns:
        MotorContextObservation: observation containing:
            - on_food: provided on-food flag
            - on_shelter: provided on-shelter flag
            - predator_visible: visibility value from `predator_view`
            - predator_certainty: certainty value from `predator_view`
            - predator_dist: provided normalized predator distance
            - day: provided day indicator
            - night: provided night indicator
            - last_move_dx: most recent horizontal move from world state
            - last_move_dy: most recent vertical move from world state
            - shelter_role_level: provided shelter role level
    """
    return MotorContextObservation(
        on_food=on_food,
        on_shelter=on_shelter,
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        predator_dist=predator_dist_norm,
        day=day,
        night=night,
        last_move_dx=float(world.state.last_move_dx),
        last_move_dy=float(world.state.last_move_dy),
        shelter_role_level=shelter_role_level,
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
            - "meta": a dict of diagnostic fields including nearest distances (food, shelter, predator), predator distance used for perception, day/night flags, on_shelter/on_food booleans, predator visibility flag, lizard position and mode, shelter role and terrain at the spider, phase and sleep metrics, heading, configured profiles, predator motion salience, per-vision PerceivedTarget entries under "vision", perceptual trace views under "percept_traces", memory vectors with TTLs under "memory_vectors", and individual memory ages (or None when absent).
    """
    from .memory import MEMORY_TTLS, memory_vector

    _, food_dist = world.nearest(world.food_positions)
    shelter_target = world.safest_shelter_target()
    _, shelter_dist = world.nearest(world.shelter_cells)
    home_dx, home_dy, home_dist = world._relative(shelter_target)
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
    food_trace_view = world._trace_view(world.state.food_trace)
    shelter_trace_view = world._trace_view(world.state.shelter_trace)
    predator_trace_view = world._trace_view(world.state.predator_trace)
    food_trace_dx = float(food_trace_view["dx"])
    food_trace_dy = float(food_trace_view["dy"])
    food_trace_strength = float(food_trace_view["strength"])
    shelter_trace_dx = float(shelter_trace_view["dx"])
    shelter_trace_dy = float(shelter_trace_view["dy"])
    shelter_trace_strength = float(shelter_trace_view["strength"])
    predator_trace_dx = float(predator_trace_view["dx"])
    predator_trace_dy = float(predator_trace_view["dy"])
    predator_trace_strength = float(predator_trace_view["strength"])

    food_view = visible_object(world, world.food_positions, radius=visible_range(world))
    shelter_view = visible_object(world, world.shelter_cells, radius=visible_range(world))
    predator_view = predator_visible_to_spider(world)
    motion_salience = predator_motion_salience(world, predator_view=predator_view)
    predator_dist_real = world.manhattan(world.spider_pos(), world.lizard_pos())
    predator_dist_norm = min(1.0, predator_dist_real / float(world.width + world.height))

    food_smell_strength, food_smell_dx, food_smell_dy, _ = smell_gradient(
        world,
        world.food_positions,
        radius=world.food_smell_range,
    )
    predator_smell_strength, predator_smell_dx, predator_smell_dy, _ = smell_gradient(
        world,
        [world.lizard_pos()],
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
        heading_dx=heading_dx,
        heading_dy=heading_dy,
        food_trace_strength=food_trace_strength,
        shelter_trace_strength=shelter_trace_strength,
        predator_trace_strength=predator_trace_strength,
        predator_motion_salience_value=motion_salience,
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
        food_trace=(food_trace_dx, food_trace_dy, food_trace_strength),
        food_memory=(food_mem_dx, food_mem_dy, food_mem_age),
    )
    sleep_observation = build_sleep_observation(
        world,
        on_shelter=on_shelter,
        night=night,
        home_dx=home_dx,
        home_dy=home_dy,
        home_dist=home_dist,
        sleep_phase_level=sleep_phase,
        rest_streak_norm=rest_norm,
        shelter_role_level=shelter_role_level,
        shelter_trace=(shelter_trace_dx, shelter_trace_dy, shelter_trace_strength),
        shelter_memory=(shelter_mem_dx, shelter_mem_dy, shelter_mem_age),
    )
    alert_observation = build_alert_observation(
        world,
        predator_view=predator_view,
        predator_dist_norm=predator_dist_norm,
        predator_smell_strength=predator_smell_strength,
        predator_motion_salience_value=motion_salience,
        home_dx=home_dx,
        home_dy=home_dy,
        on_shelter=on_shelter,
        night=night,
        predator_trace=(predator_trace_dx, predator_trace_dy, predator_trace_strength),
        predator_memory=(predator_mem_dx, predator_mem_dy, predator_mem_age),
        escape_memory=(escape_mem_dx, escape_mem_dy, escape_mem_age),
    )
    action_context_observation = build_action_context_observation(
        world,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_view=predator_view,
        predator_dist_norm=predator_dist_norm,
        day=day,
        night=night,
        shelter_role_level=shelter_role_level,
    )
    motor_context_observation = build_motor_context_observation(
        world,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_view=predator_view,
        predator_dist_norm=predator_dist_norm,
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

    obs["meta"] = {
        "food_dist": food_dist,
        "shelter_dist": shelter_dist,
        "predator_dist": predator_dist_real,
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
        "predator_motion_salience": motion_salience,
        "vision": {
            "food": asdict(food_view),
            "shelter": asdict(shelter_view),
            "predator": asdict(predator_view),
        },
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
