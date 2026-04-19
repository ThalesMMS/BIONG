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

from .perception_targets import NO_TARGET_DISTANCE, PerceivedTarget, VisibilityZone

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
