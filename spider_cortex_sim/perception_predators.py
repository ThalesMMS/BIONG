from __future__ import annotations

import sys
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

from .perception_geometry import has_line_of_sight, smell_gradient, visibility_confidence, visible_object, visible_range
from .perception_targets import DOMINANT_PREDATOR_TYPE_NONE, DOMINANT_PREDATOR_TYPE_OLFACTORY, DOMINANT_PREDATOR_TYPE_VISUAL, HasPosition, NO_TARGET_DISTANCE, PerceivedTarget

def _facade_symbol(name: str, fallback: object) -> object:
    facade = sys.modules.get("spider_cortex_sim.perception")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)

def _public_predator_visual_view(
    world: "SpiderWorld",
    predator: object,
    *,
    apply_noise: bool,
) -> PerceivedTarget:
    visual_view = _facade_symbol("_predator_visual_view", _predator_visual_view)
    return visual_view(world, predator, apply_noise=apply_noise)

def _public_has_line_of_sight(
    world: "SpiderWorld",
    source: tuple[int, int],
    target: tuple[int, int],
) -> bool:
    line_of_sight = _facade_symbol("has_line_of_sight", has_line_of_sight)
    return bool(line_of_sight(world, source, target))

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
        base_view = _public_predator_visual_view(
            world,
            predator,
            apply_noise=apply_noise,
        )
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
        id(predator): _public_predator_visual_view(
            world,
            predator,
            apply_noise=apply_noise,
        )
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
    if not _public_has_line_of_sight(world, source, world.spider_pos()):
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
            visual_view = _public_predator_visual_view(
                world,
                predator,
                apply_noise=True,
            )
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
