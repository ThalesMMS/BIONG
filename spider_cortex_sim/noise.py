from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Tuple

import numpy as np

from .maps import BLOCKED, CLUTTER, NARROW

if TYPE_CHECKING:
    from .world import SpiderWorld


_NOISE_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "visual": ("certainty_jitter", "direction_jitter", "dropout_prob"),
    "olfactory": ("strength_jitter", "direction_jitter"),
    "motor": (
        "action_flip_prob",
        "orientation_slip_factor",
        "terrain_slip_factor",
        "fatigue_slip_factor",
    ),
    "spawn": ("uniform_mix",),
    "predator": ("random_choice_prob",),
    "delay": ("certainty_decay_per_tick", "direction_jitter_per_tick"),
}

_NOISE_SECTION_DEFAULTS: dict[str, dict[str, float]] = {
    "motor": {
        "orientation_slip_factor": 0.0,
        "terrain_slip_factor": 0.0,
        "fatigue_slip_factor": 0.0,
    },
    "delay": {
        "certainty_decay_per_tick": 0.0,
        "direction_jitter_per_tick": 0.0,
    },
}

TERRAIN_DIFFICULTY: Dict[str, float] = {
    CLUTTER: 0.4,
    NARROW: 0.7,
    BLOCKED: 1.0,
}
SLIP_ADJACENT_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "MOVE_UP": ("MOVE_LEFT", "MOVE_RIGHT"),
    "MOVE_DOWN": ("MOVE_LEFT", "MOVE_RIGHT"),
    "MOVE_LEFT": ("MOVE_UP", "MOVE_DOWN"),
    "MOVE_RIGHT": ("MOVE_UP", "MOVE_DOWN"),
}


def _terrain_difficulty(terrain: str) -> float:
    return float(np.clip(TERRAIN_DIFFICULTY.get(terrain, 0.0), 0.0, 1.0))


def _compute_execution_difficulty_core(
    heading: tuple[float, float],
    intended_direction: tuple[float, float],
    *,
    terrain_difficulty: float,
    fatigue: float,
    momentum: float = 0.0,
) -> tuple[float, dict[str, float]]:
    heading_dx, heading_dy = float(heading[0]), float(heading[1])
    move_dx, move_dy = float(intended_direction[0]), float(intended_direction[1])
    heading_norm = float(np.hypot(heading_dx, heading_dy))
    move_norm = float(np.hypot(move_dx, move_dy))
    if heading_norm <= 0.0 or move_norm <= 0.0:
        orientation_alignment = 1.0
    else:
        dot = heading_dx * move_dx + heading_dy * move_dy
        cosine = float(np.clip(dot / (heading_norm * move_norm), -1.0, 1.0))
        orientation_alignment = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
    terrain_factor = float(np.clip(terrain_difficulty, 0.0, 1.0))
    fatigue_factor = float(np.clip(fatigue, 0.0, 1.0))
    momentum_factor = float(np.clip(momentum, 0.0, 1.0))
    orientation_mismatch = float(np.clip(1.0 - orientation_alignment, 0.0, 1.0))
    raw_difficulty = terrain_factor * orientation_mismatch * (1.0 + fatigue_factor)
    if orientation_alignment > 0.7:
        raw_difficulty *= 1.0 - momentum_factor * 0.5
    difficulty = float(np.clip(raw_difficulty, 0.0, 1.0))
    return difficulty, {
        "heading_dx": heading_dx,
        "heading_dy": heading_dy,
        "move_dx": move_dx,
        "move_dy": move_dy,
        "orientation_alignment": orientation_alignment,
        "orientation_mismatch": orientation_mismatch,
        "terrain_difficulty": terrain_factor,
        "fatigue_factor": fatigue_factor,
        "momentum": momentum_factor,
        "raw_difficulty": float(raw_difficulty),
    }


def compute_execution_difficulty(
    heading: tuple[float, float],
    intended_direction: tuple[float, float],
    terrain: str,
    fatigue: float,
    momentum: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """
    Compute motor execution difficulty from heading, action direction, terrain,
    fatigue, and bounded execution momentum.
    """
    return _compute_execution_difficulty_core(
        heading,
        intended_direction,
        terrain_difficulty=_terrain_difficulty(terrain),
        fatigue=fatigue,
        momentum=momentum,
    )


def sample_slip_action(action_name: str, motor_rng: Any) -> str:
    """
    Sample the substitute action for a slipped movement action.
    """
    adjacent = SLIP_ADJACENT_ACTIONS.get(action_name, ())
    if not adjacent:
        return action_name
    candidates = ("STAY", *adjacent)
    weights = np.array([0.6, *([0.4 / len(adjacent)] * len(adjacent))], dtype=float)
    index = int(motor_rng.choice(len(candidates), p=weights))
    return candidates[index]


def motor_slip_reason(components: Mapping[str, float], *, base_slip_rate: float) -> str:
    """
    Return the primary diagnostic reason for a sampled motor slip.
    """
    if base_slip_rate > 0.0 and components["raw_difficulty"] <= 0.0:
        return "base"
    scores = {
        "terrain": components["terrain_difficulty"],
        "orientation": components["orientation_mismatch"],
        "fatigue": components["fatigue_factor"],
    }
    return max(scores, key=scores.get)


def apply_motor_noise(world: "SpiderWorld", action_name: str) -> dict[str, object]:
    """
    Determine whether a requested action slips and return execution diagnostics.
    """
    from .interfaces import ACTION_DELTAS, LOCOMOTION_ACTIONS, ORIENT_HEADINGS

    if action_name not in tuple(LOCOMOTION_ACTIONS):
        raise ValueError(f"Unknown locomotion action {action_name!r}.")

    cfg = world.noise_profile.motor
    base_slip_rate = float(np.clip(cfg["action_flip_prob"], 0.0, 1.0))
    orientation_factor = max(0.0, float(cfg["orientation_slip_factor"]))
    terrain_factor = max(0.0, float(cfg["terrain_slip_factor"]))
    fatigue_factor = max(0.0, float(cfg["fatigue_slip_factor"]))
    if action_name in ORIENT_HEADINGS:
        return {
            "occurred": False,
            "reason": "none",
            "original_action": action_name,
            "executed_action": action_name,
            "slip_probability": 0.0,
            "execution_difficulty": 0.0,
            "terrain": "orientation_only",
            "components": {
                "heading_dx": 0.0,
                "heading_dy": 0.0,
                "move_dx": 0.0,
                "move_dy": 0.0,
                "orientation_alignment": 1.0,
                "orientation_mismatch": 0.0,
                "terrain_difficulty": 0.0,
                "fatigue_factor": 0.0,
                "momentum": float(np.clip(world.state.momentum, 0.0, 1.0)),
                "raw_difficulty": 0.0,
            },
            "base_slip_rate": base_slip_rate,
            "orientation_slip_factor": orientation_factor,
            "terrain_slip_factor": terrain_factor,
            "fatigue_slip_factor": fatigue_factor,
        }

    heading = (float(world.state.heading_dx), float(world.state.heading_dy))
    move_direction = ACTION_DELTAS.get(action_name, (0, 0))
    terrain = world.terrain_at(world.spider_pos())
    difficulty, components = compute_execution_difficulty(
        heading,
        move_direction,
        terrain,
        float(world.state.fatigue),
        float(world.state.momentum),
    )
    result: dict[str, object] = {
        "occurred": False,
        "reason": "none",
        "original_action": action_name,
        "executed_action": action_name,
        "slip_probability": 0.0,
        "execution_difficulty": difficulty,
        "terrain": terrain,
        "components": dict(components),
        "base_slip_rate": base_slip_rate,
        "orientation_slip_factor": orientation_factor,
        "terrain_slip_factor": terrain_factor,
        "fatigue_slip_factor": fatigue_factor,
    }
    slip_pressure = difficulty * (
        terrain_factor
        + orientation_factor * components["orientation_mismatch"]
        + fatigue_factor * components["fatigue_factor"]
    )
    if action_name == "STAY":
        slip_probability = 0.0
    else:
        slip_probability = float(np.clip(base_slip_rate + slip_pressure, 0.0, 1.0))
    result["slip_probability"] = slip_probability
    if action_name != "STAY" and slip_probability > 0.0:
        if world.motor_rng.random() < slip_probability:
            result["occurred"] = True
            result["executed_action"] = sample_slip_action(action_name, world.motor_rng)
            result["reason"] = motor_slip_reason(components, base_slip_rate=base_slip_rate)
    return result


def _copy_float_mapping(values: Mapping[str, float]) -> dict[str, float]:
    """
    Create a new dictionary from the input mapping with each key converted to `str` and each value converted to `float`.
    
    Parameters:
    	values (Mapping[str, float]): Mapping of keys and values to copy and coerce.
    
    Returns:
    	dict[str, float]: A new dictionary whose keys are strings and whose values are floats.
    """
    return {str(name): float(value) for name, value in values.items()}


def _normalize_noise_section(
    section_name: str,
    values: Mapping[str, float],
) -> Mapping[str, float]:
    """
    Validate, coerce, and return an immutable mapping for a named noise section.
    
    Parameters:
        section_name (str): Canonical section name (e.g., "visual", "olfactory", "motor", "spawn", "predator").
        values (Mapping[str, float]): Mapping of section keys to numeric values; keys will be coerced to strings and values to floats.
    
    Returns:
        Mapping[str, float]: An immutable mapping containing the section keys (as `str`) and values (as `float`).
    
    Raises:
        ValueError: If `values` is not a Mapping; if any key or value cannot be coerced to the required types; if required keys for `section_name` are missing; or if unknown extra keys are present.
    """
    if not isinstance(values, Mapping):
        raise ValueError(
            f"Invalid NoiseConfig section '{section_name}': expected Mapping[str, float]."
        )
    try:
        copied = _copy_float_mapping(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid NoiseConfig section '{section_name}': values must be coercible to float."
        ) from exc
    defaults = _NOISE_SECTION_DEFAULTS.get(section_name, {})
    required = set(_NOISE_SECTION_KEYS[section_name]) - set(defaults)
    allowed = set(_NOISE_SECTION_KEYS[section_name])
    present = set(copied.keys())
    missing = sorted(required - present)
    unknown = sorted(present - allowed)
    if missing:
        raise ValueError(
            f"Invalid NoiseConfig section '{section_name}': missing required keys {missing}."
        )
    if unknown:
        raise ValueError(
            f"Invalid NoiseConfig section '{section_name}': unknown keys {unknown}."
        )
    for key, value in defaults.items():
        copied.setdefault(key, float(value))
    return MappingProxyType(copied)


@dataclass(frozen=True)
class NoiseConfig:
    name: str
    visual: Mapping[str, float]
    olfactory: Mapping[str, float]
    motor: Mapping[str, float]
    spawn: Mapping[str, float]
    predator: Mapping[str, float]
    delay: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """
        Normalize and coerce dataclass fields to their canonical types.
        
        Coerces `name` to a `str`; validates and normalizes the noise sections `"visual"`, `"olfactory"`, `"motor"`, `"spawn"`, and `"predator"` to mappings with `str` keys and `float` values. Treats `delay=None` as an empty mapping and normalizes `"delay"`, applying section defaults where keys are missing; all section mappings are stored as immutable mappings.
        """
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "visual", _normalize_noise_section("visual", self.visual))
        object.__setattr__(self, "olfactory", _normalize_noise_section("olfactory", self.olfactory))
        object.__setattr__(self, "motor", _normalize_noise_section("motor", self.motor))
        object.__setattr__(self, "spawn", _normalize_noise_section("spawn", self.spawn))
        object.__setattr__(self, "predator", _normalize_noise_section("predator", self.predator))
        delay_values: Mapping[str, float] = {} if self.delay is None else self.delay
        object.__setattr__(self, "delay", _normalize_noise_section("delay", delay_values))

    def to_summary(self) -> dict[str, object]:
        """
        Produce a serializable mapping representing this NoiseConfig.
        
        Returns:
            dict[str, object]: Mapping with keys "name", "visual", "olfactory", "motor", "spawn", "predator", "delay", and "robustness_order". Each section value is a plain `dict` copy of the corresponding internal mapping; "name" is the config's name. Canonical robustness profiles use orders 0-3; custom profiles use -1.
        """
        robustness_order = (
            CANONICAL_ROBUSTNESS_CONDITIONS.index(self.name)
            if self.name in CANONICAL_ROBUSTNESS_CONDITIONS
            else -1
        )
        return {
            "name": self.name,
            "visual": dict(self.visual),
            "olfactory": dict(self.olfactory),
            "motor": dict(self.motor),
            "spawn": dict(self.spawn),
            "predator": dict(self.predator),
            "delay": dict(self.delay),
            "robustness_order": robustness_order,
        }

    @classmethod
    def from_summary(cls, summary: Mapping[str, object]) -> "NoiseConfig":
        """
        Construct a NoiseConfig from a serializable summary mapping.
        
        The summary must include the top-level keys "name", "visual", "olfactory", "motor", "spawn", and "predator". The optional "delay" key may be omitted; when omitted an empty delay mapping is used and defaults are applied during normalization. Each section value must be a Mapping (typically mapping str to numeric values) and "name" must be a str.
        
        Parameters:
            summary (Mapping[str, object]): Summary mapping containing required keys and section mappings.
        
        Returns:
            NoiseConfig: A new NoiseConfig built from the validated summary.
        
        Raises:
            ValueError: If any required top-level key is missing, if "name" is not a str, or if any section value (including "delay" when present) is not a Mapping.
        """
        required = ("name", "visual", "olfactory", "motor", "spawn", "predator")
        missing = [key for key in required if key not in summary]
        if missing:
            raise ValueError(
                "Invalid NoiseConfig summary: missing required keys "
                f"{missing}."
            )
        sections: dict[str, Mapping[str, float]] = {}
        for key in required[1:]:
            value = summary[key]
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"Invalid NoiseConfig summary: '{key}' must be a Mapping."
                )
            sections[key] = value
        delay_value = summary.get("delay", {})
        if not isinstance(delay_value, Mapping):
            raise ValueError("Invalid NoiseConfig summary: 'delay' must be a Mapping.")
        sections["delay"] = delay_value
        name = summary["name"]
        if not isinstance(name, str):
            raise ValueError("Invalid NoiseConfig summary: 'name' must be str.")
        return cls(
            name=name,
            visual=sections["visual"],
            olfactory=sections["olfactory"],
            motor=sections["motor"],
            spawn=sections["spawn"],
            predator=sections["predator"],
            delay=sections["delay"],
        )


NONE_NOISE_PROFILE = NoiseConfig(
    name="none",
    visual={
        "certainty_jitter": 0.0,
        "direction_jitter": 0.0,
        "dropout_prob": 0.0,
    },
    olfactory={
        "strength_jitter": 0.0,
        "direction_jitter": 0.0,
    },
    motor={
        "action_flip_prob": 0.0,
        "orientation_slip_factor": 0.0,
        "terrain_slip_factor": 0.0,
        "fatigue_slip_factor": 0.0,
    },
    spawn={
        "uniform_mix": 0.0,
    },
    predator={
        "random_choice_prob": 0.0,
    },
    delay={
        "certainty_decay_per_tick": 0.0,
        "direction_jitter_per_tick": 0.0,
    },
)

LOW_NOISE_PROFILE = NoiseConfig(
    name="low",
    visual={
        "certainty_jitter": 0.04,
        "direction_jitter": 0.03,
        "dropout_prob": 0.01,
    },
    olfactory={
        "strength_jitter": 0.04,
        "direction_jitter": 0.03,
    },
    motor={
        "action_flip_prob": 0.03,
        "orientation_slip_factor": 0.04,
        "terrain_slip_factor": 0.08,
        "fatigue_slip_factor": 0.04,
    },
    spawn={
        "uniform_mix": 0.08,
    },
    predator={
        "random_choice_prob": 0.10,
    },
    delay={
        "certainty_decay_per_tick": 0.03,
        "direction_jitter_per_tick": 0.02,
    },
)

MEDIUM_NOISE_PROFILE = NoiseConfig(
    name="medium",
    visual={
        "certainty_jitter": 0.08,
        "direction_jitter": 0.06,
        "dropout_prob": 0.03,
    },
    olfactory={
        "strength_jitter": 0.08,
        "direction_jitter": 0.06,
    },
    motor={
        "action_flip_prob": 0.08,
        "orientation_slip_factor": 0.10,
        "terrain_slip_factor": 0.18,
        "fatigue_slip_factor": 0.10,
    },
    spawn={
        "uniform_mix": 0.18,
    },
    predator={
        "random_choice_prob": 0.22,
    },
    delay={
        "certainty_decay_per_tick": 0.06,
        "direction_jitter_per_tick": 0.04,
    },
)

HIGH_NOISE_PROFILE = NoiseConfig(
    name="high",
    visual={
        "certainty_jitter": 0.14,
        "direction_jitter": 0.10,
        "dropout_prob": 0.08,
    },
    olfactory={
        "strength_jitter": 0.14,
        "direction_jitter": 0.10,
    },
    motor={
        "action_flip_prob": 0.15,
        "orientation_slip_factor": 0.18,
        "terrain_slip_factor": 0.32,
        "fatigue_slip_factor": 0.18,
    },
    spawn={
        "uniform_mix": 0.30,
    },
    predator={
        "random_choice_prob": 0.40,
    },
    delay={
        "certainty_decay_per_tick": 0.10,
        "direction_jitter_per_tick": 0.08,
    },
)

CANONICAL_ROBUSTNESS_CONDITIONS: tuple[str, ...] = (
    "none",
    "low",
    "medium",
    "high",
)


@dataclass(frozen=True)
class RobustnessMatrixSpec:
    train_conditions: tuple[str, ...]
    eval_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        train_conditions = tuple(
            str(condition).strip() for condition in self.train_conditions
        )
        eval_conditions = tuple(
            str(condition).strip() for condition in self.eval_conditions
        )
        for axis_name, conditions in (
            ("train_conditions", train_conditions),
            ("eval_conditions", eval_conditions),
        ):
            if any(not condition for condition in conditions):
                raise ValueError(
                    f"{axis_name} must not contain empty condition names."
                )
            seen: set[str] = set()
            duplicates: list[str] = []
            for condition in conditions:
                if condition in seen and condition not in duplicates:
                    duplicates.append(condition)
                seen.add(condition)
            if duplicates:
                raise ValueError(
                    f"{axis_name} must contain unique condition names; "
                    f"duplicates: {', '.join(repr(condition) for condition in duplicates)}."
                )
        object.__setattr__(
            self,
            "train_conditions",
            train_conditions,
        )
        object.__setattr__(
            self,
            "eval_conditions",
            eval_conditions,
        )

    def cells(self) -> Iterator[tuple[str, str]]:
        """
        Iterate over every ordered pair of training and evaluation conditions.
        
        Yields tuples (train_condition, eval_condition) for each combination where train_condition iterates over `self.train_conditions` (outer loop) and eval_condition iterates over `self.eval_conditions` (inner loop).
        
        Returns:
        	Iterator[tuple[str, str]]: An iterator that yields `(train_condition, eval_condition)` pairs.
        """
        for train_condition in self.train_conditions:
            for eval_condition in self.eval_conditions:
                yield train_condition, eval_condition

    def to_summary(self) -> dict[str, object]:
        """
        Produce a serializable summary of the robustness matrix specification.
        
        The returned mapping contains:
        - `train_conditions`: list of configured training condition names.
        - `eval_conditions`: list of configured evaluation condition names.
        - `cell_count`: number of train/eval cells (len(train_conditions) * len(eval_conditions)).
        - `cells`: list of cell mappings, each with `train_noise_profile` and `eval_noise_profile` keys for the pair.
        
        Returns:
            summary (dict[str, object]): A dictionary with the keys described above suitable for JSON serialization.
        """
        return {
            "train_conditions": list(self.train_conditions),
            "eval_conditions": list(self.eval_conditions),
            "cell_count": len(self.train_conditions) * len(self.eval_conditions),
            "cells": [
                {
                    "train_noise_profile": train_condition,
                    "eval_noise_profile": eval_condition,
                }
                for train_condition, eval_condition in self.cells()
            ],
        }


def canonical_robustness_matrix() -> RobustnessMatrixSpec:
    """
    Create a RobustnessMatrixSpec using canonical robustness conditions for both training and evaluation.
    
    Returns:
        spec (RobustnessMatrixSpec): Spec with `train_conditions` and `eval_conditions` set to CANONICAL_ROBUSTNESS_CONDITIONS.
    """
    return RobustnessMatrixSpec(
        train_conditions=CANONICAL_ROBUSTNESS_CONDITIONS,
        eval_conditions=CANONICAL_ROBUSTNESS_CONDITIONS,
    )

_NOISE_PROFILES = {
    "none": NONE_NOISE_PROFILE,
    "low": LOW_NOISE_PROFILE,
    "medium": MEDIUM_NOISE_PROFILE,
    "high": HIGH_NOISE_PROFILE,
}


def canonical_noise_profile_names() -> tuple[str, ...]:
    """
    Provide the canonical noise profile names.
    
    Returns:
        A tuple of strings containing the canonical robustness condition names in order (for example: "none", "low", "medium", "high").
    """
    return CANONICAL_ROBUSTNESS_CONDITIONS


def resolve_noise_profile(
    profile: str | NoiseConfig | None,
) -> NoiseConfig:
    """
    Resolve a profile specifier into a normalized NoiseConfig.
    
    Parameters:
        profile: A profile name, a NoiseConfig instance, or `None`. If `None`, the canonical "none" profile is used.
    
    Returns:
        The resolved `NoiseConfig` corresponding to the specified profile (normalized).
    
    Raises:
        ValueError: If `profile` is a string that does not match any known profile key.
    """
    if profile is None:
        return NoiseConfig.from_summary(NONE_NOISE_PROFILE.to_summary())
    if isinstance(profile, NoiseConfig):
        return NoiseConfig.from_summary(profile.to_summary())
    key = str(profile)
    if key not in _NOISE_PROFILES:
        raise ValueError(f"Invalid noise profile: {profile}")
    return NoiseConfig.from_summary(_NOISE_PROFILES[key].to_summary())
