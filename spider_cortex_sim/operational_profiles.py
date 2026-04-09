from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping


def _copy_float_mapping(values: Mapping[str, float]) -> dict[str, float]:
    """
    Create a new dict copying a mapping, coercing each key to `str` and each value to `float`.
    
    Parameters:
        values (Mapping[str, float]): Mapping whose entries will be copied; keys and values are coerced.
    
    Returns:
        dict[str, float]: A new dictionary with every key converted to `str` and every value converted to `float`.
    """
    return {str(name): float(value) for name, value in values.items()}


def _copy_nested_float_mapping(values: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """
    Create a new dict mapping string keys to copied inner dicts with float values.
    
    Parameters:
        values (Mapping[str, Mapping[str, float]]): A mapping whose keys will be cast to `str` and whose values are mappings of keys to numeric values.
    
    Returns:
        dict[str, dict[str, float]]: A new dictionary where each top-level key is a `str` and each corresponding value is a shallow copy of the original inner mapping with values coerced to `float`.
    """
    return {
        str(name): _copy_float_mapping(section)
        for name, section in values.items()
    }


@dataclass(frozen=True)
class OperationalProfile:
    name: str
    version: int
    brain_aux_weights: dict[str, float]
    brain_reflex_logit_strengths: dict[str, float]
    brain_reflex_thresholds: dict[str, dict[str, float]]
    perception: dict[str, float]
    reward: dict[str, float]

    def __post_init__(self) -> None:
        """
        Normalize and canonicalize all dataclass fields after initialization.
        
        Coerces `name` to `str` and `version` to `int`, and replaces each mapping field with a copied mapping whose keys are `str` and whose numeric values are `float`. Nested mapping fields are copied recursively so that internal dictionaries are plain `dict` instances with `str` keys and `float` values, preventing shared mutable state.
        """
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "brain_aux_weights", _copy_float_mapping(self.brain_aux_weights))
        object.__setattr__(
            self,
            "brain_reflex_logit_strengths",
            _copy_float_mapping(self.brain_reflex_logit_strengths),
        )
        object.__setattr__(
            self,
            "brain_reflex_thresholds",
            _copy_nested_float_mapping(self.brain_reflex_thresholds),
        )
        object.__setattr__(self, "perception", _copy_float_mapping(self.perception))
        object.__setattr__(self, "reward", _copy_float_mapping(self.reward))

    def to_summary(self) -> dict[str, object]:
        """
        Produce a dictionary summary of the OperationalProfile suitable for serialization or external inspection.
        
        The returned mapping contains the keys:
        - "name" (str)
        - "version" (int)
        - "brain" (dict) with keys:
          - "aux_weights" (dict[str, float])
          - "reflex_logit_strengths" (dict[str, float])
          - "reflex_thresholds" (dict[str, dict[str, float]])
        - "perception" (dict[str, float])
        - "reward" (dict[str, float])
        
        Returns:
            dict[str, object]: A deep-copied dictionary representation of the profile; mutable sub-mappings are copied to avoid sharing internal state.
        """
        return {
            "name": self.name,
            "version": self.version,
            "brain": {
                "aux_weights": deepcopy(self.brain_aux_weights),
                "reflex_logit_strengths": deepcopy(self.brain_reflex_logit_strengths),
                "reflex_thresholds": deepcopy(self.brain_reflex_thresholds),
            },
            "perception": deepcopy(self.perception),
            "reward": deepcopy(self.reward),
        }

    @classmethod
    def from_summary(cls, summary: Mapping[str, object]) -> "OperationalProfile":
        """
        Create an OperationalProfile from a summary mapping.
        
        Validates that `summary` contains the top-level keys "name", "version", "brain", "perception",
        and "reward". Ensures "name" is a `str`, "version" is `int` or coercible to `int`, and that
        "brain", "perception", and "reward" are mappings. The "brain" mapping must include the keys
        "aux_weights", "reflex_logit_strengths", and "reflex_thresholds", each a mapping. The
        "perception" mapping must include "percept_trace_ttl" and "percept_trace_decay". On success,
        returns a canonicalized OperationalProfile instance.
        
        Parameters:
            summary (Mapping[str, object]): Mapping with the required structure described above.
        
        Returns:
            OperationalProfile: An instance constructed from the validated summary.
        
        Raises:
            ValueError: If required keys are missing or any required section has an invalid type or
            cannot be coerced (e.g., `version` not coercible to `int`).
        """
        required_top_level = ("name", "version", "brain", "perception", "reward")
        missing_top_level = [key for key in required_top_level if key not in summary]
        if missing_top_level:
            raise ValueError(
                "Invalid operational profile summary: missing required keys "
                f"{missing_top_level}."
            )

        name = summary["name"]
        if not isinstance(name, str):
            raise ValueError("Invalid operational profile summary: 'name' must be str.")

        try:
            version = int(summary["version"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid operational profile summary: 'version' must be int or coercible to int."
            ) from exc

        brain = summary["brain"]
        if not isinstance(brain, Mapping):
            raise ValueError("Invalid operational profile summary: 'brain' block must be a Mapping.")

        required_brain = ("aux_weights", "reflex_logit_strengths", "reflex_thresholds")
        missing_brain = [key for key in required_brain if key not in brain]
        if missing_brain:
            raise ValueError(
                "Invalid operational profile summary: 'brain' block is missing required keys "
                f"{missing_brain}."
            )

        aux_weights = brain["aux_weights"]
        reflex_logit_strengths = brain["reflex_logit_strengths"]
        reflex_thresholds = brain["reflex_thresholds"]
        perception = summary["perception"]
        reward = summary["reward"]
        if not isinstance(aux_weights, Mapping):
            raise ValueError("Invalid operational profile summary: 'brain.aux_weights' must be a Mapping.")
        if not isinstance(reflex_logit_strengths, Mapping):
            raise ValueError(
                "Invalid operational profile summary: 'brain.reflex_logit_strengths' must be a Mapping."
            )
        if not isinstance(reflex_thresholds, Mapping):
            raise ValueError(
                "Invalid operational profile summary: 'brain.reflex_thresholds' must be a Mapping."
            )
        if not isinstance(perception, Mapping):
            raise ValueError("Invalid operational profile summary: 'perception' must be a Mapping.")
        if not isinstance(reward, Mapping):
            raise ValueError("Invalid operational profile summary: 'reward' must be a Mapping.")
        required_perception = ("percept_trace_ttl", "percept_trace_decay")
        missing_perception = [key for key in required_perception if key not in perception]
        if missing_perception:
            raise ValueError(
                "Invalid operational profile summary: 'perception' block is missing required keys "
                f"{missing_perception}."
            )

        return cls(
            name=name,
            version=version,
            brain_aux_weights=aux_weights,
            brain_reflex_logit_strengths=reflex_logit_strengths,
            brain_reflex_thresholds=reflex_thresholds,
            perception=perception,
            reward=reward,
        )


DEFAULT_OPERATIONAL_PROFILE = OperationalProfile(
    name="default_v1",
    version=1,
    brain_aux_weights={
        "visual_cortex": 0.10,
        "sensory_cortex": 0.12,
        "hunger_center": 0.22,
        "sleep_center": 0.28,
        "alert_center": 0.32,
    },
    brain_reflex_logit_strengths={
        "visual_cortex": 0.35,
        "sensory_cortex": 0.45,
        "hunger_center": 0.85,
        "sleep_center": 1.00,
        "alert_center": 1.15,
    },
    brain_reflex_thresholds={
        "visual_cortex": {
            "visible": 0.50,
            "predator_certainty": 0.45,
            "shelter_certainty": 0.35,
            "food_certainty": 0.35,
        },
        "sensory_cortex": {
            "contact": 0.10,
            "pain": 0.18,
            "predator_smell": 0.25,
            "hunger": 0.40,
            "food_smell": 0.08,
        },
        "hunger_center": {
            "on_food": 0.50,
            "stay_hunger": 0.15,
            "visible_hunger": 0.22,
            "visible_food": 0.50,
            "visible_food_certainty": 0.35,
            "occluded_hunger": 0.28,
            "occluded_food": 0.50,
            "occluded_food_smell": 0.04,
            "smell_hunger": 0.32,
            "food_smell": 0.05,
            "memory_hunger": 0.35,
            "memory_age": 0.80,
        },
        "sleep_center": {
            "on_shelter": 0.50,
            "sleep_phase": 0.25,
            "deep_shelter_level": 0.60,
            "rest_streak": 0.30,
            "rest_hunger": 0.72,
            "memory_rest_hunger": 0.78,
            "fatigue_to_hold": 0.30,
            "sleep_debt_to_hold": 0.28,
            "fatigue_to_seek": 0.26,
            "sleep_debt_to_seek": 0.20,
            "memory_fatigue": 0.22,
            "memory_age": 0.85,
        },
        "alert_center": {
            "predator_visible": 0.50,
            "predator_certainty": 0.35,
            "predator_occluded": 0.50,
            "predator_smell": 0.08,
            "contact_any": 0.0,
            "contact_threat": 0.10,
            "pain": 0.16,
            "on_shelter": 0.50,
            "predator_memory_age": 0.70,
            "escape_memory_age": 0.70,
        },
    },
    perception={
        "night_vision_range_penalty": 1.0,
        "night_vision_min_range": 2.0,
        "visibility_clutter_penalty": 0.18,
        "visibility_narrow_penalty": 0.10,
        "visibility_source_narrow_penalty": 0.08,
        "visibility_night_penalty": 0.06,
        "visibility_binary_threshold": 0.45,
        "occluded_certainty_min": 0.10,
        "occluded_certainty_base": 0.40,
        "occluded_certainty_decay_per_step": 0.08,
        "lizard_detection_range_motion_bonus": 1.0,
        "lizard_detection_range_clutter_penalty": 1.0,
        "lizard_detection_range_narrow_penalty": 1.0,
        "lizard_detection_range_inside_penalty": 1.0,
        "lizard_detection_motion_bonus": 0.12,
        "lizard_detection_inside_penalty": 0.08,
        "lizard_detection_threshold": 0.45,
        "predator_motion_bonus": 0.10,
        "percept_trace_ttl": 4.0,
        "percept_trace_decay": 0.65,
    },
    reward={
        "predator_threat_contact_threshold": 0.0,
        "predator_threat_recent_pain_threshold": 0.18,
        "predator_threat_distance_threshold": 2.0,
        "predator_threat_smell_threshold": 0.14,
        "narrow_predator_risk_max_distance": 3.0,
        "food_progress_hunger_threshold": 0.30,
        "food_progress_hunger_bias": 0.50,
        "shelter_progress_fatigue_threshold": 0.40,
        "shelter_progress_sleep_debt_threshold": 0.35,
        "shelter_progress_bias": 0.50,
        "shelter_progress_sleep_debt_weight": 0.50,
        "predator_escape_contact_threshold": 0.0,
        "day_exploration_hunger_threshold": 0.35,
        "predator_visibility_threshold": 0.50,
        "predator_escape_distance_gain_threshold": 2.0,
        "reset_sleep_predator_distance_threshold": 2.0,
        "reset_sleep_contact_threshold": 0.0,
        "reset_sleep_recent_pain_threshold": 0.18,
    },
)

OPERATIONAL_PROFILES: dict[str, OperationalProfile] = {
    DEFAULT_OPERATIONAL_PROFILE.name: DEFAULT_OPERATIONAL_PROFILE,
}


def canonical_operational_profile_names() -> tuple[str, ...]:
    """
    Get the canonical names of registered operational profiles.
    
    Returns:
        A tuple of profile names as strings, in the registry's insertion order.
    """
    return tuple(OPERATIONAL_PROFILES.keys())


def resolve_operational_profile(
    profile: str | OperationalProfile | None,
) -> OperationalProfile:
    """
    Resolve an OperationalProfile from a profile name, an existing instance, or None.
    
    If `profile` is `None`, the default operational profile is returned. If `profile` is an `OperationalProfile` instance, it is returned unchanged. If `profile` is a string, it is treated as a registry key and the corresponding profile is returned.
    
    Returns:
        The resolved OperationalProfile.
    
    Raises:
        ValueError: If `profile` is a string that does not match any registered profile name.
    """
    if profile is None:
        return DEFAULT_OPERATIONAL_PROFILE
    if isinstance(profile, OperationalProfile):
        return profile
    try:
        return OPERATIONAL_PROFILES[str(profile)]
    except KeyError as exc:
        available = ", ".join(sorted(OPERATIONAL_PROFILES))
        raise ValueError(
            f"Invalid operational profile: {profile!r}. Available: {available}"
        ) from exc
