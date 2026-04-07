from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


_NOISE_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "visual": ("certainty_jitter", "direction_jitter", "dropout_prob"),
    "olfactory": ("strength_jitter", "direction_jitter"),
    "motor": ("action_flip_prob",),
    "spawn": ("uniform_mix",),
    "predator": ("random_choice_prob",),
}


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
    """Validate, coerce, and freeze one NoiseConfig channel section."""
    if not isinstance(values, Mapping):
        raise ValueError(
            f"NoiseConfig seção '{section_name}' inválida: esperado Mapping[str, float]."
        )
    try:
        copied = _copy_float_mapping(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"NoiseConfig seção '{section_name}' inválida: valores devem ser coercíveis para float."
        ) from exc
    required = set(_NOISE_SECTION_KEYS[section_name])
    present = set(copied.keys())
    missing = sorted(required - present)
    unknown = sorted(present - required)
    if missing:
        raise ValueError(
            f"NoiseConfig seção '{section_name}' inválida: faltando chaves obrigatórias {missing}."
        )
    if unknown:
        raise ValueError(
            f"NoiseConfig seção '{section_name}' inválida: chaves desconhecidas {unknown}."
        )
    return MappingProxyType(copied)


@dataclass(frozen=True)
class NoiseConfig:
    name: str
    visual: Mapping[str, float]
    olfactory: Mapping[str, float]
    motor: Mapping[str, float]
    spawn: Mapping[str, float]
    predator: Mapping[str, float]

    def __post_init__(self) -> None:
        """
        Normalize and coerce dataclass fields to their canonical types after initialization.
        
        Coerces `name` to a `str` and replaces each section (`visual`, `olfactory`, `motor`, `spawn`, `predator`)
        with a new `dict` whose keys are `str` and values are `float`, ensuring the frozen dataclass contains
        normalized, concrete mappings.
        """
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "visual", _normalize_noise_section("visual", self.visual))
        object.__setattr__(self, "olfactory", _normalize_noise_section("olfactory", self.olfactory))
        object.__setattr__(self, "motor", _normalize_noise_section("motor", self.motor))
        object.__setattr__(self, "spawn", _normalize_noise_section("spawn", self.spawn))
        object.__setattr__(self, "predator", _normalize_noise_section("predator", self.predator))

    def to_summary(self) -> dict[str, object]:
        """
        Produce a serializable summary of this NoiseConfig.
        
        Returns:
            summary (dict[str, object]): Mapping with keys `"name"`, `"visual"`, `"olfactory"`, `"motor"`, `"spawn"`, and `"predator"`. Each section value is a deep copy of the corresponding internal dict and `"name"` is the config's name.
        """
        return {
            "name": self.name,
            "visual": dict(self.visual),
            "olfactory": dict(self.olfactory),
            "motor": dict(self.motor),
            "spawn": dict(self.spawn),
            "predator": dict(self.predator),
        }

    @classmethod
    def from_summary(cls, summary: Mapping[str, object]) -> "NoiseConfig":
        """
        Create a NoiseConfig from a summary mapping.
        
        The input mapping must contain the keys "name", "visual", "olfactory", "motor", "spawn", and "predator". The value for "name" must be a str; the five section values must be mappings (typically mapping from str to numeric values) that will be used to populate the corresponding NoiseConfig fields.
        
        Parameters:
            summary (Mapping[str, object]): Mapping containing the required keys and values.
        
        Returns:
            NoiseConfig: A new NoiseConfig constructed from the validated summary.
        
        Raises:
            ValueError: If any required key is missing, if "name" is not a str, or if any section value is not a Mapping.
        """
        required = ("name", "visual", "olfactory", "motor", "spawn", "predator")
        missing = [key for key in required if key not in summary]
        if missing:
            raise ValueError(
                "NoiseConfig resumo inválido: faltando chaves obrigatórias "
                f"{missing}."
            )
        sections: dict[str, Mapping[str, float]] = {}
        for key in required[1:]:
            value = summary[key]
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"NoiseConfig resumo inválido: '{key}' deve ser Mapping."
                )
            sections[key] = value
        name = summary["name"]
        if not isinstance(name, str):
            raise ValueError("NoiseConfig resumo inválido: 'name' deve ser str.")
        return cls(
            name=name,
            visual=sections["visual"],
            olfactory=sections["olfactory"],
            motor=sections["motor"],
            spawn=sections["spawn"],
            predator=sections["predator"],
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
    },
    spawn={
        "uniform_mix": 0.0,
    },
    predator={
        "random_choice_prob": 0.0,
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
    },
    spawn={
        "uniform_mix": 0.08,
    },
    predator={
        "random_choice_prob": 0.10,
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
    },
    spawn={
        "uniform_mix": 0.18,
    },
    predator={
        "random_choice_prob": 0.22,
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
    },
    spawn={
        "uniform_mix": 0.30,
    },
    predator={
        "random_choice_prob": 0.40,
    },
)

_NOISE_PROFILES = {
    "none": NONE_NOISE_PROFILE,
    "low": LOW_NOISE_PROFILE,
    "medium": MEDIUM_NOISE_PROFILE,
    "high": HIGH_NOISE_PROFILE,
}


def canonical_noise_profile_names() -> tuple[str, ...]:
    """
    Get canonical noise profile names.
    
    Returns:
        profile_names (tuple[str, ...]): Tuple of available profile name strings corresponding to the predefined noise profiles.
    """
    return tuple(_NOISE_PROFILES.keys())


def resolve_noise_profile(
    profile: str | NoiseConfig | None,
) -> NoiseConfig:
    """
    Resolve a noise profile specifier into a NoiseConfig.
    
    Parameters:
        profile: A profile name (string key), a NoiseConfig instance, or `None`. If `None`, the canonical "none" profile is used. If a string, it must match one of the available canonical profile names.
    
    Returns:
        The matching NoiseConfig instance.
    
    Raises:
        ValueError: If `profile` is a string that does not match any known profile key.
    """
    if profile is None:
        return NoiseConfig.from_summary(NONE_NOISE_PROFILE.to_summary())
    if isinstance(profile, NoiseConfig):
        return NoiseConfig.from_summary(profile.to_summary())
    key = str(profile)
    if key not in _NOISE_PROFILES:
        raise ValueError(f"Perfil de ruído inválido: {profile}")
    return NoiseConfig.from_summary(_NOISE_PROFILES[key].to_summary())
