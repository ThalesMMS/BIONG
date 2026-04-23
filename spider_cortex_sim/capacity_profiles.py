from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .modules import MODULE_HIDDEN_DIMS


def _normalize_module_hidden_dims(
    module_hidden_dims: Mapping[str, int],
) -> dict[str, int]:
    normalized = {
        str(name): int(value)
        for name, value in dict(module_hidden_dims).items()
    }
    missing = sorted(set(MODULE_HIDDEN_DIMS) - set(normalized))
    if missing:
        raise ValueError(
            "Capacity profile is missing module hidden dims for "
            f"{missing}."
        )
    unknown = sorted(set(normalized) - set(MODULE_HIDDEN_DIMS))
    if unknown:
        raise ValueError(
            "Capacity profile contains unknown modules "
            f"{unknown}."
        )
    non_positive = sorted(
        name for name, value in normalized.items() if value <= 0
    )
    if non_positive:
        raise ValueError(
            "Capacity profile hidden dims must be positive for "
            f"{non_positive}."
        )
    return dict(sorted(normalized.items()))


def _scaled_module_hidden_dims(scale_factor: float) -> dict[str, int]:
    return {
        name: max(1, int(round(int(hidden_dim) * float(scale_factor))))
        for name, hidden_dim in MODULE_HIDDEN_DIMS.items()
    }


@dataclass(frozen=True)
class CapacityProfile:
    name: str
    version: str
    module_hidden_dims: Mapping[str, int]
    integration_hidden_dim: int
    scale_factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "version", str(self.version))
        object.__setattr__(
            self,
            "module_hidden_dims",
            MappingProxyType(
                _normalize_module_hidden_dims(self.module_hidden_dims)
            ),
        )
        integration_hidden_dim = int(self.integration_hidden_dim)
        if integration_hidden_dim <= 0:
            raise ValueError("integration_hidden_dim must be positive.")
        object.__setattr__(
            self,
            "integration_hidden_dim",
            integration_hidden_dim,
        )
        object.__setattr__(self, "scale_factor", float(self.scale_factor))

    def to_summary(self) -> dict[str, object]:
        return {
            "profile": self.name,
            "version": self.version,
            "module_hidden_dims": dict(self.module_hidden_dims),
            "integration_hidden_dim": self.integration_hidden_dim,
            "scale_factor": self.scale_factor,
        }


CAPACITY_PROFILES: dict[str, CapacityProfile] = {
    "small": CapacityProfile(
        name="small",
        version="v1",
        module_hidden_dims=_scaled_module_hidden_dims(0.6),
        integration_hidden_dim=20,
        scale_factor=0.6,
    ),
    "current": CapacityProfile(
        name="current",
        version="v1",
        module_hidden_dims=dict(MODULE_HIDDEN_DIMS),
        integration_hidden_dim=32,
        scale_factor=1.0,
    ),
    "medium": CapacityProfile(
        name="medium",
        version="v1",
        module_hidden_dims=_scaled_module_hidden_dims(1.5),
        integration_hidden_dim=48,
        scale_factor=1.5,
    ),
    "large": CapacityProfile(
        name="large",
        version="v1",
        module_hidden_dims=_scaled_module_hidden_dims(2.0),
        integration_hidden_dim=64,
        scale_factor=2.0,
    ),
}


def canonical_capacity_profile_names() -> tuple[str, ...]:
    return tuple(CAPACITY_PROFILES.keys())


def resolve_capacity_profile(
    name: str | CapacityProfile | None,
) -> CapacityProfile:
    if name is None:
        profile = CAPACITY_PROFILES["current"]
        return CapacityProfile(
            name=profile.name,
            version=profile.version,
            module_hidden_dims=profile.module_hidden_dims,
            integration_hidden_dim=profile.integration_hidden_dim,
            scale_factor=profile.scale_factor,
        )
    if isinstance(name, CapacityProfile):
        return CapacityProfile(
            name=name.name,
            version=name.version,
            module_hidden_dims=name.module_hidden_dims,
            integration_hidden_dim=name.integration_hidden_dim,
            scale_factor=name.scale_factor,
        )
    key = str(name)
    if key not in CAPACITY_PROFILES:
        available = ", ".join(CAPACITY_PROFILES.keys())
        raise ValueError(
            f"Unknown capacity profile: {key!r}. Available: {available}"
        )
    profile = CAPACITY_PROFILES[key]
    return CapacityProfile(
        name=profile.name,
        version=profile.version,
        module_hidden_dims=profile.module_hidden_dims,
        integration_hidden_dim=profile.integration_hidden_dim,
        scale_factor=profile.scale_factor,
    )


__all__ = [
    "CAPACITY_PROFILES",
    "CapacityProfile",
    "canonical_capacity_profile_names",
    "resolve_capacity_profile",
]
