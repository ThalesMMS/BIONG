from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .modules import MODULE_HIDDEN_DIMS


CAPACITY_AXIS_NAMES: tuple[str, ...] = (
    "proposers",
    "action_center",
    "arbitration",
    "motor",
    "all",
)


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
        name: max(1, int(round(hidden_dim * float(scale_factor))))
        for name, hidden_dim in MODULE_HIDDEN_DIMS.items()
    }


@dataclass(frozen=True, init=False)
class CapacityProfile:
    name: str
    version: str
    module_hidden_dims: Mapping[str, int]
    action_center_hidden_dim: int
    arbitration_hidden_dim: int
    motor_hidden_dim: int
    scale_factor: float

    def __init__(
        self,
        *,
        name: str,
        version: str,
        module_hidden_dims: Mapping[str, int],
        action_center_hidden_dim: int | None = None,
        arbitration_hidden_dim: int | None = None,
        motor_hidden_dim: int | None = None,
        integration_hidden_dim: int | None = None,
        scale_factor: float,
    ) -> None:
        legacy_hidden_dim = (
            None if integration_hidden_dim is None else int(integration_hidden_dim)
        )
        if legacy_hidden_dim is not None and any(
            value is not None
            for value in (
                action_center_hidden_dim,
                arbitration_hidden_dim,
                motor_hidden_dim,
            )
        ):
            raise ValueError(
                "Do not mix integration_hidden_dim with axis-specific hidden dims."
            )
        if legacy_hidden_dim is None and any(
            value is None
            for value in (
                action_center_hidden_dim,
                arbitration_hidden_dim,
                motor_hidden_dim,
            )
        ):
            raise ValueError(
                "Provide all axis-specific hidden dims or integration_hidden_dim."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "module_hidden_dims", module_hidden_dims)
        object.__setattr__(
            self,
            "action_center_hidden_dim",
            legacy_hidden_dim
            if action_center_hidden_dim is None
            else int(action_center_hidden_dim),
        )
        object.__setattr__(
            self,
            "arbitration_hidden_dim",
            legacy_hidden_dim
            if arbitration_hidden_dim is None
            else int(arbitration_hidden_dim),
        )
        object.__setattr__(
            self,
            "motor_hidden_dim",
            legacy_hidden_dim if motor_hidden_dim is None else int(motor_hidden_dim),
        )
        object.__setattr__(self, "scale_factor", scale_factor)
        self.__post_init__()

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
        for field_name in (
            "action_center_hidden_dim",
            "arbitration_hidden_dim",
            "motor_hidden_dim",
        ):
            hidden_dim = int(getattr(self, field_name))
            if hidden_dim <= 0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, hidden_dim)
        object.__setattr__(self, "scale_factor", float(self.scale_factor))

    @property
    def integration_hidden_dim(self) -> int:
        """
        Legacy alias for callers that have not split integration capacity yet.

        `action_center_hidden_dim` is the intentional fallback here, rather
        than an average, minimum, or other aggregate across integration axes,
        because the legacy single integration path fed the action-center
        capacity directly. Keeping that value preserves checkpoint and summary
        compatibility for older callers.
        """
        return int(self.action_center_hidden_dim)

    def to_summary(self) -> dict[str, object]:
        summary = {
            "profile": self.name,
            "version": self.version,
            "module_hidden_dims": dict(self.module_hidden_dims),
            "action_center_hidden_dim": self.action_center_hidden_dim,
            "arbitration_hidden_dim": self.arbitration_hidden_dim,
            "motor_hidden_dim": self.motor_hidden_dim,
            "scale_factor": self.scale_factor,
        }
        if (
            self.action_center_hidden_dim
            == self.arbitration_hidden_dim
            == self.motor_hidden_dim
        ):
            summary["integration_hidden_dim"] = self.integration_hidden_dim
        return summary


CAPACITY_PROFILES: dict[str, CapacityProfile] = {
    "small": CapacityProfile(
        name="small",
        version="v1",
        module_hidden_dims=_scaled_module_hidden_dims(0.6),
        action_center_hidden_dim=20,
        arbitration_hidden_dim=20,
        motor_hidden_dim=20,
        scale_factor=0.6,
    ),
    "current": CapacityProfile(
        name="current",
        version="v1",
        module_hidden_dims=dict(MODULE_HIDDEN_DIMS),
        action_center_hidden_dim=32,
        arbitration_hidden_dim=32,
        motor_hidden_dim=32,
        scale_factor=1.0,
    ),
    "medium": CapacityProfile(
        name="medium",
        version="v1",
        module_hidden_dims=_scaled_module_hidden_dims(1.5),
        action_center_hidden_dim=48,
        arbitration_hidden_dim=48,
        motor_hidden_dim=48,
        scale_factor=1.5,
    ),
    "large": CapacityProfile(
        name="large",
        version="v1",
        module_hidden_dims=_scaled_module_hidden_dims(2.0),
        action_center_hidden_dim=64,
        arbitration_hidden_dim=64,
        motor_hidden_dim=64,
        scale_factor=2.0,
    ),
}


def canonical_capacity_axis_names() -> tuple[str, ...]:
    return CAPACITY_AXIS_NAMES


def canonical_capacity_profile_names() -> tuple[str, ...]:
    return tuple(CAPACITY_PROFILES.keys())


def resolve_capacity_profile(
    name: str | CapacityProfile | None,
) -> CapacityProfile:
    if isinstance(name, CapacityProfile):
        return name
    key = "current" if name is None else str(name)
    if key not in CAPACITY_PROFILES:
        available = ", ".join(CAPACITY_PROFILES.keys())
        raise ValueError(
            f"Unknown capacity profile: {key!r}. Available: {available}"
        )
    profile = CAPACITY_PROFILES[key]
    return CapacityProfile(
        name=profile.name,
        version=profile.version,
        module_hidden_dims=dict(profile.module_hidden_dims),
        action_center_hidden_dim=profile.action_center_hidden_dim,
        arbitration_hidden_dim=profile.arbitration_hidden_dim,
        motor_hidden_dim=profile.motor_hidden_dim,
        scale_factor=profile.scale_factor,
    )


def capacity_profile_for_axis(
    *,
    axis: str,
    target_profile: str | CapacityProfile,
    base_profile: str | CapacityProfile | None = "current",
) -> CapacityProfile:
    axis_name = str(axis)
    if axis_name not in CAPACITY_AXIS_NAMES:
        available = ", ".join(CAPACITY_AXIS_NAMES)
        raise ValueError(f"Unknown capacity axis: {axis_name!r}. Available: {available}")
    target = resolve_capacity_profile(target_profile)
    base = resolve_capacity_profile(base_profile)
    mixed_profile = axis_name != "all"
    return CapacityProfile(
        name=(
            target.name
            if not mixed_profile
            else f"{axis_name}_{target.name}_from_{base.name}"
        ),
        version=(
            f"{target.version}+axis:{axis_name}"
            if not mixed_profile
            else f"{target.version}+axis:{axis_name}+base:{base.name}:{base.version}"
        ),
        module_hidden_dims=(
            target.module_hidden_dims
            if axis_name in {"proposers", "all"}
            else base.module_hidden_dims
        ),
        action_center_hidden_dim=(
            target.action_center_hidden_dim
            if axis_name in {"action_center", "all"}
            else base.action_center_hidden_dim
        ),
        arbitration_hidden_dim=(
            target.arbitration_hidden_dim
            if axis_name in {"arbitration", "all"}
            else base.arbitration_hidden_dim
        ),
        motor_hidden_dim=(
            target.motor_hidden_dim
            if axis_name in {"motor", "all"}
            else base.motor_hidden_dim
        ),
        scale_factor=(
            target.scale_factor
            if axis_name == "all"
            else base.scale_factor
        ),
    )


__all__ = [
    "CAPACITY_PROFILES",
    "CAPACITY_AXIS_NAMES",
    "CapacityProfile",
    "canonical_capacity_axis_names",
    "canonical_capacity_profile_names",
    "capacity_profile_for_axis",
    "resolve_capacity_profile",
]
