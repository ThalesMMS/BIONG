from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Callable, Dict, Sequence

from ..metrics import BehaviorCheckSpec, BehavioralEpisodeScore, EpisodeStats
from ..predator import VISUAL_HUNTER_PROFILE
from ..world import SpiderWorld

BehaviorScoreFn = Callable[[EpisodeStats, Sequence[Dict[str, object]]], BehavioralEpisodeScore]

TRACE_MAP_WIDTH_FALLBACK = 12
TRACE_MAP_HEIGHT_FALLBACK = 12
FAST_VISUAL_HUNTER_PROFILE = replace(
    VISUAL_HUNTER_PROFILE,
    name="fast_visual_hunter",
    move_interval=1,
    detection_threshold=0.45,
)

NIGHT_REST_INITIAL_SLEEP_DEBT = 0.60
FOOD_DEPRIVATION_INITIAL_HUNGER = 0.96
SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT = 0.92
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class ProbeType(str, Enum):
    EMERGENCE_GATE = "emergence_gate"
    CAPABILITY_PROBE = "capability_probe"
    CURRICULUM_FOCUS = "curriculum_focus"


class BenchmarkTier(str, Enum):
    PRIMARY = "primary"
    CAPABILITY = "capability"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    objective: str
    behavior_checks: Sequence[BehaviorCheckSpec]
    diagnostic_focus: str
    success_interpretation: str
    failure_interpretation: str
    budget_note: str
    max_steps: int
    map_template: str
    setup: Callable[[SpiderWorld], None]
    score_episode: BehaviorScoreFn
    probe_type: str | ProbeType = ProbeType.EMERGENCE_GATE.value
    target_skill: str | None = None
    geometry_assumptions: str | None = None
    benchmark_tier: str | BenchmarkTier = BenchmarkTier.PRIMARY.value
    acceptable_partial_progress: str | None = None
    expected_owner_modules: Sequence[str] = ()

    def __post_init__(self) -> None:
        """
        Normalize and validate probe and benchmark fields after dataclass initialization.
        
        Converts `probe_type` and `benchmark_tier` enum instances to their
        string values, verifies each is one of the allowed enum values, enforces
        that capability probes provide non-empty `target_skill`,
        `geometry_assumptions`, and `acceptable_partial_progress` and that
        `benchmark_tier` equals "capability", and validates and normalizes
        `expected_owner_modules` to an immutable tuple of strings.
        
        Raises:
            ValueError: If `probe_type` or `benchmark_tier` is not an allowed
                value, if a capability probe is missing required metadata or has
                an incorrect `benchmark_tier`, or if `expected_owner_modules`
                is malformed or contains non-string entries.
        """
        probe_type = (
            self.probe_type.value
            if isinstance(self.probe_type, ProbeType)
            else self.probe_type
        )
        benchmark_tier = (
            self.benchmark_tier.value
            if isinstance(self.benchmark_tier, BenchmarkTier)
            else self.benchmark_tier
        )
        valid_probe_types = {item.value for item in ProbeType}
        if probe_type not in valid_probe_types:
            raise ValueError(
                f"Invalid probe_type {self.probe_type!r}; expected one of "
                f"{sorted(valid_probe_types)}"
            )
        valid_benchmark_tiers = {item.value for item in BenchmarkTier}
        if benchmark_tier not in valid_benchmark_tiers:
            raise ValueError(
                f"Invalid benchmark_tier {self.benchmark_tier!r}; expected one of "
                f"{sorted(valid_benchmark_tiers)}"
            )
        if probe_type == ProbeType.CAPABILITY_PROBE.value:
            missing_fields: list[str] = []
            for field in (
                "target_skill",
                "geometry_assumptions",
                "acceptable_partial_progress",
            ):
                value = getattr(self, field)
                if value is None or not isinstance(value, str) or not value.strip():
                    missing_fields.append(field)
            invalid_fields: list[str] = []
            if benchmark_tier != BenchmarkTier.CAPABILITY.value:
                invalid_fields.append("benchmark_tier")
            if missing_fields or invalid_fields:
                details: list[str] = []
                if missing_fields:
                    details.append(
                        "missing required fields: " + ", ".join(missing_fields)
                    )
                if invalid_fields:
                    details.append(
                        f"benchmark_tier must be {BenchmarkTier.CAPABILITY.value!r}"
                    )
                raise ValueError(
                    "Invalid capability probe metadata: " + "; ".join(details)
                )
        object.__setattr__(self, "probe_type", probe_type)
        object.__setattr__(self, "benchmark_tier", benchmark_tier)
        expected_owner_modules = self.expected_owner_modules
        if expected_owner_modules is None:
            raise ValueError(
                "expected_owner_modules must be a sequence of module names."
            )
        if isinstance(expected_owner_modules, str):
            raise ValueError(
                "expected_owner_modules must be a sequence of module names, "
                "not a string."
            )
        if isinstance(expected_owner_modules, bytes):
            raise ValueError(
                "expected_owner_modules must be a sequence of module names, "
                "not bytes."
            )
        try:
            expected_owner_iter = iter(expected_owner_modules)
        except TypeError as exc:
            raise ValueError(
                "expected_owner_modules must be an iterable sequence of module names."
            ) from exc
        normalized_expected_owner_modules: list[str] = []
        seen_expected_owner_modules: set[str] = set()
        for name in expected_owner_iter:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "expected_owner_modules entries must be non-empty strings."
                )
            module_name = name.strip()
            if MODULE_NAME_PATTERN.fullmatch(module_name) is None:
                raise ValueError(
                    "expected_owner_modules entries must start with a letter or "
                    "underscore and contain only letters, digits, '.', or '_'."
                )
            if module_name in seen_expected_owner_modules:
                raise ValueError(
                    f"expected_owner_modules contains duplicate module {module_name!r}."
                )
            seen_expected_owner_modules.add(module_name)
            normalized_expected_owner_modules.append(module_name)
        object.__setattr__(
            self,
            "expected_owner_modules",
            tuple(normalized_expected_owner_modules),
        )

__all__ = [
    'FAST_VISUAL_HUNTER_PROFILE',
    'FOOD_DEPRIVATION_INITIAL_HUNGER',
    'NIGHT_REST_INITIAL_SLEEP_DEBT',
    'SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT',
    'TRACE_MAP_HEIGHT_FALLBACK',
    'TRACE_MAP_WIDTH_FALLBACK',
    'BehaviorScoreFn',
    'BenchmarkTier',
    'ProbeType',
    'ScenarioSpec',
]
