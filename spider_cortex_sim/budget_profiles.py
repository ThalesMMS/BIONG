from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

CHECKPOINT_SELECTION_NAMES: tuple[str, ...] = ("none", "best")
CHECKPOINT_METRIC_NAMES: tuple[str, ...] = (
    "scenario_success_rate",
    "episode_success_rate",
    "mean_reward",
)


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    benchmark_strength: str
    episodes: int
    eval_episodes: int
    max_steps: int
    scenario_episodes: int
    comparison_seeds: tuple[int, ...]
    checkpoint_interval: int
    selection_scenario_episodes: int
    requires_checkpoint_selection: bool = False

    def __post_init__(self) -> None:
        """
        Canonicalize and normalize BudgetProfile fields to their expected types after initialization.
        
        Converts `name` and `benchmark_strength` to `str`; casts `episodes`, `eval_episodes`, `max_steps`,
        `scenario_episodes`, `checkpoint_interval`, and `selection_scenario_episodes` to `int`; converts
        `comparison_seeds` to `tuple[int, ...]` by casting each element to `int`; and coerces
        `requires_checkpoint_selection` to `bool`.
        """
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "benchmark_strength", str(self.benchmark_strength))
        object.__setattr__(self, "episodes", int(self.episodes))
        object.__setattr__(self, "eval_episodes", int(self.eval_episodes))
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "scenario_episodes", int(self.scenario_episodes))
        object.__setattr__(
            self,
            "comparison_seeds",
            tuple(int(seed) for seed in self.comparison_seeds),
        )
        object.__setattr__(self, "checkpoint_interval", int(self.checkpoint_interval))
        object.__setattr__(
            self,
            "selection_scenario_episodes",
            int(self.selection_scenario_episodes),
        )
        object.__setattr__(
            self,
            "requires_checkpoint_selection",
            bool(self.requires_checkpoint_selection),
        )

    def to_summary(self) -> dict[str, object]:
        """
        Return a serializable summary of this budget profile.
        
        The returned dictionary has top-level keys "profile" (profile name), "benchmark_strength",
        and "resolved". The "resolved" mapping contains the numeric budget fields, "comparison_seeds"
        as a list of ints, and "requires_checkpoint_selection" as a bool.
        
        Returns:
            dict[str, object]: Summary dictionary with structure:
                {
                    "profile": str,
                    "benchmark_strength": str,
                    "resolved": {
                        "episodes": int,
                        "eval_episodes": int,
                        "max_steps": int,
                        "scenario_episodes": int,
                        "comparison_seeds": list[int],
                        "checkpoint_interval": int,
                        "selection_scenario_episodes": int,
                        "requires_checkpoint_selection": bool,
                    }
                }
        """
        return {
            "profile": self.name,
            "benchmark_strength": self.benchmark_strength,
            "resolved": {
                "episodes": self.episodes,
                "eval_episodes": self.eval_episodes,
                "max_steps": self.max_steps,
                "scenario_episodes": self.scenario_episodes,
                "comparison_seeds": list(self.comparison_seeds),
                "checkpoint_interval": self.checkpoint_interval,
                "selection_scenario_episodes": self.selection_scenario_episodes,
                "requires_checkpoint_selection": self.requires_checkpoint_selection,
            },
        }


@dataclass(frozen=True)
class ResolvedBudget:
    profile: str
    benchmark_strength: str
    episodes: int
    eval_episodes: int
    max_steps: int
    scenario_episodes: int
    comparison_seeds: tuple[int, ...]
    checkpoint_interval: int
    selection_scenario_episodes: int
    behavior_seeds: tuple[int, ...]
    ablation_seeds: tuple[int, ...]
    overrides: dict[str, object]
    requires_checkpoint_selection: bool = False

    def __post_init__(self) -> None:
        """
        Normalize and canonicalize ResolvedBudget fields after dataclass initialization.
        
        This method coerces the instance's attributes to their canonical types and makes a defensive copy of mutable data:
        - ensures `profile` and `benchmark_strength` are `str`
        - ensures `episodes`, `eval_episodes`, `max_steps`, `scenario_episodes`, `checkpoint_interval`, and `selection_scenario_episodes` are `int`
        - ensures `comparison_seeds`, `behavior_seeds`, and `ablation_seeds` are `tuple[int, ...]`
        - coerces `requires_checkpoint_selection` to `bool`
        - deep-copies `overrides` into a new `dict` to prevent external mutation
        """
        object.__setattr__(self, "profile", str(self.profile))
        object.__setattr__(self, "benchmark_strength", str(self.benchmark_strength))
        object.__setattr__(self, "episodes", int(self.episodes))
        object.__setattr__(self, "eval_episodes", int(self.eval_episodes))
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "scenario_episodes", int(self.scenario_episodes))
        object.__setattr__(
            self,
            "comparison_seeds",
            tuple(int(seed) for seed in self.comparison_seeds),
        )
        object.__setattr__(self, "checkpoint_interval", int(self.checkpoint_interval))
        object.__setattr__(
            self,
            "selection_scenario_episodes",
            int(self.selection_scenario_episodes),
        )
        object.__setattr__(
            self,
            "behavior_seeds",
            tuple(int(seed) for seed in self.behavior_seeds),
        )
        object.__setattr__(
            self,
            "ablation_seeds",
            tuple(int(seed) for seed in self.ablation_seeds),
        )
        object.__setattr__(self, "overrides", deepcopy(dict(self.overrides)))
        object.__setattr__(
            self,
            "requires_checkpoint_selection",
            bool(self.requires_checkpoint_selection),
        )

    def to_summary(self) -> dict[str, object]:
        """
        Return a serializable summary of the resolved budget.
        
        The summary includes top-level `profile` and `benchmark_strength`, a `resolved` mapping with resolved numeric fields and seed lists (all seed tuples converted to lists), and an `overrides` mapping which is a deep copy of any explicit overrides.
        
        Returns:
            dict[str, object]: {
                "profile": str,
                "benchmark_strength": str,
                "resolved": {
                    "episodes": int,
                    "eval_episodes": int,
                    "max_steps": int,
                    "scenario_episodes": int,
                    "comparison_seeds": list[int],
                    "checkpoint_interval": int,
                    "selection_scenario_episodes": int,
                    "behavior_seeds": list[int],
                    "ablation_seeds": list[int],
                    "requires_checkpoint_selection": bool,
                },
                "overrides": dict[str, object],
            }
        """
        return {
            "profile": self.profile,
            "benchmark_strength": self.benchmark_strength,
            "resolved": {
                "episodes": self.episodes,
                "eval_episodes": self.eval_episodes,
                "max_steps": self.max_steps,
                "scenario_episodes": self.scenario_episodes,
                "comparison_seeds": list(self.comparison_seeds),
                "checkpoint_interval": self.checkpoint_interval,
                "selection_scenario_episodes": self.selection_scenario_episodes,
                "behavior_seeds": list(self.behavior_seeds),
                "ablation_seeds": list(self.ablation_seeds),
                "requires_checkpoint_selection": self.requires_checkpoint_selection,
            },
            "overrides": deepcopy(self.overrides),
        }


SMOKE_BUDGET_PROFILE = BudgetProfile(
    name="smoke",
    benchmark_strength="quick",
    episodes=6,
    eval_episodes=1,
    max_steps=60,
    scenario_episodes=1,
    comparison_seeds=(7,),
    checkpoint_interval=2,
    selection_scenario_episodes=1,
)

DEV_BUDGET_PROFILE = BudgetProfile(
    name="dev",
    benchmark_strength="quick",
    episodes=12,
    eval_episodes=2,
    max_steps=90,
    scenario_episodes=1,
    comparison_seeds=(7, 17, 29),
    checkpoint_interval=4,
    selection_scenario_episodes=1,
)

REPORT_BUDGET_PROFILE = BudgetProfile(
    name="report",
    benchmark_strength="strong",
    episodes=24,
    eval_episodes=4,
    max_steps=120,
    scenario_episodes=2,
    comparison_seeds=(7, 17, 29, 41, 53),
    checkpoint_interval=6,
    selection_scenario_episodes=1,
)

CUSTOM_BUDGET_PROFILE = BudgetProfile(
    name="custom",
    benchmark_strength="custom",
    episodes=180,
    eval_episodes=3,
    max_steps=120,
    scenario_episodes=1,
    comparison_seeds=(7, 17, 29),
    checkpoint_interval=10,
    selection_scenario_episodes=1,
)

PAPER_BUDGET_PROFILE = BudgetProfile(
    name="paper",
    benchmark_strength="publication",
    episodes=48,
    eval_episodes=8,
    max_steps=150,
    scenario_episodes=4,
    comparison_seeds=(7, 17, 29, 41, 53, 67, 79),
    checkpoint_interval=8,
    selection_scenario_episodes=2,
    requires_checkpoint_selection=True,
)

BUDGET_PROFILES: dict[str, BudgetProfile] = {
    SMOKE_BUDGET_PROFILE.name: SMOKE_BUDGET_PROFILE,
    DEV_BUDGET_PROFILE.name: DEV_BUDGET_PROFILE,
    REPORT_BUDGET_PROFILE.name: REPORT_BUDGET_PROFILE,
    PAPER_BUDGET_PROFILE.name: PAPER_BUDGET_PROFILE,
}


def canonical_budget_profile_names() -> tuple[str, ...]:
    """
    Return the canonical budget profile names available in the module.
    
    Returns:
        tuple[str, ...]: Tuple of profile name strings present in BUDGET_PROFILES, in insertion order.
    """
    return tuple(BUDGET_PROFILES.keys())


def resolve_budget_profile(profile: str | BudgetProfile | None) -> BudgetProfile:
    """
    Resolve a budget profile specification into a concrete BudgetProfile instance.
    
    Parameters:
        profile (str | BudgetProfile | None): A profile name, an existing `BudgetProfile`, or `None`.
            If `None`, the custom budget profile is returned.
    
    Returns:
        BudgetProfile: The resolved budget profile corresponding to the input.
    
    Raises:
        ValueError: If `profile` is a string that does not match any available profile names;
            the exception message lists the available profiles.
    """
    if profile is None:
        return CUSTOM_BUDGET_PROFILE
    if isinstance(profile, BudgetProfile):
        return profile
    try:
        return BUDGET_PROFILES[str(profile)]
    except KeyError as exc:
        available = ", ".join(sorted(BUDGET_PROFILES))
        raise ValueError(
            f"Invalid budget profile: {profile!r}. Available: {available}"
        ) from exc


def _normalize_seed_sequence(seeds: Sequence[int] | None) -> tuple[int, ...] | None:
    """
    Convert an optional sequence of seed values into a tuple of integers, or return None.
    
    Parameters:
        seeds (Sequence[int] | None): A sequence of values convertible to `int`, or `None`.
    
    Returns:
        tuple[int, ...] | None: A tuple of `int` cast from `seeds` if provided, otherwise `None`.
    """
    if seeds is None:
        return None
    return tuple(int(seed) for seed in seeds)


def resolve_budget(
    *,
    profile: str | BudgetProfile | None,
    episodes: int | None,
    eval_episodes: int | None,
    max_steps: int | None,
    scenario_episodes: int | None,
    checkpoint_interval: int | None,
    behavior_seeds: Sequence[int] | None,
    ablation_seeds: Sequence[int] | None,
) -> ResolvedBudget:
    """
    Resolve a budget profile into a fully specified ResolvedBudget, applying any explicit overrides.
    
    Uses `profile` (name or BudgetProfile) as the base and replaces any base numeric fields or seed sets with the non-None arguments supplied to this call. If `behavior_seeds` or `ablation_seeds` are not provided, each defaults to the base profile's `comparison_seeds`. The returned `overrides` dictionary contains only parameters that were explicitly supplied.
    
    Parameters:
        profile: Profile name, a BudgetProfile instance, or None to use the custom profile.
        episodes: If provided, override for the number of training episodes.
        eval_episodes: If provided, override for the number of evaluation episodes.
        max_steps: If provided, override for the maximum steps per episode.
        scenario_episodes: If provided, override for the number of episodes per scenario.
        checkpoint_interval: If provided, override for how often to checkpoint.
        behavior_seeds: If provided, sequence of integers to use as behavior seeds instead of the profile's comparison_seeds.
        ablation_seeds: If provided, sequence of integers to use as ablation seeds instead of the profile's comparison_seeds.
    
    Returns:
        ResolvedBudget: A dataclass with all budget fields resolved to concrete values; seed sets are tuples and `overrides` is a dict of the explicitly supplied override values.
    """
    base = resolve_budget_profile(profile)
    explicit_behavior_seeds = _normalize_seed_sequence(behavior_seeds)
    explicit_ablation_seeds = _normalize_seed_sequence(ablation_seeds)
    resolved_episodes = int(episodes if episodes is not None else base.episodes)
    resolved_eval_episodes = int(
        eval_episodes if eval_episodes is not None else base.eval_episodes
    )
    resolved_max_steps = int(max_steps if max_steps is not None else base.max_steps)
    resolved_scenario_episodes = int(
        scenario_episodes if scenario_episodes is not None else base.scenario_episodes
    )
    resolved_checkpoint_interval = int(
        checkpoint_interval
        if checkpoint_interval is not None
        else base.checkpoint_interval
    )

    if resolved_episodes < 0:
        raise ValueError("episodes must be >= 0.")
    if resolved_eval_episodes < 0:
        raise ValueError("eval_episodes must be >= 0.")
    if resolved_max_steps <= 0:
        raise ValueError("max_steps must be > 0.")
    if resolved_scenario_episodes <= 0:
        raise ValueError("scenario_episodes must be > 0.")
    if resolved_checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1.")
    if explicit_behavior_seeds is not None and not explicit_behavior_seeds:
        raise ValueError("behavior_seeds cannot be empty.")
    if explicit_ablation_seeds is not None and not explicit_ablation_seeds:
        raise ValueError("ablation_seeds cannot be empty.")

    overrides: dict[str, object] = {}
    if episodes is not None:
        overrides["episodes"] = resolved_episodes
    if eval_episodes is not None:
        overrides["eval_episodes"] = resolved_eval_episodes
    if max_steps is not None:
        overrides["max_steps"] = resolved_max_steps
    if scenario_episodes is not None:
        overrides["scenario_episodes"] = resolved_scenario_episodes
    if checkpoint_interval is not None:
        overrides["checkpoint_interval"] = resolved_checkpoint_interval
    if explicit_behavior_seeds is not None:
        overrides["behavior_seeds"] = list(explicit_behavior_seeds)
    if explicit_ablation_seeds is not None:
        overrides["ablation_seeds"] = list(explicit_ablation_seeds)

    resolved_comparison_seeds = tuple(base.comparison_seeds)
    resolved_behavior_seeds = explicit_behavior_seeds or resolved_comparison_seeds
    resolved_ablation_seeds = explicit_ablation_seeds or resolved_comparison_seeds

    return ResolvedBudget(
        profile=base.name,
        benchmark_strength=base.benchmark_strength,
        episodes=resolved_episodes,
        eval_episodes=resolved_eval_episodes,
        max_steps=resolved_max_steps,
        scenario_episodes=resolved_scenario_episodes,
        comparison_seeds=resolved_comparison_seeds,
        checkpoint_interval=resolved_checkpoint_interval,
        selection_scenario_episodes=base.selection_scenario_episodes,
        behavior_seeds=resolved_behavior_seeds,
        ablation_seeds=resolved_ablation_seeds,
        overrides=overrides,
        requires_checkpoint_selection=base.requires_checkpoint_selection,
    )
