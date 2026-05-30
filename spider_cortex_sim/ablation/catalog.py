from __future__ import annotations

from ._catalog_shared import *
from ._catalog_canonical import canonical_ablation_configs
from ._catalog_diagnostic_b0_b6 import diagnostic_b0_b6_configs
from ._catalog_diagnostic_b7_b18 import diagnostic_b7_b18_configs
from ._catalog_diagnostic_b19_b30 import diagnostic_b19_b30_configs
from ._catalog_diagnostic_b31_b38 import diagnostic_b31_b38_configs
from ._catalog_diagnostic_b39_b51 import diagnostic_b39_b51_configs
from ._catalog_diagnostic_b52_b62 import diagnostic_b52_b62_configs
from ._catalog_diagnostic_direct_core import diagnostic_direct_core_configs
from ._catalog_diagnostic_direct_teacher_replay import diagnostic_direct_teacher_replay_configs
from ._catalog_diagnostic_direct_local_branches import diagnostic_direct_local_branches_configs


def diagnostic_ablation_configs(
    *,
    module_dropout: float = 0.05,
    capacity_profile: str | CapacityProfile | None = None,
) -> Dict[str, BrainAblationConfig]:
    """Build opt-in diagnostic variants that should not become canonical defaults."""
    profile_fields = {"capacity_profile": capacity_profile}
    configs: Dict[str, BrainAblationConfig] = {}
    configs.update(diagnostic_b0_b6_configs(profile_fields))
    configs.update(diagnostic_b7_b18_configs(profile_fields))
    configs.update(diagnostic_b19_b30_configs(profile_fields))
    configs.update(diagnostic_b31_b38_configs(profile_fields))
    configs.update(diagnostic_b39_b51_configs(profile_fields))
    configs.update(diagnostic_b52_b62_configs(profile_fields))
    configs.update(diagnostic_direct_core_configs(profile_fields))
    configs.update(diagnostic_direct_teacher_replay_configs(profile_fields))
    configs.update(diagnostic_direct_local_branches_configs(profile_fields))
    return configs


def canonical_ablation_variant_names(*, module_dropout: float = 0.05) -> tuple[str, ...]:
    """
    Produce the ordered tuple of canonical ablation variant names.
    
    Returns:
        tuple[str, ...]: Ordered variant names starting with "modular_full", then
        "no_module_dropout", "no_module_reflexes", and recurrent/credit/arbitration
        variants, followed by "drop_<module>" for each fine-grained A4
        proposer, and
        ending with the monolithic baselines.
    """
    return tuple(canonical_ablation_configs(module_dropout=module_dropout).keys())


def resolve_ablation_configs(
    names: Sequence[str] | None,
    *,
    module_dropout: float = 0.05,
    capacity_profile: str | CapacityProfile | None = None,
) -> list[BrainAblationConfig]:
    """
    Resolve ablation variant names to their corresponding BrainAblationConfig objects in the requested order.
    
    Parameters:
        names (Sequence[str] | None): Variant names to resolve. If `None`, all canonical variant names are used in their canonical order.
        module_dropout (float): Dropout value forwarded to the canonical registry construction.
        capacity_profile (str | CapacityProfile | None): Optional capacity profile forwarded to the canonical registry construction.
    
    Returns:
        list[BrainAblationConfig]: Config objects for the requested variant names, ordered as in `names` or canonical order when `names` is `None`.
    
    Raises:
        KeyError: If any requested names are unknown; the exception message lists the unknown names and the available canonical variant names.
    """
    canonical_registry = canonical_ablation_configs(
        module_dropout=module_dropout,
        capacity_profile=capacity_profile,
    )
    registry = dict(canonical_registry)
    registry.update(
        diagnostic_ablation_configs(
            module_dropout=module_dropout,
            capacity_profile=capacity_profile,
        )
    )
    requested = list(names) if names is not None else list(canonical_registry)
    unknown = sorted({name for name in requested if name not in registry})
    if unknown:
        available = ", ".join(registry.keys())
        raise KeyError(f"Unknown ablation variants: {unknown}. Available: {available}")
    return [registry[name] for name in requested]


def canonical_ablation_scenario_groups() -> dict[str, tuple[str, ...]]:
    """
    Get mapping of canonical multi-predator scenario group names to their ordered scenario tuples.

    Returns:
        dict[str, tuple[str, ...]]: Mapping from group name to a tuple of scenario name strings in canonical order.
    """
    return dict(MULTI_PREDATOR_SCENARIO_GROUPS)


def resolve_ablation_scenario_group(name: str) -> tuple[str, ...]:
    """
    Resolve a named ablation scenario group into its ordered scenario tuple.
    
    Parameters:
        name (str): The name of the scenario group to resolve.
    
    Returns:
        tuple[str, ...]: Ordered tuple of scenario names belonging to the group.
    
    Raises:
        KeyError: If `name` is not present in the available scenario groups; the exception message lists available group names.
    """
    if name not in MULTI_PREDATOR_SCENARIO_GROUPS:
        available = ", ".join(sorted(MULTI_PREDATOR_SCENARIO_GROUPS))
        raise KeyError(f"Unknown ablation scenario group: {name!r}. Available: {available}")
    return MULTI_PREDATOR_SCENARIO_GROUPS[name]

__all__ = [
    "canonical_ablation_configs",
    "canonical_ablation_scenario_groups",
    "canonical_ablation_variant_names",
    "resolve_ablation_configs",
    "resolve_ablation_scenario_group",
]
