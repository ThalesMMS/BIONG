from __future__ import annotations

from .config import *

def canonical_ablation_configs(
    *,
    module_dropout: float = 0.05,
    capacity_profile: str | CapacityProfile | None = None,
) -> Dict[str, BrainAblationConfig]:
    """
    Builds the canonical registry of named brain ablation configurations.
    
    The registry includes standard modular and monolithic presets, credit- and arbitration- variants, reflex-scale variants, and per-module "drop_<module_name>" variants for each A4 fine-grained module.
    
    Parameters:
        module_dropout (float): Default module dropout rate applied to modular variants.
        capacity_profile (str | CapacityProfile | None): Optional capacity profile forwarded into each config.
    
    Returns:
        dict: Mapping from canonical variant name (str) to its BrainAblationConfig.
    """
    profile_fields = {"capacity_profile": capacity_profile}
    four_center_disabled_modules = (
        "alert_center",
        "hunger_center",
        "perception_center",
        "sleep_center",
    )
    three_center_disabled_modules = (
        "alert_center",
        "hunger_center",
        "sensory_cortex",
        "sleep_center",
        "visual_cortex",
    )

    def modular_preset_fields(
        *,
        module_dropout_value: float = module_dropout,
        enable_reflexes: bool = True,
        enable_auxiliary_targets: bool = True,
        reflex_scale: float = 1.0,
        recurrent_modules: tuple[str, ...] = (),
        arbitration_fields: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "architecture": "modular",
            "module_dropout": module_dropout_value,
            "enable_reflexes": enable_reflexes,
            "enable_auxiliary_targets": enable_auxiliary_targets,
            **(
                arbitration_fields
                if arbitration_fields is not None
                else _arbitration_fields()
            ),
            "recurrent_modules": recurrent_modules,
            "reflex_scale": reflex_scale,
            "module_reflex_scales": {},
            **profile_fields,
        }

    def make_four_center_variant(
        name: str,
        credit_strategy: str,
    ) -> BrainAblationConfig:
        """
        Builds a canonical four-center modular BrainAblationConfig using the given name and credit strategy.
        
        Parameters:
            name (str): Identifier for the returned config.
            credit_strategy (str): Credit assignment strategy to apply (for
            example 'broadcast', 'route_mask', 'local_only', or
            'counterfactual').
        
        Returns:
            BrainAblationConfig: A modular config with the four-center modules disabled, reflexes and auxiliary targets enabled, no recurrent modules, reflex scale set to 1.0, arbitration fields and provided capacity profile applied.
        """
        return BrainAblationConfig(
            name=name,
            **modular_preset_fields(),
            credit_strategy=credit_strategy,
            disabled_modules=four_center_disabled_modules,
        )

    variants: Dict[str, BrainAblationConfig] = {
        "modular_full": default_brain_config(
            module_dropout=module_dropout,
            capacity_profile=capacity_profile,
        ),
        "modular_full_broadcast": BrainAblationConfig(
            name="modular_full_broadcast",
            **modular_preset_fields(),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "no_module_dropout": BrainAblationConfig(
            name="no_module_dropout",
            **modular_preset_fields(module_dropout_value=0.0),
            credit_strategy="route_mask",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "no_module_reflexes": BrainAblationConfig(
            name="no_module_reflexes",
            **modular_preset_fields(
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                reflex_scale=0.0,
            ),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "three_center_modular": BrainAblationConfig(
            name="three_center_modular",
            **modular_preset_fields(),
            credit_strategy="route_mask",
            disabled_modules=three_center_disabled_modules,
        ),
        "three_center_modular_broadcast": BrainAblationConfig(
            name="three_center_modular_broadcast",
            **modular_preset_fields(),
            credit_strategy="broadcast",
            disabled_modules=three_center_disabled_modules,
        ),
        "three_center_modular_local_credit": BrainAblationConfig(
            name="three_center_modular_local_credit",
            **modular_preset_fields(),
            credit_strategy="local_only",
            disabled_modules=three_center_disabled_modules,
        ),
        "three_center_modular_counterfactual": BrainAblationConfig(
            name="three_center_modular_counterfactual",
            **modular_preset_fields(),
            credit_strategy="counterfactual",
            disabled_modules=three_center_disabled_modules,
        ),
        "four_center_modular": make_four_center_variant(
            "four_center_modular",
            "route_mask",
        ),
        "four_center_modular_broadcast": make_four_center_variant(
            "four_center_modular_broadcast",
            "broadcast",
        ),
        "four_center_modular_local_credit": make_four_center_variant(
            "four_center_modular_local_credit",
            "local_only",
        ),
        "four_center_modular_counterfactual": make_four_center_variant(
            "four_center_modular_counterfactual",
            "counterfactual",
        ),
        "modular_recurrent": BrainAblationConfig(
            name="modular_recurrent",
            **modular_preset_fields(
                recurrent_modules=("alert_center", "sleep_center", "hunger_center")
            ),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "modular_recurrent_all": BrainAblationConfig(
            name="modular_recurrent_all",
            **modular_preset_fields(recurrent_modules=A4_FINE_MODULES),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "local_credit_only": BrainAblationConfig(
            name="local_credit_only",
            **modular_preset_fields(),
            credit_strategy="local_only",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "counterfactual_credit": BrainAblationConfig(
            name="counterfactual_credit",
            **modular_preset_fields(),
            credit_strategy="counterfactual",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "constrained_arbitration": BrainAblationConfig(
            name="constrained_arbitration",
            **modular_preset_fields(
                arbitration_fields=_arbitration_fields(
                    enable_deterministic_guards=True,
                    enable_food_direction_bias=True,
                )
            ),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "weaker_prior_arbitration": BrainAblationConfig(
            name="weaker_prior_arbitration",
            **modular_preset_fields(
                arbitration_fields=_arbitration_fields(warm_start_scale=0.5)
            ),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "minimal_arbitration": BrainAblationConfig(
            name="minimal_arbitration",
            **modular_preset_fields(
                arbitration_fields=_arbitration_fields(
                    warm_start_scale=0.0,
                    gate_adjustment_bounds=(0.1, 2.0),
                )
            ),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "fixed_arbitration_baseline": BrainAblationConfig(
            name="fixed_arbitration_baseline",
            **modular_preset_fields(
                arbitration_fields=_arbitration_fields(use_learned_arbitration=False)
            ),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "learned_arbitration_no_regularization": BrainAblationConfig(
            name="learned_arbitration_no_regularization",
            **modular_preset_fields(),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "reflex_scale_0_25": BrainAblationConfig(
            name="reflex_scale_0_25",
            **modular_preset_fields(reflex_scale=0.25),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "reflex_scale_0_50": BrainAblationConfig(
            name="reflex_scale_0_50",
            **modular_preset_fields(reflex_scale=0.50),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "reflex_scale_0_75": BrainAblationConfig(
            name="reflex_scale_0_75",
            **modular_preset_fields(reflex_scale=0.75),
            credit_strategy="broadcast",
            disabled_modules=COARSE_ROLLUP_MODULES,
        ),
        "monolithic_policy": BrainAblationConfig(
            name=MONOLITHIC_POLICY_NAME,
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            **_arbitration_fields(),
            credit_strategy="broadcast",
            disabled_modules=(),
            reflex_scale=0.0,
            module_reflex_scales={},
            **profile_fields,
        ),
        "true_monolithic_policy": BrainAblationConfig(
            name=TRUE_MONOLITHIC_POLICY_NAME,
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            **_arbitration_fields(use_learned_arbitration=False, warm_start_scale=0.0),
            credit_strategy="broadcast",
            disabled_modules=(),
            reflex_scale=0.0,
            module_reflex_scales={},
            **profile_fields,
        ),
    }
    for module_name in A4_FINE_MODULES:
        variants[f"drop_{module_name}"] = BrainAblationConfig(
            name=f"drop_{module_name}",
            **modular_preset_fields(),
            credit_strategy="broadcast",
            disabled_modules=(*COARSE_ROLLUP_MODULES, module_name),
        )
    return variants


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
    registry = canonical_ablation_configs(
        module_dropout=module_dropout,
        capacity_profile=capacity_profile,
    )
    requested = list(names) if names is not None else list(registry)
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
