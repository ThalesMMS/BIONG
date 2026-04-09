from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence

from .interfaces import MODULE_INTERFACES


MODULE_NAMES: tuple[str, ...] = tuple(spec.name for spec in MODULE_INTERFACES)
MONOLITHIC_POLICY_NAME: str = "monolithic_policy"
PROPOSAL_SOURCE_NAMES: tuple[str, ...] = MODULE_NAMES + (MONOLITHIC_POLICY_NAME,)
REFLEX_MODULE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in MODULE_INTERFACES if spec.role == "proposal"
)


@dataclass(frozen=True)
class BrainAblationConfig:
    name: str
    architecture: str = "modular"
    module_dropout: float = 0.05
    enable_reflexes: bool = True
    enable_auxiliary_targets: bool = True
    disabled_modules: tuple[str, ...] = ()
    reflex_scale: float = 1.0
    module_reflex_scales: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and normalize dataclass fields after initialization.
        
        Normalizes and persists these fields on the frozen dataclass:
        - `architecture`: coerced to `str` and must be either "modular" or "monolithic".
        - `module_dropout`: coerced to `float` and must be finite.
        - `reflex_scale`: coerced to `float`, must be finite and >= 0.0.
        - `disabled_modules`: coerced to `str`, deduplicated, sorted into a `tuple`, and each name must be in `MODULE_NAMES`.
        - `module_reflex_scales`: keys coerced to `str`, values coerced to `float`, sorted by key into a `dict`; each key must be in `REFLEX_MODULE_NAMES`, each value must be finite and >= 0.0.
        
        Raises:
            ValueError: If `architecture` is not "modular" or "monolithic";
                        if `module_dropout` or any `module_reflex_scales` value is not finite;
                        if `reflex_scale` or any `module_reflex_scales` value is negative;
                        if `disabled_modules` or `module_reflex_scales` contain unknown module names;
                        or if `architecture == "monolithic"` while `disabled_modules` or `module_reflex_scales` are non-empty.
        """
        architecture = str(self.architecture)
        if architecture not in {"modular", "monolithic"}:
            raise ValueError(f"Invalid ablation architecture: {architecture!r}.")
        object.__setattr__(self, "architecture", architecture)
        module_dropout = float(self.module_dropout)
        if not math.isfinite(module_dropout):
            raise ValueError("module_dropout must be finite.")
        object.__setattr__(self, "module_dropout", module_dropout)
        reflex_scale = float(self.reflex_scale)
        if not math.isfinite(reflex_scale):
            raise ValueError("reflex_scale must be finite.")
        if reflex_scale < 0.0:
            raise ValueError("reflex_scale must be non-negative.")
        object.__setattr__(self, "reflex_scale", reflex_scale)

        normalized_modules = tuple(sorted({str(name) for name in self.disabled_modules}))
        invalid_modules = [name for name in normalized_modules if name not in MODULE_NAMES]
        if invalid_modules:
            raise ValueError(f"Invalid ablation modules: {invalid_modules}.")
        if architecture == "monolithic" and normalized_modules:
            raise ValueError("The monolithic baseline does not accept disabled modules.")
        object.__setattr__(self, "disabled_modules", normalized_modules)
        normalized_reflex_scales = {
            name: float(value)
            for name, value in sorted(
                (str(name), value)
                for name, value in dict(self.module_reflex_scales).items()
            )
        }
        invalid_reflex_modules = [
            name for name in normalized_reflex_scales if name not in REFLEX_MODULE_NAMES
        ]
        if invalid_reflex_modules:
            raise ValueError(
                f"Invalid modules in module_reflex_scales: {invalid_reflex_modules}."
            )
        non_finite_reflex_scales = [
            name
            for name, value in normalized_reflex_scales.items()
            if not math.isfinite(value)
        ]
        if non_finite_reflex_scales:
            raise ValueError(
                "module_reflex_scales must contain only finite values."
            )
        negative_reflex_scales = [
            name
            for name, value in normalized_reflex_scales.items()
            if value < 0.0
        ]
        if negative_reflex_scales:
            raise ValueError(
                "module_reflex_scales must contain only non-negative values."
            )
        if architecture == "monolithic" and normalized_reflex_scales:
            raise ValueError(
                "The monolithic baseline does not accept per-module module_reflex_scales."
            )
        object.__setattr__(self, "module_reflex_scales", normalized_reflex_scales)

    @property
    def is_modular(self) -> bool:
        """
        Whether the configuration uses the modular architecture.
        
        Returns:
            `true` if `architecture` equals "modular", `false` otherwise.
        """
        return self.architecture == "modular"

    @property
    def is_monolithic(self) -> bool:
        """
        Return whether the configuration uses the monolithic architecture.
        
        Returns:
            `true` if `architecture` is "monolithic", `false` otherwise.
        """
        return self.architecture == "monolithic"

    @property
    def uses_local_credit_only(self) -> bool:
        """
        Return whether this ablation uses only local proposal-credit updates.

        The behavior is keyed off the variant name so we can add the experiment
        without changing the saved ablation-config schema.
        """
        return self.name == "local_credit_only"

    def to_summary(self) -> Dict[str, object]:
        """
        Produce a serialization-friendly dictionary summarizing the ablation configuration.
        
        Returns:
            summary (dict): Mapping with keys:
                - "name" (str): Configuration identifier.
                - "architecture" (str): "modular" or "monolithic".
                - "module_dropout" (float): Dropout probability applied to modules.
                - "enable_reflexes" (bool): Whether reflexes are enabled.
                - "enable_auxiliary_targets" (bool): Whether auxiliary targets are enabled.
                - "disabled_modules" (list[str]): List of disabled module names.
                - "reflex_scale" (float): Global reflex scaling factor (>= 0).
                - "module_reflex_scales" (dict[str, float]): Per-module reflex scales (keys are module names).
        """
        return {
            "name": self.name,
            "architecture": self.architecture,
            "module_dropout": self.module_dropout,
            "enable_reflexes": self.enable_reflexes,
            "enable_auxiliary_targets": self.enable_auxiliary_targets,
            "disabled_modules": list(self.disabled_modules),
            "reflex_scale": self.reflex_scale,
            "module_reflex_scales": dict(self.module_reflex_scales),
        }


def default_brain_config(*, module_dropout: float = 0.05) -> BrainAblationConfig:
    """
    Create the canonical "modular_full" brain ablation configuration.
    
    Parameters:
        module_dropout (float): Module-level dropout probability to set on the configuration.
    
    Returns:
        BrainAblationConfig: A `modular_full` configuration with reflexes and auxiliary targets enabled and no disabled modules.
    """
    return BrainAblationConfig(
        name="modular_full",
        architecture="modular",
        module_dropout=module_dropout,
        enable_reflexes=True,
        enable_auxiliary_targets=True,
        disabled_modules=(),
        reflex_scale=1.0,
        module_reflex_scales={},
    )


def canonical_ablation_configs(*, module_dropout: float = 0.05) -> Dict[str, BrainAblationConfig]:
    """
    Builds the canonical registry of named brain ablation configurations.
    
    Creates presets for common ablations (including "modular_full", "no_module_dropout",
    "no_module_reflexes", and "monolithic_policy") and one per-module variant named
    "drop_<module_name>" that disables exactly that module.
    
    Parameters:
        module_dropout (float): Default dropout rate applied to modular variants.
    
    Returns:
        Dict[str, BrainAblationConfig]: Mapping from variant name to its configuration.
    """
    variants: Dict[str, BrainAblationConfig] = {
        "modular_full": default_brain_config(module_dropout=module_dropout),
        "no_module_dropout": BrainAblationConfig(
            name="no_module_dropout",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "no_module_reflexes": BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
            reflex_scale=0.0,
            module_reflex_scales={},
        ),
        "local_credit_only": BrainAblationConfig(
            name="local_credit_only",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "reflex_scale_0_25": BrainAblationConfig(
            name="reflex_scale_0_25",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=0.25,
            module_reflex_scales={},
        ),
        "reflex_scale_0_50": BrainAblationConfig(
            name="reflex_scale_0_50",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=0.50,
            module_reflex_scales={},
        ),
        "reflex_scale_0_75": BrainAblationConfig(
            name="reflex_scale_0_75",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=0.75,
            module_reflex_scales={},
        ),
        "monolithic_policy": BrainAblationConfig(
            name=MONOLITHIC_POLICY_NAME,
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
            reflex_scale=0.0,
            module_reflex_scales={},
        ),
    }
    for module_name in MODULE_NAMES:
        variants[f"drop_{module_name}"] = BrainAblationConfig(
            name=f"drop_{module_name}",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(module_name,),
            reflex_scale=1.0,
            module_reflex_scales={},
        )
    return variants


def canonical_ablation_variant_names(*, module_dropout: float = 0.05) -> tuple[str, ...]:
    """
    Produce the ordered tuple of canonical ablation variant names.
    
    Returns:
        tuple[str, ...]: Ordered variant names starting with "modular_full", then
        "no_module_dropout" and "no_module_reflexes", followed by "drop_<module>"
        for each name in MODULE_NAMES, and ending with "monolithic_policy".
    """
    return tuple(canonical_ablation_configs(module_dropout=module_dropout).keys())


def resolve_ablation_configs(
    names: Sequence[str] | None,
    *,
    module_dropout: float = 0.05,
) -> list[BrainAblationConfig]:
    """
    Resolve a sequence of ablation variant names into their corresponding BrainAblationConfig objects in the requested order.
    
    Parameters:
        names (Sequence[str] | None): Variant names to resolve. If `None`, all canonical variant names are used in their canonical order.
        module_dropout (float): Dropout value used when constructing the canonical registry of variants.
    
    Returns:
        list[BrainAblationConfig]: Config objects corresponding to the requested variant names, in the same order as `names` (or canonical order when `names` is `None`).
    
    Raises:
        KeyError: If any requested names are unknown. The error message lists the unknown names and the available canonical variant names.
    """
    registry = canonical_ablation_configs(module_dropout=module_dropout)
    requested = list(names) if names is not None else list(canonical_ablation_variant_names(module_dropout=module_dropout))
    unknown = sorted({name for name in requested if name not in registry})
    if unknown:
        available = ", ".join(registry.keys())
        raise KeyError(f"Unknown ablation variants: {unknown}. Available: {available}")
    return [registry[name] for name in requested]
