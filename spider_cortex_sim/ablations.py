from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

from .interfaces import MODULE_INTERFACES


MODULE_NAMES: tuple[str, ...] = tuple(spec.name for spec in MODULE_INTERFACES)
MONOLITHIC_POLICY_NAME: str = "monolithic_policy"
PROPOSAL_SOURCE_NAMES: tuple[str, ...] = MODULE_NAMES + (MONOLITHIC_POLICY_NAME,)
REFLEX_MODULE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in MODULE_INTERFACES if spec.role == "proposal"
)
MULTI_PREDATOR_SCENARIOS: tuple[str, ...] = (
    "visual_olfactory_pincer",
    "olfactory_ambush",
    "visual_hunter_open_field",
)
VISUAL_PREDATOR_SCENARIOS: tuple[str, ...] = (
    "visual_olfactory_pincer",
    "visual_hunter_open_field",
)
OLFACTORY_PREDATOR_SCENARIOS: tuple[str, ...] = (
    "visual_olfactory_pincer",
    "olfactory_ambush",
)
MULTI_PREDATOR_SCENARIO_GROUPS: dict[str, tuple[str, ...]] = {
    "multi_predator_ecology": MULTI_PREDATOR_SCENARIOS,
    "visual_predator_scenarios": VISUAL_PREDATOR_SCENARIOS,
    "olfactory_predator_scenarios": OLFACTORY_PREDATOR_SCENARIOS,
}


def _normalize_module_names(
    names: Sequence[str],
    *,
    preserve_order: bool,
    invalid_msg: str,
    monolithic_msg: str,
    architecture: str,
) -> tuple[str, ...]:
    """Normalize, validate, and enforce architecture constraints for a module name sequence.

    Parameters:
        names: Raw module names to normalize.
        preserve_order: If True, deduplicate while preserving first-occurrence order (for recurrent_modules).
                        If False, deduplicate and sort (for disabled_modules).
        invalid_msg: Prefix used in the ValueError when unknown names are found.
        monolithic_msg: Message used in the ValueError when the architecture is monolithic.
        architecture: The configured architecture string; rejects non-empty results when "monolithic".

    Returns:
        Normalized tuple of unique module names.

    Raises:
        ValueError: If any name is not in MODULE_NAMES, or if architecture is "monolithic" and names is non-empty.
    """
    if preserve_order:
        normalized = tuple(dict.fromkeys(str(n) for n in names))
    else:
        normalized = tuple(sorted({str(n) for n in names}))
    invalid = [n for n in normalized if n not in MODULE_NAMES]
    if invalid:
        raise ValueError(f"{invalid_msg}: {invalid}.")
    if architecture == "monolithic" and normalized:
        raise ValueError(monolithic_msg)
    return normalized


@dataclass(frozen=True)
class BrainAblationConfig:
    name: str = "custom"
    architecture: str = "modular"
    module_dropout: float = 0.05
    enable_reflexes: bool = True
    enable_auxiliary_targets: bool = True
    use_learned_arbitration: bool = True
    enable_deterministic_guards: bool = False
    enable_food_direction_bias: bool = False
    warm_start_scale: float = 1.0
    gate_adjustment_bounds: tuple[float, float] = (0.5, 1.5)
    credit_strategy: str = "broadcast"
    disabled_modules: tuple[str, ...] = ()
    recurrent_modules: Tuple[str, ...] = ()
    reflex_scale: float = 1.0
    module_reflex_scales: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and normalize dataclass fields after initialization.
        
        Normalizes and persists these fields on the frozen dataclass:
        - `architecture`: coerced to `str` and must be either "modular" or "monolithic".
        - `module_dropout`: coerced to `float` and must be finite.
        - `use_learned_arbitration`: coerced to `bool`.
        - `enable_deterministic_guards`: coerced to `bool`.
        - `enable_food_direction_bias`: coerced to `bool`.
        - `warm_start_scale`: coerced to `float`, must be finite and in [0.0, 1.0].
        - `gate_adjustment_bounds`: coerced to two finite floats ordered as min < max.
        - `credit_strategy`: coerced to `str` and must be "broadcast", "local_only", or "counterfactual".
        - `reflex_scale`: coerced to `float`, must be finite and >= 0.0.
        - `disabled_modules`: coerced to `str`, deduplicated, sorted into a `tuple`, and each name must be in `MODULE_NAMES`.
        - `recurrent_modules`: coerced to `str`, deduplicated while preserving order into a `tuple`, and each name must be in `MODULE_NAMES`.
        - `module_reflex_scales`: keys coerced to `str`, values coerced to `float`, sorted by key into a `dict`; each key must be in `REFLEX_MODULE_NAMES`, each value must be finite and >= 0.0.
        
        Raises:
            ValueError: If `architecture` is not "modular" or "monolithic";
                        if `module_dropout` or any `module_reflex_scales` value is not finite;
                        if `warm_start_scale` is outside [0.0, 1.0];
                        if `gate_adjustment_bounds` does not contain two finite ordered values;
                        if `reflex_scale` or any `module_reflex_scales` value is negative;
                        if `disabled_modules`, `recurrent_modules`, or `module_reflex_scales` contain unknown module names;
                        or if `architecture == "monolithic"` while `disabled_modules`, `recurrent_modules`, or `module_reflex_scales` are non-empty.
        """
        architecture = str(self.architecture)
        if architecture not in {"modular", "monolithic"}:
            raise ValueError(f"Invalid ablation architecture: {architecture!r}.")
        object.__setattr__(self, "architecture", architecture)
        module_dropout = float(self.module_dropout)
        if not math.isfinite(module_dropout):
            raise ValueError("module_dropout must be finite.")
        object.__setattr__(self, "module_dropout", module_dropout)
        object.__setattr__(
            self,
            "use_learned_arbitration",
            bool(self.use_learned_arbitration),
        )
        object.__setattr__(
            self,
            "enable_deterministic_guards",
            bool(self.enable_deterministic_guards),
        )
        object.__setattr__(
            self,
            "enable_food_direction_bias",
            bool(self.enable_food_direction_bias),
        )
        warm_start_scale = float(self.warm_start_scale)
        if not math.isfinite(warm_start_scale):
            raise ValueError("warm_start_scale must be finite.")
        if warm_start_scale < 0.0 or warm_start_scale > 1.0:
            raise ValueError("warm_start_scale must be in [0.0, 1.0].")
        object.__setattr__(self, "warm_start_scale", warm_start_scale)
        try:
            gate_adjustment_bounds = tuple(
                float(value)
                for value in self.gate_adjustment_bounds
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("gate_adjustment_bounds must contain two finite values.") from exc
        if len(gate_adjustment_bounds) != 2:
            raise ValueError("gate_adjustment_bounds must contain exactly two values.")
        gate_adjustment_min, gate_adjustment_max = gate_adjustment_bounds
        if (
            not math.isfinite(gate_adjustment_min)
            or not math.isfinite(gate_adjustment_max)
        ):
            raise ValueError("gate_adjustment_bounds must contain only finite values.")
        if gate_adjustment_min >= gate_adjustment_max:
            raise ValueError("gate_adjustment_bounds must be ordered as min < max.")
        object.__setattr__(self, "gate_adjustment_bounds", gate_adjustment_bounds)
        credit_strategy = str(self.credit_strategy)
        if credit_strategy not in {"broadcast", "local_only", "counterfactual"}:
            raise ValueError(f"Invalid credit_strategy: {credit_strategy!r}.")
        object.__setattr__(self, "credit_strategy", credit_strategy)
        reflex_scale = float(self.reflex_scale)
        if not math.isfinite(reflex_scale):
            raise ValueError("reflex_scale must be finite.")
        if reflex_scale < 0.0:
            raise ValueError("reflex_scale must be non-negative.")
        object.__setattr__(self, "reflex_scale", reflex_scale)

        object.__setattr__(self, "disabled_modules", _normalize_module_names(
            self.disabled_modules,
            preserve_order=False,
            invalid_msg="Invalid ablation modules",
            monolithic_msg="The monolithic baseline does not accept disabled modules.",
            architecture=architecture,
        ))
        object.__setattr__(self, "recurrent_modules", _normalize_module_names(
            self.recurrent_modules,
            preserve_order=True,
            invalid_msg="Invalid recurrent modules",
            monolithic_msg="The monolithic baseline does not accept recurrent modules.",
            architecture=architecture,
        ))
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
        Indicates whether the configuration applies local-only proposal-credit updates.
        
        @returns
            `true` if `credit_strategy` is "local_only", `false` otherwise.
        """
        return self.credit_strategy == "local_only"

    @property
    def uses_counterfactual_credit(self) -> bool:
        """
        Indicates whether the ablation uses counterfactual credit.
        
        Returns:
            True if `credit_strategy` is "counterfactual", False otherwise.
        """
        return self.credit_strategy == "counterfactual"

    @property
    def is_recurrent(self) -> bool:
        """
        Indicates whether any module proposer uses recurrent state.

        Returns:
            True if `recurrent_modules` is non-empty, False otherwise.
        """
        return bool(self.recurrent_modules)

    def to_summary(self) -> Dict[str, object]:
        """
        Produce a JSON-friendly summary of this ablation configuration.
        
        Returns:
            mapping (dict): Dictionary with the configuration fields:
                - name: str
                - architecture: str
                - module_dropout: float
                - enable_reflexes: bool
                - enable_auxiliary_targets: bool
                - use_learned_arbitration: bool
                - enable_deterministic_guards: bool
                - enable_food_direction_bias: bool
                - warm_start_scale: float
                - gate_adjustment_bounds: list[float]
                - credit_strategy: str
                - disabled_modules: list[str]
                - recurrent_modules: list[str]
                - is_recurrent: bool
                - reflex_scale: float
                - module_reflex_scales: dict[str, float]
        """
        return {
            "name": self.name,
            "architecture": self.architecture,
            "module_dropout": self.module_dropout,
            "enable_reflexes": self.enable_reflexes,
            "enable_auxiliary_targets": self.enable_auxiliary_targets,
            "use_learned_arbitration": self.use_learned_arbitration,
            "enable_deterministic_guards": self.enable_deterministic_guards,
            "enable_food_direction_bias": self.enable_food_direction_bias,
            "warm_start_scale": self.warm_start_scale,
            "gate_adjustment_bounds": list(self.gate_adjustment_bounds),
            "credit_strategy": self.credit_strategy,
            "disabled_modules": list(self.disabled_modules),
            "recurrent_modules": sorted(self.recurrent_modules, key=str),
            "is_recurrent": self.is_recurrent,
            "reflex_scale": self.reflex_scale,
            "module_reflex_scales": dict(self.module_reflex_scales),
        }


def _arbitration_fields(
    *,
    use_learned_arbitration: bool = True,
    enable_deterministic_guards: bool = False,
    enable_food_direction_bias: bool = False,
    warm_start_scale: float = 1.0,
    gate_adjustment_bounds: tuple[float, float] = (0.5, 1.5),
) -> dict[str, object]:
    return {
        "use_learned_arbitration": use_learned_arbitration,
        "enable_deterministic_guards": enable_deterministic_guards,
        "enable_food_direction_bias": enable_food_direction_bias,
        "warm_start_scale": warm_start_scale,
        "gate_adjustment_bounds": gate_adjustment_bounds,
    }


def default_brain_config(*, module_dropout: float = 0.05) -> BrainAblationConfig:
    """
    Provide the canonical "modular_full" brain ablation configuration.
    
    Parameters:
        module_dropout (float): Module-level dropout probability to apply to the configuration.
    
    Returns:
        BrainAblationConfig: A `modular_full` configuration with reflexes and auxiliary targets enabled, deterministic guards and food-direction bias disabled, broadcast credit strategy, and no disabled or recurrent modules.
    """
    return BrainAblationConfig(
        name="modular_full",
        architecture="modular",
        module_dropout=module_dropout,
        enable_reflexes=True,
        enable_auxiliary_targets=True,
        **_arbitration_fields(),
        credit_strategy="broadcast",
        disabled_modules=(),
        recurrent_modules=(),
        reflex_scale=1.0,
        module_reflex_scales={},
    )


def canonical_ablation_configs(*, module_dropout: float = 0.05) -> Dict[str, BrainAblationConfig]:
    """
    Create the canonical registry of named brain ablation configurations.
    
    The registry contains commonly used presets (for example: "modular_full",
    "no_module_dropout", "no_module_reflexes", "modular_recurrent",
    "monolithic_policy"), credit variants, named arbitration-prior variants,
    reflex-scale variants, and one per-module variant named "drop_<module_name>"
    that disables exactly that module.
    
    Parameters:
        module_dropout (float): Default module dropout rate applied to modular variants.
    
    Returns:
        A dict mapping variant names to their corresponding BrainAblationConfig objects.
    """
    variants: Dict[str, BrainAblationConfig] = {
        "modular_full": default_brain_config(module_dropout=module_dropout),
        "no_module_dropout": BrainAblationConfig(
            name="no_module_dropout",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(),
            credit_strategy="broadcast",
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
            **_arbitration_fields(),
            credit_strategy="broadcast",
            disabled_modules=(),
            reflex_scale=0.0,
            module_reflex_scales={},
        ),
        "modular_recurrent": BrainAblationConfig(
            name="modular_recurrent",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(),
            credit_strategy="broadcast",
            disabled_modules=(),
            recurrent_modules=("alert_center", "sleep_center", "hunger_center"),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "modular_recurrent_all": BrainAblationConfig(
            name="modular_recurrent_all",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(),
            credit_strategy="broadcast",
            disabled_modules=(),
            recurrent_modules=MODULE_NAMES,
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "local_credit_only": BrainAblationConfig(
            name="local_credit_only",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(),
            credit_strategy="local_only",
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "counterfactual_credit": BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(),
            credit_strategy="counterfactual",
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "constrained_arbitration": BrainAblationConfig(
            name="constrained_arbitration",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(
                enable_deterministic_guards=True,
                enable_food_direction_bias=True,
            ),
            credit_strategy="broadcast",
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "weaker_prior_arbitration": BrainAblationConfig(
            name="weaker_prior_arbitration",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(warm_start_scale=0.5),
            credit_strategy="broadcast",
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "minimal_arbitration": BrainAblationConfig(
            name="minimal_arbitration",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(
                warm_start_scale=0.0,
                gate_adjustment_bounds=(0.1, 2.0),
            ),
            credit_strategy="broadcast",
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "fixed_arbitration_baseline": BrainAblationConfig(
            name="fixed_arbitration_baseline",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(use_learned_arbitration=False),
            credit_strategy="broadcast",
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        ),
        "learned_arbitration_no_regularization": BrainAblationConfig(
            name="learned_arbitration_no_regularization",
            architecture="modular",
            module_dropout=module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            **_arbitration_fields(),
            credit_strategy="broadcast",
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
            **_arbitration_fields(),
            credit_strategy="broadcast",
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
            **_arbitration_fields(),
            credit_strategy="broadcast",
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
            **_arbitration_fields(),
            credit_strategy="broadcast",
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
            **_arbitration_fields(),
            credit_strategy="broadcast",
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
            **_arbitration_fields(),
            credit_strategy="broadcast",
            disabled_modules=(module_name,),
            recurrent_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        )
    return variants


def canonical_ablation_variant_names(*, module_dropout: float = 0.05) -> tuple[str, ...]:
    """
    Produce the ordered tuple of canonical ablation variant names.
    
    Returns:
        tuple[str, ...]: Ordered variant names starting with "modular_full", then
        "no_module_dropout", "no_module_reflexes", and recurrent/credit/arbitration
        variants, followed by "drop_<module>" for each name in MODULE_NAMES, and
        ending with "monolithic_policy".
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


def _safe_float(value: object) -> float:
    """
    Coerces a value to a finite float, falling back to 0.0 for invalid or non-finite inputs.
    
    Parameters:
        value (object): The value to convert to float.
    
    Returns:
        float: The finite float representation of `value`, or `0.0` if `value` cannot be converted or is not finite.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _finite_float_or_none(value: object) -> float | None:
    """Coerce a value to a finite float, returning None for invalid inputs."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: Sequence[float]) -> float:
    """
    Return the arithmetic mean of a sequence of floats.
    
    If the sequence is empty, returns 0.0.
    
    Parameters:
        values (Sequence[float]): Numeric values to average.
    
    Returns:
        float: The arithmetic mean of the input values, or 0.0 if the sequence is empty.
    """
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _scenario_success_rate(payload: Mapping[str, object], scenario_name: str) -> float | None:
    """
    Extracts a scenario's success rate from a payload that may use current or legacy keys.
    
    Parameters:
        payload (Mapping[str, object]): Mapping expected to contain either a "suite" mapping or a "legacy_scenarios" mapping where each scenario name maps to a dict that may include "success_rate".
        scenario_name (str): The scenario key to look up.
    
    Returns:
        float | None: The scenario's success rate as a finite float if present and valid, otherwise `None`.
    """
    suite = payload.get("suite", {})
    if isinstance(suite, Mapping):
        scenario_payload = suite.get(scenario_name)
        if isinstance(scenario_payload, Mapping):
            if "success_rate" in scenario_payload:
                return _finite_float_or_none(scenario_payload.get("success_rate"))
    legacy_scenarios = payload.get("legacy_scenarios", {})
    if isinstance(legacy_scenarios, Mapping):
        scenario_payload = legacy_scenarios.get(scenario_name)
        if isinstance(scenario_payload, Mapping):
            if "success_rate" in scenario_payload:
                return _finite_float_or_none(scenario_payload.get("success_rate"))
    return None


def _group_metric_value(
    group: object,
    metric_name: str,
    *,
    count_key: str,
) -> float | None:
    """Return a group's metric when the group has real data for that metric."""
    if not isinstance(group, Mapping):
        return None
    if int(group.get(count_key, 0)) <= 0:
        return None
    return _finite_float_or_none(group.get(metric_name))


def compare_predator_type_ablation_performance(
    ablations_payload: Mapping[str, object],
    *,
    variant_names: Sequence[str] = ("drop_visual_cortex", "drop_sensory_cortex"),
) -> dict[str, object]:
    """
    Compute per-variant comparisons of mean success rates and success-rate deltas across visual and olfactory predator scenario groups.
    
    Given an ablations payload mapping (expected to contain "variants" and optional "deltas_vs_reference"), this function iterates the requested variant names and, for each variant present, computes for each multi-predator scenario group:
    - scenario_names: list of scenarios in the group
    - scenario_count: number of scenarios with a valid success rate
    - mean_success_rate: arithmetic mean of available success rates
    - mean_success_rate_delta: arithmetic mean of available success-rate deltas
    
    For each variant the result also includes:
    - visual_minus_olfactory_success_rate: difference between the visual and olfactory group mean success rates, rounded to 6 decimals
    - visual_minus_olfactory_success_rate_delta: difference between the visual and olfactory group mean deltas, rounded to 6 decimals
    
    If the payload does not contain a mapping under "variants", the function returns available=False and an empty comparisons mapping.
    
    Parameters:
        ablations_payload (Mapping[str, object]): Mapping expected to contain "variants" (mapping of variant name to variant payload) and optionally "deltas_vs_reference" (mapping of variant name to per-scenario deltas).
        variant_names (Sequence[str]): Ordered sequence of variant names to process; only variants present and mapping-like in the payload are included.
    
    Returns:
        dict[str, object]: A dictionary with:
            - "available" (bool): True if any comparisons were produced, False otherwise.
            - "scenario_groups" (dict): The canonical mapping of named scenario groups to their scenario tuples.
            - "comparisons" (dict): Mapping from variant name to the computed per-group rows and the visual-minus-olfactory differences.
    """
    variants = ablations_payload.get("variants", {})
    deltas_vs_reference = ablations_payload.get("deltas_vs_reference", {})
    if not isinstance(variants, Mapping):
        return {
            "available": False,
            "scenario_groups": canonical_ablation_scenario_groups(),
            "comparisons": {},
        }

    comparisons: dict[str, object] = {}
    for variant_name in variant_names:
        payload = variants.get(variant_name, {})
        if not isinstance(payload, Mapping):
            continue
        variant_delta_payload = (
            deltas_vs_reference.get(variant_name, {})
            if isinstance(deltas_vs_reference, Mapping)
            else {}
        )
        variant_delta_scenarios = (
            variant_delta_payload.get("scenarios", {})
            if isinstance(variant_delta_payload, Mapping)
            else {}
        )
        group_rows: dict[str, object] = {}
        for group_name, scenario_names in MULTI_PREDATOR_SCENARIO_GROUPS.items():
            scenario_rates = [
                rate
                for scenario_name in scenario_names
                if (rate := _scenario_success_rate(payload, scenario_name)) is not None
            ]
            scenario_deltas: list[float] = []
            for scenario_name in scenario_names:
                if not isinstance(variant_delta_scenarios, Mapping):
                    continue
                scenario_delta = variant_delta_scenarios.get(scenario_name, {})
                if not isinstance(scenario_delta, Mapping):
                    continue
                raw_delta = scenario_delta.get("success_rate_delta")
                parsed_delta = _finite_float_or_none(raw_delta)
                if parsed_delta is not None:
                    scenario_deltas.append(parsed_delta)
            group_rows[group_name] = {
                "scenario_names": list(scenario_names),
                "scenario_count": len(scenario_rates),
                "scenario_delta_count": len(scenario_deltas),
                "mean_success_rate": _mean(scenario_rates),
                "mean_success_rate_delta": _mean(scenario_deltas),
            }
        if not any(
            int(group.get("scenario_count", 0)) > 0
            for group in group_rows.values()
            if isinstance(group, Mapping)
        ):
            continue
        visual_group = group_rows.get("visual_predator_scenarios", {})
        olfactory_group = group_rows.get("olfactory_predator_scenarios", {})
        visual_success_rate = _group_metric_value(
            visual_group,
            "mean_success_rate",
            count_key="scenario_count",
        )
        olfactory_success_rate = _group_metric_value(
            olfactory_group,
            "mean_success_rate",
            count_key="scenario_count",
        )
        visual_success_rate_delta = _group_metric_value(
            visual_group,
            "mean_success_rate_delta",
            count_key="scenario_delta_count",
        )
        olfactory_success_rate_delta = _group_metric_value(
            olfactory_group,
            "mean_success_rate_delta",
            count_key="scenario_delta_count",
        )
        comparisons[str(variant_name)] = {
            **group_rows,
            "visual_minus_olfactory_success_rate": (
                round(visual_success_rate - olfactory_success_rate, 6)
                if visual_success_rate is not None
                and olfactory_success_rate is not None
                else None
            ),
            "visual_minus_olfactory_success_rate_delta": (
                round(
                    visual_success_rate_delta - olfactory_success_rate_delta,
                    6,
                )
                if visual_success_rate_delta is not None
                and olfactory_success_rate_delta is not None
                else None
            ),
        }

    return {
        "available": bool(comparisons),
        "scenario_groups": canonical_ablation_scenario_groups(),
        "comparisons": comparisons,
    }
