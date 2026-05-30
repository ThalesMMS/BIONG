from __future__ import annotations

from ._config_model import *


def _arbitration_fields(
    *,
    use_learned_arbitration: bool = True,
    enable_deterministic_guards: bool = False,
    enable_food_direction_bias: bool = False,
    warm_start_scale: float = 1.0,
    gate_adjustment_bounds: tuple[float, float] = (0.5, 1.5),
) -> dict[str, object]:
    """
    Collects arbitration-related configuration values into a dictionary.
    
    Parameters:
        use_learned_arbitration (bool): Whether learned arbitration is enabled.
        enable_deterministic_guards (bool): Whether deterministic guard behavior is enabled.
        enable_food_direction_bias (bool): Whether a bias toward food direction is applied.
        warm_start_scale (float): Scaling factor applied during warm-start of arbitration.
        gate_adjustment_bounds (tuple[float, float]): Lower and upper bounds used to adjust gate values (min, max).
    
    Returns:
        dict[str, object]: Mapping of arbitration field names to the provided values:
            {
                "use_learned_arbitration": ...,
                "enable_deterministic_guards": ...,
                "enable_food_direction_bias": ...,
                "warm_start_scale": ...,
                "gate_adjustment_bounds": ...,
            }
    """
    return {
        "use_learned_arbitration": use_learned_arbitration,
        "enable_deterministic_guards": enable_deterministic_guards,
        "enable_food_direction_bias": enable_food_direction_bias,
        "warm_start_scale": warm_start_scale,
        "gate_adjustment_bounds": gate_adjustment_bounds,
    }


def default_brain_config(
    *,
    module_dropout: float = 0.05,
    capacity_profile: str | CapacityProfile | None = None,
) -> BrainAblationConfig:
    """
    Provide the canonical "modular_full" brain ablation configuration.
    
    Parameters:
        module_dropout (float): Module-level dropout probability to apply to the configuration.
        capacity_profile (str | CapacityProfile | None): Capacity profile to
            apply to the returned configuration, or None to use the default.
    
    Returns:
        BrainAblationConfig: A `modular_full` A4 configuration with the
        requested capacity profile, reflexes and auxiliary targets enabled,
        deterministic guards and food-direction bias disabled, route-mask credit
        strategy, coarse rollup modules disabled, and no recurrent modules.
    """
    return BrainAblationConfig(
        name="modular_full",
        architecture="modular",
        module_dropout=module_dropout,
        enable_reflexes=True,
        enable_auxiliary_targets=True,
        **_arbitration_fields(),
        credit_strategy="route_mask",
        disabled_modules=COARSE_ROLLUP_MODULES,
        recurrent_modules=(),
        reflex_scale=1.0,
        module_reflex_scales={},
        capacity_profile=capacity_profile,
    )
