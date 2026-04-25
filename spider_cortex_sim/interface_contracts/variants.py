from __future__ import annotations

from typing import Dict, Tuple

from .module_specs import *

HUNGER_CENTER_V1_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["hunger_center"],
    name="hunger_v1",
    signal_names=(
        "hunger",
        "on_food",
        "food_visible",
        "food_dx",
        "food_dy",
    ),
    description="Diagnostic sufficiency variant for hunger_center with hunger and direct visual food direction only.",
)

HUNGER_CENTER_V2_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["hunger_center"],
    name="hunger_v2",
    signal_names=(
        *HUNGER_CENTER_V1_INTERFACE.signal_names,
        "food_smell_strength",
        "food_smell_dx",
        "food_smell_dy",
    ),
    description="Diagnostic sufficiency variant for hunger_center with direct food cues plus smell gradient.",
)

HUNGER_CENTER_V3_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["hunger_center"],
    name="hunger_v3",
    signal_names=(
        *HUNGER_CENTER_V2_INTERFACE.signal_names,
        "food_trace_dx",
        "food_trace_dy",
        "food_trace_strength",
        "food_trace_heading_dx",
        "food_trace_heading_dy",
    ),
    description="Diagnostic sufficiency variant for hunger_center with direct, olfactory, and short food trace cues.",
)

SLEEP_CENTER_V1_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["sleep_center"],
    name="sleep_v1",
    signal_names=(
        "fatigue",
        "night",
        "on_shelter",
    ),
    description="Diagnostic sufficiency variant for sleep_center with fatigue, night state, and current shelter occupancy only.",
)

SLEEP_CENTER_V2_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["sleep_center"],
    name="sleep_v2",
    signal_names=(
        *SLEEP_CENTER_V1_INTERFACE.signal_names,
        "shelter_role_level",
    ),
    description="Diagnostic sufficiency variant for sleep_center adding shelter depth or role level.",
)

SLEEP_CENTER_V3_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["sleep_center"],
    name="sleep_v3",
    signal_names=(
        *SLEEP_CENTER_V2_INTERFACE.signal_names,
        "shelter_trace_dx",
        "shelter_trace_dy",
        "shelter_trace_strength",
        "shelter_trace_heading_dx",
        "shelter_trace_heading_dy",
    ),
    description="Diagnostic sufficiency variant for sleep_center adding short shelter trace cues.",
)

ALERT_CENTER_V1_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["alert_center"],
    name="alert_v1",
    signal_names=(
        "predator_visible",
        "predator_dx",
        "predator_dy",
    ),
    description="Diagnostic sufficiency variant for alert_center with direct predator visibility and direction only.",
)

ALERT_CENTER_V2_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["alert_center"],
    name="alert_v2",
    signal_names=(
        *ALERT_CENTER_V1_INTERFACE.signal_names,
        "predator_certainty",
        "predator_occluded",
    ),
    description="Diagnostic sufficiency variant for alert_center adding visual certainty and occlusion cues.",
)

ALERT_CENTER_V3_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["alert_center"],
    name="alert_v3",
    signal_names=(
        *ALERT_CENTER_V2_INTERFACE.signal_names,
        "predator_smell_strength",
        "recent_pain",
        "recent_contact",
        "on_shelter",
    ),
    description="Diagnostic sufficiency variant for alert_center adding smell, pain, contact, and shelter occupancy cues.",
)

VISUAL_CORTEX_V1_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["visual_cortex"],
    name="visual_v1",
    signal_names=(
        "food_visible",
        "food_dx",
        "food_dy",
        "heading_dx",
        "heading_dy",
    ),
    description="Diagnostic sufficiency variant for visual_cortex with visible food direction and current heading only.",
)

VISUAL_CORTEX_V2_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["visual_cortex"],
    name="visual_v2",
    signal_names=(
        *VISUAL_CORTEX_V1_INTERFACE.signal_names,
        "shelter_visible",
        "shelter_dx",
        "shelter_dy",
        "predator_visible",
        "predator_dx",
        "predator_dy",
    ),
    description="Diagnostic sufficiency variant for visual_cortex adding direct shelter and predator direction cues.",
)

VISUAL_CORTEX_V3_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["visual_cortex"],
    name="visual_v3",
    signal_names=(
        *VISUAL_CORTEX_V2_INTERFACE.signal_names,
        "foveal_scan_age",
        "food_trace_strength",
        "food_trace_heading_dx",
        "food_trace_heading_dy",
        "shelter_trace_strength",
        "shelter_trace_heading_dx",
        "shelter_trace_heading_dy",
        "predator_trace_strength",
        "predator_trace_heading_dx",
        "predator_trace_heading_dy",
        "predator_motion_salience",
        "visual_predator_threat",
        "olfactory_predator_threat",
        "day",
        "night",
    ),
    description="Diagnostic sufficiency variant for visual_cortex adding scan age, percept traces, threat salience, and day-night context.",
)

SENSORY_CORTEX_V1_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["sensory_cortex"],
    name="sensory_v1",
    signal_names=(
        "recent_pain",
        "recent_contact",
        "predator_smell_strength",
        "predator_smell_dx",
        "predator_smell_dy",
    ),
    description="Diagnostic sufficiency variant for sensory_cortex with acute pain/contact and predator smell cues only.",
)

SENSORY_CORTEX_V2_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["sensory_cortex"],
    name="sensory_v2",
    signal_names=(
        *SENSORY_CORTEX_V1_INTERFACE.signal_names,
        "food_smell_strength",
        "food_smell_dx",
        "food_smell_dy",
    ),
    description="Diagnostic sufficiency variant for sensory_cortex adding food-smell cues.",
)

SENSORY_CORTEX_V3_INTERFACE = _subset_interface(
    MODULE_INTERFACE_BY_NAME["sensory_cortex"],
    name="sensory_v3",
    signal_names=(
        *SENSORY_CORTEX_V2_INTERFACE.signal_names,
        "health",
        "hunger",
        "fatigue",
    ),
    description="Diagnostic sufficiency variant for sensory_cortex adding core homeostatic state without ambient light.",
)

INTERFACE_VARIANTS: Dict[Tuple[str, int], ModuleInterface] = {
    ("visual_cortex", 1): VISUAL_CORTEX_V1_INTERFACE,
    ("visual_cortex", 2): VISUAL_CORTEX_V2_INTERFACE,
    ("visual_cortex", 3): VISUAL_CORTEX_V3_INTERFACE,
    ("sensory_cortex", 1): SENSORY_CORTEX_V1_INTERFACE,
    ("sensory_cortex", 2): SENSORY_CORTEX_V2_INTERFACE,
    ("sensory_cortex", 3): SENSORY_CORTEX_V3_INTERFACE,
    ("hunger_center", 1): HUNGER_CENTER_V1_INTERFACE,
    ("hunger_center", 2): HUNGER_CENTER_V2_INTERFACE,
    ("hunger_center", 3): HUNGER_CENTER_V3_INTERFACE,
    ("sleep_center", 1): SLEEP_CENTER_V1_INTERFACE,
    ("sleep_center", 2): SLEEP_CENTER_V2_INTERFACE,
    ("sleep_center", 3): SLEEP_CENTER_V3_INTERFACE,
    ("alert_center", 1): ALERT_CENTER_V1_INTERFACE,
    ("alert_center", 2): ALERT_CENTER_V2_INTERFACE,
    ("alert_center", 3): ALERT_CENTER_V3_INTERFACE,
}
VARIANT_INTERFACES: tuple[ModuleInterface, ...] = tuple(INTERFACE_VARIANTS.values())


def get_interface_variant(module_name: str, level: int) -> ModuleInterface:
    """
    Retrieve a diagnostic-sufficiency ModuleInterface variant for a named proposal module.
    
    Levels 1–3 return the corresponding registered variant; level 4 returns the module's canonical interface.
    
    Parameters:
        module_name (str): Name of the proposal module (must be one of the recognized variant modules).
        level (int): Variant level to retrieve (1-4).
    
    Returns:
        ModuleInterface: The interface specification for the given module and level.
    
    Raises:
        ValueError: If `module_name` is not a recognized variant module or if `level` is not available for that module.
    """
    if module_name not in VARIANT_MODULES:
        raise ValueError(f"Unknown interface variant module '{module_name}'.")
    if level == 4:
        try:
            return MODULE_INTERFACE_BY_NAME[module_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown interface variant module '{module_name}'."
            ) from exc
    try:
        return INTERFACE_VARIANTS[(module_name, level)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown interface variant level {level} for module '{module_name}'."
        ) from exc


def get_variant_levels(module_name: str) -> tuple[int, ...]:
    """
    List available variant levels for the given module, including level 4 for the canonical interface.
    
    Parameters:
        module_name (str): Name of a module that supports interface variants.
    
    Returns:
        tuple[int, ...]: Tuple of available levels (subset of 1–4) in ascending order; level 4 is included when the canonical interface applies.
    
    Raises:
        ValueError: If `module_name` is not a known variant module.
    """
    if module_name not in VARIANT_MODULES:
        raise ValueError(f"Unknown interface variant module '{module_name}'.")
    return tuple(
        level
        for level in (1, 2, 3, 4)
        if level == 4 or (module_name, level) in INTERFACE_VARIANTS
    )


def is_variant_interface(interface: ModuleInterface) -> bool:
    """
    Check whether the given ModuleInterface object is one of the registered diagnostic sufficiency variants.
    
    Returns:
        True if the provided interface is identical to a registered variant object, False otherwise.
    """
    return any(variant is interface for variant in INTERFACE_VARIANTS.values())


def validate_variant_interfaces() -> None:
    """
    Ensure every registered variant interface is consistent with its parent module interface.
    
    Performs three checks for each (module_name, level) in INTERFACE_VARIANTS:
    - all signal names in the variant are present in the parent interface,
    - the variant's observation_key matches the parent's observation_key,
    - the variant's signal_names appear in the same order as in the parent's inputs.
    
    Raises:
        ValueError: if any variant fails these checks; the exception message lists all detected inconsistencies.
    """
    errors: list[str] = []
    for (module_name, level), interface in INTERFACE_VARIANTS.items():
        parent = MODULE_INTERFACE_BY_NAME[module_name]
        missing = [name for name in interface.signal_names if name not in parent.signal_names]
        if missing:
            errors.append(
                f"{interface.name} (module={module_name}, level={level}) has signals absent from "
                f"{parent.name}: {missing}."
            )
        if interface.observation_key != parent.observation_key:
            errors.append(
                f"{interface.name} (module={module_name}, level={level}) observation_key "
                f"'{interface.observation_key}' does not match parent '{parent.observation_key}'."
            )
        expected_order = tuple(
            signal.name for signal in parent.inputs if signal.name in interface.signal_names
        )
        if interface.signal_names != expected_order:
            errors.append(
                f"{interface.name} (module={module_name}, level={level}) signal order does not "
                f"match parent '{parent.name}'. Expected {expected_order}, got {interface.signal_names}."
            )
    if errors:
        raise ValueError("Invalid interface variants:\n- " + "\n- ".join(errors))


validate_variant_interfaces()

OBSERVATION_INTERFACE_BY_KEY: Dict[str, ModuleInterface] = {
    spec.observation_key: spec for spec in ALL_INTERFACES
}

__all__ = [
    "ALERT_CENTER_V1_INTERFACE",
    "ALERT_CENTER_V2_INTERFACE",
    "ALERT_CENTER_V3_INTERFACE",
    "HUNGER_CENTER_V1_INTERFACE",
    "HUNGER_CENTER_V2_INTERFACE",
    "HUNGER_CENTER_V3_INTERFACE",
    "INTERFACE_VARIANTS",
    "OBSERVATION_INTERFACE_BY_KEY",
    "SENSORY_CORTEX_V1_INTERFACE",
    "SENSORY_CORTEX_V2_INTERFACE",
    "SENSORY_CORTEX_V3_INTERFACE",
    "SLEEP_CENTER_V1_INTERFACE",
    "SLEEP_CENTER_V2_INTERFACE",
    "SLEEP_CENTER_V3_INTERFACE",
    "VARIANT_INTERFACES",
    "VISUAL_CORTEX_V1_INTERFACE",
    "VISUAL_CORTEX_V2_INTERFACE",
    "VISUAL_CORTEX_V3_INTERFACE",
    "get_interface_variant",
    "get_variant_levels",
    "is_variant_interface",
    "validate_variant_interfaces",
]
