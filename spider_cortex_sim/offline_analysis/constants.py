from __future__ import annotations

from ..ablations import MONOLITHIC_POLICY_NAME, TRUE_MONOLITHIC_POLICY_NAME
from ..noise import canonical_noise_profile_names
from ..reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD

DEFAULT_MODULE_NAMES: tuple[str, ...] = (
    "visual_cortex",
    "sensory_cortex",
    "hunger_center",
    "sleep_center",
    "alert_center",
    "perception_center",
    "homeostasis_center",
    "threat_center",
    "monolithic_policy",
    TRUE_MONOLITHIC_POLICY_NAME,
)
MODULAR_CREDIT_RUNGS: tuple[str, ...] = ("A2", "A3", "A4")
LADDER_RUNG_MAPPING: dict[str, str] = {
    TRUE_MONOLITHIC_POLICY_NAME: "A0",
    MONOLITHIC_POLICY_NAME: "A1",
    "three_center_modular": "A2",
    "three_center_modular_local_credit": "A2",
    "three_center_modular_counterfactual": "A2",
    "four_center_modular": "A3",
    "four_center_modular_local_credit": "A3",
    "four_center_modular_counterfactual": "A3",
    "modular_full": "A4",
    "local_credit_only": "A4",
    "counterfactual_credit": "A4",
}
LADDER_PRIMARY_VARIANT_BY_RUNG: dict[str, str] = {
    "A0": TRUE_MONOLITHIC_POLICY_NAME,
    "A1": MONOLITHIC_POLICY_NAME,
    "A2": "three_center_modular",
    "A3": "four_center_modular",
    "A4": "modular_full",
}
LADDER_PROTOCOL_NAMES: dict[str, str] = {
    "A0": "A0_true_monolithic",
    "A1": "A1_monolithic_with_action_motor",
    "A2": "A2_three_center",
    "A3": "A3_four_center",
    "A4": "A4_current_full_modular",
}
LADDER_ACTIVE_RUNGS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4")
LADDER_ADJACENT_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A0", "A1"),
    ("A1", "A2"),
    ("A2", "A3"),
    ("A3", "A4"),
)
LADDER_RUNG_DESCRIPTIONS: dict[str, str] = {
    "A0": "Direct monolithic control: one network maps shared observations straight to action logits with no action_center or motor_cortex stage.",
    "A1": "Monolithic representation with the existing action_center plus motor_cortex pipeline, isolating whether separating decision from execution helps even without modular sensory processing.",
    "A2": "Three proposer centers with shared action_center and motor_cortex, isolating whether a coarse first modular split is helpful before the full five-module architecture.",
    "A3": "Four proposer centers that split visual and olfactory/sensory pathways while keeping homeostasis and threat unified, testing whether perception separation helps before the full modular split.",
    "A4": "Full modular architecture with the current proposer set and shared downstream action_center plus motor_cortex pipeline.",
}

REFLEX_OVERRIDE_WARNING_THRESHOLD = 0.10
REFLEX_DOMINANCE_WARNING_THRESHOLD = 0.25
SHAPING_DEPENDENCE_WARNING_THRESHOLD = 0.20
DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD = MINIMAL_SHAPING_SURVIVAL_THRESHOLD
AUSTERE_SURVIVAL_THRESHOLD = MINIMAL_SHAPING_SURVIVAL_THRESHOLD
SHAPING_DEPENDENCE_THRESHOLD = SHAPING_DEPENDENCE_WARNING_THRESHOLD
FAILS_WITH_SHAPING_THRESHOLD = 0.30
CLASSIFICATION_LABELS: tuple[str, ...] = (
    "austere_survivor",
    "shaping_dependent",
    "fails_with_shaping",
    "mixed_or_borderline",
)
CANONICAL_NOISE_CONDITIONS: tuple[str, ...] = canonical_noise_profile_names()
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_RANDOM_SEED = 0
