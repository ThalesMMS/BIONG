from __future__ import annotations

from ..noise import canonical_noise_profile_names

DEFAULT_MODULE_NAMES: tuple[str, ...] = (
    "visual_cortex",
    "sensory_cortex",
    "hunger_center",
    "sleep_center",
    "alert_center",
    "monolithic_policy",
)

REFLEX_OVERRIDE_WARNING_THRESHOLD = 0.10
REFLEX_DOMINANCE_WARNING_THRESHOLD = 0.25
SHAPING_DEPENDENCE_WARNING_THRESHOLD = 0.20
DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD = 0.50
CANONICAL_NOISE_CONDITIONS: tuple[str, ...] = canonical_noise_profile_names()
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_RANDOM_SEED = 0
