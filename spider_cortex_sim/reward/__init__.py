"""Compatibility exports for reward profiles, audits, shaping policy, and computation."""

from .audit import (
    REWARD_COMPONENT_AUDIT,
    reward_component_audit,
    reward_profile_audit,
    shaping_disposition_summary,
)
from .audit import _roadmap_status_for_profile
from ._utils import policy_float as _policy_float
from .computation import (
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_total,
)
from .profiles import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_COMPONENT_NAMES,
    REWARD_PROFILES,
)
from .shaping import (
    DISPOSITION_EVIDENCE_CRITERIA,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_DISPOSITIONS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    DispositionEvidenceEntry,
    DispositionName,
    ReductionPriority,
    ScenarioAustereRequirement,
    ScenarioRequirementLevel,
    ShapingGapPolicy,
    ShapingRoadmapEntry,
    shaping_reduction_roadmap,
    validate_gap_policy,
    validate_shaping_disposition,
)

__all__ = [
    "DISPOSITION_EVIDENCE_CRITERIA",
    "DispositionEvidenceEntry",
    "DispositionName",
    "MINIMAL_SHAPING_SURVIVAL_THRESHOLD",
    "REWARD_COMPONENT_AUDIT",
    "REWARD_COMPONENT_NAMES",
    "REWARD_PROFILES",
    "ReductionPriority",
    "SCENARIO_AUSTERE_REQUIREMENTS",
    "SHAPING_DISPOSITIONS",
    "SHAPING_GAP_POLICY",
    "SHAPING_REDUCTION_ROADMAP",
    "ScenarioAustereRequirement",
    "ScenarioRequirementLevel",
    "ShapingGapPolicy",
    "ShapingRoadmapEntry",
    "apply_action_and_terrain_effects",
    "apply_pressure_penalties",
    "apply_progress_and_event_rewards",
    "compute_predator_threat",
    "copy_reward_components",
    "empty_reward_components",
    "reward_component_audit",
    "reward_profile_audit",
    "reward_total",
    "shaping_disposition_summary",
    "shaping_reduction_roadmap",
    "validate_gap_policy",
    "validate_shaping_disposition",
]
