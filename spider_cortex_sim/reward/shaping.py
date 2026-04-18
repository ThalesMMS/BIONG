"""Reward-shaping disposition policy and reduction-roadmap helpers."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Literal, Mapping, Sequence, TypedDict

from ._utils import policy_float
from .profiles import MINIMAL_SHAPING_SURVIVAL_THRESHOLD

SHAPING_DISPOSITIONS: Sequence[str] = (
    "defended",
    "weakened",
    "removed",
    "outcome_signal",
    "under_investigation",
)

DispositionName = Literal[
    "defended",
    "weakened",
    "removed",
    "outcome_signal",
    "under_investigation",
]
ReductionPriority = Literal["high", "medium", "low"]
ScenarioRequirementLevel = Literal["gate", "warning", "diagnostic", "excluded"]


class ShapingGapPolicy(TypedDict):
    minimal_profile: str
    max_scenario_success_rate_delta: Dict[str, float]
    max_mean_reward_delta: Dict[str, float]
    min_austere_survival_rate: float
    warning_threshold_multiplier: float
    notes: List[str]


class DispositionEvidenceEntry(TypedDict):
    definition: str
    required_evidence: List[str]
    example_components: List[str]
    transition_conditions: List[str]


class ShapingRoadmapEntry(TypedDict):
    current_disposition: DispositionName
    target_disposition: DispositionName
    reduction_priority: ReductionPriority
    evidence_required: List[str]
    blocking_scenarios: List[str]
    notes: str


class ScenarioAustereRequirement(TypedDict):
    requirement_level: ScenarioRequirementLevel
    rationale: str
    claim_test_linkage: List[str]


SHAPING_GAP_POLICY: ShapingGapPolicy = {
    "minimal_profile": "austere",
    "max_scenario_success_rate_delta": {
        "classic_minus_austere": 0.20,
        "ecological_minus_austere": 0.15,
    },
    "max_mean_reward_delta": {
        "classic_minus_austere": 0.50,
        "ecological_minus_austere": 0.40,
    },
    "min_austere_survival_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    "warning_threshold_multiplier": 0.80,
    "notes": [
        "Positive scenario-success deltas above the profile-specific limit mark shaping dependence.",
        "Mean-reward gaps are treated as absolute profile-scale gaps because dense shaping can shift reward scale in either direction.",
        "Warnings fire before hard gates: upper-bound metrics warn at limit * warning_threshold_multiplier, while austere survival warns when it falls near the minimum gate.",
    ],
}

DISPOSITION_EVIDENCE_CRITERIA: Dict[str, DispositionEvidenceEntry] = {
    "defended": {
        "definition": (
            "A retained reward signal with a documented physiological or ecological necessity, "
            "not merely behavioral convenience."
        ),
        "required_evidence": [
            "The component maps to an external hazard, energy cost, physiological state, or terminal outcome.",
            "Austere scenario survival remains at or above the configured threshold for relevant claim scenarios.",
            "Classic/ecological versus austere gaps remain within the shaping-gap policy limits.",
        ],
        "example_components": [
            "action_cost",
            "night_exposure",
            "hunger_pressure",
            "fatigue_pressure",
        ],
        "transition_conditions": [
            "Move to weakened when the term is necessary but its profile gap exceeds policy limits.",
            "Move to under_investigation when the ecological or physiological rationale is not supported by scenario evidence.",
        ],
    },
    "weakened": {
        "definition": (
            "A reward signal still present after dense shaping reduction, but bounded or split so "
            "austere behavior is not explained primarily by the signal."
        ),
        "required_evidence": [
            "Austere survival reaches the threshold on a majority of relevant scenarios.",
            "Performance gaps are bounded by the shaping-gap policy or documented as warnings.",
            "Any remaining dense portion has a concrete follow-up reduction target.",
        ],
        "example_components": ["resting"],
        "transition_conditions": [
            "Move to removed when all relevant scenarios survive austere and claim tests show no regression.",
            "Move to defended only when the remaining signal is proven to be physiological or ecological rather than scaffolded guidance.",
        ],
    },
    "removed": {
        "definition": (
            "A direct guidance or auxiliary scaffold that is zeroed in the austere profile and treated "
            "as removable from experiment-of-record claims unless defended by new evidence."
        ),
        "required_evidence": [
            "Austere survival is at or above threshold on all relevant scenarios.",
            "No claim test regresses when the component is absent or zeroed in austere.",
            "The component does not anchor a sparse outcome such as feeding, predator contact, or death.",
        ],
        "example_components": [
            "food_progress",
            "shelter_progress",
            "predator_escape",
            "day_exploration",
        ],
        "transition_conditions": [
            "Remain removed while austere survival and claim tests hold.",
            "Move to under_investigation if removing the term exposes a reproducible claim regression.",
        ],
    },
    "outcome_signal": {
        "definition": (
            "A sparse outcome anchor tied to consummatory success, acute damage, or terminal failure "
            "rather than incremental behavioral steering."
        ),
        "required_evidence": [
            "The signal fires on an outcome event rather than on progress toward an outcome.",
            "It is not repeatedly emitted as a dense proximity or occupancy bonus.",
            "Its audit notes remain separate from reduction candidates unless new steering evidence appears.",
        ],
        "example_components": ["feeding", "predator_contact", "death_penalty"],
        "transition_conditions": [
            "Move to under_investigation if the outcome begins firing densely enough to steer intermediate behavior.",
            "Do not classify as removed solely because it is high magnitude; sparse outcome anchors are policy-exempt.",
        ],
    },
    "under_investigation": {
        "definition": (
            "A term that needs more scenario evidence before it can be defended, weakened, or removed."
        ),
        "required_evidence": [
            "Identify blocking scenarios and compare them under classic, ecological, and austere profiles.",
            "Measure whether failures are due to missing ecological consequence, reward-scale gap, or policy weakness.",
            "Resolve the term into defended, weakened, or removed before using it as support for core claims.",
        ],
        "example_components": [
            "terrain_cost",
            "homeostasis_penalty",
        ],
        "transition_conditions": [
            "Move to defended when physiological or ecological necessity is documented and gaps are within policy.",
            "Move to weakened or removed when bounded austere performance shows the dense signal is not required.",
        ],
    },
}

SHAPING_REDUCTION_ROADMAP: Dict[str, ShapingRoadmapEntry] = {
    "resting": {
        "current_disposition": "weakened",
        "target_disposition": "weakened",
        "reduction_priority": "high",
        "evidence_required": [
            "Split sparse rest outcome evidence from configurable rest/deep-shelter bonuses.",
            "Show night_rest, two_shelter_tradeoff, and sleep_vs_exploration_conflict survive austere at or above threshold.",
            "Keep classic/ecological reward gaps within policy after further rest-bonus trimming.",
        ],
        "blocking_scenarios": [
            "night_rest",
            "two_shelter_tradeoff",
            "sleep_vs_exploration_conflict",
        ],
        "notes": (
            "Keep a sparse rest outcome available, but keep configurable rest bonuses on the "
            "short list for further weakening if austere survival remains stable."
        ),
    },
    "night_exposure": {
        "current_disposition": "defended",
        "target_disposition": "defended",
        "reduction_priority": "medium",
        "evidence_required": [
            "Demonstrate the term behaves as ecological hazard pressure rather than shelter waypoint guidance.",
            "Check austere survival gates for night_rest and shelter-return scenarios.",
            "Bound classic/ecological versus austere success deltas within policy.",
        ],
        "blocking_scenarios": [
            "night_rest",
            "two_shelter_tradeoff",
            "sleep_vs_exploration_conflict",
        ],
        "notes": (
            "Currently defended as exposure hazard, but dense per-tick pressure must be "
            "kept tied to night exposure rather than navigation convenience."
        ),
    },
    "hunger_pressure": {
        "current_disposition": "defended",
        "target_disposition": "defended",
        "reduction_priority": "medium",
        "evidence_required": [
            "Show the pressure represents physiological need without pointing directly to food.",
            "Check weak-signal foraging diagnostics under austere before further strengthening.",
            "Keep mean-reward and success-rate gaps inside the shaping-gap policy.",
        ],
        "blocking_scenarios": [
            "food_deprivation",
            "open_field_foraging",
            "exposed_day_foraging",
            "food_vs_predator_conflict",
        ],
        "notes": (
            "Treat as physiological necessity for now, but require bounded gaps because "
            "continuous hunger pressure can still become homeostatic shaping."
        ),
    },
    "fatigue_pressure": {
        "current_disposition": "defended",
        "target_disposition": "defended",
        "reduction_priority": "medium",
        "evidence_required": [
            "Show the pressure represents exhaustion cost rather than direct shelter guidance.",
            "Check rest and shelter-return scenarios under austere.",
            "Confirm pressure changes do not become the primary driver of scenario success deltas.",
        ],
        "blocking_scenarios": [
            "night_rest",
            "two_shelter_tradeoff",
            "recover_after_failed_chase",
        ],
        "notes": (
            "Remain defended as physiological pressure, with explicit gap bounds before "
            "using it as evidence for sleep or shelter-return claims."
        ),
    },
    "sleep_debt_pressure": {
        "current_disposition": "defended",
        "target_disposition": "defended",
        "reduction_priority": "high",
        "evidence_required": [
            "Separate physiological sleep debt from dense pressure that indirectly guides shelter seeking.",
            "Require austere survival in night_rest and sleep_vs_exploration_conflict.",
            "Inspect whether sleep pressure explains gaps in shelter-return and rest scenarios.",
        ],
        "blocking_scenarios": [
            "night_rest",
            "sleep_vs_exploration_conflict",
            "two_shelter_tradeoff",
        ],
        "notes": (
            "Defended only if it remains physiological pressure; if it substitutes for "
            "learned shelter return, the term should be weakened."
        ),
    },
    "homeostasis_penalty": {
        "current_disposition": "defended",
        "target_disposition": "under_investigation",
        "reduction_priority": "medium",
        "evidence_required": [
            "Quantify hardcoded and configured mass separately.",
            "Show deterioration penalties are physiological consequences, not hidden route guidance.",
            "Check survival and reward-gap policy on rest, hunger, and conflict scenarios.",
        ],
        "blocking_scenarios": [
            "food_deprivation",
            "night_rest",
            "sleep_vs_exploration_conflict",
        ],
        "notes": (
            "Currently defended, but the mixed hardcoded/configured mass needs stronger "
            "evidence before it remains permanently defended."
        ),
    },
    "terrain_cost": {
        "current_disposition": "defended",
        "target_disposition": "under_investigation",
        "reduction_priority": "medium",
        "evidence_required": [
            "Confirm clutter and narrow-passage costs are neutral ecological costs.",
            "Audit narrow_predator_risk separately because it may steer away from bottlenecks under threat.",
            "Run corridor and ambush scenarios under austere before defending the full terrain family.",
        ],
        "blocking_scenarios": [
            "corridor_gauntlet",
            "entrance_ambush",
            "shelter_blockade",
        ],
        "notes": (
            "Terrain cost is plausibly ecological, but narrow_predator_risk needs evidence "
            "that it is hazard modeling rather than behavior-directing scaffold."
        ),
    },
    "action_cost": {
        "current_disposition": "defended",
        "target_disposition": "defended",
        "reduction_priority": "low",
        "evidence_required": [
            "Confirm stay and move costs remain small universal energy costs across profiles.",
            "Check that action-cost changes do not decide conflict or shelter-return outcomes.",
            "Keep deltas within policy before treating action cost as neutral.",
        ],
        "blocking_scenarios": [
            "recover_after_failed_chase",
            "food_vs_predator_conflict",
            "sleep_vs_exploration_conflict",
        ],
        "notes": (
            "Lowest-priority reduction target because it is universal rather than directional, "
            "but it still needs periodic neutrality checks."
        ),
    },
}

SCENARIO_AUSTERE_REQUIREMENTS: Dict[str, ScenarioAustereRequirement] = {
    "night_rest": {
        "requirement_level": "gate",
        "rationale": "Core shelter/rest claim; dense rest and night-shelter shaping must not be required.",
        "claim_test_linkage": [
            "learning_without_privileged_signals",
            "memory_improves_shelter_return",
        ],
    },
    "predator_edge": {
        "requirement_level": "gate",
        "rationale": "Core predator-response claim; escape behavior must survive without progress escape rewards.",
        "claim_test_linkage": [
            "learning_without_privileged_signals",
            "escape_without_reflex_support",
            "noise_preserves_threat_valence",
        ],
    },
    "entrance_ambush": {
        "requirement_level": "gate",
        "rationale": "Ambush response must be supported by sensory threat evidence and outcomes, not dense escape guidance.",
        "claim_test_linkage": [
            "learning_without_privileged_signals",
            "escape_without_reflex_support",
            "noise_preserves_threat_valence",
        ],
    },
    "shelter_blockade": {
        "requirement_level": "gate",
        "rationale": "Shelter access under threat is a core no-reflex predator-response gate.",
        "claim_test_linkage": [
            "learning_without_privileged_signals",
            "escape_without_reflex_support",
            "noise_preserves_threat_valence",
        ],
    },
    "two_shelter_tradeoff": {
        "requirement_level": "gate",
        "rationale": "Shelter memory and tradeoff behavior must survive without distance-to-shelter scaffolding.",
        "claim_test_linkage": [
            "learning_without_privileged_signals",
            "memory_improves_shelter_return",
        ],
    },
    "visual_olfactory_pincer": {
        "requirement_level": "gate",
        "rationale": "Multi-predator specialization should survive minimal shaping across dual threat cues.",
        "claim_test_linkage": [
            "specialization_emerges_with_multiple_predators",
            "noise_preserves_threat_valence",
        ],
    },
    "olfactory_ambush": {
        "requirement_level": "gate",
        "rationale": "Olfactory threat handling must not depend on dense predator-escape progress shaping.",
        "claim_test_linkage": [
            "specialization_emerges_with_multiple_predators",
            "noise_preserves_threat_valence",
        ],
    },
    "visual_hunter_open_field": {
        "requirement_level": "gate",
        "rationale": "Visual predator response in exposed terrain is a core specialization and threat-valence gate.",
        "claim_test_linkage": [
            "specialization_emerges_with_multiple_predators",
            "noise_preserves_threat_valence",
        ],
    },
    "recover_after_failed_chase": {
        "requirement_level": "warning",
        "rationale": "Recovery after threat is important, but current evidence should warn before gating the whole program.",
        "claim_test_linkage": ["predator_response_recovery"],
    },
    "food_vs_predator_conflict": {
        "requirement_level": "warning",
        "rationale": "Threat-over-food arbitration should be reported under austere, with failure treated as reduction risk.",
        "claim_test_linkage": ["noise_preserves_threat_valence", "action_center_arbitration"],
    },
    "sleep_vs_exploration_conflict": {
        "requirement_level": "warning",
        "rationale": "Sleep-over-exploration arbitration directly informs rest shaping but is not yet a hard claim gate.",
        "claim_test_linkage": ["action_center_arbitration", "sleep_rest_policy"],
    },
    "open_field_foraging": {
        "requirement_level": "diagnostic",
        "rationale": "Weak-signal foraging is a capability probe; austere survival is reported to guide hunger-shaping reductions.",
        "claim_test_linkage": ["weak_signal_foraging"],
    },
    "corridor_gauntlet": {
        "requirement_level": "diagnostic",
        "rationale": "Corridor navigation diagnoses terrain and predator-risk costs without gating core claims yet.",
        "claim_test_linkage": ["corridor_navigation_under_threat"],
    },
    "exposed_day_foraging": {
        "requirement_level": "diagnostic",
        "rationale": "Exposed foraging reports whether hunger pressure works without restoring food-progress shaping.",
        "claim_test_linkage": ["weak_signal_foraging"],
    },
    "food_deprivation": {
        "requirement_level": "diagnostic",
        "rationale": "Hunger commitment and recovery diagnose homeostatic pressure but remain calibration probes.",
        "claim_test_linkage": ["weak_signal_foraging", "hunger_driven_commitment"],
    },
}

def shaping_reduction_roadmap() -> dict[str, object]:
    """
    Produce a defensive payload containing the austere-profile reduction roadmap, gap policy, evidence criteria, and scenario requirements.
    
    The dictionary is a deep-copied snapshot of module-level catalogs so callers may inspect or attach it to reports without mutating internal state.
    
    Returns:
        roadmap_payload (dict[str, object]): Payload with the following keys:
            - minimal_profile: the canonical minimal profile name (fixed to "austere").
            - survival_threshold: numeric austere survival threshold (MINIMAL_SHAPING_SURVIVAL_THRESHOLD).
            - reduction_targets: deep copy of SHAPING_REDUCTION_ROADMAP (per-component targets and priorities).
            - gap_policy: deep copy of SHAPING_GAP_POLICY (thresholds and multipliers for violations/warnings).
            - evidence_criteria: deep copy of DISPOSITION_EVIDENCE_CRITERIA (required evidence per disposition).
            - scenario_requirements: deep copy of SCENARIO_AUSTERE_REQUIREMENTS (per-scenario requirement levels and rationale).
            - notes: list of human-readable notes describing gate/warning/diagnostic semantics and reduction intent.
    """
    gap_policy = deepcopy(SHAPING_GAP_POLICY)
    return {
        "minimal_profile": gap_policy["minimal_profile"],
        "survival_threshold": gap_policy["min_austere_survival_rate"],
        "reduction_targets": deepcopy(SHAPING_REDUCTION_ROADMAP),
        "gap_policy": gap_policy,
        "evidence_criteria": deepcopy(DISPOSITION_EVIDENCE_CRITERIA),
        "scenario_requirements": deepcopy(SCENARIO_AUSTERE_REQUIREMENTS),
        "notes": [
            "Roadmap targets focus on dense signals that remain defended, weakened, or unresolved after austere removed direct progress guidance.",
            "Gate scenarios must meet austere survival before they support core claims.",
            "Warning and diagnostic scenarios remain reportable evidence for deciding which terms to weaken next.",
        ],
    }


def validate_shaping_disposition(
    metadata: Mapping[str, object],
    component_name: str,
) -> str:
    """
    Validate and return the shaping disposition declared in a component's metadata.
    
    Parameters:
        metadata (Mapping[str, object]): Mapping expected to contain the key `"shaping_disposition"`.
        component_name (str): Human-readable name of the component used in error messages.
    
    Returns:
        str: The validated disposition name from `metadata["shaping_disposition"]`.
    
    Raises:
        ValueError: If `"shaping_disposition"` is missing or its value is not one of the allowed dispositions.
    """
    disposition = metadata.get("shaping_disposition")
    if disposition is None:
        raise ValueError(
            f"Reward component {component_name!r} is missing shaping_disposition."
        )
    disposition_name = str(disposition)
    if disposition_name not in SHAPING_DISPOSITIONS:
        raise ValueError(
            f"Reward component {component_name!r} has invalid shaping disposition "
            f"{disposition_name!r}."
        )
    return disposition_name


def validate_gap_policy(comparison_data: Mapping[str, object]) -> dict[str, object]:
    """
    Validate a reward-profile comparison payload against the shaping-gap policy.
    
    This function accepts the comparison payload shape produced by
    `spider_cortex_sim.comparison.build_reward_audit_comparison` and tolerates missing
    or partial sections. It evaluates configured thresholds for per-profile
    scenario success-rate deltas, mean-reward deltas, and austere survival rates
    (global and per-scenario) and classifies observed violations and warnings.
    
    Parameters:
        comparison_data (Mapping[str, object]): Comparison payload to validate. Expected
            sections (optional) include `deltas_vs_minimal` and `behavior_survival`.
    
    Returns:
        result (dict[str, object]): Validation summary containing:
            - passes (bool): `True` when no violations were found, `False` otherwise.
            - policy (dict): A defensive copy of the shaping-gap policy used for checks.
            - violations (list[dict]): Observations that exceed strict policy limits.
            - warnings (list[dict]): Observations that exceed warning thresholds but not strict limits.
            - checked_profiles (list[str]): Profile names that were examined.
            - checked_scenarios (list[str]): Scenario names that were examined.
            - notes (list[str]): Human-readable notes describing the check semantics.
    """
    policy = deepcopy(SHAPING_GAP_POLICY)
    warning_multiplier = policy_float(
        policy.get("warning_threshold_multiplier"),
        0.80,
    )
    if warning_multiplier <= 0.0:
        warning_multiplier = 0.80
    warning_multiplier = min(1.0, warning_multiplier)
    min_survival_rate = policy_float(
        policy.get("min_austere_survival_rate"),
        MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    )
    survival_warning_floor = min(
        1.0,
        min_survival_rate / warning_multiplier,
    )
    violations: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    checked_profiles: list[str] = []
    checked_scenarios: list[str] = []

    deltas = comparison_data.get("deltas_vs_minimal", {})
    if isinstance(deltas, Mapping):
        for profile_name, delta_payload in sorted(deltas.items()):
            if not isinstance(delta_payload, Mapping):
                continue
            profile_key = f"{profile_name}_minus_austere"
            checked_profiles.append(str(profile_name))
            success_limits = policy["max_scenario_success_rate_delta"]
            if isinstance(success_limits, Mapping) and profile_key in success_limits:
                limit = policy_float(success_limits[profile_key])
                observed = max(
                    0.0,
                    policy_float(delta_payload.get("scenario_success_rate_delta")),
                )
                item = {
                    "profile": str(profile_name),
                    "comparison": profile_key,
                    "metric": "scenario_success_rate_delta",
                    "observed_delta": round(float(observed), 6),
                    "limit": round(float(limit), 6),
                }
                if observed > limit:
                    violations.append(item)
                elif observed > limit * warning_multiplier:
                    warnings.append(item)
            reward_limits = policy["max_mean_reward_delta"]
            if isinstance(reward_limits, Mapping) and profile_key in reward_limits:
                limit = policy_float(reward_limits[profile_key])
                observed = abs(policy_float(delta_payload.get("mean_reward_delta")))
                item = {
                    "profile": str(profile_name),
                    "comparison": profile_key,
                    "metric": "mean_reward_delta",
                    "observed_delta": round(float(observed), 6),
                    "limit": round(float(limit), 6),
                }
                if observed > limit:
                    violations.append(item)
                elif observed > limit * warning_multiplier:
                    warnings.append(item)

    behavior_survival = comparison_data.get("behavior_survival", {})
    if isinstance(behavior_survival, Mapping):
        survival_available = bool(behavior_survival.get("available", False))
        if survival_available and "survival_rate" in behavior_survival:
            survival_rate = policy_float(behavior_survival.get("survival_rate"))
            survival_item = {
                "metric": "austere_survival_rate",
                "observed_rate": round(float(survival_rate), 6),
                "minimum": round(float(min_survival_rate), 6),
            }
            if survival_rate < min_survival_rate:
                violations.append(survival_item)
            elif survival_rate < survival_warning_floor:
                warnings.append(
                    {
                        **survival_item,
                        "warning_floor": round(float(survival_warning_floor), 6),
                    }
                )
        scenarios = behavior_survival.get("scenarios", {})
        if isinstance(scenarios, Mapping):
            for scenario_name, scenario_payload in sorted(scenarios.items()):
                if not isinstance(scenario_payload, Mapping):
                    continue
                requirement = SCENARIO_AUSTERE_REQUIREMENTS.get(str(scenario_name))
                if requirement is None:
                    continue
                requirement_level = str(requirement["requirement_level"])
                raw = scenario_payload.get("austere_success_rate")
                if raw is None:
                    continue
                try:
                    observed = float(raw)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(observed):
                    continue
                if requirement_level == "diagnostic":
                    checked_scenarios.append(str(scenario_name))
                    continue
                if requirement_level == "excluded":
                    continue
                if requirement_level not in {"gate", "warning"}:
                    continue
                item = {
                    "scenario": str(scenario_name),
                    "requirement_level": requirement_level,
                    "metric": "austere_success_rate",
                    "observed_rate": round(float(observed), 6),
                    "minimum": round(float(min_survival_rate), 6),
                }
                checked_scenarios.append(str(scenario_name))
                if requirement_level == "gate":
                    if observed < min_survival_rate:
                        violations.append(item)
                    elif observed < survival_warning_floor:
                        warnings.append(
                            {
                                **item,
                                "warning_floor": round(float(survival_warning_floor), 6),
                            }
                        )
                elif requirement_level == "warning" and observed < min_survival_rate:
                    warnings.append(item)

    return {
        "passes": not violations,
        "policy": policy,
        "violations": violations,
        "warnings": warnings,
        "checked_profiles": sorted(set(checked_profiles)),
        "checked_scenarios": sorted(set(checked_scenarios)),
        "notes": [
            "Positive dense-profile success deltas above policy limits are violations.",
            "Mean reward deltas use absolute gap size because reward scale can drift in either direction.",
            "Gate scenario austere survival below the minimum is a violation; warning scenarios report warnings only.",
        ],
    }
