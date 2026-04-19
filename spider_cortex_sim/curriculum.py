from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence


CURRICULUM_PROFILE_NAMES: tuple[str, ...] = (
    "none",
    "ecological_v1",
    "ecological_v2",
)
CURRICULUM_COLUMNS: tuple[str, ...] = (
    "training_regime",
    "training_regime_name",
    "curriculum_profile",
    "curriculum_phase",
    "curriculum_skill",
    "curriculum_phase_status",
    "curriculum_promotion_reason",
)
CURRICULUM_FOCUS_SCENARIOS: tuple[str, ...] = (
    "open_field_foraging",
    "corridor_gauntlet",
    "exposed_day_foraging",
    "food_deprivation",
)


@dataclass(frozen=True)
class PromotionCheckCriteria:
    """Promotion gate for a named behavior check.

    `aggregation` controls how a phase combines its check specs: "all" requires
    every spec to pass, while "any" promotes when at least one spec passes.
    """

    scenario: str
    check_name: str
    required_pass_rate: float
    aggregation: str = "all"


@dataclass(frozen=True)
class CurriculumPhaseDefinition:
    name: str
    training_scenarios: tuple[str, ...]
    promotion_scenarios: tuple[str, ...]
    success_threshold: float
    max_episodes: int
    min_episodes: int
    skill_name: str = ""
    promotion_check_specs: tuple[PromotionCheckCriteria, ...] = ()


SUBSKILL_CHECK_MAPPINGS: Dict[str, tuple[PromotionCheckCriteria, ...]] = {
    "shelter_exit": (
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="commits_to_foraging",
            required_pass_rate=1.0,
        ),
    ),
    "food_approach": (
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="approaches_food",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="open_field_foraging",
            check_name="made_food_progress",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="exposed_day_foraging",
            check_name="day_food_progress",
            required_pass_rate=1.0,
        ),
    ),
    "predator_response": (
        PromotionCheckCriteria(
            scenario="predator_edge",
            check_name="predator_detected",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="predator_edge",
            check_name="predator_reacted",
            required_pass_rate=1.0,
        ),
    ),
    "corridor_navigation": (
        PromotionCheckCriteria(
            scenario="corridor_gauntlet",
            check_name="corridor_survives",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="corridor_gauntlet",
            check_name="corridor_food_progress",
            required_pass_rate=1.0,
        ),
    ),
    "hunger_commitment": (
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="hunger_reduced",
            required_pass_rate=1.0,
        ),
        PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="survives_deprivation",
            required_pass_rate=1.0,
        ),
    ),
}


def _curriculum_profile_error_message() -> str:
    """
    Constructs an error message listing the allowed curriculum profile names.
    
    Returns:
        error_message (str): A message indicating the provided curriculum_profile is invalid and listing the available profiles.
    """
    available = ", ".join(repr(name) for name in CURRICULUM_PROFILE_NAMES)
    return f"Invalid curriculum_profile. Available profiles: {available}."


def validate_curriculum_profile(curriculum_profile: str) -> str:
    """
    Validate and normalize a curriculum profile name.
    
    Converts the input to a string and verifies it is one of the allowed curriculum profile identifiers.
    
    Returns:
        The validated profile name as a string.
    
    Raises:
        ValueError: If the profile name is not one of CURRICULUM_PROFILE_NAMES.
    """
    profile_name = str(curriculum_profile)
    if profile_name not in CURRICULUM_PROFILE_NAMES:
        raise ValueError(_curriculum_profile_error_message())
    return profile_name


def resolve_curriculum_phase_budgets(total_episodes: int) -> list[int]:
    """
    Allocate a total episode budget across four ordered curriculum phases.
    
    Parameters:
        total_episodes (int): Total episodes (coerced to an integer and clamped to zero if negative).
    
    Returns:
        list[int]: Four integers [phase1, phase2, phase3, phase4] giving the episode budget for each phase.
    """
    total = max(0, int(total_episodes))
    if total <= 0:
        return [0, 0, 0, 0]
    remaining = total
    phase_1 = min(remaining, max(1, total // 6))
    remaining -= phase_1
    phase_2 = min(remaining, max(1, total // 6))
    remaining -= phase_2
    phase_3 = min(remaining, max(1, total // 3))
    remaining -= phase_3
    phase_4 = remaining
    return [phase_1, phase_2, phase_3, phase_4]


def resolve_curriculum_profile(
    *,
    curriculum_profile: str,
    total_episodes: int,
) -> list[CurriculumPhaseDefinition]:
    """
    Builds an ordered list of curriculum phase definitions for the given profile and training budget.
    
    Parameters:
        curriculum_profile (str): Curriculum profile identifier (e.g., "none", "ecological_v1", "ecological_v2").
        total_episodes (int): Total training episodes to allocate across the curriculum phases.
    
    Returns:
        list[CurriculumPhaseDefinition]: Ordered phase definitions for the selected profile. Returns an empty list when the profile is "none". Each phase contains its training and promotion scenario tuples, success threshold, and per-phase episode min/max values and promotion check specifications.
    
    Raises:
        ValueError: If `curriculum_profile` is not one of the supported profile names.
    """
    profile_name = validate_curriculum_profile(curriculum_profile)
    if profile_name == "none":
        return []
    budgets = resolve_curriculum_phase_budgets(total_episodes)

    if profile_name == "ecological_v1":
        phase_specs = (
            (
                "phase_1_night_rest_predator_edge",
                "predator_response",
                ("night_rest", "predator_edge"),
                ("night_rest", "predator_edge"),
                1.0,
                (),
            ),
            (
                "phase_2_entrance_ambush_shelter_blockade",
                "shelter_exit",
                ("entrance_ambush", "shelter_blockade"),
                ("entrance_ambush", "shelter_blockade"),
                1.0,
                (),
            ),
            (
                "phase_3_open_field_exposed_day",
                "food_approach",
                ("open_field_foraging", "exposed_day_foraging"),
                ("open_field_foraging", "exposed_day_foraging"),
                0.5,
                (),
            ),
            (
                "phase_4_corridor_food_deprivation",
                "corridor_gauntlet+food_deprivation",
                ("corridor_gauntlet", "food_deprivation"),
                ("corridor_gauntlet", "food_deprivation"),
                0.5,
                (),
            ),
        )
    else:
        phase_specs = (
            (
                "phase_1_shelter_safety_predator_awareness",
                "predator_response",
                ("night_rest", "predator_edge", "entrance_ambush"),
                ("predator_edge",),
                1.0,
                tuple(SUBSKILL_CHECK_MAPPINGS["predator_response"]),
            ),
            (
                "phase_2_shelter_exit_commitment",
                "shelter_exit",
                ("entrance_ambush", "shelter_blockade", "food_deprivation"),
                ("food_deprivation",),
                1.0,
                tuple(SUBSKILL_CHECK_MAPPINGS["shelter_exit"]),
            ),
            (
                "phase_3_food_approach_under_exposure",
                "food_approach",
                (
                    "food_deprivation",
                    "open_field_foraging",
                    "exposed_day_foraging",
                ),
                (
                    "food_deprivation",
                    "open_field_foraging",
                    "exposed_day_foraging",
                ),
                1.0,
                tuple(SUBSKILL_CHECK_MAPPINGS["food_approach"]),
            ),
            (
                "phase_4_corridor_navigation_hunger_survival",
                "corridor_navigation+hunger_commitment",
                ("corridor_gauntlet", "food_deprivation"),
                ("corridor_gauntlet", "food_deprivation"),
                1.0,
                (
                    *SUBSKILL_CHECK_MAPPINGS["corridor_navigation"],
                    *SUBSKILL_CHECK_MAPPINGS["hunger_commitment"],
                ),
            ),
        )

    phases: list[CurriculumPhaseDefinition] = []
    if len(budgets) != len(phase_specs):
        raise ValueError("Curriculum budgets and phase specs must have the same length.")
    for (
        budget,
        (
            name,
            skill_name,
            training_scenarios,
            promotion_scenarios,
            threshold,
            promotion_check_specs,
        ),
    ) in zip(budgets, phase_specs):
        phases.append(
            CurriculumPhaseDefinition(
                name=name,
                training_scenarios=tuple(training_scenarios),
                promotion_scenarios=tuple(promotion_scenarios),
                success_threshold=float(threshold),
                max_episodes=int(budget),
                min_episodes=max(1, int(budget) // 2) if int(budget) > 0 else 0,
                skill_name=str(skill_name),
                promotion_check_specs=tuple(promotion_check_specs),
            )
        )
    return phases


def promotion_check_spec_records(
    specs: Sequence[PromotionCheckCriteria],
) -> list[Dict[str, object]]:
    """Serialize promotion check criteria into plain dict records."""
    return [
        {
            "scenario": str(spec.scenario),
            "check_name": str(spec.check_name),
            "required_pass_rate": float(spec.required_pass_rate),
            "aggregation": str(spec.aggregation),
        }
        for spec in specs
    ]


def evaluate_promotion_check_specs(
    payload: Dict[str, object],
    specs: Sequence[PromotionCheckCriteria],
) -> tuple[Dict[str, Dict[str, Dict[str, object]]], bool, str]:
    """
    Determine whether a curriculum phase's promotion checks are satisfied by a behavior-suite payload.
    
    Validates that all specs use a single aggregation mode ('all' or 'any') and that no duplicate (scenario, check) pairs are present. Each spec's `required_pass_rate` must be convertible to a float between 0.0 and 1.0. For each spec, the payload's suite is inspected at payload["suite"][scenario]["checks"][check]["pass_rate"]; a missing or invalid pass rate is treated as 0.0. Aggregation behavior:
    - "all": promotion succeeds only if every check passes.
    - "any": promotion succeeds if at least one check passes.
    If `specs` is empty the function returns immediately with reason "no_checks_specified".
    
    Parameters:
        payload (Dict[str, object]): A behavior-suite payload; expected to contain a mapping under the "suite" key.
        specs (Sequence[PromotionCheckCriteria]): Iterable of promotion check specifications to evaluate.
    
    Returns:
        tuple:
            results (Dict[str, Dict[str, Dict[str, object]]]): Nested mapping of scenario -> check -> detail dict containing:
                - "scenario": scenario name (str)
                - "pass_rate": observed pass rate (float)
                - "required": required pass rate (float)
                - "passed": whether the check met the required pass rate (bool)
            promotion_passed (bool): `true` if the phase passes according to the aggregation mode, `false` otherwise.
            reason (str): One of "all_checks_passed", "any_check_passed", "check_failed:<first_failed_check>", or "no_checks_specified".
    """
    if not specs:
        return {}, False, "no_checks_specified"
    suite = payload.get("suite", {})
    if not isinstance(suite, dict):
        suite = {}
    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    first_failed_check = ""
    aggregations = {str(spec.aggregation).lower() for spec in specs}
    invalid_aggregations = aggregations - {"all", "any"}
    if invalid_aggregations:
        invalid = sorted(invalid_aggregations)[0]
        raise ValueError(
            "Invalid promotion check aggregation. Use 'all' or 'any': "
            f"{invalid!r}."
        )
    if len(aggregations) > 1:
        raise ValueError(
            "Mixed promotion check aggregations are not supported within one phase."
        )
    aggregation = next(iter(aggregations), "all")
    seen_specs: set[tuple[str, str]] = set()
    for spec in specs:
        scenario_name = str(spec.scenario)
        check_name = str(spec.check_name)
        spec_key = (scenario_name, check_name)
        if spec_key in seen_specs:
            raise ValueError(
                "Duplicate promotion check spec for scenario "
                f"{scenario_name!r} check {check_name!r}."
            )
        seen_specs.add(spec_key)
        try:
            required = float(spec.required_pass_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid promotion check required_pass_rate for scenario "
                f"{scenario_name!r} check {check_name!r}: "
                f"{spec.required_pass_rate!r}."
            ) from exc
        if not 0.0 <= required <= 1.0:
            raise ValueError(
                "Promotion check required_pass_rate must be between 0.0 "
                f"and 1.0 for scenario {scenario_name!r} check "
                f"{check_name!r}: {required!r}."
            )
        pass_rate = 0.0
        scenario_data = suite.get(scenario_name)
        checks = (
            scenario_data.get("checks", {})
            if isinstance(scenario_data, dict)
            else {}
        )
        check_data = checks.get(check_name) if isinstance(checks, dict) else None
        if isinstance(check_data, dict) and "pass_rate" in check_data:
            try:
                pass_rate = float(check_data.get("pass_rate", 0.0))
            except (TypeError, ValueError):
                pass_rate = 0.0
        passed = bool(pass_rate >= required)
        if not passed and not first_failed_check:
            first_failed_check = check_name
        scenario_results = results.setdefault(scenario_name, {})
        scenario_results[check_name] = {
            "scenario": scenario_name,
            "pass_rate": pass_rate,
            "required": required,
            "passed": passed,
        }
    passed_values = [
        bool(result["passed"])
        for scenario_results in results.values()
        for result in scenario_results.values()
    ]
    if aggregation == "any":
        promotion_passed = any(passed_values)
    else:
        promotion_passed = bool(passed_values) and all(passed_values)
    if promotion_passed:
        reason = "any_check_passed" if aggregation == "any" else "all_checks_passed"
    else:
        reason = f"check_failed:{first_failed_check}"
    return results, promotion_passed, reason


def empty_curriculum_summary(
    curriculum_profile: str,
    total_episodes: int,
) -> Dict[str, object]:
    """
    Create a minimal curriculum summary dictionary for seeding metadata or representing a zero-budget curriculum.
    
    Parameters:
        curriculum_profile (str): Profile identifier; coerced to `str` in the result.
        total_episodes (int): Total training episodes; coerced to `int` and clamped to be >= 0.
    
    Returns:
        Dict[str, object]: A summary dict with keys:
            - "profile": the coerced profile name
            - "total_training_episodes": the resolved non-negative episode count
            - "executed_training_episodes": always 0
            - "status": "not_started" if `total_training_episodes` > 0, otherwise "no_training_budget"
            - "phases": an empty list
    """
    resolved_episodes = max(0, int(total_episodes))
    return {
        "profile": str(curriculum_profile),
        "total_training_episodes": resolved_episodes,
        "executed_training_episodes": 0,
        "status": "not_started" if resolved_episodes > 0 else "no_training_budget",
        "phases": [],
    }


def regime_row_metadata_from_summary(
    training_regime: Dict[str, object],
    curriculum_summary: Dict[str, object] | None,
) -> Dict[str, object]:
    """Build flat training-regime and latest-curriculum-phase CSV metadata."""
    latest_phase: Dict[str, object] | None = None
    if isinstance(curriculum_summary, dict):
        try:
            executed_training_episodes = int(
                curriculum_summary.get("executed_training_episodes", 0)
            )
        except (TypeError, ValueError):
            executed_training_episodes = 0
        if executed_training_episodes > 0:
            phases = curriculum_summary.get("phases", [])
            if isinstance(phases, list) and phases:
                for phase in reversed(phases):
                    if not isinstance(phase, dict):
                        continue
                    try:
                        episodes_executed = int(
                            phase.get("episodes_executed", 0)
                        )
                    except (TypeError, ValueError):
                        episodes_executed = 0
                    if episodes_executed > 0:
                        latest_phase = phase
                        break

    def _phase_str(key: str) -> str:
        """
        Return the string value for `key` from the currently selected latest phase, or an empty string if no latest phase is set or the key is missing.
        
        Parameters:
            key (str): The dictionary key to retrieve from the latest phase.
        
        Returns:
            str: The stringified value of the requested key, or `""` if not available.
        """
        return str(latest_phase.get(key, "")) if latest_phase is not None else ""

    return {
        "training_regime": str(training_regime.get("mode", "flat")),
        "training_regime_name": str(training_regime.get("name", "baseline")),
        "curriculum_profile": str(training_regime.get("curriculum_profile", "none")),
        "curriculum_phase": _phase_str("name"),
        "curriculum_skill": _phase_str("skill_name"),
        "curriculum_phase_status": _phase_str("status"),
        "curriculum_promotion_reason": _phase_str("promotion_reason"),
    }
