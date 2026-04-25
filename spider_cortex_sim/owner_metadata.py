from __future__ import annotations

from collections.abc import Mapping


def expected_owner_modules_for(scenario: str | None) -> tuple[str, ...]:
    """
    Retrieve the tuple of module names expected to own the given scenario.
    
    Parameters:
        scenario (str | None): Scenario identifier to look up; falsy values (None or empty) are treated as no scenario.
    
    Returns:
        tuple[str, ...]: Tuple of expected owner module names for the scenario, or an empty tuple if the scenario is missing or has no mapping.
    """
    if not scenario:
        return ()
    from .scenarios import SCENARIOS

    spec = SCENARIOS.get(str(scenario))
    if spec is None:
        return ()
    return tuple(spec.expected_owner_modules)


def owner_alignment_metrics(
    scenario: str | None,
    *,
    dominant_module_distribution: Mapping[str, float],
    module_contribution_share: Mapping[str, float],
) -> dict[str, object]:
    """
    Compute alignment metrics for expected owner modules for a given scenario.
    
    Calculates how much of the dominant distribution and contribution share is covered by the scenario's expected owner modules, determines the best (lowest) rank among those modules in a deterministic ranking, and computes the suppressed rate as the complement of their combined contribution.
    
    Parameters:
        scenario (str | None): Scenario identifier used to look up expected owner modules; when falsy or unmapped, no expected modules apply.
        dominant_module_distribution (Mapping[str, float]): Mapping of module name to its dominant weight used to compute owner_alignment and to rank modules.
        module_contribution_share (Mapping[str, float]): Mapping of module name to its contribution share used to compute owner_suppressed_rate.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - "expected_owner_modules" (list[str]): List of expected owner module names (empty if none apply).
            - "owner_alignment" (float): Sum of dominant weights for expected modules, clamped to the range [0.0, 1.0].
            - "owner_rank" (int): The smallest 1-based rank among expected modules in a descending-weight (tie-broken by name) ordering; 0 if none of the expected modules appear.
            - "owner_suppressed_rate" (float): 1.0 minus the summed contribution share for expected modules, where the summed contribution is clamped to [0.0, 1.0].
    """
    expected = expected_owner_modules_for(scenario)
    if not expected:
        return {
            "expected_owner_modules": [],
            "owner_alignment": 0.0,
            "owner_rank": 0,
            "owner_suppressed_rate": 0.0,
        }
    alignment = float(
        sum(float(dominant_module_distribution.get(name, 0.0)) for name in expected)
    )
    contribution = float(
        sum(float(module_contribution_share.get(name, 0.0)) for name in expected)
    )
    ranked_modules = [
        name
        for name, _ in sorted(
            dominant_module_distribution.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    ]
    owner_rank = 0
    for name in expected:
        if name in ranked_modules:
            candidate_rank = ranked_modules.index(name) + 1
            owner_rank = (
                candidate_rank
                if owner_rank == 0
                else min(owner_rank, candidate_rank)
            )
    return {
        "expected_owner_modules": list(expected),
        "owner_alignment": float(max(0.0, min(1.0, alignment))),
        "owner_rank": int(owner_rank),
        "owner_suppressed_rate": float(1.0 - max(0.0, min(1.0, contribution))),
    }


__all__ = [
    "expected_owner_modules_for",
    "owner_alignment_metrics",
]
