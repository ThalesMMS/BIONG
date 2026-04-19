"""Threshold parsing, registry lookup, and claim-summary compatibility helpers."""

from __future__ import annotations

import re
from typing import Dict

from ..claim_tests import ClaimTestSpec

def condense_claim_test_summary(*args: object, **kwargs: object) -> Dict[str, object]:
    """Compatibility wrapper for claim-test summary condensation."""
    from .summary import condense_claim_test_summary as _condense_claim_test_summary

    return _condense_claim_test_summary(*args, **kwargs)

def claim_test_source(spec: ClaimTestSpec) -> str | None:
    """
    Map a claim-test specification to the primitive payload type it requires.

    Parameters:
        spec (ClaimTestSpec): Claim-test specification object with a `.protocol` attribute.

    Returns:
        str | None: `'learning_evidence'`, `'noise_robustness'`, or `'ablation'` when the protocol name references that primitive; `None` if no known primitive is referenced.
    """
    protocol = spec.protocol.lower()
    if "learning-evidence" in protocol:
        return "learning_evidence"
    if "noise-robustness" in protocol:
        return "noise_robustness"
    if "ablation" in protocol:
        return "ablation"
    return None

def claim_skip_result(spec: ClaimTestSpec, reason: str) -> Dict[str, object]:
    """
    Produce a standardized result dictionary representing a skipped claim test.
    
    Parameters:
        spec (ClaimTestSpec): Claim test specification; used to populate `primary_metric`, `scenarios_evaluated`, and `notes`.
        reason (str): Human-readable explanation for why the claim test was skipped.
    
    Returns:
        Dict[str, object]: Result dictionary with keys:
            - "status": "skipped"
            - "passed": False
            - "reason": provided reason string
            - "reference_value": None
            - "comparison_values": {}
            - "delta": {}
            - "effect_size": None
            - "reference_uncertainty": None
            - "comparison_uncertainty": {}
            - "delta_uncertainty": {}
            - "effect_size_uncertainty": {}
            - "cohens_d": {}
            - "effect_magnitude": {}
            - "primary_metric": value from `spec.primary_metric`
            - "scenarios_evaluated": list of scenarios from `spec.scenarios`
            - "notes": list containing `spec.success_criterion`
    """
    return {
        "status": "skipped",
        "passed": False,
        "reason": str(reason),
        "reference_value": None,
        "comparison_values": {},
        "delta": {},
        "effect_size": None,
        "reference_uncertainty": None,
        "comparison_uncertainty": {},
        "delta_uncertainty": {},
        "effect_size_uncertainty": {},
        "cohens_d": {},
        "effect_magnitude": {},
        "primary_metric": spec.primary_metric,
        "scenarios_evaluated": list(spec.scenarios),
        "notes": [spec.success_criterion],
    }

def claim_threshold_from_operator(
    success_criterion: str,
    operator: str,
) -> float | None:
    """
    Parses the first numeric literal immediately following a given operator token in a criterion string.
    
    Parameters:
        success_criterion (str): Criterion text to search.
        operator (str): Literal operator token to match (e.g., ">=", "<", "==").
    
    Returns:
        float | None: Parsed number (supports negative and decimal values) if found, otherwise None.
    """
    match = re.search(
        rf"{re.escape(operator)}\s*(-?\d+(?:\.\d+)?)",
        success_criterion,
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None

def claim_threshold_from_phrase(
    success_criterion: str,
    phrase: str,
) -> float | None:
    """
    Extracts a numeric value immediately following a given literal phrase in a criterion string.
    
    Parameters:
        success_criterion (str): Text to search for the phrase and subsequent number.
        phrase (str): Literal phrase to locate; the function returns the number that appears directly after this phrase.
    
    Returns:
        float | None: The parsed numeric value if a number immediately follows `phrase`, `None` if no match or parsing fails.
    """
    match = re.search(
        rf"{re.escape(phrase)}\s*(-?\d+(?:\.\d+)?)",
        success_criterion,
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None

def claim_count_threshold(success_criterion: str) -> int | None:
    """
    Extracts the integer count N from a success-criterion phrase of the form "at least N of the M".
    
    Parameters:
        success_criterion (str): Text containing the success-criterion to inspect.
    
    Returns:
        int | None: The parsed integer N if the pattern is present and valid, otherwise `None`.
    """
    match = re.search(r"at least\s+(\d+)\s+of\s+the\s+\d+", success_criterion)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None

def claim_registry_entry(
    payload: Dict[str, object] | None,
    *,
    registry_key: str,
    entry_name: str,
) -> tuple[Dict[str, object] | None, str | None]:
    """
    Locate a named entry in a registry payload and verify it was evaluated.
    
    Parameters:
        payload (Dict[str, object] | None): Top-level payload expected to contain one or more registries.
        registry_key (str): Top-level key in `payload` expected to contain the registry mapping.
        entry_name (str): Name of the entry to fetch from the registry.
    
    Returns:
        tuple[Dict[str, object] | None, str | None]: `(entry, error)` where `entry` is the registry entry dict when present and not marked skipped, otherwise `None`. `error` is a human-readable message if the registry or entry is missing or if the entry was skipped, otherwise `None`.
    """
    if not isinstance(payload, dict):
        return None, f"Missing {registry_key} payload."
    registry = payload.get(registry_key, {})
    if not isinstance(registry, dict):
        return None, f"{registry_key!r} payload is missing its registry."
    entry = registry.get(entry_name)
    if not isinstance(entry, dict):
        return None, f"Missing {registry_key[:-1]} {entry_name!r}."
    if bool(entry.get("skipped")):
        return None, str(
            entry.get("reason", f"{registry_key[:-1].capitalize()} {entry_name!r} was skipped.")
        )
    return entry, None

__all__ = [
    "claim_count_threshold",
    "claim_registry_entry",
    "claim_skip_result",
    "claim_test_source",
    "claim_threshold_from_operator",
    "claim_threshold_from_phrase",
    "condense_claim_test_summary",
]
