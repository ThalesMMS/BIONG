"""Leakage audit summaries and scaffold-assessment config extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

from ..claim_tests import ClaimTestSpec
from ..memory import memory_leakage_audit
from ..perception import observation_leakage_audit

from .subsets import claim_payload_config_summary, claim_payload_eval_reflex_scale
from .thresholds import claim_registry_entry, claim_test_source

def claim_leakage_audit_summary() -> Dict[str, object]:
    """
    Summarizes unresolved privileged or world-derived leakage findings.

    Returns:
        summary (dict): A mapping with:
            - finding_count (int): Number of unresolved findings.
            - findings (list[str]): List of findings formatted as "<audit_name>:<signal_name>".
    """
    findings: list[str] = []
    for audit_name, audit_entries in (
        ("observation", observation_leakage_audit()),
        ("memory", memory_leakage_audit()),
    ):
        for signal_name, metadata in audit_entries.items():
            classification = str(metadata.get("classification", ""))
            risk = str(metadata.get("risk", ""))
            if classification in {
                "privileged_world_signal",
                "world_derived_navigation_hint",
            } and risk != "resolved":
                findings.append(f"{audit_name}:{signal_name}")
    return {
        "finding_count": len(findings),
        "findings": findings,
    }

def extract_claim_config_for_scaffold_assessment(
    spec: ClaimTestSpec,
    payloads: dict,
) -> tuple[dict, float | None]:
    """
    Extract the scaffold configuration summary and the evaluation reflex scale needed for scaffold assessment.
    
    Parameters:
        spec (ClaimTestSpec): Claim test specification whose protocol determines the payload source and which reference/comparison entries to consult.
        payloads (dict): Mapping of primitive payload types ("learning_evidence", "ablation", "noise_robustness") to their loaded payload dictionaries.
    
    Returns:
        tuple[dict, float | None]: A pair where the first element is a shallow config summary dictionary extracted from the selected reference/baseline payload (empty dict if missing or invalid), and the second element is the extracted `eval_reflex_scale` as a finite float or `None` when unavailable.
    """
    source = claim_test_source(spec)
    if source == "learning_evidence":
        payload = payloads.get("learning_evidence")
        reference_payload, _ = claim_registry_entry(
            payload,
            registry_key="conditions",
            entry_name=spec.reference_condition,
        )
        config_summary = claim_payload_config_summary(reference_payload)
        if not config_summary:
            return {}, None
        determining_comparison_name: str | None = None
        if "trained_without_reflex_support" in spec.comparison_conditions:
            determining_comparison_name = "trained_without_reflex_support"
        elif spec.comparison_conditions:
            determining_comparison_name = spec.comparison_conditions[0]
        determining_comparison_payload = None
        if determining_comparison_name is not None:
            determining_comparison_payload, _ = claim_registry_entry(
                payload,
                registry_key="conditions",
                entry_name=determining_comparison_name,
            )
        eval_reflex_scale = claim_payload_eval_reflex_scale(
            determining_comparison_payload
        )
        if eval_reflex_scale is None:
            eval_reflex_scale = claim_payload_eval_reflex_scale(
                reference_payload
            )
        return (config_summary, eval_reflex_scale)

    if source == "ablation":
        payload = payloads.get("ablation")
        reference_payload, _ = claim_registry_entry(
            payload,
            registry_key="variants",
            entry_name=spec.reference_condition,
        )
        config_summary = claim_payload_config_summary(reference_payload)
        eval_reflex_scale = claim_payload_eval_reflex_scale(reference_payload)
        if not config_summary and isinstance(reference_payload, dict):
            without_reflex_payload = reference_payload.get(
                "without_reflex_support"
            )
            config_summary = claim_payload_config_summary(
                without_reflex_payload
            )
            if eval_reflex_scale is None:
                eval_reflex_scale = claim_payload_eval_reflex_scale(
                    without_reflex_payload
                )
        if not config_summary:
            return {}, None
        return config_summary, eval_reflex_scale

    if source == "noise_robustness":
        payload = payloads.get("noise_robustness")
        if not isinstance(payload, dict):
            return {}, None
        matrix = payload.get("matrix", {})
        if not isinstance(matrix, dict):
            return {}, None
        baseline_condition: str | None = None
        matrix_spec = payload.get("matrix_spec", {})
        if isinstance(matrix_spec, dict):
            train_conditions = matrix_spec.get("train_conditions", [])
            eval_conditions = matrix_spec.get("eval_conditions", [])
            if isinstance(train_conditions, (list, tuple)) and isinstance(
                eval_conditions,
                (list, tuple),
            ):
                train_names = [str(condition) for condition in train_conditions]
                eval_names = [str(condition) for condition in eval_conditions]
                if "none" in train_names and "none" in eval_names:
                    baseline_condition = "none"
                else:
                    eval_name_set = set(eval_names)
                    for condition in train_names:
                        if condition in eval_name_set:
                            baseline_condition = condition
                            break
        if baseline_condition is None:
            for train_condition, row in matrix.items():
                if isinstance(row, dict) and str(train_condition) in row:
                    baseline_condition = str(train_condition)
                    break
        if baseline_condition is None:
            return {}, None
        row = matrix.get(baseline_condition, {})
        if not isinstance(row, dict):
            return {}, None
        baseline_payload = row.get(baseline_condition)
        config_summary = claim_payload_config_summary(baseline_payload)
        if not config_summary:
            return {}, None
        return (
            config_summary,
            claim_payload_eval_reflex_scale(baseline_payload),
        )

    return {}, None
