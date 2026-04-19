"""Claim-test evaluation workflows and helpers.

``condense_claim_test_summary`` is the public home for the CLI helper formerly
named ``_short_claim_test_suite_summary``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Sequence

from ..ablations import BrainAblationConfig, compare_predator_type_ablation_performance
from ..benchmark_types import SeedLevelResult
from ..budget_profiles import BudgetProfile
from ..checkpointing import CheckpointPenaltyMode
from ..claim_tests import (
    ClaimTestSpec,
    assess_scaffold_support,
    primary_claim_test_names,
    resolve_claim_tests,
)
from ..comparison import (
    aggregate_with_uncertainty,
    austere_comparison_from_payloads,
    compare_ablation_suite,
    compare_behavior_suite,
    compare_learning_evidence,
    compare_noise_robustness,
    fallback_seed_values,
    metric_seed_values_from_payload,
    paired_seed_delta_rows,
    paired_seed_effect_size_rows,
    representation_specialization_from_payload,
    safe_float,
    values_only,
    visual_minus_olfactory_seed_rows,
)
from ..memory import memory_leakage_audit
from ..noise import NoiseConfig, RobustnessMatrixSpec
from ..operational_profiles import OperationalProfile
from ..perception import observation_leakage_audit
from ..reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD, SCENARIO_AUSTERE_REQUIREMENTS
from ..statistics import cohens_d

from .scaffold import claim_leakage_audit_summary, extract_claim_config_for_scaffold_assessment
from .subsets import claim_austere_survival_gate, claim_comparison_statistics, claim_noise_subset_scores, claim_noise_subset_seed_values, claim_specialization_engagement, claim_subset_scenario_success_rate, claim_subset_seed_values
from .thresholds import claim_count_threshold, claim_registry_entry, claim_skip_result, claim_test_source, claim_threshold_from_operator, claim_threshold_from_phrase

def finalize_claim_result(
    spec: ClaimTestSpec,
    result: Dict[str, object],
    payloads: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Finalize a claim result by applying austere survival gating and scaffold-support assessment.
    
    Attaches the computed austere survival gate to the result, forces failure when the claim requires an austere gate that did not pass, performs scaffold-support assessment (extracting a config summary and eval reflex scale from provided payloads), populates scaffold-related metadata (`scaffold_support_level`, `scaffold_findings`, `benchmark_of_record_eligible`, `claim_severity`), and appends scaffold findings to the result notes. If the claim is a primary claim that only passes under `scaffolded_runtime` support, the result is downgraded to failed and an explanatory reason is appended.
    
    Parameters:
        spec (ClaimTestSpec): The claim test specification driving gating and scaffold extraction.
        result (Dict[str, object]): The intermediate claim result to finalize; a copy is returned and original is not mutated.
        payloads (Dict[str, Dict[str, object]]): Available primitive payloads used to compute the austere gate and scaffold assessment inputs.
    
    Returns:
        Dict[str, object]: A finalized claim result dictionary with added keys:
            - `austere_survival_required` (bool)
            - `austere_survival_gate` (dict)
            - `scaffold_support_level` (str)
            - `scaffold_findings` (list[str])
            - `benchmark_of_record_eligible` (bool)
            - `claim_severity` (str)
          The returned result may have `status`, `passed`, `reason`, and `notes` modified when gates or scaffold rules change the outcome.
    """
    finalized = dict(result)
    gate = claim_austere_survival_gate(spec, payloads)
    finalized["austere_survival_required"] = bool(
        spec.austere_survival_required
    )
    finalized["austere_survival_gate"] = gate
    if (
        spec.austere_survival_required
        and str(finalized.get("status")) != "skipped"
        and not bool(gate.get("passed", False))
    ):
        finalized["status"] = "failed"
        finalized["passed"] = False
        finalized["reason"] = (
            "Austere survival gate failed: "
            f"{gate.get('reason') or 'required gate did not pass'}."
        )
        notes = list(finalized.get("notes", []))
        notes.append(str(finalized["reason"]))
        finalized["notes"] = notes
    config_summary, eval_reflex_scale = (
        extract_claim_config_for_scaffold_assessment(spec, payloads)
    )
    severity_by_support_level = {
        "minimal_manual": "full",
        "standard_constrained": "qualified",
        "scaffolded_runtime": "non_benchmark",
        "missing_inputs": "non_benchmark",
    }
    if not config_summary or eval_reflex_scale is None:
        support_level = "missing_inputs"
        scaffold_findings: list[str] = []
        if not config_summary:
            scaffold_findings.append("scaffold_config_missing")
        if eval_reflex_scale is None:
            scaffold_findings.append("scaffold_eval_reflex_scale_missing")
        if not scaffold_findings:
            scaffold_findings.append("scaffold_assessment_inputs_missing")
        benchmark_of_record_eligible = False
    else:
        scaffold_assessment = assess_scaffold_support(
            config_summary,
            eval_reflex_scale,
        )
        support_level = scaffold_assessment.support_level.value
        scaffold_findings = list(scaffold_assessment.findings)
        benchmark_of_record_eligible = bool(
            scaffold_assessment.benchmark_of_record_eligible
        )
    finalized["scaffold_support_level"] = support_level
    finalized["scaffold_findings"] = list(scaffold_findings)
    finalized["benchmark_of_record_eligible"] = benchmark_of_record_eligible
    finalized["claim_severity"] = severity_by_support_level[support_level]
    notes = list(finalized.get("notes", []))
    notes.extend(
        f"Scaffold finding: {finding}."
        for finding in scaffold_findings
    )
    finalized["notes"] = notes
    if (
        spec.primary
        and str(finalized.get("status")) == "passed"
        and support_level == "scaffolded_runtime"
    ):
        findings_clause = ""
        if scaffold_findings:
            findings_clause = (
                f" Findings: {', '.join(scaffold_findings)}."
            )
        finalized["status"] = "failed"
        finalized["passed"] = False
        finalized["reason"] = (
            "Primary claim passed only under scaffolded runtime conditions "
            "and is not eligible for benchmark-of-record evidence."
            f"{findings_clause}"
        )
        notes.append(str(finalized["reason"]))
        finalized["notes"] = notes
    return finalized

def evaluate_claim_test(
    spec: ClaimTestSpec,
    payloads: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Evaluate a single claim test specification using provided primitive payloads and produce a standardized result record.
    
    This function selects the primitive payload type required by `spec`, validates and extracts the needed reference and comparison data from `payloads`, computes reference/comparison metrics, per-seed uncertainties, deltas, effect-size statistics (including Cohen's d and magnitude), and determines pass/fail/skipped status according to the claim's `success_criterion` and any claim-specific rules. The returned result is finalized with austere-survival and scaffold-support assessments.
    
    Parameters:
        spec (ClaimTestSpec): The canonical claim test specification to evaluate; provides protocol, name, primary metric, scenarios, reference and comparison condition names, and success criterion.
        payloads (Dict[str, Dict[str, object]]): Mapping of primitive source names ("learning_evidence", "ablation", "noise_robustness", etc.) to their payload registries used to compute metrics and seed-level values.
    
    Returns:
        Dict[str, object]: A standardized claim result record containing at least:
          - `status` ("passed", "failed", or "skipped") and `passed` (bool),
          - metric values (`reference_value`, `comparison_values`, `delta`, `effect_size`),
          - uncertainty summaries (`reference_uncertainty`, `comparison_uncertainty`, `delta_uncertainty`, `effect_size_uncertainty`),
          - effect statistics (`cohens_d`, `effect_magnitude`),
          - `primary_metric`, `scenarios_evaluated`, `notes`, and scaffold/austere-related fields added by finalization.
    """
    def finish(result: Dict[str, object]) -> Dict[str, object]:
        """
        Finalize a single claim result by applying austere survival gating, scaffold-support assessment, and primary-claim eligibility rules.
        
        Parameters:
            result (Dict[str, object]): Partial claim result produced by evaluation (may be mutated or extended).
        
        Returns:
            Dict[str, object]: The finalized claim result record, with `austere_survival_gate`, `scaffold_support_level`, `scaffold_findings`, `benchmark_of_record_eligible`, `claim_severity`, and potentially updated `status`, `passed`, `reason`, and `notes`.
        """
        return finalize_claim_result(spec, result, payloads)

    def skip(reason: str) -> Dict[str, object]:
        """
        Create a finalized skipped claim result for the current claim spec using the provided reason.
        
        Parameters:
        	reason (str): Explanation for skipping the claim.
        
        Returns:
        	Dict[str, object]: A claim result record with `status` set to "skipped", `passed` set to False, `reason` populated with the given text, and standardized empty or placeholder fields for comparison, delta, effect-size, and uncertainty.
        """
        return finish(claim_skip_result(spec, reason))

    source = claim_test_source(spec)
    if source is None:
        return skip(
            f"Could not determine a primitive payload source from protocol {spec.protocol!r}.",
        )

    if source == "learning_evidence":
        payload = payloads.get("learning_evidence")
        reference_payload, reason = claim_registry_entry(
            payload,
            registry_key="conditions",
            entry_name=spec.reference_condition,
        )
        if reference_payload is None:
            return skip(str(reason))
        reference_value, reason = claim_subset_scenario_success_rate(
            reference_payload,
            scenarios=spec.scenarios,
        )
        if reference_value is None:
            return skip(str(reason))
        reference_seed_values = claim_subset_seed_values(
            reference_payload,
            scenarios=spec.scenarios,
            fallback_value=reference_value,
        )
        reference_uncertainty = aggregate_with_uncertainty(
            [
                SeedLevelResult(
                    metric_name=spec.primary_metric,
                    seed=seed,
                    value=value,
                    condition=spec.reference_condition,
                )
                for seed, value in reference_seed_values
            ]
        )
        comparison_values: Dict[str, float] = {}
        comparison_uncertainty: Dict[str, object] = {}
        deltas: Dict[str, float] = {}
        delta_uncertainty: Dict[str, object] = {}
        effect_size_uncertainty: Dict[str, object] = {}
        cohens_d_values: Dict[str, float] = {}
        effect_magnitudes: Dict[str, str] = {}
        for comparison_name in spec.comparison_conditions:
            comparison_payload, reason = claim_registry_entry(
                payload,
                registry_key="conditions",
                entry_name=comparison_name,
            )
            if comparison_payload is None:
                return skip(str(reason))
            comparison_value, reason = claim_subset_scenario_success_rate(
                comparison_payload,
                scenarios=spec.scenarios,
            )
            if comparison_value is None:
                return skip(str(reason))
            comparison_values[comparison_name] = comparison_value
            deltas[comparison_name] = round(comparison_value - reference_value, 6)
            comparison_seed_values = claim_subset_seed_values(
                comparison_payload,
                scenarios=spec.scenarios,
                fallback_value=comparison_value,
            )
            comparison_uncertainty[comparison_name] = (
                aggregate_with_uncertainty(
                    [
                        SeedLevelResult(
                            metric_name=spec.primary_metric,
                            seed=seed,
                            value=value,
                            condition=comparison_name,
                        )
                        for seed, value in comparison_seed_values
                    ]
                )
            )
            stats = claim_comparison_statistics(
                reference_seed_values=reference_seed_values,
                comparison_seed_values=comparison_seed_values,
                comparison_name=comparison_name,
                fallback_delta=deltas[comparison_name],
            )
            delta_uncertainty[comparison_name] = stats["delta_uncertainty"]
            effect_size_uncertainty[comparison_name] = stats[
                "effect_size_uncertainty"
            ]
            cohens_d_values[comparison_name] = float(stats["cohens_d"])
            effect_magnitudes[comparison_name] = str(stats["effect_magnitude"])

        if spec.name == "learning_without_privileged_signals":
            delta_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "by at least",
            )
            if delta_threshold is None:
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            leakage_audit = claim_leakage_audit_summary()
            trained_value = comparison_values.get("trained_without_reflex_support")
            trained_delta = deltas.get("trained_without_reflex_support")
            if trained_value is None or trained_delta is None:
                return skip(
                    "Missing trained_without_reflex_support comparison data.",
                )
            passed = bool(
                trained_delta >= delta_threshold
                and int(leakage_audit["finding_count"]) == 0
            )
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": reference_value,
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": dict(deltas),
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": [
                    spec.success_criterion,
                    f"Leakage audit unresolved findings: {leakage_audit['finding_count']}.",
                ],
            })

        if spec.name == "escape_without_reflex_support":
            minimum_success = claim_threshold_from_operator(
                spec.success_criterion,
                ">=",
            )
            delta_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "by at least",
            )
            if minimum_success is None or delta_threshold is None:
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            trained_value = comparison_values.get("trained_without_reflex_support")
            trained_delta = deltas.get("trained_without_reflex_support")
            if trained_value is None or trained_delta is None:
                return skip(
                    "Missing trained_without_reflex_support comparison data.",
                )
            passed = bool(
                trained_value >= minimum_success
                and trained_delta >= delta_threshold
            )
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": reference_value,
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": dict(deltas),
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": [spec.success_criterion],
            })

        return skip(
            f"Unsupported learning-evidence claim test {spec.name!r}.",
        )

    if source == "ablation":
        payload = payloads.get("ablation")
        reference_payload, reason = claim_registry_entry(
            payload,
            registry_key="variants",
            entry_name=spec.reference_condition,
        )
        if reference_payload is None:
            return skip(str(reason))

        if spec.name == "memory_improves_shelter_return":
            reference_value, reason = claim_subset_scenario_success_rate(
                reference_payload,
                scenarios=spec.scenarios,
            )
            if reference_value is None:
                return skip(str(reason))
            reference_seed_values = claim_subset_seed_values(
                reference_payload,
                scenarios=spec.scenarios,
                fallback_value=reference_value,
            )
            reference_uncertainty = aggregate_with_uncertainty(
                [
                    SeedLevelResult(
                        metric_name=spec.primary_metric,
                        seed=seed,
                        value=value,
                        condition=spec.reference_condition,
                    )
                    for seed, value in reference_seed_values
                ]
            )
            comparison_values: Dict[str, float] = {}
            comparison_uncertainty: Dict[str, object] = {}
            deltas: Dict[str, float] = {}
            delta_uncertainty: Dict[str, object] = {}
            effect_size_uncertainty: Dict[str, object] = {}
            cohens_d_values: Dict[str, float] = {}
            effect_magnitudes: Dict[str, str] = {}
            for comparison_name in spec.comparison_conditions:
                comparison_payload, reason = claim_registry_entry(
                    payload,
                    registry_key="variants",
                    entry_name=comparison_name,
                )
                if comparison_payload is None:
                    return skip(str(reason))
                comparison_value, reason = claim_subset_scenario_success_rate(
                    comparison_payload,
                    scenarios=spec.scenarios,
                )
                if comparison_value is None:
                    return skip(str(reason))
                comparison_values[comparison_name] = comparison_value
                deltas[comparison_name] = round(comparison_value - reference_value, 6)
                comparison_seed_values = claim_subset_seed_values(
                    comparison_payload,
                    scenarios=spec.scenarios,
                    fallback_value=comparison_value,
                )
                comparison_uncertainty[comparison_name] = (
                    aggregate_with_uncertainty(
                        [
                            SeedLevelResult(
                                metric_name=spec.primary_metric,
                                seed=seed,
                                value=value,
                                condition=comparison_name,
                            )
                            for seed, value in comparison_seed_values
                        ]
                    )
                )
                stats = claim_comparison_statistics(
                    reference_seed_values=reference_seed_values,
                    comparison_seed_values=comparison_seed_values,
                    comparison_name=comparison_name,
                    fallback_delta=deltas[comparison_name],
                )
                delta_uncertainty[comparison_name] = stats["delta_uncertainty"]
                effect_size_uncertainty[comparison_name] = stats[
                    "effect_size_uncertainty"
                ]
                cohens_d_values[comparison_name] = float(stats["cohens_d"])
                effect_magnitudes[comparison_name] = str(stats["effect_magnitude"])
            delta_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "by at least",
            )
            if delta_threshold is None:
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            comparison_name = spec.comparison_conditions[0]
            passed = bool(deltas.get(comparison_name, 0.0) >= delta_threshold)
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": reference_value,
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": dict(deltas),
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": [spec.success_criterion],
            })

        if spec.name == "specialization_emerges_with_multiple_predators":
            comparison_summary = compare_predator_type_ablation_performance(
                payload or {},
                variant_names=(spec.reference_condition, *spec.comparison_conditions),
            )
            comparisons = comparison_summary.get("comparisons", {})
            if not isinstance(comparisons, dict):
                return skip(
                    "Predator-type ablation comparison did not return comparison rows.",
                )
            reference_comparison = comparisons.get(spec.reference_condition)
            if not isinstance(reference_comparison, dict):
                return skip(
                    f"Predator-type comparison is missing reference variant {spec.reference_condition!r}.",
                )
            reference_value = reference_comparison.get(
                "visual_minus_olfactory_success_rate"
            )
            if reference_value is None:
                return skip(
                    f"Reference variant {spec.reference_condition!r} is missing "
                    "visual_minus_olfactory_success_rate.",
                )
            reference_value = round(float(reference_value), 6)
            reference_rows = visual_minus_olfactory_seed_rows(
                reference_payload,
                condition=spec.reference_condition,
                metric_name=spec.primary_metric,
                fallback_value=reference_value,
            )
            reference_seed_values = [
                (row.seed, row.value) for row in reference_rows
            ]
            reference_uncertainty = aggregate_with_uncertainty(
                reference_rows
            )
            comparison_values: Dict[str, float] = {}
            comparison_uncertainty: Dict[str, object] = {}
            deltas: Dict[str, float] = {}
            delta_uncertainty: Dict[str, object] = {}
            effect_sizes: Dict[str, float | None] = {}
            effect_size_uncertainty: Dict[str, object] = {}
            cohens_d_values: Dict[str, float] = {}
            effect_magnitudes: Dict[str, str] = {}
            for comparison_name in spec.comparison_conditions:
                comparison_payload = comparisons.get(comparison_name)
                if not isinstance(comparison_payload, dict):
                    return skip(
                        f"Predator-type comparison is missing {comparison_name!r}.",
                    )
                raw_value = comparison_payload.get("visual_minus_olfactory_success_rate")
                if raw_value is None:
                    return skip(
                        f"Comparison {comparison_name!r} is missing "
                        "visual_minus_olfactory_success_rate.",
                    )
                comparison_value = round(float(raw_value), 6)
                comparison_values[comparison_name] = comparison_value
                deltas[comparison_name] = round(
                    comparison_value - reference_value,
                    6,
                )
                raw_effect_size = comparison_payload.get(
                    "visual_minus_olfactory_success_rate_delta"
                )
                effect_sizes[comparison_name] = (
                    round(float(raw_effect_size), 6)
                    if raw_effect_size is not None
                    else None
                )
                variant_payload, variant_reason = claim_registry_entry(
                    payload,
                    registry_key="variants",
                    entry_name=comparison_name,
                )
                if variant_payload is None:
                    return skip(str(variant_reason))
                comparison_rows = visual_minus_olfactory_seed_rows(
                    variant_payload,
                    condition=comparison_name,
                    metric_name=spec.primary_metric,
                    fallback_value=comparison_value,
                )
                comparison_seed_values = [
                    (row.seed, row.value) for row in comparison_rows
                ]
                comparison_uncertainty[comparison_name] = (
                    aggregate_with_uncertainty(comparison_rows)
                )
                stats = claim_comparison_statistics(
                    reference_seed_values=reference_seed_values,
                    comparison_seed_values=comparison_seed_values,
                    comparison_name=comparison_name,
                    fallback_delta=effect_sizes[comparison_name],
                )
                delta_uncertainty[comparison_name] = stats["delta_uncertainty"]
                effect_size_uncertainty[comparison_name] = stats[
                    "effect_size_uncertainty"
                ]
                cohens_d_values[comparison_name] = float(stats["cohens_d"])
                effect_magnitudes[comparison_name] = str(stats["effect_magnitude"])
            engagement_threshold = claim_count_threshold(spec.success_criterion)
            negative_threshold = claim_threshold_from_operator(
                spec.success_criterion,
                "<=",
            )
            positive_threshold = claim_threshold_from_operator(
                spec.success_criterion,
                ">=",
            )
            representation_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "representation_specialization_score >=",
            )
            if (
                engagement_threshold is None
                or negative_threshold is None
                or positive_threshold is None
                or representation_threshold is None
            ):
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            engagement_count, engagement_pass_rates, reason = (
                claim_specialization_engagement(
                    payload,
                    variant_name=spec.reference_condition,
                    scenarios=spec.scenarios,
                )
            )
            if engagement_count is None or engagement_pass_rates is None:
                return skip(str(reason))
            visual_drop = comparison_values.get("drop_visual_cortex")
            sensory_drop = comparison_values.get("drop_sensory_cortex")
            if visual_drop is None or sensory_drop is None:
                return skip(
                    "Missing drop_visual_cortex or drop_sensory_cortex comparison data.",
                )
            representation_metrics = representation_specialization_from_payload(
                reference_payload,
            )
            if not bool(representation_metrics.get("available")):
                return skip(
                    f"Reference variant {spec.reference_condition!r} is missing "
                    "representation specialization evidence.",
                )
            representation_score = round(
                safe_float(
                    representation_metrics.get(
                        "representation_specialization_score"
                    )
                ),
                6,
            )
            behavior_tier_passed = bool(
                visual_drop <= negative_threshold
                and sensory_drop >= positive_threshold
            )
            engagement_tier_passed = bool(
                engagement_count >= engagement_threshold
            )
            representation_tier_passed = bool(
                representation_score >= representation_threshold
            )
            passed = bool(
                behavior_tier_passed
                and engagement_tier_passed
                and representation_tier_passed
            )
            notes = [spec.success_criterion]
            if behavior_tier_passed and engagement_tier_passed and not representation_tier_passed:
                notes.append(
                    "Behavioral specialization passed while representation separation stayed below the emerging threshold; interpret this as possible downstream gating or scenario asymmetry rather than clean proposer separation."
                )
            elif (
                representation_tier_passed
                and not (behavior_tier_passed and engagement_tier_passed)
            ):
                notes.append(
                    "Representation separation cleared the emerging threshold without full behavioral specialization; interpret this as early internal differentiation that has not yet stabilized into policy outcomes."
                )
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": {
                    "visual_minus_olfactory_success_rate": reference_value,
                    "type_specific_cortex_engagement_count": engagement_count,
                    "type_specific_cortex_engagement_pass_rates": engagement_pass_rates,
                    "proposer_divergence_by_module": dict(
                        representation_metrics.get(
                            "proposer_divergence_by_module",
                            {},
                        )
                    ),
                    "action_center_gate_differential": dict(
                        representation_metrics.get(
                            "action_center_gate_differential",
                            {},
                        )
                    ),
                    "action_center_contribution_differential": dict(
                        representation_metrics.get(
                            "action_center_contribution_differential",
                            {},
                        )
                    ),
                    "representation_specialization_score": representation_score,
                    "behavior_tier_passed": behavior_tier_passed,
                    "engagement_tier_passed": engagement_tier_passed,
                    "representation_tier_passed": representation_tier_passed,
                },
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": effect_sizes,
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": notes,
            })

        return skip(
            f"Unsupported ablation-backed claim test {spec.name!r}.",
        )

    if source == "noise_robustness":
        payload = payloads.get("noise_robustness")
        diagonal_score, off_diagonal_score, reason = claim_noise_subset_scores(
            payload,
            scenarios=spec.scenarios,
        )
        if diagonal_score is None or off_diagonal_score is None:
            return skip(str(reason))
        minimum_off_diagonal = claim_threshold_from_operator(
            spec.success_criterion,
            ">=",
        )
        maximum_gap = claim_threshold_from_operator(
            spec.success_criterion,
            "<=",
        )
        if minimum_off_diagonal is None or maximum_gap is None:
            return skip(
                f"Could not parse success criterion {spec.success_criterion!r}.",
            )
        effect_size = round(diagonal_score - off_diagonal_score, 6)
        passed = bool(
            off_diagonal_score >= minimum_off_diagonal
            and effect_size <= maximum_gap
        )
        diagonal_seed_values, off_diagonal_seed_values = (
            claim_noise_subset_seed_values(
                payload,
                scenarios=spec.scenarios,
                diagonal_fallback=diagonal_score,
                off_diagonal_fallback=off_diagonal_score,
            )
        )
        reference_uncertainty = aggregate_with_uncertainty(
            [
                SeedLevelResult(
                    metric_name="diagonal_score",
                    seed=seed,
                    value=value,
                    condition=spec.reference_condition,
                )
                for seed, value in diagonal_seed_values
            ]
        )
        comparison_uncertainty = {
            "off_diagonal": aggregate_with_uncertainty(
                [
                    SeedLevelResult(
                        metric_name="off_diagonal_score",
                        seed=seed,
                        value=value,
                        condition="off_diagonal",
                    )
                    for seed, value in off_diagonal_seed_values
                ]
            )
        }
        noise_delta_stats = claim_comparison_statistics(
            reference_seed_values=diagonal_seed_values,
            comparison_seed_values=off_diagonal_seed_values,
            comparison_name="off_diagonal",
            fallback_delta=round(off_diagonal_score - diagonal_score, 6),
        )
        noise_effect_stats = claim_comparison_statistics(
            reference_seed_values=off_diagonal_seed_values,
            comparison_seed_values=diagonal_seed_values,
            comparison_name="diagonal_minus_off_diagonal",
            fallback_delta=effect_size,
        )
        return finish({
            "status": "passed" if passed else "failed",
            "passed": passed,
            "reference_value": diagonal_score,
            "comparison_values": {"off_diagonal": off_diagonal_score},
            "delta": {"off_diagonal": round(off_diagonal_score - diagonal_score, 6)},
            "effect_size": effect_size,
            "reference_uncertainty": reference_uncertainty,
            "comparison_uncertainty": comparison_uncertainty,
            "delta_uncertainty": {
                "off_diagonal": noise_delta_stats["delta_uncertainty"],
            },
            "effect_size_uncertainty": {
                "off_diagonal": noise_effect_stats["effect_size_uncertainty"],
            },
            "cohens_d": {"off_diagonal": noise_effect_stats["cohens_d"]},
            "effect_magnitude": {
                "off_diagonal": noise_effect_stats["effect_magnitude"],
            },
            "primary_metric": spec.primary_metric,
            "scenarios_evaluated": list(spec.scenarios),
            "notes": [spec.success_criterion],
        })

    return skip(
        f"Unsupported claim-test source {source!r}.",
    )
