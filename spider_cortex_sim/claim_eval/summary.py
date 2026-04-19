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

from .evaluate import evaluate_claim_test
from .scaffold import claim_leakage_audit_summary
from .thresholds import claim_test_source

def build_claim_test_summary(
    claim_results: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Summarizes claim-test results into pass/fail counts and scaffold-tier metadata.

    Parameters:
        claim_results (Dict[str, Dict[str, object]]): Mapping from claim-test name to its result record.
            Each result record is expected to include a `status` field (commonly `"passed"`, `"failed"`, or `"skipped"`)
            and may include scaffold metadata used for benchmark-of-record summaries.

    Returns:
        Dict[str, object]: Summary dictionary with the following keys:
            - `claim_count`: total number of claim tests processed.
            - `claims_passed`: number of tests whose `status` equals `"passed"`.
            - `claims_failed`: number of tests whose `status` equals `"failed"`.
            - `claims_skipped`: number of tests whose `status` equals `"skipped"`.
            - `claims_at_minimal_manual`: number of tests classified at the minimal-manual scaffold level.
            - `claims_at_standard_constrained`: number of tests classified at the standard-constrained level.
            - `claims_at_scaffolded_runtime`: number of tests classified at the scaffolded-runtime level.
            - `benchmark_of_record_claims`: number of passing tests marked benchmark-of-record eligible.
            - `all_primary_claims_passed`: `true` if every executed primary claim has a truthy `passed` value in
              `claim_results`, `false` otherwise.
            - `all_primary_claims_benchmark_eligible`: `true` if every passed primary claim is classified as
              minimal manual and benchmark-of-record eligible, `false` otherwise.
            - `primary_claims`: list of canonical primary claim names derived from the claim registry.
    """
    claims_passed = sum(
        1
        for result in claim_results.values()
        if str(result.get("status")) == "passed"
    )
    claims_failed = sum(
        1
        for result in claim_results.values()
        if str(result.get("status")) == "failed"
    )
    claims_skipped = sum(
        1
        for result in claim_results.values()
        if str(result.get("status")) == "skipped"
    )
    claims_at_minimal_manual = sum(
        1
        for result in claim_results.values()
        if str(result.get("scaffold_support_level")) == "minimal_manual"
    )
    claims_at_standard_constrained = sum(
        1
        for result in claim_results.values()
        if str(result.get("scaffold_support_level")) == "standard_constrained"
    )
    claims_at_scaffolded_runtime = sum(
        1
        for result in claim_results.values()
        if str(result.get("scaffold_support_level")) == "scaffolded_runtime"
    )
    benchmark_of_record_claims = sum(
        1
        for result in claim_results.values()
        if bool(result.get("passed", False))
        and bool(result.get("benchmark_of_record_eligible", False))
    )
    primary_claims = primary_claim_test_names()
    executed_primary_claims = [
        name for name in primary_claims if name in claim_results
    ]
    all_primary_claims_passed = bool(executed_primary_claims) and all(
        bool(claim_results[name].get("passed", False))
        for name in executed_primary_claims
    )
    all_primary_claims_benchmark_eligible = bool(executed_primary_claims) and all(
        bool(claim_results[name].get("passed", False))
        and str(claim_results[name].get("scaffold_support_level"))
        == "minimal_manual"
        and bool(claim_results[name].get("benchmark_of_record_eligible", False))
        for name in executed_primary_claims
    )
    return {
        "claim_count": len(claim_results),
        "claims_passed": claims_passed,
        "claims_failed": claims_failed,
        "claims_skipped": claims_skipped,
        "claims_at_minimal_manual": claims_at_minimal_manual,
        "claims_at_standard_constrained": claims_at_standard_constrained,
        "claims_at_scaffolded_runtime": claims_at_scaffolded_runtime,
        "benchmark_of_record_claims": benchmark_of_record_claims,
        "all_primary_claims_passed": bool(all_primary_claims_passed),
        "all_primary_claims_benchmark_eligible": bool(
            all_primary_claims_benchmark_eligible
        ),
        "primary_claims": list(primary_claims),
    }

def condense_claim_test_summary(
    claim_test_payload: object,
) -> dict[str, object]:
    """
    Produce a compact, CLI-friendly summary of a claim-test payload.
    
    Parameters:
        claim_test_payload (object): A mapping-like payload expected to contain
            "claims" (per-claim result rows) and "summary" (aggregate counts).
            Non-dict inputs are treated as empty.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - "claims": mapping of claim name -> {"passed": bool, "skipped": bool}
              where "passed" is True only if the claim is marked passed and not skipped.
            - "claims_passed": int count of passed claims (falls back to 0).
            - "claims_failed": int count of failed claims (falls back to 0).
            - "claims_skipped": int count of skipped claims (falls back to 0).
            - "all_primary_claims_passed": bool flag indicating whether all executed
              primary claims passed (defaults to False).
    """
    payload = claim_test_payload if isinstance(claim_test_payload, dict) else {}
    claims = payload.get("claims", {})
    summary = payload.get("summary", {})
    claim_rows = claims if isinstance(claims, dict) else {}
    summary_row = summary if isinstance(summary, dict) else {}
    condensed_claims: dict[str, dict[str, bool]] = {}
    for name, data in claim_rows.items():
        if not isinstance(data, dict):
            continue
        skipped = bool(data.get("skipped")) or str(data.get("status")) == "skipped"
        condensed_claims[str(name)] = {
            "passed": bool(data.get("passed", False)) and not skipped,
            "skipped": skipped,
        }

    def summary_count(key: str) -> int:
        """
        Parse an integer count stored in the enclosing `summary_row` under `key`.
        
        Parameters:
            key (str): Key name to look up in the surrounding `summary_row`.
        
        Returns:
            int: The integer value of `summary_row.get(key)` if it can be parsed, otherwise 0.
        """
        try:
            return int(summary_row.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "claims": condensed_claims,
        "claims_passed": summary_count("claims_passed"),
        "claims_failed": summary_count("claims_failed"),
        "claims_skipped": summary_count("claims_skipped"),
        "all_primary_claims_passed": bool(
            summary_row.get("all_primary_claims_passed", False)
        ),
    }

def run_claim_test_suite(
    *,
    claim_tests: Sequence[str] | None = None,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    brain_config: BrainAblationConfig | None = None,
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    long_budget_profile: str | BudgetProfile | None = "report",
    seeds: Sequence[int] | None = None,
    episodes_per_scenario: int | None = None,
    robustness_matrix: RobustnessMatrixSpec | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    ablation_payload: Dict[str, object] | None = None,
    learning_evidence_payload: Dict[str, object] | None = None,
    noise_robustness_payload: Dict[str, object] | None = None,
    austere_survival_payload: Dict[str, object] | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Run selected claim tests by synthesizing or reusing primitive comparison payloads.

    This method resolves requested claim-test specs, ensures required primitive comparison data (learning evidence, ablation, noise robustness, and austere survival gates for primary claims) are available by reusing provided payloads or invoking the corresponding comparison routines, evaluates each claim test to produce structured pass/skip/fail results, and returns a combined claims payload plus CSV-like row records suitable for export.

    Parameters:
        claim_tests: Optional sequence of claim-test identifiers or specs to run; if None all canonical claim tests are evaluated.
        width, height, food_count, day_length, night_length, max_steps:
            Environment layout and step-limit overrides for any generated comparison runs.
        episodes, evaluation_episodes, episodes_per_scenario:
            Training / evaluation budget overrides used for generated primitive comparisons.
        gamma, module_lr, motor_lr, module_dropout:
            Learning hyperparameter overrides applied when running training comparisons.
        reward_profile, map_template, brain_config, operational_profile, noise_profile:
            Configuration overrides used when generating comparison payloads.
        budget_profile, long_budget_profile:
            Budget profile names for base and long-form comparisons where applicable.
        seeds:
            Sequence of RNG seeds to use for generated comparisons; when omitted comparison helpers choose defaults.
        robustness_matrix:
            Optional robustness-matrix spec to use for noise-robustness comparisons.
        checkpoint_selection, checkpoint_metric, checkpoint_interval, checkpoint_dir:
            Checkpointing controls passed through to comparison runs that support candidate selection.
        ablation_payload, learning_evidence_payload, noise_robustness_payload, austere_survival_payload:
            Optional precomputed primitive payloads to reuse; when provided the method will not regenerate that source.

    Returns:
        tuple:
            - claims_payload (dict): A mapping with keys "claims" (per-claim result dicts), "summary" (aggregate pass/skip/fail counts and primary-claims gating), and "metadata" (requested tests, required sources, per-source metadata, seeds, noise profile mapping, and leakage-audit summary).
            - rows (list[dict]): CSV-ready row dictionaries, one per evaluated claim test, containing serialized reference/comparison values, deltas, effect sizes, evaluated scenarios, status, reason, and notes.
    """
    resolved_claim_tests = resolve_claim_tests(claim_tests)
    required_sources = {
        source
        for spec in resolved_claim_tests
        if (source := claim_test_source(spec)) is not None
    }
    austere_survival_required = any(
        spec.austere_survival_required for spec in resolved_claim_tests
    )
    austere_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if spec.austere_survival_required
            for scenario_name in spec.scenarios
            if SCENARIO_AUSTERE_REQUIREMENTS.get(
                scenario_name,
                {},
            ).get("requirement_level") == "gate"
        )
    )
    if austere_survival_required:
        required_sources.add("austere_survival")
    learning_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "learning_evidence"
            for scenario_name in spec.scenarios
        )
    )
    learning_conditions = list(
        dict.fromkeys(
            condition_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "learning_evidence"
            for condition_name in (spec.reference_condition, *spec.comparison_conditions)
        )
    )
    ablation_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "ablation"
            for scenario_name in spec.scenarios
        )
    )
    ablation_variants = list(
        dict.fromkeys(
            variant_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "ablation"
            for variant_name in (spec.reference_condition, *spec.comparison_conditions)
        )
    )
    noise_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "noise_robustness"
            for scenario_name in spec.scenarios
        )
    )

    payloads: Dict[str, Dict[str, object]] = {}
    source_reused = {
        "ablation": ablation_payload is not None,
        "austere_survival": austere_survival_payload is not None,
        "learning_evidence": learning_evidence_payload is not None,
        "noise_robustness": noise_robustness_payload is not None,
    }
    if "learning_evidence" in required_sources:
        if learning_evidence_payload is None:
            learning_evidence_payload, _ = compare_learning_evidence(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                brain_config=brain_config,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                budget_profile=budget_profile,
                long_budget_profile=long_budget_profile,
                seeds=seeds,
                names=learning_scenarios or None,
                condition_names=learning_conditions or None,
                episodes_per_scenario=episodes_per_scenario,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["learning_evidence"] = learning_evidence_payload
    if "ablation" in required_sources:
        if ablation_payload is None:
            ablation_payload, _ = compare_ablation_suite(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                budget_profile=budget_profile,
                seeds=seeds,
                names=ablation_scenarios or None,
                variant_names=ablation_variants or None,
                episodes_per_scenario=episodes_per_scenario,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["ablation"] = ablation_payload
    if "noise_robustness" in required_sources:
        if noise_robustness_payload is None:
            noise_robustness_payload, _ = compare_noise_robustness(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                operational_profile=operational_profile,
                budget_profile=budget_profile,
                seeds=seeds,
                names=noise_scenarios or None,
                episodes_per_scenario=episodes_per_scenario,
                robustness_matrix=robustness_matrix,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["noise_robustness"] = noise_robustness_payload
    if "austere_survival" in required_sources:
        if austere_survival_payload is None:
            comparison_profiles = tuple(
                dict.fromkeys((str(reward_profile), "austere"))
            )
            austere_survival_payload, _ = compare_behavior_suite(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                budget_profile=budget_profile,
                reward_profiles=comparison_profiles,
                map_templates=(map_template,),
                seeds=seeds,
                names=austere_scenarios or None,
                episodes_per_scenario=episodes_per_scenario,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["austere_survival"] = austere_survival_payload

    claim_results: Dict[str, Dict[str, object]] = {
        spec.name: evaluate_claim_test(spec, payloads)
        for spec in resolved_claim_tests
    }
    rows: List[Dict[str, object]] = []
    for spec in resolved_claim_tests:
        result = claim_results[spec.name]
        austere_gate = result.get("austere_survival_gate")
        if not isinstance(austere_gate, dict):
            austere_gate = {}
        row = {
            "claim_test": spec.name,
            "claim_test_status": result.get("status"),
            "claim_test_passed": bool(result.get("passed", False)),
            "claim_test_austere_survival_required": bool(
                result.get("austere_survival_required", False)
            ),
            "claim_test_austere_survival_passed": bool(
                austere_gate.get("passed", False)
            ),
            "claim_test_austere_survival_gate": json.dumps(
                austere_gate,
                sort_keys=True,
            ),
            "claim_test_scaffold_support_level": result.get(
                "scaffold_support_level"
            ),
            "claim_test_scaffold_findings": json.dumps(
                result.get("scaffold_findings", []),
                sort_keys=True,
            ),
            "claim_test_benchmark_of_record_eligible": bool(
                result.get("benchmark_of_record_eligible", False)
            ),
            "claim_test_severity": result.get("claim_severity"),
            "claim_test_primary_metric": result.get("primary_metric"),
            "claim_test_reference_condition": spec.reference_condition,
            "claim_test_comparison_conditions": json.dumps(
                list(spec.comparison_conditions),
                sort_keys=True,
            ),
            "claim_test_reference_value": json.dumps(
                result.get("reference_value"),
                sort_keys=True,
            ),
            "claim_test_comparison_values": json.dumps(
                result.get("comparison_values", {}),
                sort_keys=True,
            ),
            "claim_test_delta": json.dumps(
                result.get("delta", {}),
                sort_keys=True,
            ),
            "claim_test_effect_size": json.dumps(
                result.get("effect_size"),
                sort_keys=True,
            ),
            "claim_test_reference_uncertainty": json.dumps(
                result.get("reference_uncertainty"),
                sort_keys=True,
            ),
            "claim_test_comparison_uncertainty": json.dumps(
                result.get("comparison_uncertainty", {}),
                sort_keys=True,
            ),
            "claim_test_delta_uncertainty": json.dumps(
                result.get("delta_uncertainty", {}),
                sort_keys=True,
            ),
            "claim_test_effect_size_uncertainty": json.dumps(
                result.get("effect_size_uncertainty", {}),
                sort_keys=True,
            ),
            "claim_test_cohens_d": json.dumps(
                result.get("cohens_d", {}),
                sort_keys=True,
            ),
            "claim_test_effect_magnitude": json.dumps(
                result.get("effect_magnitude", {}),
                sort_keys=True,
            ),
            "claim_test_scenarios": json.dumps(
                result.get("scenarios_evaluated", []),
                sort_keys=True,
            ),
            "claim_test_reason": str(result.get("reason", "")),
            "claim_test_notes": json.dumps(result.get("notes", []), sort_keys=True),
        }
        rows.append(row)

    def _metadata_sequence_or_empty(value: object) -> list[object]:
        """
        Convert a list or tuple into a new list, or return an empty list for any other input.
        
        Parameters:
            value (object): The value to coerce; only `list` and `tuple` inputs are converted.
        
        Returns:
            list[object]: A new list containing the elements of `value` when it is a `list` or `tuple`, otherwise an empty list.
        """
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    def _metadata_payload_or_empty(value: object) -> dict[str, object]:
        """
        Return a dict copy of `value` if it implements the Mapping protocol, otherwise an empty dict.
        
        Parameters:
            value (object): Value to convert to a dict if it is a Mapping.
        
        Returns:
            dict[str, object]: A shallow dict copy of `value` when it's a Mapping; otherwise an empty dict.
        """
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    source_metadata: Dict[str, object] = {}
    metadata_seeds: set[int] = set()
    noise_profiles: Dict[str, object] = {}
    for source_name, raw_payload in payloads.items():
        payload = _metadata_payload_or_empty(raw_payload)
        source_metadata[source_name] = {
            "reused": bool(source_reused.get(source_name, False)),
            "budget_profile": payload.get("budget_profile"),
            "benchmark_strength": payload.get("benchmark_strength"),
            "noise_profile": payload.get("noise_profile"),
            "seeds": _metadata_sequence_or_empty(payload.get("seeds")),
            "scenario_names": _metadata_sequence_or_empty(
                payload.get("scenario_names")
            ),
            "episodes_per_scenario": payload.get("episodes_per_scenario"),
        }
        for seed in _metadata_sequence_or_empty(payload.get("seeds")):
            if isinstance(seed, int):
                metadata_seeds.add(int(seed))
        noise_profile = payload.get("noise_profile")
        if noise_profile is not None:
            noise_profiles[source_name] = noise_profile
    metadata = {
        "requested_claim_tests": [spec.name for spec in resolved_claim_tests],
        "required_sources": sorted(required_sources),
        "sources": source_metadata,
        "seeds": sorted(metadata_seeds),
        "noise_profiles": noise_profiles,
        "leakage_audit": claim_leakage_audit_summary(),
    }
    return {
        "claims": claim_results,
        "summary": build_claim_test_summary(claim_results),
        "metadata": metadata,
    }, rows
