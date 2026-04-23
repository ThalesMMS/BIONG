from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .ingestion import load_behavior_csv, load_summary
from .report import write_report
from .report_data import build_report_data
from .utils import _coerce_float, _mapping_or_empty


def _load_json_object(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            "combined offline-analysis JSON inputs must be top-level objects; "
            f"found {type(payload).__name__}."
        )
    return payload


def _module_local_sufficiency_summary(
    payload: Mapping[str, object],
) -> dict[str, object]:
    variant_modules = _mapping_or_empty(payload.get("variant_modules"))
    canonical_only_modules = _mapping_or_empty(payload.get("canonical_only_modules"))
    if not variant_modules and not canonical_only_modules:
        return {
            "available": False,
            "artifact_state": "expected_but_missing",
            "paper_gate_pass": False,
            "rows": [],
            "partial_variant_coverage": [],
            "blocked_reasons": ["No module-local sufficiency payload was available."],
        }

    rows: list[dict[str, object]] = []
    partial_variant_coverage = {
        str(name) for name in payload.get("partial_variant_coverage", [])
    }

    def add_rows(module_mapping: Mapping[str, object], mode: str) -> None:
        for module_name, runs in sorted(module_mapping.items()):
            run_list = runs if isinstance(runs, list) else []
            seeds: list[int] = []
            minimal_levels: list[int] = []
            canonical_v4_pass = True
            for run in run_list:
                if not isinstance(run, Mapping):
                    continue
                seeds.append(int(_coerce_float(run.get("seed"), 0.0)))
                report = _mapping_or_empty(run.get("report"))
                minimal_level = report.get("minimal_sufficient_level")
                if minimal_level is not None:
                    minimal_levels.append(int(_coerce_float(minimal_level, 0.0)))
                level_four = None
                for level_entry in report.get("levels", []):
                    if (
                        isinstance(level_entry, Mapping)
                        and int(_coerce_float(level_entry.get("level"), 0.0)) == 4
                    ):
                        level_four = level_entry
                        break
                if level_four is None or not bool(level_four.get("all_tasks_passed")):
                    canonical_v4_pass = False
            rows.append(
                {
                    "module": str(module_name),
                    "coverage_mode": mode,
                    "seed_count": len(seeds),
                    "seeds": ", ".join(str(seed) for seed in sorted(seeds)),
                    "minimal_sufficient_level": (
                        min(minimal_levels) if minimal_levels else None
                    ),
                    "canonical_v4_pass": canonical_v4_pass,
                    "partial_variant_coverage": str(module_name) in partial_variant_coverage,
                }
            )

    add_rows(variant_modules, "variant_ladder")
    add_rows(canonical_only_modules, "canonical_only")

    return {
        "available": bool(rows),
        "artifact_state": "available" if rows else "expected_but_missing",
        "paper_gate_pass": bool(payload.get("paper_gate_pass")),
        "rows": rows,
        "partial_variant_coverage": sorted(partial_variant_coverage),
        "blocked_reasons": [
            str(reason)
            for reason in payload.get("blocked_reasons", [])
            if reason
        ],
    }


def _distillation_analysis(summary: Mapping[str, object]) -> dict[str, object]:
    distillation = _mapping_or_empty(summary.get("distillation"))
    comparison = _mapping_or_empty(distillation.get("post_distillation_comparison"))
    assessment = _mapping_or_empty(comparison.get("assessment"))
    if not distillation:
        return {
            "available": False,
            "artifact_state": "not_in_this_artifact",
            "assessment": {},
            "rows": [],
            "limitations": ["No distillation summary was included in this artifact."],
        }

    rows: list[dict[str, object]] = []
    for label, key in (
        ("teacher", "teacher"),
        ("modular_rl_from_scratch", "modular_rl_from_scratch"),
        ("modular_distilled", "modular_distilled"),
        (
            "modular_distilled_plus_rl_finetuning",
            "modular_distilled_plus_rl_finetuning",
        ),
    ):
        payload = _mapping_or_empty(comparison.get(key))
        if not payload:
            continue
        rows.append(
            {
                "condition": label,
                "episodes": int(_coerce_float(payload.get("episodes"), 0.0)),
                "survival_rate": _coerce_float(payload.get("survival_rate"), 0.0),
                "mean_reward": _coerce_float(payload.get("mean_reward"), 0.0),
                "mean_food_distance_delta": _coerce_float(
                    payload.get("mean_food_distance_delta"),
                    0.0,
                ),
            }
        )

    return {
        "available": bool(assessment or rows),
        "artifact_state": "available" if (assessment or rows) else "expected_but_missing",
        "assessment": dict(assessment),
        "rows": rows,
        "limitations": []
        if (assessment or rows)
        else ["Distillation artifact was present but carried no post-distillation comparison."],
    }


def build_combined_ladder_report(
    *,
    ablation_report: Mapping[str, object],
    profile_report: Mapping[str, object],
    capacity_report: Mapping[str, object],
    module_local_payload: Mapping[str, object],
    distillation_summary: Mapping[str, object],
    ablation_summary_path: str | Path,
    ablation_behavior_csv_path: str | Path,
    profile_summary_path: str | Path,
    profile_behavior_csv_path: str | Path,
    capacity_summary_path: str | Path,
    capacity_behavior_csv_path: str | Path,
    module_local_report_path: str | Path,
    distillation_summary_path: str | Path | None,
) -> dict[str, object]:
    module_local_sufficiency = _module_local_sufficiency_summary(module_local_payload)
    distillation_analysis = _distillation_analysis(distillation_summary)

    limitations: list[str] = []
    seen_limitations: set[str] = set()
    for source in (
        ablation_report,
        profile_report,
        capacity_report,
        module_local_sufficiency,
        distillation_analysis,
    ):
        for item in source.get("limitations", []):
            text = str(item)
            if not text or text in seen_limitations:
                continue
            seen_limitations.add(text)
            limitations.append(text)
    for item in module_local_sufficiency.get("blocked_reasons", []):
        text = str(item)
        if not text or text in seen_limitations:
            continue
        seen_limitations.add(text)
        limitations.append(text)

    return {
        "inputs": {
            "summary_path": "combined_ladder_report",
            "trace_path": None,
            "behavior_csv_path": "combined_ladder_report",
            "summary_loaded": True,
            "trace_loaded": False,
            "behavior_csv_loaded": True,
        },
        "combined_inputs": {
            "ablation_summary_path": str(ablation_summary_path),
            "ablation_behavior_csv_path": str(ablation_behavior_csv_path),
            "profile_summary_path": str(profile_summary_path),
            "profile_behavior_csv_path": str(profile_behavior_csv_path),
            "capacity_summary_path": str(capacity_summary_path),
            "capacity_behavior_csv_path": str(capacity_behavior_csv_path),
            "module_local_report_path": str(module_local_report_path),
            "distillation_summary_path": (
                str(distillation_summary_path)
                if distillation_summary_path is not None
                else None
            ),
        },
        "training_eval": dict(_mapping_or_empty(ablation_report.get("training_eval"))),
        "scenario_success": dict(_mapping_or_empty(ablation_report.get("scenario_success"))),
        "primary_benchmark": dict(_mapping_or_empty(ablation_report.get("primary_benchmark"))),
        "shaping_program": dict(
            _mapping_or_empty(
                profile_report.get("shaping_program")
                or ablation_report.get("shaping_program")
            )
        ),
        "comparisons": dict(_mapping_or_empty(ablation_report.get("comparisons"))),
        "noise_robustness": dict(_mapping_or_empty(ablation_report.get("noise_robustness"))),
        "ablations": dict(_mapping_or_empty(ablation_report.get("ablations"))),
        "ladder_comparison": dict(_mapping_or_empty(ablation_report.get("ladder_comparison"))),
        "ladder_profile_comparison": dict(
            _mapping_or_empty(profile_report.get("ladder_profile_comparison"))
        ),
        "architectural_ladder": dict(
            _mapping_or_empty(ablation_report.get("architectural_ladder"))
        ),
        "predator_type_specialization": dict(
            _mapping_or_empty(ablation_report.get("predator_type_specialization"))
        ),
        "representation_specialization": dict(
            _mapping_or_empty(ablation_report.get("representation_specialization"))
        ),
        "parameter_counts": dict(
            _mapping_or_empty(
                ablation_report.get("parameter_counts")
                or profile_report.get("parameter_counts")
                or capacity_report.get("parameter_counts")
            )
        ),
        "model_capacity": dict(
            _mapping_or_empty(
                ablation_report.get("model_capacity")
                if _mapping_or_empty(ablation_report.get("model_capacity")).get(
                    "available"
                )
                else capacity_report.get("model_capacity")
            )
        ),
        "capacity_sweeps": dict(_mapping_or_empty(capacity_report.get("capacity_sweeps"))),
        "capacity_analysis": dict(
            _mapping_or_empty(
                ablation_report.get("capacity_analysis")
                if _mapping_or_empty(ablation_report.get("capacity_analysis")).get(
                    "available"
                )
                else capacity_report.get("capacity_analysis")
            )
        ),
        "aggregate_benchmark_tables": dict(
            _mapping_or_empty(ablation_report.get("aggregate_benchmark_tables"))
        ),
        "claim_test_tables": dict(_mapping_or_empty(ablation_report.get("claim_test_tables"))),
        "effect_size_tables": dict(_mapping_or_empty(ablation_report.get("effect_size_tables"))),
        "reward_profile_ladder_tables": dict(
            _mapping_or_empty(profile_report.get("reward_profile_ladder_tables"))
        ),
        "unified_ladder_report": dict(
            _mapping_or_empty(ablation_report.get("unified_ladder_report"))
        ),
        "reflex_frequency": dict(_mapping_or_empty(ablation_report.get("reflex_frequency"))),
        "credit_assignment": dict(_mapping_or_empty(ablation_report.get("credit_assignment"))),
        "credit_analysis": dict(_mapping_or_empty(ablation_report.get("credit_analysis"))),
        "reflex_dependence": dict(_mapping_or_empty(ablation_report.get("reflex_dependence"))),
        "scenario_checks": list(ablation_report.get("scenario_checks", [])),
        "reward_components": list(ablation_report.get("reward_components", [])),
        "diagnostics": list(ablation_report.get("diagnostics", [])),
        "module_local_sufficiency": module_local_sufficiency,
        "distillation_analysis": distillation_analysis,
        "limitations": limitations,
    }


def run_combined_offline_analysis(
    *,
    ablation_summary_path: str | Path,
    ablation_behavior_csv_path: str | Path,
    profile_summary_path: str | Path,
    profile_behavior_csv_path: str | Path,
    capacity_summary_path: str | Path,
    capacity_behavior_csv_path: str | Path,
    module_local_report_path: str | Path,
    output_dir: str | Path,
    distillation_summary_path: str | Path | None = None,
) -> dict[str, object]:
    ablation_report = build_report_data(
        summary=load_summary(ablation_summary_path),
        trace=[],
        behavior_rows=load_behavior_csv(ablation_behavior_csv_path),
        summary_path=ablation_summary_path,
        behavior_csv_path=ablation_behavior_csv_path,
    )
    profile_report = build_report_data(
        summary=load_summary(profile_summary_path),
        trace=[],
        behavior_rows=load_behavior_csv(profile_behavior_csv_path),
        summary_path=profile_summary_path,
        behavior_csv_path=profile_behavior_csv_path,
    )
    capacity_report = build_report_data(
        summary=load_summary(capacity_summary_path),
        trace=[],
        behavior_rows=load_behavior_csv(capacity_behavior_csv_path),
        summary_path=capacity_summary_path,
        behavior_csv_path=capacity_behavior_csv_path,
    )
    module_local_payload = _load_json_object(module_local_report_path)
    distillation_summary = load_summary(distillation_summary_path)
    report = build_combined_ladder_report(
        ablation_report=ablation_report,
        profile_report=profile_report,
        capacity_report=capacity_report,
        module_local_payload=module_local_payload,
        distillation_summary=distillation_summary,
        ablation_summary_path=ablation_summary_path,
        ablation_behavior_csv_path=ablation_behavior_csv_path,
        profile_summary_path=profile_summary_path,
        profile_behavior_csv_path=profile_behavior_csv_path,
        capacity_summary_path=capacity_summary_path,
        capacity_behavior_csv_path=capacity_behavior_csv_path,
        module_local_report_path=module_local_report_path,
        distillation_summary_path=distillation_summary_path,
    )
    report["generated_files"] = write_report(output_dir, report)
    return report
