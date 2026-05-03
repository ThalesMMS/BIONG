"""Standardized behavior-gate reporting helpers.

This module is intentionally lightweight and only depends on the artifacts
already produced by the main CLI:

- summary JSON (--summary)
- per-episode behavior metrics CSV (--behavior-csv)

The goal is to provide a stable, easy-to-compare, human-readable summary
across gates without requiring the broader offline-analysis pipeline.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateReportPaths:
    report_json_path: Path
    report_md_path: Path
    report: dict[str, object]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return data


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _as_float(value: str | None) -> float | str:
    if value is None:
        return ""
    value = value.strip()
    if not value:
        return ""
    try:
        result = float(value)
    except ValueError:
        return ""
    if not math.isfinite(result):
        return ""
    return result


def _group_counts(values: list[str]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_gate_report(
    *,
    gate_name: str,
    summary_path: str | Path,
    behavior_csv_path: str | Path,
) -> dict[str, object]:
    summary_path = Path(summary_path)
    behavior_csv_path = Path(behavior_csv_path)
    summary = _load_json(summary_path)
    rows = _load_csv_rows(behavior_csv_path)

    be = summary.get("behavior_evaluation")
    if isinstance(be, dict):
        suite = be.get("suite", {})
        scenario_results = suite if isinstance(suite, dict) else {}
    else:
        scenario_results = {}
    scenario_entry = scenario_results.get(gate_name)

    per_seed: dict[int, dict[str, object]] = {}
    failure_labels: list[str] = []
    representative_failures: list[dict[str, object]] = []

    for row in rows:
        scenario = row.get("scenario")
        if scenario != gate_name:
            continue
        seed = row.get("simulation_seed") or row.get("seed") or row.get("episode_seed")
        if seed is None:
            continue
        try:
            seed_int = int(str(seed).strip())
        except ValueError:
            continue
        passed_raw = row.get("passed")
        if passed_raw is None:
            passed_raw = row.get("success")
        if isinstance(passed_raw, bool):
            passed = passed_raw
        else:
            passed_token = str(passed_raw).strip()
            if passed_token in ("True", "true", "1"):
                passed = True
            elif passed_token in ("False", "false", "0"):
                passed = False
            else:
                continue
        per_seed.setdefault(seed_int, {"episodes": 0, "passed": 0})
        per_seed[seed_int]["episodes"] = int(per_seed[seed_int]["episodes"]) + 1
        per_seed[seed_int]["passed"] = int(per_seed[seed_int]["passed"]) + (1 if passed else 0)

        if not passed:
            failure_mode_label = (
                row.get("metric_failure_mode")
                or row.get("metric_failure_attribution")
                or "unknown"
            )
            failure_attribution_label = (
                row.get("metric_failure_attribution")
                or row.get("metric_failure_mode")
                or "unknown"
            )
            failure_labels.append(failure_mode_label)
            if len(representative_failures) < 10:
                representative_failures.append(
                    {
                        "seed": seed_int,
                        "episode": row.get("episode"),
                        "failure_attribution": failure_attribution_label,
                        "food_distance_delta": _as_float(row.get("metric_food_distance_delta")),
                        "shelter_distance_delta": _as_float(
                            row.get("metric_shelter_distance_delta")
                        ),
                        "night_shelter_occupancy_rate": _as_float(
                            row.get("metric_night_shelter_occupancy_rate")
                        ),
                    }
                )

    seeds_sorted = sorted(per_seed)
    per_seed_list = []
    for seed_int in seeds_sorted:
        episodes = int(per_seed[seed_int]["episodes"])
        passed_eps = int(per_seed[seed_int]["passed"])
        per_seed_list.append(
            {
                "seed": seed_int,
                "episodes": episodes,
                "passed": passed_eps,
                "pass_rate": (passed_eps / episodes) if episodes else None,
            }
        )

    overall_episodes = sum(item["episodes"] for item in per_seed_list)
    overall_passed = sum(item["passed"] for item in per_seed_list)
    report: dict[str, object] = {
        "gate": gate_name,
        "inputs": {
            "summary_path": str(summary_path),
            "behavior_csv_path": str(behavior_csv_path),
        },
        "overall": {
            "episodes": overall_episodes,
            "passed": overall_passed,
            "pass_rate": (overall_passed / overall_episodes) if overall_episodes else None,
        },
        "per_seed": per_seed_list,
        "failure_modes": _group_counts(failure_labels),
        "representative_failures": representative_failures,
        "scenario_entry": scenario_entry,
    }
    return report


def write_gate_report(*, output_dir: str | Path, report: dict[str, object]) -> GateReportPaths:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = output_dir / "gate_report.json"
    report_md_path = output_dir / "gate_report.md"
    new_report = dict(report)
    new_report["generated_files"] = {
        "json": str(report_json_path),
        "md": str(report_md_path),
    }

    report_json_path.write_text(json.dumps(new_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append(f"# Gate report: {new_report.get('gate')}\n")
    overall = new_report.get("overall", {})
    md_lines.append(
        f"Overall: passed {overall.get('passed')} / {overall.get('episodes')} "
        f"(pass_rate={overall.get('pass_rate')})\n"
    )

    md_lines.append("## Per-seed pass rates\n\n")
    md_lines.append("| seed | episodes | passed | pass_rate |\n")
    md_lines.append("| ---: | ---: | ---: | ---: |\n")
    for item in new_report.get("per_seed", []):
        md_lines.append(
            f"| {item.get('seed')} | {item.get('episodes')} | {item.get('passed')} | {item.get('pass_rate')} |\n"
        )

    md_lines.append("\n## Failure modes\n\n")
    md_lines.append("| label | count |\n")
    md_lines.append("| --- | ---: |\n")
    for item in new_report.get("failure_modes", []):
        md_lines.append(f"| {item.get('label')} | {item.get('count')} |\n")

    md_lines.append("\n## Representative failing episodes\n\n")
    md_lines.append(
        "| seed | episode | failure_attribution | night_shelter_occupancy_rate | food_distance_delta | shelter_distance_delta |\n"
    )
    md_lines.append("| ---: | ---: | --- | ---: | ---: | ---: |\n")
    for item in new_report.get("representative_failures", []):
        md_lines.append(
            "| {seed} | {episode} | {failure_attribution} | {night_shelter_occupancy_rate} | {food_distance_delta} | {shelter_distance_delta} |\n".format(
                seed=item.get("seed", ""),
                episode=item.get("episode", ""),
                failure_attribution=item.get("failure_attribution", ""),
                night_shelter_occupancy_rate=item.get("night_shelter_occupancy_rate", ""),
                food_distance_delta=item.get("food_distance_delta", ""),
                shelter_distance_delta=item.get("shelter_distance_delta", ""),
            )
        )

    report_md_path.write_text("".join(md_lines), encoding="utf-8")

    return GateReportPaths(report_json_path=report_json_path, report_md_path=report_md_path, report=new_report)
