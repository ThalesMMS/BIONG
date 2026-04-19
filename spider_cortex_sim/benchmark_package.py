"""Benchmark-of-record package assembly utilities.

The package format is intentionally file-based and reproducible: JSON tables
carry machine-readable benchmark data, CSVs preserve seed and behavior rows,
Markdown provides the human report, and the manifest records file hashes so a
published package can be audited without rerunning the simulation.

``summarize_benchmark_manifest`` and ``build_and_record_benchmark_package``
are the public homes for the benchmark-package helpers previously kept in the
CLI entrypoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .offline_analysis import (
    build_aggregate_benchmark_tables,
    build_claim_test_tables,
    build_effect_size_tables,
    build_report_data,
    load_behavior_csv,
    write_report,
)


BENCHMARK_PACKAGE_VERSION = "1.0"

# Canonical inventory expected in a benchmark-of-record output directory. Some
# files can be empty or absent from the manifest when the source run lacks the
# corresponding input, but these names define the package contract.
BENCHMARK_PACKAGE_CONTENTS: tuple[str, ...] = (
    "benchmark_manifest.json",
    "resolved_config.json",
    "seed_level_rows.csv",
    "aggregate_benchmark_tables.json",
    "claim_test_tables.json",
    "effect_size_tables.json",
    "report.md",
    "report.json",
    "plots/training_eval.svg",
    "plots/scenario_success.svg",
    "plots/robustness_matrix.svg",
    "plots/ablation_comparison.svg",
    "plots/reflex_frequency.svg",
    "supporting_csvs/behavior_rows.csv",
    "supporting_csvs/scenario_checks.csv",
    "supporting_csvs/reward_components.csv",
    "limitations.txt",
)


@dataclass(frozen=True)
class BenchmarkManifest:
    """JSON-serializable manifest for an assembled benchmark package."""

    package_version: str
    created_at: str
    command_metadata: Mapping[str, object]
    budget_profile: Mapping[str, object]
    checkpoint_selection: Mapping[str, object]
    contents: Sequence[Mapping[str, object]]
    seed_count: int
    confidence_level: float
    limitations: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.package_version, str) or not self.package_version:
            raise ValueError("package_version must be a non-empty string.")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at must be a non-empty string.")
        if not isinstance(self.command_metadata, Mapping):
            raise ValueError("command_metadata must be a mapping.")
        if not isinstance(self.budget_profile, Mapping):
            raise ValueError("budget_profile must be a mapping.")
        if not isinstance(self.checkpoint_selection, Mapping):
            raise ValueError("checkpoint_selection must be a mapping.")
        if isinstance(self.seed_count, bool) or not isinstance(self.seed_count, int):
            raise ValueError("seed_count must be an integer.")
        if self.seed_count < 0:
            raise ValueError("seed_count must be non-negative.")
        if isinstance(self.confidence_level, bool) or not isinstance(
            self.confidence_level,
            (int, float),
        ):
            raise ValueError("confidence_level must be numeric.")
        if not 0.0 <= float(self.confidence_level) <= 1.0:
            raise ValueError("confidence_level must be between 0.0 and 1.0.")
        if isinstance(self.contents, (str, bytes)) or not isinstance(
            self.contents,
            Sequence,
        ):
            raise ValueError("contents must be a sequence of mappings.")
        for item in self.contents:
            if not isinstance(item, Mapping):
                raise ValueError("contents entries must be mappings.")
        if isinstance(self.limitations, (str, bytes)) or not isinstance(
            self.limitations,
            Sequence,
        ):
            raise ValueError("limitations must be a sequence of strings.")
        for item in self.limitations:
            if not isinstance(item, str):
                raise ValueError("limitations entries must be strings.")

    def to_dict(self) -> dict[str, object]:
        return {
            "package_version": str(self.package_version),
            "created_at": str(self.created_at),
            "command_metadata": dict(self.command_metadata),
            "budget_profile": dict(self.budget_profile),
            "checkpoint_selection": dict(self.checkpoint_selection),
            "contents": [dict(item) for item in self.contents],
            "seed_count": int(self.seed_count),
            "confidence_level": float(self.confidence_level),
            "limitations": [str(item) for item in self.limitations],
        }


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(output_path: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in output_path.rglob("*") if item.is_file()):
        if path.name == "benchmark_manifest.json":
            continue
        relative_path = path.relative_to(output_path).as_posix()
        files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _budget_profile_from_summary(summary: Mapping[str, object]) -> dict[str, object]:
    config = _mapping_or_empty(summary.get("config"))
    budget = config.get("budget")
    if isinstance(budget, Mapping):
        return dict(budget)
    return {}


def _checkpoint_selection_from_summary(summary: Mapping[str, object]) -> dict[str, object]:
    checkpointing = summary.get("checkpointing")
    if isinstance(checkpointing, Mapping):
        return dict(checkpointing)
    return {}


def _seed_level_rows_from_summary(summary: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            seed_level = value.get("seed_level")
            if isinstance(seed_level, list):
                for item in seed_level:
                    if not isinstance(item, Mapping):
                        continue
                    if "seed" not in item or "value" not in item:
                        continue
                    rows.append(
                        {
                            "source_path": ".".join(path),
                            "metric_name": str(item.get("metric_name") or ""),
                            "seed": item.get("seed"),
                            "value": item.get("value"),
                            "condition": str(item.get("condition") or ""),
                            "scenario": str(item.get("scenario") or ""),
                        }
                    )
            for key, nested_value in value.items():
                visit(nested_value, (*path, str(key)))
        elif isinstance(value, list):
            for index, nested_value in enumerate(value):
                visit(nested_value, (*path, str(index)))

    visit(summary, ("summary",))
    return rows


def _seed_level_rows_from_behavior_rows(
    behavior_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, str, str], list[float]] = {}
    for row in behavior_rows:
        seed = row.get("simulation_seed")
        if seed in (None, ""):
            continue
        success = row.get("success")
        if isinstance(success, bool):
            value = 1.0 if success else 0.0
        else:
            value = 1.0 if str(success).strip().lower() in {"1", "true", "yes"} else 0.0
        condition = str(row.get("ablation_variant") or row.get("condition") or "")
        scenario = str(row.get("scenario") or "")
        grouped.setdefault((seed, condition, scenario), []).append(value)

    rows: list[dict[str, object]] = []
    for (seed, condition, scenario), values in sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), item[0][1], item[0][2]),
    ):
        rows.append(
            {
                "source_path": "behavior_csv",
                "metric_name": "episode_success_rate",
                "seed": seed,
                "value": sum(values) / len(values),
                "condition": condition,
                "scenario": scenario,
            }
        )
    return rows


def _write_seed_level_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    fieldnames = ("source_path", "metric_name", "seed", "value", "condition", "scenario")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_behavior_rows(
    path: Path,
    behavior_rows: Sequence[Mapping[str, object]],
    behavior_csv: str | Path | None,
    *,
    explicit_behavior_rows: Sequence[Mapping[str, object]] | None,
) -> None:
    if (
        explicit_behavior_rows is None
        and behavior_csv is not None
        and Path(behavior_csv).exists()
    ):
        source = Path(behavior_csv)
        if source.resolve() != path.resolve():
            shutil.copyfile(source, path)
        return
    fieldnames = sorted(
        {
            str(key)
            for row in behavior_rows
            for key in row.keys()
        }
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in behavior_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _copy_if_present(source: str | Path | None, target: Path) -> bool:
    if not source:
        return False
    source_path = Path(source)
    if not source_path.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)
    return True


def _copy_generated_report_artifacts(
    generated_files: Mapping[str, str],
    *,
    output_path: Path,
) -> None:
    for key, filename in (
        ("report_md", "report.md"),
        ("report_json", "report.json"),
    ):
        _copy_if_present(generated_files.get(key), output_path / filename)

    plot_targets = {
        "training_eval_svg": "training_eval.svg",
        "scenario_success_svg": "scenario_success.svg",
        "robustness_matrix_svg": "robustness_matrix.svg",
        "ablation_comparison_svg": "ablation_comparison.svg",
        "reflex_frequency_svg": "reflex_frequency.svg",
    }
    for key, filename in plot_targets.items():
        _copy_if_present(generated_files.get(key), output_path / "plots" / filename)

    csv_targets = {
        "scenario_checks_csv": "scenario_checks.csv",
        "reward_components_csv": "reward_components.csv",
    }
    for key, filename in csv_targets.items():
        _copy_if_present(
            generated_files.get(key),
            output_path / "supporting_csvs" / filename,
        )


def _seed_count(seed_level_rows: Sequence[Mapping[str, object]]) -> int:
    seeds = {
        str(row.get("seed"))
        for row in seed_level_rows
        if row.get("seed") not in (None, "")
    }
    return len(seeds)


def assemble_benchmark_package(
    output_dir: str | Path,
    summary: Mapping[str, object],
    behavior_csv: str | Path | None = None,
    *,
    behavior_rows: Sequence[Mapping[str, object]] | None = None,
    trace: Sequence[Mapping[str, object]] | None = None,
    command_metadata: Mapping[str, object] | None = None,
    confidence_level: float = 0.95,
    limitations: Sequence[str] | None = None,
) -> BenchmarkManifest:
    """
    Write a benchmark-of-record package to output_dir and return its manifest.

    The package is assembled from an already-completed simulation summary plus
    optional behavior rows or a behavior CSV. It does not run simulations. The
    generated manifest records the resolved budget, checkpoint-selection
    metadata, seed count, confidence level, limitations, and SHA-256 hashes for
    package files so downstream readers can reproduce exactly which artifacts
    were reported.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "supporting_csvs").mkdir(exist_ok=True)

    loaded_behavior_rows = (
        [dict(row) for row in behavior_rows]
        if behavior_rows is not None
        else load_behavior_csv(behavior_csv)
    )
    loaded_trace = [dict(item) for item in trace] if trace is not None else []

    report = build_report_data(
        summary=summary,
        trace=loaded_trace,
        behavior_rows=loaded_behavior_rows,
        behavior_csv_path=behavior_csv,
    )
    aggregate_tables = build_aggregate_benchmark_tables(summary)
    claim_tables = build_claim_test_tables(summary)
    effect_size_tables = build_effect_size_tables(summary)
    report["aggregate_benchmark_tables"] = aggregate_tables
    report["claim_test_tables"] = claim_tables
    report["effect_size_tables"] = effect_size_tables

    _json_dump(output_path / "resolved_config.json", summary.get("config", {}))
    _json_dump(output_path / "aggregate_benchmark_tables.json", aggregate_tables)
    _json_dump(output_path / "claim_test_tables.json", claim_tables)
    _json_dump(output_path / "effect_size_tables.json", effect_size_tables)

    seed_level_rows = _seed_level_rows_from_summary(summary)
    if not seed_level_rows:
        seed_level_rows = _seed_level_rows_from_behavior_rows(loaded_behavior_rows)
    _write_seed_level_rows(output_path / "seed_level_rows.csv", seed_level_rows)
    _write_behavior_rows(
        output_path / "supporting_csvs" / "behavior_rows.csv",
        loaded_behavior_rows,
        behavior_csv,
        explicit_behavior_rows=behavior_rows,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        generated = write_report(tmpdir, report)
        _copy_generated_report_artifacts(generated, output_path=output_path)

    package_limitations = [
        str(item)
        for item in (
            *(report.get("limitations", []) if isinstance(report.get("limitations"), list) else []),
            *(aggregate_tables.get("limitations", []) if isinstance(aggregate_tables.get("limitations"), list) else []),
            *(claim_tables.get("limitations", []) if isinstance(claim_tables.get("limitations"), list) else []),
            *(effect_size_tables.get("limitations", []) if isinstance(effect_size_tables.get("limitations"), list) else []),
            *(limitations or ()),
        )
        if item
    ]
    (output_path / "limitations.txt").write_text(
        "\n".join(package_limitations) + ("\n" if package_limitations else ""),
        encoding="utf-8",
    )

    manifest = BenchmarkManifest(
        package_version=BENCHMARK_PACKAGE_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        command_metadata=dict(command_metadata or {}),
        budget_profile=_budget_profile_from_summary(summary),
        checkpoint_selection=_checkpoint_selection_from_summary(summary),
        contents=_file_inventory(output_path),
        seed_count=_seed_count(seed_level_rows),
        confidence_level=float(confidence_level),
        limitations=package_limitations,
    )
    _json_dump(output_path / "benchmark_manifest.json", manifest.to_dict())
    return manifest


def summarize_benchmark_manifest(
    manifest: BenchmarkManifest,
    *,
    output_dir: Path,
) -> dict[str, object]:
    """Return compact manifest fields suitable for user-facing summaries."""
    manifest_paths = {
        str(item.get("path"))
        for item in manifest.contents
        if isinstance(item, Mapping)
    }
    # _file_inventory() omits benchmark_manifest.json while the manifest
    # variable represents a package that has already written that file.
    file_count = len(manifest.contents) + (
        0 if "benchmark_manifest.json" in manifest_paths else 1
    )
    return {
        "output_dir": str(output_dir),
        "package_version": manifest.package_version,
        "created_at": manifest.created_at,
        "seed_count": manifest.seed_count,
        "confidence_level": manifest.confidence_level,
        "file_count": file_count,
        "limitations": list(manifest.limitations),
    }


def build_and_record_benchmark_package(
    *,
    output_dir: Path,
    summary: dict[str, object],
    behavior_csv: Path | None,
    behavior_rows: Sequence[Mapping[str, object]],
    command_metadata: Mapping[str, object],
    trace: Sequence[Mapping[str, object]] = (),
) -> BenchmarkManifest:
    """Assemble a benchmark package and record its compact summary payload."""
    manifest = assemble_benchmark_package(
        output_dir,
        summary,
        behavior_csv,
        behavior_rows=behavior_rows,
        trace=trace,
        command_metadata=command_metadata,
    )
    summary["benchmark_package"] = summarize_benchmark_manifest(
        manifest,
        output_dir=output_dir,
    )
    return manifest


__all__ = [
    "BENCHMARK_PACKAGE_CONTENTS",
    "BenchmarkManifest",
    "assemble_benchmark_package",
    "build_and_record_benchmark_package",
    "summarize_benchmark_manifest",
]
