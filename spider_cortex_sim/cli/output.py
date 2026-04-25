from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..benchmark_package import (
    BenchmarkManifest,
    assemble_benchmark_package,
    summarize_benchmark_manifest,
)

_benchmark_package_manifest_summary = summarize_benchmark_manifest


def _build_and_record_benchmark_package(
    *,
    output_dir: Path,
    summary: dict[str, object],
    behavior_csv: Path | None,
    behavior_rows: Sequence[Mapping[str, object]],
    command_metadata: Mapping[str, object],
    trace: Sequence[Mapping[str, object]] = (),
) -> BenchmarkManifest:
    """
    Assemble a benchmark package and record its summarized manifest into the provided summary dictionary.
    
    Parameters:
        output_dir (Path): Directory where the assembled package will be written.
        summary (dict[str, object]): Mutable mapping that will be updated with a "benchmark_package" key containing the manifest summary.
        behavior_csv (Path | None): Optional path to a behavior CSV file to include in the package.
        behavior_rows (Sequence[Mapping[str, object]]): Sequence of behavior row mappings to include in the package.
        command_metadata (Mapping[str, object]): Metadata describing the command/environment to include in the package.
        trace (Sequence[Mapping[str, object]]): Optional sequence of trace mappings to include in the package.
    
    Returns:
        BenchmarkManifest: The assembled benchmark manifest.
    """
    manifest = assemble_benchmark_package(
        output_dir,
        summary,
        behavior_csv,
        behavior_rows=behavior_rows,
        trace=trace,
        command_metadata=command_metadata,
    )
    summary["benchmark_package"] = _benchmark_package_manifest_summary(
        manifest,
        output_dir=output_dir,
    )
    return manifest
