"""CLI for standardized behavior-gate reports.

Example:
    python3 -m spider_cortex_sim.scenarios.gate_report_cli \
        --gate night_rest \
        --summary /tmp/night_rest_summary.json \
        --behavior-csv /tmp/night_rest.csv \
        --output-dir /tmp/night_rest_gate_report

This is intended to be a small, stable wrapper around the scenario gate
artifacts so we can compare before/after behavior runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gate_reports import build_gate_report, write_gate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a standardized gate report.")
    parser.add_argument("--gate", required=True, help="Behavior scenario name (gate name).")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary JSON.")
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        required=True,
        help="Path to per-episode behavior CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where gate_report.json/md will be written.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = build_gate_report(
        gate_name=args.gate,
        summary_path=args.summary,
        behavior_csv_path=args.behavior_csv,
    )
    paths = write_gate_report(output_dir=args.output_dir, report=report)
    print(json.dumps(paths.report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
