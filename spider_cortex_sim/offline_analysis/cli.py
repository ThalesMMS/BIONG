from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingestion import load_behavior_csv, load_summary, load_trace
from .report import build_report_data, write_report

def run_offline_analysis(
    *,
    summary_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    behavior_csv_path: str | Path | None = None,
    output_dir: str | Path,
) -> dict[str, object]:
    """
    Run the offline analysis pipeline and produce a consolidated report.
    
    Loads optional input artifacts (summary, trace, behavior CSV), builds the report data, writes generated report files to the specified output directory, and returns the final report structure.
    
    Parameters:
        summary_path (str | Path | None): Path to a summary input file, or `None` to skip.
        trace_path (str | Path | None): Path to a trace input file, or `None` to skip.
        behavior_csv_path (str | Path | None): Path to a behavior CSV file, or `None` to skip.
        output_dir (str | Path): Directory where generated report files will be written.
    
    Returns:
        report (dict[str, object]): The consolidated report dictionary, augmented with a
        `generated_files` entry containing the paths of files written to `output_dir`.
    """
    summary = load_summary(summary_path)
    trace = load_trace(trace_path)
    behavior_rows = load_behavior_csv(behavior_csv_path)
    report = build_report_data(
        summary=summary,
        trace=trace,
        behavior_rows=behavior_rows,
        summary_path=summary_path,
        trace_path=trace_path,
        behavior_csv_path=behavior_csv_path,
    )
    report["generated_files"] = write_report(output_dir, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line ArgumentParser for the offline analysis tool.
    
    The parser exposes the following options:
    - --summary: path to an optional summary.json input.
    - --trace: path to an optional trace.jsonl input.
    - --behavior-csv: path to an optional behavior CSV exported by the main CLI.
    - --output-dir: required path where the report and artifacts will be written.
    
    Returns:
        argparse.ArgumentParser: A configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Generate reproducible offline analysis from summary, trace, and behavior_csv.",
    )
    parser.add_argument("--summary", type=Path, default=None, help="Path to summary.json.")
    parser.add_argument("--trace", type=Path, default=None, help="Path to trace.jsonl.")
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=None,
        help="Path to behavior CSV exported by the main CLI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the report and artifacts will be generated.",
    )
    return parser


def main() -> None:
    """
    Parse command-line arguments, run the offline analysis pipeline with the provided inputs, write report artifacts to the output directory, and print the resulting report as JSON to stdout.
    
    If none of --summary, --trace, or --behavior-csv are supplied, the parser terminates with an error.
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.summary is None and args.trace is None and args.behavior_csv is None:
        parser.error(
            "At least one of --summary, --trace, or --behavior-csv is required."
        )
    report = run_offline_analysis(
        summary_path=args.summary,
        trace_path=args.trace,
        behavior_csv_path=args.behavior_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
