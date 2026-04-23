from __future__ import annotations

import argparse
import json
from pathlib import Path

from .combined import run_combined_offline_analysis
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
    parser.add_argument("--ablation-summary", type=Path, default=None)
    parser.add_argument("--ablation-behavior-csv", type=Path, default=None)
    parser.add_argument("--profile-summary", type=Path, default=None)
    parser.add_argument("--profile-behavior-csv", type=Path, default=None)
    parser.add_argument("--capacity-summary", type=Path, default=None)
    parser.add_argument("--capacity-behavior-csv", type=Path, default=None)
    parser.add_argument("--module-local-report", type=Path, default=None)
    parser.add_argument("--distillation-summary", type=Path, default=None)
    return parser


def main() -> None:
    """
    Parse command-line arguments, run the offline analysis pipeline with the provided inputs, write report artifacts to the output directory, and print the resulting report as JSON to stdout.
    
    If none of --summary, --trace, or --behavior-csv are supplied, the parser terminates with an error.
    """
    parser = build_parser()
    args = parser.parse_args()
    combined_args = (
        args.ablation_summary,
        args.ablation_behavior_csv,
        args.profile_summary,
        args.profile_behavior_csv,
        args.capacity_summary,
        args.capacity_behavior_csv,
        args.module_local_report,
        args.distillation_summary,
    )
    combined_mode = any(item is not None for item in combined_args)
    if combined_mode:
        if args.summary is not None or args.trace is not None or args.behavior_csv is not None:
            parser.error(
                "Use either the single-artifact flags (--summary/--trace/--behavior-csv) "
                "or the combined ladder flags, not both."
            )
        required_missing = [
            flag
            for flag, value in (
                ("--ablation-summary", args.ablation_summary),
                ("--ablation-behavior-csv", args.ablation_behavior_csv),
                ("--profile-summary", args.profile_summary),
                ("--profile-behavior-csv", args.profile_behavior_csv),
                ("--capacity-summary", args.capacity_summary),
                ("--capacity-behavior-csv", args.capacity_behavior_csv),
                ("--module-local-report", args.module_local_report),
            )
            if value is None
        ]
        if required_missing:
            parser.error(
                "Combined ladder mode requires "
                + ", ".join(required_missing)
                + "."
            )
        report = run_combined_offline_analysis(
            ablation_summary_path=args.ablation_summary,
            ablation_behavior_csv_path=args.ablation_behavior_csv,
            profile_summary_path=args.profile_summary,
            profile_behavior_csv_path=args.profile_behavior_csv,
            capacity_summary_path=args.capacity_summary,
            capacity_behavior_csv_path=args.capacity_behavior_csv,
            module_local_report_path=args.module_local_report,
            distillation_summary_path=args.distillation_summary,
            output_dir=args.output_dir,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
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
