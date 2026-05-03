"""Repeatable before/after artifact generation for remediated behavior gates.

This script is intentionally small and only shells out to the existing main CLI.

It runs each requested gate twice:
- before: current code + provided label (typically "before")
- after:  current code + provided label (typically "after")

Outputs are written to:
    <output_dir>/<gate>/<label>/{summary.json, behavior.csv}
    <output_dir>/<gate>/<label>/gate_report/{gate_report.json, gate_report.md}

Example:
    python3 -m spider_cortex_sim.scenarios.generate_gate_artifacts \
        --gates night_rest two_shelter_tradeoff \
        --seeds 0 1 2 3 4 \
        --episodes 10 \
        --output-dir artifacts/gates \
        --before-label before \
        --after-label after

Note: This does *not* attempt to checkout old commits for a true baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        raise SystemExit(proc.returncode)


def _safe_path_segment(value: str, *, field: str) -> str:
    segment = str(value)
    if not segment or segment in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty path segment.")
    if "/" in segment or "\\" in segment or Path(segment).name != segment:
        raise ValueError(f"{field} must not contain path separators: {value!r}.")
    return segment


def _run_gate(*, gate: str, label: str, seeds: list[int], episodes: int, output_dir: Path) -> None:
    gate_segment = _safe_path_segment(gate, field="gate")
    label_segment = _safe_path_segment(label, field="label")
    run_dir = output_dir / gate_segment / label_segment
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    behavior_csv_path = run_dir / "behavior.csv"

    cmd = [
        sys.executable,
        "-m",
        "spider_cortex_sim",
        "--behavior-scenario",
        gate,
        "--behavior-seeds",
        *[str(seed) for seed in seeds],
        "--scenario-episodes",
        str(episodes),
        "--summary",
        str(summary_path),
        "--behavior-csv",
        str(behavior_csv_path),
    ]
    _run(cmd)

    report_dir = run_dir / "gate_report"
    report_cmd = [
        sys.executable,
        "-m",
        "spider_cortex_sim.scenarios.gate_report_cli",
        "--gate",
        gate,
        "--summary",
        str(summary_path),
        "--behavior-csv",
        str(behavior_csv_path),
        "--output-dir",
        str(report_dir),
    ]
    _run(report_cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate before/after behavior-gate artifacts.")
    parser.add_argument(
        "--gates",
        nargs="+",
        default=["night_rest", "two_shelter_tradeoff"],
        help="Behavior scenario (gate) names to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Seeds to run.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "behavior_gates",
        help="Root output directory for artifacts.",
    )
    parser.add_argument("--before-label", default="before", help="Directory label for the baseline run.")
    parser.add_argument("--after-label", default="after", help="Directory label for the remediated run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "gates": args.gates,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "output_dir": str(output_dir),
        "runs": [],
    }

    for gate in args.gates:
        for label in (args.before_label, args.after_label):
            _run_gate(gate=gate, label=label, seeds=args.seeds, episodes=args.episodes, output_dir=output_dir)
            manifest["runs"].append({"gate": gate, "label": label})

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
