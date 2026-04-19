"""CLI parser construction and argument normalization helpers.

The former private ``_parse_module_reflex_scales`` helper is now the public
``parse_module_reflex_scales`` function in this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..ablations import canonical_ablation_variant_names
from ..budget_profiles import (
    CHECKPOINT_METRIC_NAMES,
    CHECKPOINT_SELECTION_NAMES,
    canonical_budget_profile_names,
)
from ..checkpointing import CHECKPOINT_PENALTY_MODE_NAMES
from ..claim_tests import claim_test_names
from ..curriculum import CURRICULUM_PROFILE_NAMES
from ..maps import MAP_TEMPLATE_NAMES
from ..noise import canonical_noise_profile_names
from ..operational_profiles import canonical_operational_profile_names
from ..scenarios import SCENARIO_NAMES
from ..training_regimes import canonical_training_regime_names
from ..world import REWARD_PROFILES


def collect_requested_scenarios(args: argparse.Namespace) -> list[str]:
    """Collect the ordered set of explicitly requested behavior scenarios."""
    requested_scenarios = list(args.scenario or [])
    for name in args.behavior_scenario or []:
        if name not in requested_scenarios:
            requested_scenarios.append(name)
    if args.scenario_suite or args.behavior_suite:
        for name in SCENARIO_NAMES:
            if name not in requested_scenarios:
                requested_scenarios.append(name)
    return requested_scenarios


def parse_module_reflex_scales(values: list[str] | None) -> dict[str, float]:
    """Parse per-module reflex scale specs of the form ``module=scale``."""
    parsed: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(
                "--module-reflex-scale must use the format module=scale."
            )
        module_name, scale_text = raw.split("=", 1)
        module_name = module_name.strip()
        if not module_name:
            raise ValueError(
                "--module-reflex-scale requires a module name before '='."
            )
        try:
            scale = float(scale_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid reflex scale for module {module_name!r}: {scale_text!r}."
            ) from exc
        parsed[module_name] = scale
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the command-line argument parser for the SpiderSimulation CLI.

    Configures options for training/evaluation sizing and budget, world and map configuration, reward/operational/noise profiles, learning hyperparameters and reflex/module overrides, deterministic scenarios and behavioral evaluation (including ablation, comparison, claim-test, and noise-robustness workflows), checkpointing controls, GUI/rendering, model load/save, and summary/trace/CSV persistence and debug flags.

    Returns:
        argparse.ArgumentParser: A configured ArgumentParser for the SpiderSimulation command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Online reward-based training for independent cortical modules in a simulated spider."
    )
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Number of greedy evaluation episodes.")
    parser.add_argument(
        "--curriculum-profile",
        choices=list(CURRICULUM_PROFILE_NAMES),
        default="none",
        help="Optional curriculum profile for organizing training into reproducible phases.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps per episode.")
    parser.add_argument("--width", type=int, default=12, help="World width.")
    parser.add_argument("--height", type=int, default=12, help="World height.")
    parser.add_argument("--food-count", type=int, default=4, help="Number of active food sources.")
    parser.add_argument("--day-length", type=int, default=18, help="Day length in ticks.")
    parser.add_argument("--night-length", type=int, default=12, help="Night length in ticks.")
    parser.add_argument("--seed", type=int, default=7, help="Primary seed.")
    parser.add_argument(
        "--reward-profile",
        choices=sorted(REWARD_PROFILES.keys()),
        default="classic",
        help="World reward profile.",
    )
    parser.add_argument(
        "--map-template",
        choices=list(MAP_TEMPLATE_NAMES),
        default="central_burrow",
        help="World map template.",
    )
    parser.add_argument(
        "--budget-profile",
        choices=list(canonical_budget_profile_names()),
        default=None,
        help="Reproducible budget profile for training and benchmarking.",
    )
    parser.add_argument(
        "--operational-profile",
        choices=list(canonical_operational_profile_names()),
        default="default_v1",
        help="Versioned operational profile for reflexes, perception, and reward heuristics.",
    )
    parser.add_argument(
        "--noise-profile",
        choices=list(canonical_noise_profile_names()),
        default="none",
        help="Explicit experimental noise or stochasticity profile.",
    )
    parser.add_argument("--module-lr", type=float, default=0.010, help="Learning rate for the specialized modules.")
    parser.add_argument("--motor-lr", type=float, default=0.012, help="Learning rate for the motor or critic cortex.")
    parser.add_argument("--module-dropout", type=float, default=0.05, help="Per-module dropout probability during training.")
    parser.add_argument("--reflex-scale", type=float, default=1.0, help="Global reflex scale for the modular architecture.")
    parser.add_argument(
        "--module-reflex-scale",
        action="append",
        default=None,
        help="Per-module override in module=scale format. Can be used multiple times.",
    )
    reflex_training_group = parser.add_mutually_exclusive_group()
    reflex_training_group.add_argument(
        "--reflex-anneal-final-scale",
        type=float,
        default=None,
        help="Final reflex scale for linear annealing during training.",
    )
    reflex_training_group.add_argument(
        "--training-regime",
        choices=list(canonical_training_regime_names()),
        default=None,
        help="Named training regime for reflex annealing and late fine-tuning.",
    )
    parser.add_argument(
        "--experiment-of-record",
        action="store_true",
        help=(
            "Run the canonical no-reflex benchmark workflow: late fine-tuning, "
            "best-checkpoint selection, and direct reflex-dependence penalties."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.96, help="Online TD discount factor.")
    parser.add_argument("--summary", type=Path, default=None, help="JSON file path for writing the summary.")
    parser.add_argument("--trace", type=Path, default=None, help="JSONL file path for writing an evaluation trace.")
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Render the final evaluation episode in ASCII after training.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the Pygame graphical interface to visualize the simulation.",
    )
    parser.add_argument(
        "--save-brain",
        type=Path,
        default=None,
        help="Directory where brain weights should be saved after training.",
    )
    parser.add_argument(
        "--load-brain",
        type=Path,
        default=None,
        help="Directory from which brain weights should be loaded before training.",
    )
    parser.add_argument(
        "--load-modules",
        nargs="+",
        default=None,
        help="List of specific modules to load (for example: visual_cortex hunger_center). "
             "Requires --load-brain.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list(SCENARIO_NAMES),
        default=None,
        help="Run a deterministic evaluation scenario. Can be used multiple times.",
    )
    parser.add_argument(
        "--scenario-suite",
        action="store_true",
        help="Run the full deterministic scenario suite.",
    )
    parser.add_argument(
        "--behavior-scenario",
        action="append",
        choices=list(SCENARIO_NAMES),
        default=None,
        help="Run behavioral evaluation scorecards for a specific scenario. Can be used multiple times.",
    )
    parser.add_argument(
        "--behavior-suite",
        action="store_true",
        help="Run the full behavioral evaluation suite.",
    )
    parser.add_argument(
        "--behavior-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit seeds for reproducible behavioral comparisons.",
    )
    parser.add_argument(
        "--behavior-compare-profiles",
        action="store_true",
        help="Compare the behavioral suite across reward profiles.",
    )
    parser.add_argument(
        "--behavior-compare-maps",
        action="store_true",
        help="Compare the behavioral suite across map templates.",
    )
    parser.add_argument(
        "--noise-robustness",
        "--behavior-noise-robustness",
        dest="noise_robustness",
        action="store_true",
        help="Train across canonical noise conditions and evaluate the full train-noise x eval-noise matrix.",
    )
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=None,
        help="CSV file path for exporting flat behavioral evaluation results.",
    )
    parser.add_argument(
        "--benchmark-package",
        type=Path,
        default=None,
        help=(
            "Directory for writing the benchmark-of-record package. Requires "
            "--budget-profile paper and --checkpoint-selection best."
        ),
    )
    parser.add_argument(
        "--scenario-episodes",
        type=int,
        default=None,
        help="Number of repetitions per scenario.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=list(CHECKPOINT_SELECTION_NAMES),
        default="none",
        help="Automatically select the best checkpoint for behavioral workflows.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=list(CHECKPOINT_METRIC_NAMES),
        default="scenario_success_rate",
        help="Metric used to choose the best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-override-penalty",
        type=float,
        default=0.0,
        help="Direct-mode penalty weight for final reflex override rate.",
    )
    parser.add_argument(
        "--checkpoint-dominance-penalty",
        type=float,
        default=0.0,
        help="Direct-mode penalty weight for mean reflex dominance.",
    )
    parser.add_argument(
        "--checkpoint-penalty-mode",
        choices=list(CHECKPOINT_PENALTY_MODE_NAMES),
        default="tiebreaker",
        help="How reflex-dependence penalties influence checkpoint ranking.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for persisting the best and last checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Episode interval between intermediate checkpoints.",
    )
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Compare the behavioral suite across architecture and ablation variants.",
    )
    parser.add_argument(
        "--ablation-variant",
        action="append",
        choices=list(canonical_ablation_variant_names()),
        default=None,
        help="Select a specific ablation variant. Can be used multiple times.",
    )
    parser.add_argument(
        "--ablation-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit seeds for the reproducible ablation suite.",
    )
    parser.add_argument(
        "--learning-evidence",
        action="store_true",
        help="Compare the trained checkpoint against controls to measure evidence of learning.",
    )
    parser.add_argument(
        "--claim-test-suite",
        action="store_true",
        help="Run the canonical claim-test suite that composes the core scientific comparisons.",
    )
    parser.add_argument(
        "--claim-test",
        action="append",
        choices=list(claim_test_names()),
        default=None,
        help="Select an individual canonical claim test. Can be used multiple times.",
    )
    parser.add_argument(
        "--learning-evidence-long-budget-profile",
        choices=list(canonical_budget_profile_names()),
        default="report",
        help="Long-budget profile used by the trained_long_budget condition.",
    )
    parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="Run an aggregated comparison across reward profiles using fixed seeds.",
    )
    parser.add_argument(
        "--compare-maps",
        action="store_true",
        help="Run an aggregated comparison across map templates using fixed seeds.",
    )
    parser.add_argument(
        "--full-summary",
        action="store_true",
        help="Print the full JSON summary instead of the default short summary.",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Enrich the trace with derived observations, memory, and the predator's internal state.",
    )
    return parser


__all__ = [
    "build_parser",
    "collect_requested_scenarios",
    "parse_module_reflex_scales",
]
