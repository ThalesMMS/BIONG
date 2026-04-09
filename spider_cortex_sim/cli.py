from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .ablations import BrainAblationConfig, canonical_ablation_variant_names
from .budget_profiles import (
    CHECKPOINT_METRIC_NAMES,
    CHECKPOINT_SELECTION_NAMES,
    canonical_budget_profile_names,
    resolve_budget,
)
from .maps import MAP_TEMPLATE_NAMES
from .noise import canonical_noise_profile_names
from .operational_profiles import canonical_operational_profile_names
from .scenarios import SCENARIO_NAMES
from .simulation import CURRICULUM_PROFILE_NAMES, SpiderSimulation
from .world import REWARD_PROFILES


def _default_behavior_evaluation() -> dict[str, object]:
    """
    Create the default behavior-evaluation payload structure.
    
    Returns:
        dict[str, object]: A payload with empty `suite` and `legacy_scenarios`, and a `summary` containing zeroed counts (`scenario_count`, `episode_count`), zero success rates (`scenario_success_rate`, `episode_success_rate`), and an empty `regressions` list.
    """
    return {
        "suite": {},
        "legacy_scenarios": {},
        "summary": {
            "scenario_count": 0,
            "episode_count": 0,
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "regressions": [],
        },
    }


def _ensure_behavior_evaluation(summary: dict[str, object]) -> None:
    """Initialise the behavior_evaluation key in summary if absent."""
    if "behavior_evaluation" not in summary:
        summary["behavior_evaluation"] = _default_behavior_evaluation()


def _default_checkpointing_summary(
    *,
    selection: str,
    metric: str,
    checkpoint_interval: int,
    selection_scenario_episodes: int,
) -> dict[str, object]:
    """
    Builds the checkpointing metadata payload for inclusion in a run summary.
    
    Parameters:
        selection (str): Checkpoint selection policy name; use "none" to disable checkpointing.
        metric (str): Metric name used to choose the selected checkpoint.
        checkpoint_interval (int): Number of episodes between generated checkpoints.
        selection_scenario_episodes (int): Episodes per scenario used when evaluating checkpoints for selection.
    
    Returns:
        dict: Payload containing:
            - `enabled` (bool): True if `selection` is not "none", False otherwise.
            - `selection` (str): Echoes the provided selection policy.
            - `metric` (str): Echoes the provided metric name.
            - `checkpoint_interval` (int): Echoes the provided checkpoint interval.
            - `evaluation_source` (str): Always "behavior_suite".
            - `selection_scenario_episodes` (int): Echoes the provided episodes-per-scenario value.
            - `generated_checkpoints` (list): Initially empty list reserved for generated checkpoint records.
            - `selected_checkpoint` (dict): Metadata for the chosen checkpoint (includes `scope` = "per_run").
    """
    return {
        "enabled": selection != "none",
        "selection": selection,
        "metric": metric,
        "checkpoint_interval": checkpoint_interval,
        "evaluation_source": "behavior_suite",
        "selection_scenario_episodes": selection_scenario_episodes,
        "generated_checkpoints": [],
        "selected_checkpoint": {
            "scope": "per_run",
        },
    }


def _build_compare_training_kwargs(
    args: argparse.Namespace,
    budget,
) -> dict[str, object]:
    """Collect shared runtime kwargs for training-regime comparisons."""
    return {
        "width": args.width,
        "height": args.height,
        "food_count": args.food_count,
        "day_length": args.day_length,
        "night_length": args.night_length,
        "max_steps": budget.max_steps,
        "episodes": budget.episodes,
        "evaluation_episodes": budget.eval_episodes,
        "gamma": args.gamma,
        "module_lr": args.module_lr,
        "motor_lr": args.motor_lr,
        "module_dropout": args.module_dropout,
        "reward_profile": args.reward_profile,
        "map_template": args.map_template,
        "operational_profile": args.operational_profile,
        "noise_profile": args.noise_profile,
        "budget_profile": args.budget_profile,
        "seeds": tuple(budget.behavior_seeds),
        "episodes_per_scenario": budget.scenario_episodes,
        "checkpoint_selection": args.checkpoint_selection,
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_interval": budget.checkpoint_interval,
        "checkpoint_dir": args.checkpoint_dir,
        "curriculum_profile": args.curriculum_profile,
    }


def _parse_module_reflex_scales(values: list[str] | None) -> dict[str, float]:
    """
    Parse per-module reflex scale specifications of the form "module=scale" into a mapping.
    
    Parameters:
        values (list[str] | None): Iterable of strings each formatted as "module=scale". If `None`, no entries are parsed.
    
    Returns:
        dict[str, float]: Mapping from module name to its parsed scale as a float.
    
    Raises:
        ValueError: If an entry is missing the '=' separator, the module name is empty, or the scale cannot be converted to a float.
    """
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
    Create a preconfigured command-line parser for the SpiderSimulation CLI.
    
    The parser includes options for training/evaluation sizing, world and map configuration, budget and operational/noise profiles, learning hyperparameters, reflex/module overrides, deterministic scenario and behavior evaluation (including ablations and comparisons), checkpointing controls, GUI/rendering, model load/save, and output persistence/debugging flags.
    
    Returns:
        argparse.ArgumentParser: Configured parser for the SpiderSimulation command-line interface.
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
    parser.add_argument(
        "--reflex-anneal-final-scale",
        type=float,
        default=None,
        help="Final reflex scale for linear annealing during training.",
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
        "--behavior-csv",
        type=Path,
        default=None,
        help="CSV file path for exporting flat behavioral evaluation results.",
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


def main() -> None:
    """
    Parse CLI arguments and run the requested SpiderSimulation workflows.
    
    Builds and validates the CLI, resolves runtime budget and brain configuration, then either launches the GUI or executes headless workflows (training/evaluation, deterministic scenario evaluations, behavior and configuration comparisons, ablation analyses, learning-evidence comparisons, and curriculum comparisons). Manages checkpoint-selection metadata, optionally persists summary/trace/behavior-CSV/brain files, and prints either a condensed or full JSON report to stdout.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.load_modules and not args.load_brain:
        parser.error("--load-modules requires --load-brain.")
    if not math.isfinite(float(args.reflex_scale)):
        parser.error("--reflex-scale must be finite.")
    if (
        args.reflex_anneal_final_scale is not None
        and not math.isfinite(float(args.reflex_anneal_final_scale))
    ):
        parser.error("--reflex-anneal-final-scale must be finite.")

    budget = resolve_budget(
        profile=args.budget_profile,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        scenario_episodes=args.scenario_episodes,
        checkpoint_interval=args.checkpoint_interval,
        behavior_seeds=args.behavior_seeds,
        ablation_seeds=args.ablation_seeds,
    )
    try:
        module_reflex_scales = _parse_module_reflex_scales(args.module_reflex_scale)
    except ValueError as exc:
        parser.error(str(exc))
    custom_reflex_config_requested = (
        bool(module_reflex_scales)
        or not math.isclose(float(args.reflex_scale), 1.0, rel_tol=0.0, abs_tol=0.0)
    )
    unsupported_reflex_workflows: list[str] = []
    if args.gui:
        unsupported_reflex_workflows.append("--gui")
    if args.compare_profiles or args.compare_maps:
        unsupported_reflex_workflows.append("--compare-profiles/--compare-maps")
    if args.behavior_compare_profiles or args.behavior_compare_maps:
        unsupported_reflex_workflows.append(
            "--behavior-compare-profiles/--behavior-compare-maps"
        )
    if args.ablation_suite or args.ablation_variant:
        unsupported_reflex_workflows.append("--ablation-suite/--ablation-variant")
    if unsupported_reflex_workflows and (
        custom_reflex_config_requested or args.reflex_anneal_final_scale is not None
    ):
        parser.error(
            "Custom reflex flags are not supported with "
            + ", ".join(unsupported_reflex_workflows)
            + "."
        )

    try:
        base_brain_config = BrainAblationConfig(
            name="modular_full",
            architecture="modular",
            module_dropout=args.module_dropout,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=args.reflex_scale,
            module_reflex_scales=module_reflex_scales,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.gui:
        from .gui import run_gui

        run_gui(
            episodes=budget.episodes,
            eval_episodes=budget.eval_episodes,
            width=args.width,
            height=args.height,
            food_count=args.food_count,
            day_length=args.day_length,
            night_length=args.night_length,
            max_steps=budget.max_steps,
            seed=args.seed,
            gamma=args.gamma,
            module_lr=args.module_lr,
            motor_lr=args.motor_lr,
            module_dropout=args.module_dropout,
            reward_profile=args.reward_profile,
            map_template=args.map_template,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            load_brain=args.load_brain,
            load_modules=args.load_modules,
        )
        return

    sim = SpiderSimulation(
        width=args.width,
        height=args.height,
        food_count=args.food_count,
        day_length=args.day_length,
        night_length=args.night_length,
        max_steps=budget.max_steps,
        seed=args.seed,
        gamma=args.gamma,
        module_lr=args.module_lr,
        motor_lr=args.motor_lr,
        module_dropout=args.module_dropout,
        brain_config=base_brain_config,
        reward_profile=args.reward_profile,
        map_template=args.map_template,
        operational_profile=args.operational_profile,
        noise_profile=args.noise_profile,
        budget_profile_name=budget.profile,
        benchmark_strength=budget.benchmark_strength,
        budget_summary=budget.to_summary(),
    )
    if args.load_brain:
        loaded = sim.brain.load(args.load_brain, modules=args.load_modules)
        print(f"Loaded modules: {loaded}")

    behavior_rows = []
    ablation_active = bool(args.ablation_suite or args.ablation_variant)
    learning_evidence_active = bool(args.learning_evidence)
    checkpoint_selection_run = False
    behavior_flags_active = bool(
        args.behavior_scenario
        or args.behavior_suite
        or args.behavior_compare_profiles
        or args.behavior_compare_maps
    )

    requested_scenarios = []
    if not ablation_active:
        requested_scenarios = list(args.scenario or [])
        for name in args.behavior_scenario or []:
            if name not in requested_scenarios:
                requested_scenarios.append(name)
        if args.scenario_suite or args.behavior_suite:
            for name in SCENARIO_NAMES:
                if name not in requested_scenarios:
                    requested_scenarios.append(name)

    ablation_scenarios = []
    if ablation_active:
        for name in args.behavior_scenario or []:
            if name not in ablation_scenarios:
                ablation_scenarios.append(name)
        if args.behavior_suite:
            for name in SCENARIO_NAMES:
                if name not in ablation_scenarios:
                    ablation_scenarios.append(name)
        if not ablation_scenarios:
            ablation_scenarios = list(SCENARIO_NAMES)

    learning_evidence_scenarios = []
    if learning_evidence_active:
        for name in args.behavior_scenario or []:
            if name not in learning_evidence_scenarios:
                learning_evidence_scenarios.append(name)
        if args.behavior_suite or not learning_evidence_scenarios:
            for name in SCENARIO_NAMES:
                if name not in learning_evidence_scenarios:
                    learning_evidence_scenarios.append(name)

    primary_checkpoint_selection = (
        args.checkpoint_selection
        if args.checkpoint_selection != "none" and behavior_flags_active and requested_scenarios
        else "none"
    )
    should_train_or_eval = (
        budget.episodes > 0
        or budget.eval_episodes > 0
        or primary_checkpoint_selection != "none"
    )
    if args.reflex_anneal_final_scale is not None and not should_train_or_eval:
        parser.error(
            "--reflex-anneal-final-scale requires a workflow that trains or evaluates the base brain."
        )
    if should_train_or_eval:
        checkpoint_selection_run = primary_checkpoint_selection != "none"
        summary, trace = sim.train(
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            render_last_evaluation=args.render_eval,
            capture_evaluation_trace=args.trace is not None,
            debug_trace=args.debug_trace,
            checkpoint_selection=primary_checkpoint_selection,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir if primary_checkpoint_selection != "none" else None,
            checkpoint_scenario_names=requested_scenarios or list(SCENARIO_NAMES),
            selection_scenario_episodes=budget.selection_scenario_episodes,
            reflex_anneal_final_scale=args.reflex_anneal_final_scale,
            curriculum_profile=args.curriculum_profile,
        )
    else:
        sim._set_runtime_budget(
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            scenario_episodes=budget.scenario_episodes,
            behavior_seeds=budget.behavior_seeds,
            ablation_seeds=budget.ablation_seeds,
            checkpoint_interval=budget.checkpoint_interval,
        )
        summary, trace = sim._build_summary([], []), []

    if requested_scenarios:
        behavior_suite, scenario_trace, scenario_rows = sim.evaluate_behavior_suite(
            requested_scenarios,
            episodes_per_scenario=max(1, budget.scenario_episodes),
            capture_trace=args.trace is not None and not trace,
            debug_trace=args.debug_trace,
        )
        summary["scenarios"] = behavior_suite["legacy_scenarios"]
        summary["behavior_evaluation"] = {
            "suite": behavior_suite["suite"],
            "summary": behavior_suite["summary"],
        }
        behavior_rows.extend(scenario_rows)
        if not trace and scenario_trace:
            trace = scenario_trace

    if args.compare_profiles or args.compare_maps:
        comparison_profiles = sorted(REWARD_PROFILES.keys()) if args.compare_profiles else [args.reward_profile]
        comparison_maps = list(MAP_TEMPLATE_NAMES) if args.compare_maps else [args.map_template]
        summary["comparisons"] = SpiderSimulation.compare_configurations(
            width=args.width,
            height=args.height,
            food_count=args.food_count,
            day_length=args.day_length,
            night_length=args.night_length,
            max_steps=budget.max_steps,
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            gamma=args.gamma,
            module_lr=args.module_lr,
            motor_lr=args.motor_lr,
            module_dropout=args.module_dropout,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            budget_profile=args.budget_profile,
            reward_profiles=comparison_profiles,
            map_templates=comparison_maps,
            seeds=tuple(budget.comparison_seeds),
        )

    if args.behavior_compare_profiles or args.behavior_compare_maps:
        comparison_profiles = sorted(REWARD_PROFILES.keys()) if args.behavior_compare_profiles else [args.reward_profile]
        comparison_maps = list(MAP_TEMPLATE_NAMES) if args.behavior_compare_maps else [args.map_template]
        checkpoint_selection_run = checkpoint_selection_run or args.checkpoint_selection != "none"
        behavior_comparisons, comparison_rows = SpiderSimulation.compare_behavior_suite(
            width=args.width,
            height=args.height,
            food_count=args.food_count,
            day_length=args.day_length,
            night_length=args.night_length,
            max_steps=budget.max_steps,
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            gamma=args.gamma,
            module_lr=args.module_lr,
            motor_lr=args.motor_lr,
            module_dropout=args.module_dropout,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            budget_profile=args.budget_profile,
            reward_profiles=comparison_profiles,
            map_templates=comparison_maps,
            seeds=tuple(budget.behavior_seeds),
            names=requested_scenarios or None,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=args.checkpoint_selection,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["comparisons"] = behavior_comparisons
        if "reward_audit" in behavior_comparisons:
            summary["behavior_evaluation"]["shaping_audit"] = behavior_comparisons[
                "reward_audit"
            ]
        behavior_rows.extend(comparison_rows)

    if ablation_active:
        checkpoint_selection_run = checkpoint_selection_run or args.checkpoint_selection != "none"
        ablation_payload, ablation_rows = SpiderSimulation.compare_ablation_suite(
            width=args.width,
            height=args.height,
            food_count=args.food_count,
            day_length=args.day_length,
            night_length=args.night_length,
            max_steps=budget.max_steps,
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            gamma=args.gamma,
            module_lr=args.module_lr,
            motor_lr=args.motor_lr,
            module_dropout=args.module_dropout,
            reward_profile=args.reward_profile,
            map_template=args.map_template,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            budget_profile=args.budget_profile,
            seeds=tuple(budget.ablation_seeds),
            names=ablation_scenarios or None,
            variant_names=args.ablation_variant,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=args.checkpoint_selection,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["ablations"] = ablation_payload
        behavior_rows.extend(ablation_rows)

    if learning_evidence_active:
        checkpoint_selection_run = checkpoint_selection_run or args.checkpoint_selection != "none"
        learning_evidence_payload, learning_evidence_rows = SpiderSimulation.compare_learning_evidence(
            width=args.width,
            height=args.height,
            food_count=args.food_count,
            day_length=args.day_length,
            night_length=args.night_length,
            max_steps=budget.max_steps,
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            gamma=args.gamma,
            module_lr=args.module_lr,
            motor_lr=args.motor_lr,
            module_dropout=args.module_dropout,
            reward_profile=args.reward_profile,
            map_template=args.map_template,
            brain_config=base_brain_config,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            budget_profile=args.budget_profile,
            long_budget_profile=args.learning_evidence_long_budget_profile,
            seeds=tuple(budget.behavior_seeds),
            names=learning_evidence_scenarios or None,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=args.checkpoint_selection,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["learning_evidence"] = learning_evidence_payload
        behavior_rows.extend(learning_evidence_rows)

    if (
        args.curriculum_profile != "none"
        and not ablation_active
        and not learning_evidence_active
        and (budget.episodes > 0 or budget.eval_episodes > 0)
    ):
        compare_training_kwargs = _build_compare_training_kwargs(args, budget)
        curriculum_payload, curriculum_rows = SpiderSimulation.compare_training_regimes(
            names=requested_scenarios or None,
            **compare_training_kwargs,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["curriculum_comparison"] = curriculum_payload
        behavior_rows.extend(curriculum_rows)

    if (
        checkpoint_selection_run
        and args.checkpoint_selection != "none"
        and "checkpointing" not in summary
    ):
        summary["checkpointing"] = _default_checkpointing_summary(
            selection=args.checkpoint_selection,
            metric=args.checkpoint_metric,
            checkpoint_interval=budget.checkpoint_interval,
            selection_scenario_episodes=budget.selection_scenario_episodes,
        )

    if args.summary is not None:
        SpiderSimulation.save_summary(summary, args.summary)
    if args.trace is not None:
        SpiderSimulation.save_trace(trace, args.trace)
    if args.behavior_csv is not None:
        SpiderSimulation.save_behavior_csv(behavior_rows, args.behavior_csv)
    if args.save_brain:
        sim.brain.save(args.save_brain)
        print(f"Brain saved to: {args.save_brain}")

    if args.full_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    printable = {
        "reward_profile": summary["config"]["world"]["reward_profile"],
        "map_template": summary["config"]["world"]["map_template"],
        "budget_profile": summary["config"]["budget"]["profile"],
        "benchmark_strength": summary["config"]["budget"]["benchmark_strength"],
        "training_regime": summary["config"]["training_regime"],
        "operational_profile": summary["config"]["operational_profile"]["name"],
        "noise_profile": summary["config"]["noise_profile"]["name"],
        "training_mean_reward_last_window": summary["training_last_window"]["mean_reward"],
        "training_mean_night_shelter_occupancy_rate_last_window": summary["training_last_window"]["mean_night_shelter_occupancy_rate"],
        "training_mean_night_stillness_rate_last_window": summary["training_last_window"]["mean_night_stillness_rate"],
        "evaluation_mean_reward": summary["evaluation"]["mean_reward"],
        "evaluation_mean_food": summary["evaluation"]["mean_food"],
        "evaluation_mean_sleep": summary["evaluation"]["mean_sleep"],
        "evaluation_mean_sleep_debt": summary["evaluation"]["mean_sleep_debt"],
        "evaluation_mean_predator_contacts": summary["evaluation"]["mean_predator_contacts"],
        "evaluation_mean_predator_escapes": summary["evaluation"]["mean_predator_escapes"],
        "evaluation_mean_night_shelter_occupancy_rate": summary["evaluation"]["mean_night_shelter_occupancy_rate"],
        "evaluation_mean_night_stillness_rate": summary["evaluation"]["mean_night_stillness_rate"],
        "evaluation_mean_predator_response_events": summary["evaluation"]["mean_predator_response_events"],
        "evaluation_mean_predator_response_latency": summary["evaluation"]["mean_predator_response_latency"],
        "evaluation_mean_predator_mode_transitions": summary["evaluation"]["mean_predator_mode_transitions"],
        "evaluation_dominant_predator_state": summary["evaluation"]["dominant_predator_state"],
        "evaluation_mean_food_distance_delta": summary["evaluation"]["mean_food_distance_delta"],
        "evaluation_mean_shelter_distance_delta": summary["evaluation"]["mean_shelter_distance_delta"],
        "evaluation_survival_rate": summary["evaluation"]["survival_rate"],
        "evaluation_mean_night_role_distribution": summary["evaluation"]["mean_night_role_distribution"],
    }
    if "scenarios" in summary:
        printable["scenarios"] = {
            name: {
                "mean_reward": data["mean_reward"],
                "survival_rate": data["survival_rate"],
                "dominant_predator_state": data["dominant_predator_state"],
                "mean_predator_mode_transitions": data["mean_predator_mode_transitions"],
                "mean_food_distance_delta": data["mean_food_distance_delta"],
                "mean_shelter_distance_delta": data["mean_shelter_distance_delta"],
            }
            for name, data in summary["scenarios"].items()
        }
    if "comparisons" in summary:
        printable["comparisons"] = {
            "seeds": summary["comparisons"]["seeds"],
            "reward_profiles": {
                name: {
                    "mean_reward": data["mean_reward"],
                    "mean_food": data["mean_food"],
                    "survival_rate": data["survival_rate"],
                }
                for name, data in summary["comparisons"]["reward_profiles"].items()
            },
            "map_templates": {
                name: {
                    "mean_reward": data["mean_reward"],
                    "mean_food": data["mean_food"],
                    "survival_rate": data["survival_rate"],
                }
                for name, data in summary["comparisons"]["map_templates"].items()
            },
        }
    if "reward_audit" in summary:
        reward_audit = summary.get("reward_audit", {})
        observation_signals = reward_audit.get("observation_signals", {})
        memory_signals = reward_audit.get("memory_signals", {})
        printable["reward_audit"] = {
            "current_profile": reward_audit.get("current_profile"),
            "minimal_profile": reward_audit.get("minimal_profile"),
            "high_risk_observation_signals": sorted(
                name
                for name, data in observation_signals.items()
                if isinstance(data, dict) and data.get("risk") == "high"
            ),
            "high_risk_memory_signals": sorted(
                name
                for name, data in memory_signals.items()
                if isinstance(data, dict) and data.get("risk") == "high"
            ),
        }
    if "behavior_evaluation" in summary:
        printable["behavior_evaluation"] = {
            "scenario_success_rate": summary["behavior_evaluation"]["summary"]["scenario_success_rate"],
            "episode_success_rate": summary["behavior_evaluation"]["summary"]["episode_success_rate"],
            "regressions": summary["behavior_evaluation"]["summary"]["regressions"],
        }
        if summary["behavior_evaluation"]["suite"]:
            printable["behavior_evaluation"]["suite"] = {
                name: {
                    "success_rate": data["success_rate"],
                    "failures": data["failures"],
                }
                for name, data in summary["behavior_evaluation"]["suite"].items()
            }
        if "comparisons" in summary["behavior_evaluation"]:
            printable["behavior_evaluation"]["comparisons"] = {
                "seeds": summary["behavior_evaluation"]["comparisons"]["seeds"],
                "reward_profiles": {
                    name: {
                        "scenario_success_rate": data["summary"]["scenario_success_rate"],
                        "episode_success_rate": data["summary"]["episode_success_rate"],
                    }
                    for name, data in summary["behavior_evaluation"]["comparisons"]["reward_profiles"].items()
                },
                "map_templates": {
                    name: {
                        "scenario_success_rate": data["summary"]["scenario_success_rate"],
                        "episode_success_rate": data["summary"]["episode_success_rate"],
                    }
                    for name, data in summary["behavior_evaluation"]["comparisons"]["map_templates"].items()
                },
            }
        if "shaping_audit" in summary["behavior_evaluation"]:
            shaping = summary.get("behavior_evaluation", {}).get("shaping_audit", {})
            printable["behavior_evaluation"]["shaping_audit"] = {
                "minimal_profile": shaping.get("minimal_profile"),
                "deltas_vs_minimal": shaping.get(
                    "comparison",
                    {},
                ).get("deltas_vs_minimal", {}),
            }
        if "ablations" in summary["behavior_evaluation"]:
            printable["behavior_evaluation"]["ablations"] = {
                "reference_variant": summary["behavior_evaluation"]["ablations"]["reference_variant"],
                "seeds": summary["behavior_evaluation"]["ablations"]["seeds"],
                "variants": {
                    name: {
                        "architecture": data["config"]["architecture"],
                        "scenario_success_rate": data["summary"]["scenario_success_rate"],
                        "episode_success_rate": data["summary"]["episode_success_rate"],
                    }
                    for name, data in summary["behavior_evaluation"]["ablations"]["variants"].items()
                },
                "deltas_vs_reference": summary["behavior_evaluation"]["ablations"]["deltas_vs_reference"],
            }
        if "curriculum_comparison" in summary["behavior_evaluation"]:
            printable["behavior_evaluation"]["curriculum_comparison"] = {
                "curriculum_profile": summary["behavior_evaluation"]["curriculum_comparison"]["curriculum_profile"],
                "reference_regime": summary["behavior_evaluation"]["curriculum_comparison"]["reference_regime"],
                "focus_scenarios": summary["behavior_evaluation"]["curriculum_comparison"]["focus_scenarios"],
                "regimes": {
                    name: {
                        "scenario_success_rate": data["summary"]["scenario_success_rate"],
                        "episode_success_rate": data["summary"]["episode_success_rate"],
                    }
                    for name, data in summary["behavior_evaluation"]["curriculum_comparison"]["regimes"].items()
                },
            }
        if "learning_evidence" in summary["behavior_evaluation"]:
            printable["behavior_evaluation"]["learning_evidence"] = {
                "reference_condition": summary["behavior_evaluation"]["learning_evidence"]["reference_condition"],
                "seeds": summary["behavior_evaluation"]["learning_evidence"]["seeds"],
                "evidence_summary": summary["behavior_evaluation"]["learning_evidence"]["evidence_summary"],
                "conditions": {
                    name: (
                        {
                            "skipped": True,
                            "reason": data.get("reason", ""),
                        }
                        if data.get("skipped")
                        else {
                            "policy_mode": data["policy_mode"],
                            "train_episodes": data["train_episodes"],
                            "checkpoint_source": data["checkpoint_source"],
                            "scenario_success_rate": data["summary"]["scenario_success_rate"],
                            "episode_success_rate": data["summary"]["episode_success_rate"],
                        }
                    )
                    for name, data in summary["behavior_evaluation"]["learning_evidence"]["conditions"].items()
                },
            }
    if "checkpointing" in summary:
        printable["checkpointing"] = {
            "selection": summary["checkpointing"]["selection"],
            "metric": summary["checkpointing"]["metric"],
            "checkpoint_interval": summary["checkpointing"]["checkpoint_interval"],
            "selected_checkpoint": summary["checkpointing"]["selected_checkpoint"],
        }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
