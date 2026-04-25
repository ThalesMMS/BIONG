"""CLI command dispatch and workflow orchestration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..ablations import BrainAblationConfig, COARSE_ROLLUP_MODULES
from ..benchmark_package import (
    BenchmarkManifest,
    assemble_benchmark_package,
    summarize_benchmark_manifest,
)
from ..budget_profiles import ResolvedBudget, resolve_budget, resolve_budget_profile
from ..capacity_profiles import canonical_capacity_axis_names
from ..checkpointing import CheckpointSelectionConfig, build_checkpointing_summary
from ..claim_evaluation import run_claim_test_suite
from ..comparison import (
    compare_ablation_suite,
    compare_behavior_suite,
    compare_capacity_axis_sweep,
    compare_capacity_sweep,
    compare_configurations,
    compare_ladder_under_profiles,
    compare_learning_evidence,
    compare_noise_robustness,
    compare_training_regimes,
)
from ..distillation import DistillationConfig
from ..maps import MAP_TEMPLATE_NAMES
from ..scenarios import SCENARIO_NAMES
from ..simulation import (
    EXPERIMENT_OF_RECORD_REGIME,
    SpiderSimulation,
    default_behavior_evaluation,
    ensure_behavior_evaluation,
)
from ..training_regimes import resolve_training_regime
from ..world import REWARD_PROFILES
from .formatting import format_run_summary
from .parser import collect_requested_scenarios, parse_module_reflex_scales
from .output import _build_and_record_benchmark_package
from .validation import _argument_error
from .workflow_args import _build_checkpoint_selection_config, _build_compare_training_kwargs

EXPERIMENT_OF_RECORD_CHECKPOINT_OVERRIDE_PENALTY = 1.0
EXPERIMENT_OF_RECORD_CHECKPOINT_DOMINANCE_PENALTY = 1.0

_benchmark_package_manifest_summary = summarize_benchmark_manifest
_default_behavior_evaluation = default_behavior_evaluation
_default_checkpointing_summary = build_checkpointing_summary
_ensure_behavior_evaluation = ensure_behavior_evaluation




def run_cli(args: argparse.Namespace) -> None:
    """
    Dispatch CLI workflows to run SpiderSimulation or launch the GUI.
    
    Validate CLI arguments, budget, and brain configuration, then execute the selected workflow(s)—GUI, headless training/evaluation, deterministic scenario evaluations, behavior and configuration comparisons, ablation analyses, ladder/capacity sweeps, learning-evidence comparisons, claim-test suites, curriculum comparisons, and noise-robustness studies. Optionally produce benchmark packages, summaries, traces, CSVs, and saved brains, and emit a consolidated JSON run summary to stdout.
    """
    if args.claim_test and not args.claim_test_suite:
        args.claim_test_suite = True

    if args.load_modules and not args.load_brain:
        _argument_error(args, "--load-modules requires --load-brain.")
    if args.experiment_of_record:
        if (
            bool(args.module_reflex_scale)
            or float(args.reflex_scale) != 1.0
        ):
            _argument_error(args,
                "--experiment-of-record forbids custom reflex settings "
                "(--reflex-scale/--module-reflex-scale); it selects "
                f"{EXPERIMENT_OF_RECORD_REGIME!r} and applies checkpoint "
                "penalties "
                f"{EXPERIMENT_OF_RECORD_CHECKPOINT_OVERRIDE_PENALTY} "
                "and "
                f"{EXPERIMENT_OF_RECORD_CHECKPOINT_DOMINANCE_PENALTY}."
            )
        if args.reflex_anneal_final_scale is not None:
            _argument_error(args,
                "--experiment-of-record cannot be combined with "
                "--reflex-anneal-final-scale."
            )
        if args.training_regime not in (None, EXPERIMENT_OF_RECORD_REGIME):
            _argument_error(args,
                "--experiment-of-record selects "
                f"{EXPERIMENT_OF_RECORD_REGIME!r}; do not combine it with a "
                "different --training-regime."
            )
        args.training_regime = EXPERIMENT_OF_RECORD_REGIME
        args.checkpoint_selection = "best"
        args.checkpoint_penalty_mode = "direct"
        if float(args.checkpoint_override_penalty) == 0.0:
            args.checkpoint_override_penalty = (
                EXPERIMENT_OF_RECORD_CHECKPOINT_OVERRIDE_PENALTY
            )
        if float(args.checkpoint_dominance_penalty) == 0.0:
            args.checkpoint_dominance_penalty = (
                EXPERIMENT_OF_RECORD_CHECKPOINT_DOMINANCE_PENALTY
            )
    if args.benchmark_package is not None:
        if args.budget_profile != "paper":
            _argument_error(args, "--benchmark-package requires --budget-profile paper.")
        if args.checkpoint_selection != "best":
            _argument_error(args, "--benchmark-package requires --checkpoint-selection best.")
    if not math.isfinite(float(args.reflex_scale)):
        _argument_error(args, "--reflex-scale must be finite.")
    if (
        args.reflex_anneal_final_scale is not None
        and not math.isfinite(float(args.reflex_anneal_final_scale))
    ):
        _argument_error(args, "--reflex-anneal-final-scale must be finite.")
    for value, flag_name in (
        (args.checkpoint_override_penalty, "--checkpoint-override-penalty"),
        (args.checkpoint_dominance_penalty, "--checkpoint-dominance-penalty"),
    ):
        if not math.isfinite(float(value)):
            _argument_error(args, f"{flag_name} must be finite.")
        if float(value) < 0.0:
            _argument_error(args, f"{flag_name} must be non-negative.")
    checkpoint_selection_config = _build_checkpoint_selection_config(args)

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
    if (
        budget.requires_checkpoint_selection
        and args.checkpoint_selection in (None, "none")
    ):
        _argument_error(args,
            f"Budget profile {budget.profile!r} requires checkpoint selection; "
            "specify --checkpoint-selection best."
        )
    if (
        args.learning_evidence
        and args.learning_evidence_long_budget_profile is not None
    ):
        long_budget_profile = resolve_budget_profile(
            args.learning_evidence_long_budget_profile
        )
        if (
            long_budget_profile.requires_checkpoint_selection
            and args.checkpoint_selection in (None, "none")
        ):
            _argument_error(args,
                f"Budget profile {long_budget_profile.name!r} requires checkpoint "
                "selection; specify --checkpoint-selection best."
            )
    try:
        module_reflex_scales = parse_module_reflex_scales(args.module_reflex_scale)
    except ValueError as exc:
        _argument_error(args, str(exc))
    custom_reflex_config_requested = (
        bool(module_reflex_scales)
        or float(args.reflex_scale) != 1.0
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
    if args.noise_robustness:
        unsupported_reflex_workflows.append("--noise-robustness")
    if args.ablation_suite or args.ablation_variant:
        unsupported_reflex_workflows.append("--ablation-suite/--ablation-variant")
    if args.ladder_reward_profiles:
        unsupported_reflex_workflows.append("--ladder-reward-profiles")
    if args.capacity_sweep:
        unsupported_reflex_workflows.append("--capacity-sweep")
    if args.capacity_axis_sweep is not None:
        unsupported_reflex_workflows.append("--capacity-axis-sweep")
    if args.claim_test_suite:
        unsupported_reflex_workflows.append("--claim-test-suite")
    if unsupported_reflex_workflows and (
        custom_reflex_config_requested
        or args.reflex_anneal_final_scale is not None
        or args.training_regime is not None
    ):
        _argument_error(args,
            "Custom reflex flags, reflex annealing, and named training "
            "regime / experiment-of-record settings (--training-regime/"
            "--experiment-of-record) are not supported with "
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
            credit_strategy="route_mask",
            disabled_modules=COARSE_ROLLUP_MODULES,
            recurrent_modules=(),
            reflex_scale=args.reflex_scale,
            module_reflex_scales=module_reflex_scales,
            capacity_profile=args.capacity_profile,
        )
    except ValueError as exc:
        _argument_error(args, str(exc))

    unsupported_capacity_workflows: list[str] = []
    if args.gui:
        unsupported_capacity_workflows.append("--gui")
    if args.noise_robustness:
        unsupported_capacity_workflows.append("--noise-robustness")
    if args.compare_profiles or args.compare_maps:
        unsupported_capacity_workflows.append("--compare-profiles/--compare-maps")
    if args.behavior_compare_profiles or args.behavior_compare_maps:
        unsupported_capacity_workflows.append(
            "--behavior-compare-profiles/--behavior-compare-maps"
        )
    if args.curriculum_profile != "none":
        unsupported_capacity_workflows.append("--curriculum-profile")
    if args.capacity_profile is not None and unsupported_capacity_workflows:
        _argument_error(
            args,
            "--capacity-profile is not supported with "
            + ", ".join(unsupported_capacity_workflows)
            + ".",
        )

    if args.gui:
        from ..gui import run_gui

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
            capacity_profile=args.capacity_profile,
            reward_profile=args.reward_profile,
            map_template=args.map_template,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            load_brain=args.load_brain,
            load_modules=args.load_modules,
        )
        return

    if args.noise_robustness:
        if args.capacity_profile is not None:
            _argument_error(
                args,
                "--capacity-profile is not supported with --noise-robustness.",
            )
        if args.trace is not None:
            _argument_error(args, "--trace is not supported with --noise-robustness.")
        if args.render_eval:
            _argument_error(args, "--render-eval is not supported with --noise-robustness.")
        if args.debug_trace:
            _argument_error(args, "--debug-trace is not supported with --noise-robustness.")
        if args.curriculum_profile != "none":
            _argument_error(args,
                "--curriculum-profile is not supported with --noise-robustness."
            )
        if args.noise_profile != "none":
            _argument_error(args,
                "--noise-profile is not supported with --noise-robustness."
            )
        for enabled, flag_name in (
            (args.compare_profiles, "--compare-profiles"),
            (args.compare_maps, "--compare-maps"),
            (args.behavior_compare_profiles, "--behavior-compare-profiles"),
            (args.behavior_compare_maps, "--behavior-compare-maps"),
            (args.ablation_suite, "--ablation-suite"),
            (bool(args.ablation_variant), "--ablation-variant"),
            (args.learning_evidence, "--learning-evidence"),
            (args.claim_test_suite, "--claim-test-suite"),
        ):
            if enabled:
                _argument_error(args,
                    f"{flag_name} is not supported with --noise-robustness."
                )
        requested_scenarios = collect_requested_scenarios(args)
        robustness_payload, robustness_rows = compare_noise_robustness(
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
            budget_profile=args.budget_profile,
            seeds=tuple(budget.behavior_seeds),
            names=requested_scenarios or None,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=args.checkpoint_selection,
            checkpoint_selection_config=checkpoint_selection_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            load_brain=args.load_brain,
            load_modules=args.load_modules,
            save_brain=args.save_brain,
        )
        behavior_evaluation = _default_behavior_evaluation()
        behavior_evaluation["robustness_matrix"] = robustness_payload
        summary = {
            "config": {
                "world": {
                    "width": args.width,
                    "height": args.height,
                    "food_count": args.food_count,
                    "day_length": args.day_length,
                    "night_length": args.night_length,
                    "max_steps": budget.max_steps,
                    "reward_profile": args.reward_profile,
                    "map_template": args.map_template,
                },
                "budget": budget.to_summary(),
                "training_regime": {"name": "noise_robustness"},
                "operational_profile": {"name": args.operational_profile},
            },
            "behavior_evaluation": behavior_evaluation,
        }
        if args.checkpoint_selection != "none":
            summary["checkpointing"] = _default_checkpointing_summary(
                selection=args.checkpoint_selection,
                checkpoint_interval=budget.checkpoint_interval,
                selection_scenario_episodes=budget.selection_scenario_episodes,
                selection_config=checkpoint_selection_config,
            )
        if args.benchmark_package is not None:
            _build_and_record_benchmark_package(
                output_dir=args.benchmark_package,
                summary=summary,
                behavior_csv=args.behavior_csv,
                behavior_rows=robustness_rows,
                command_metadata={
                    "argv": list(sys.argv[1:]),
                    "entrypoint": "spider_cortex_sim",
                },
            )
        if args.summary is not None:
            SpiderSimulation.save_summary(summary, args.summary)
        if args.behavior_csv is not None:
            SpiderSimulation.save_behavior_csv(robustness_rows, args.behavior_csv)
        print(
            json.dumps(
                format_run_summary(summary, full=args.full_summary),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.capacity_sweep and args.capacity_axis_sweep is not None:
        _argument_error(args, "--capacity-sweep cannot be combined with --capacity-axis-sweep.")

    if args.capacity_sweep or args.capacity_axis_sweep is not None:
        capacity_flag = (
            "--capacity-axis-sweep"
            if args.capacity_axis_sweep is not None
            else "--capacity-sweep"
        )
        if args.trace is not None:
            _argument_error(args, f"--trace is not supported with {capacity_flag}.")
        if args.render_eval:
            _argument_error(args, f"--render-eval is not supported with {capacity_flag}.")
        if args.debug_trace:
            _argument_error(args, f"--debug-trace is not supported with {capacity_flag}.")
        for enabled, flag_name in (
            (args.compare_profiles, "--compare-profiles"),
            (args.compare_maps, "--compare-maps"),
            (args.behavior_compare_profiles, "--behavior-compare-profiles"),
            (args.behavior_compare_maps, "--behavior-compare-maps"),
            (args.ablation_suite, "--ablation-suite"),
            (bool(args.ablation_variant), "--ablation-variant"),
            (args.learning_evidence, "--learning-evidence"),
            (args.claim_test_suite, "--claim-test-suite"),
            (bool(args.capacity_profile), "--capacity-profile"),
            (bool(args.curriculum_profile and args.curriculum_profile != "none"), "--curriculum-profile"),
        ):
            if enabled:
                _argument_error(
                    args,
                    f"{flag_name} is not supported with {capacity_flag}."
                )
        capacity_scenarios: list[str] = []
        for name in args.behavior_scenario or []:
            if name not in capacity_scenarios:
                capacity_scenarios.append(name)
        if args.behavior_suite or not capacity_scenarios:
            for name in SCENARIO_NAMES:
                if name not in capacity_scenarios:
                    capacity_scenarios.append(name)
        capacity_runner = (
            compare_capacity_axis_sweep
            if args.capacity_axis_sweep is not None
            else compare_capacity_sweep
        )
        capacity_kwargs = {}
        if args.capacity_axis_sweep is not None:
            capacity_kwargs["capacity_axes"] = (
                tuple(args.capacity_axis_sweep)
                if args.capacity_axis_sweep
                else tuple(canonical_capacity_axis_names())
            )
        try:
            capacity_payload, capacity_rows = capacity_runner(
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
                names=capacity_scenarios or None,
                checkpoint_selection=args.checkpoint_selection,
                checkpoint_selection_config=checkpoint_selection_config,
                checkpoint_interval=budget.checkpoint_interval,
                checkpoint_dir=args.checkpoint_dir,
                **capacity_kwargs,
            )
        except ValueError as exc:
            _argument_error(args, str(exc))
        summary = {
            "config": {
                "brain": base_brain_config.to_summary(),
                "world": {
                    "width": args.width,
                    "height": args.height,
                    "food_count": args.food_count,
                    "day_length": args.day_length,
                    "night_length": args.night_length,
                    "max_steps": budget.max_steps,
                    "reward_profile": args.reward_profile,
                    "map_template": args.map_template,
                },
                "budget": budget.to_summary(),
                "training_regime": {
                    "name": (
                        "capacity_axis_sweep"
                        if args.capacity_axis_sweep is not None
                        else "capacity_sweep"
                    ),
                },
                "operational_profile": {"name": args.operational_profile},
                "noise_profile": {"name": args.noise_profile},
            },
            "behavior_evaluation": {
                "capacity_sweeps": capacity_payload,
            },
        }
        if args.checkpoint_selection != "none":
            summary["checkpointing"] = _default_checkpointing_summary(
                selection=args.checkpoint_selection,
                checkpoint_interval=budget.checkpoint_interval,
                selection_scenario_episodes=budget.selection_scenario_episodes,
                selection_config=checkpoint_selection_config,
            )
        if args.benchmark_package is not None:
            _build_and_record_benchmark_package(
                output_dir=args.benchmark_package,
                summary=summary,
                behavior_csv=args.behavior_csv,
                behavior_rows=capacity_rows,
                command_metadata={
                    "argv": list(sys.argv[1:]),
                    "entrypoint": "spider_cortex_sim",
                },
            )
        if args.summary is not None:
            SpiderSimulation.save_summary(summary, args.summary)
        if args.behavior_csv is not None:
            SpiderSimulation.save_behavior_csv(capacity_rows, args.behavior_csv)
        print(
            json.dumps(
                format_run_summary(summary, full=args.full_summary),
                indent=2,
                ensure_ascii=False,
            )
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
        capacity_profile=args.capacity_profile,
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
    ablation_payload: dict[str, object] | None = None
    ladder_profile_payload: dict[str, object] | None = None
    learning_evidence_payload: dict[str, object] | None = None
    ablation_active = bool(args.ablation_suite or args.ablation_variant)
    ladder_profile_active = bool(args.ladder_reward_profiles)
    learning_evidence_active = bool(args.learning_evidence)
    claim_test_suite_active = bool(args.claim_test_suite)
    checkpoint_selection_run = False
    behavior_flags_active = bool(
        args.behavior_scenario
        or args.behavior_suite
        or args.behavior_compare_profiles
        or args.behavior_compare_maps
    )

    requested_scenarios = []
    if not ablation_active and not ladder_profile_active:
        requested_scenarios = collect_requested_scenarios(args)

    ablation_scenarios = []
    if ablation_active or ladder_profile_active:
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

    claim_test_seeds: tuple[int, ...] | None = None
    if budget.behavior_seeds and budget.ablation_seeds:
        claim_test_seeds = tuple(
            dict.fromkeys((*budget.behavior_seeds, *budget.ablation_seeds))
        )
    elif budget.behavior_seeds:
        claim_test_seeds = tuple(budget.behavior_seeds)
    elif budget.ablation_seeds:
        claim_test_seeds = tuple(budget.ablation_seeds)

    if args.experiment_of_record:
        primary_checkpoint_selection = args.checkpoint_selection
    else:
        primary_checkpoint_selection = (
            args.checkpoint_selection
            if args.checkpoint_selection != "none"
            and behavior_flags_active
            and requested_scenarios
            else "none"
        )
    should_train_or_eval = (
        budget.episodes > 0
        or budget.eval_episodes > 0
        or primary_checkpoint_selection != "none"
    )
    if (
        args.reflex_anneal_final_scale is not None
        or args.training_regime is not None
    ) and not should_train_or_eval:
        _argument_error(args,
            "--reflex-anneal-final-scale/--training-regime requires a workflow "
            "that trains or evaluates the base brain."
        )
    lr_overrides = [
        value
        for value in (
            args.distillation_module_lr,
            args.distillation_motor_lr,
            args.distillation_arbitration_lr,
        )
        if value is not None
    ]
    effective_distillation_lr = (
        lr_overrides[0]
        if lr_overrides
        else None
    )
    if lr_overrides and any(
        abs(float(value) - float(lr_overrides[0])) > 1e-12
        for value in lr_overrides[1:]
    ):
        _argument_error(
            args,
            "Distillation now uses a single regime-level learning rate; provide matching distillation LR overrides.",
        )
    distillation_config = (
        None
        if args.distillation_teacher_checkpoint is None
        else DistillationConfig(
            teacher_checkpoint=args.distillation_teacher_checkpoint,
            shuffle=args.distillation_shuffle,
            match_local_proposals=args.distillation_match_local_proposals,
        )
    )
    distillation_training_regime = None
    if args.distillation_teacher_checkpoint is not None:
        base_distillation_regime = resolve_training_regime("distillation")
        distillation_training_regime = type(base_distillation_regime)(
            name=base_distillation_regime.name,
            annealing_schedule=base_distillation_regime.annealing_schedule,
            anneal_target_scale=base_distillation_regime.anneal_target_scale,
            anneal_warmup_fraction=base_distillation_regime.anneal_warmup_fraction,
            finetuning_episodes=base_distillation_regime.finetuning_episodes,
            finetuning_reflex_scale=base_distillation_regime.finetuning_reflex_scale,
            loss_override_penalty_weight=base_distillation_regime.loss_override_penalty_weight,
            loss_dominance_penalty_weight=base_distillation_regime.loss_dominance_penalty_weight,
            distillation_epochs=args.distillation_epochs,
            distillation_temperature=args.distillation_temperature,
            distillation_lr=(
                effective_distillation_lr
                if effective_distillation_lr is not None
                else base_distillation_regime.distillation_lr
            ),
            distillation_enabled=True,
        )
    resolved_training_regime = args.training_regime
    if args.training_regime == "distillation":
        if args.distillation_teacher_checkpoint is None:
            _argument_error(
                args,
                "--training-regime distillation requires --distillation-teacher-checkpoint.",
            )
        resolved_training_regime = distillation_training_regime
    if should_train_or_eval:
        checkpoint_selection_run = primary_checkpoint_selection != "none"
        summary, trace = sim.train(
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            render_last_evaluation=args.render_eval,
            capture_evaluation_trace=args.trace is not None,
            debug_trace=args.debug_trace,
            checkpoint_selection=primary_checkpoint_selection,
            checkpoint_selection_config=checkpoint_selection_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir if primary_checkpoint_selection != "none" else None,
            checkpoint_scenario_names=requested_scenarios or list(SCENARIO_NAMES),
            selection_scenario_episodes=budget.selection_scenario_episodes,
            reflex_anneal_final_scale=args.reflex_anneal_final_scale,
            curriculum_profile=args.curriculum_profile,
            training_regime=resolved_training_regime,
            teacher_checkpoint=args.distillation_teacher_checkpoint,
            distillation_config=distillation_config,
        )
    else:
        sim.set_runtime_budget(
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            scenario_episodes=budget.scenario_episodes,
            behavior_seeds=budget.behavior_seeds,
            ablation_seeds=budget.ablation_seeds,
            checkpoint_interval=budget.checkpoint_interval,
        )
        summary, trace = sim.build_summary([], []), []

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
        summary["comparisons"] = compare_configurations(
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
        behavior_comparisons, comparison_rows = compare_behavior_suite(
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
            checkpoint_selection_config=checkpoint_selection_config,
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
        ablation_payload, ablation_rows = compare_ablation_suite(
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
            capacity_profile=args.capacity_profile,
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
            checkpoint_selection_config=checkpoint_selection_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["ablations"] = ablation_payload
        behavior_rows.extend(ablation_rows)

    if ladder_profile_active:
        checkpoint_selection_run = checkpoint_selection_run or args.checkpoint_selection != "none"
        ladder_profile_payload, ladder_profile_rows = compare_ladder_under_profiles(
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
            capacity_profile=args.capacity_profile,
            map_template=args.map_template,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            budget_profile=args.budget_profile,
            seeds=tuple(budget.ablation_seeds),
            names=ablation_scenarios or None,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=args.checkpoint_selection,
            checkpoint_selection_config=checkpoint_selection_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            enforce_benchmark_policy=args.benchmark_package is not None,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["ladder_under_profiles"] = ladder_profile_payload
        summary["behavior_evaluation"]["ladder_profile_comparison"] = (
            ladder_profile_payload
        )
        behavior_rows.extend(ladder_profile_rows)

    if learning_evidence_active:
        checkpoint_selection_run = checkpoint_selection_run or args.checkpoint_selection != "none"
        learning_evidence_payload, learning_evidence_rows = compare_learning_evidence(
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
            checkpoint_selection_config=checkpoint_selection_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            distillation_config=distillation_config,
            distillation_training_regime=distillation_training_regime,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["learning_evidence"] = learning_evidence_payload
        behavior_rows.extend(learning_evidence_rows)

    if claim_test_suite_active:
        checkpoint_selection_run = checkpoint_selection_run or args.checkpoint_selection != "none"
        claim_test_payload, claim_test_rows = run_claim_test_suite(
            claim_tests=args.claim_test,
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
            seeds=claim_test_seeds,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=args.checkpoint_selection,
            checkpoint_selection_config=checkpoint_selection_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            ablation_payload=ablation_payload,
            learning_evidence_payload=learning_evidence_payload,
        )
        _ensure_behavior_evaluation(summary)
        summary["behavior_evaluation"]["claim_tests"] = claim_test_payload
        behavior_rows.extend(claim_test_rows)

    if (
        args.curriculum_profile != "none"
        and not ablation_active
        and not learning_evidence_active
        and (budget.episodes > 0 or budget.eval_episodes > 0)
    ):
        compare_training_kwargs = _build_compare_training_kwargs(
            args,
            budget,
            checkpoint_selection_config,
        )
        curriculum_payload, curriculum_rows = compare_training_regimes(
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
            checkpoint_interval=budget.checkpoint_interval,
            selection_scenario_episodes=budget.selection_scenario_episodes,
            selection_config=checkpoint_selection_config,
        )

    if args.benchmark_package is not None:
        _build_and_record_benchmark_package(
            output_dir=args.benchmark_package,
            summary=summary,
            behavior_csv=args.behavior_csv,
            behavior_rows=behavior_rows,
            trace=trace,
            command_metadata={
                "argv": list(sys.argv[1:]),
                "entrypoint": "spider_cortex_sim",
            },
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

    print(
        json.dumps(
            format_run_summary(summary, full=args.full_summary),
            indent=2,
            ensure_ascii=False,
        )
    )
