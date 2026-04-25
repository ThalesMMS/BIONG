from __future__ import annotations

import argparse

from ..budget_profiles import ResolvedBudget
from ..checkpointing import CheckpointSelectionConfig
from .validation import _argument_error


def _build_compare_training_kwargs(
    args: argparse.Namespace,
    budget: ResolvedBudget,
    checkpoint_selection_config: CheckpointSelectionConfig,
) -> dict[str, object]:
    """
    Build a dictionary of training and comparison configuration values from CLI args, a resolved budget, and a checkpoint selection config.
    
    Parameters:
        args (argparse.Namespace): Parsed CLI arguments providing model, environment, and checkpoint options (e.g., width, height, learning rates, profiles, checkpoint_dir).
        budget (ResolvedBudget): Resolved budget values providing counts and intervals (e.g., max_steps, episodes, eval_episodes, behavior_seeds, scenario_episodes, checkpoint_interval).
        checkpoint_selection_config (CheckpointSelectionConfig): Configuration object determining checkpoint selection behavior; included verbatim under the 'checkpoint_selection_config' key.
    
    Returns:
        dict[str, object]: A mapping of training/configuration keys to their values derived from the inputs. The 'seeds' entry is the budget.behavior_seeds converted to a tuple.
    """
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
        "checkpoint_selection_config": checkpoint_selection_config,
        "checkpoint_interval": budget.checkpoint_interval,
        "checkpoint_dir": args.checkpoint_dir,
        "curriculum_profile": args.curriculum_profile,
    }


def _build_checkpoint_selection_config(
    args: argparse.Namespace,
) -> CheckpointSelectionConfig:
    """
    Create a CheckpointSelectionConfig from checkpoint-related CLI arguments.
    
    Parameters:
        args (argparse.Namespace): Namespace containing checkpoint configuration fields:
            - checkpoint_metric: metric name used to select checkpoints
            - checkpoint_override_penalty: numeric weight for override penalty
            - checkpoint_dominance_penalty: numeric weight for dominance penalty
            - checkpoint_penalty_mode: mode identifier for applying penalties
    
    Returns:
        CheckpointSelectionConfig: Configured checkpoint selection policy.
    
    Raises:
        AssertionError: If the provided arguments are invalid; an argument error is reported via `_argument_error` before this is raised.
    """
    try:
        return CheckpointSelectionConfig(
            metric=args.checkpoint_metric,
            override_penalty_weight=args.checkpoint_override_penalty,
            dominance_penalty_weight=args.checkpoint_dominance_penalty,
            penalty_mode=args.checkpoint_penalty_mode,
        )
    except ValueError as exc:
        _argument_error(args, str(exc))
        raise AssertionError("unreachable") from exc
