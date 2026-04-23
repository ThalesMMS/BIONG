from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Sequence

from .training_regimes import TrainingRegimeSpec, resolve_training_regime


@dataclass(frozen=True)
class LearningEvidenceConditionSpec:
    name: str
    description: str
    policy_mode: str = "normal"
    train_budget: str = "base"
    training_regime: str | TrainingRegimeSpec | None = None
    eval_reflex_scale: float | None = None
    checkpoint_source: str = "final"
    supports_architectures: tuple[str, ...] = ("modular", "monolithic")


def canonical_learning_evidence_conditions() -> Dict[str, LearningEvidenceConditionSpec]:
    """
    Canonical registry of learning-evidence condition specifications.
    
    Each key is a canonical condition name and each value is a corresponding
    LearningEvidenceConditionSpec describing policy mode, training budget,
    optional training_regime, evaluation reflex scaling, checkpoint source, and
    supported architectures. The registry includes the canonical conditions used
    by the system: "trained_final", "trained_without_reflex_support",
    "trained_reflex_annealed", "trained_late_finetuning", "random_init",
    "reflex_only", "freeze_half_budget", and "trained_long_budget".
    
    Returns:
        Dict[str, LearningEvidenceConditionSpec]: Mapping from canonical condition
            name to its LearningEvidenceConditionSpec.
    """
    distillation_regime = resolve_training_regime("distillation")
    distilled_only_regime = replace(
        distillation_regime,
        finetuning_episodes=0,
    )
    return {
        "trained_final": LearningEvidenceConditionSpec(
            name="trained_final",
            description="Final checkpoint trained with the base budget and evaluated under the default runtime configuration as a secondary diagnostic.",
            train_budget="base",
            checkpoint_source="final",
        ),
        "trained_without_reflex_support": LearningEvidenceConditionSpec(
            name="trained_without_reflex_support",
            description="Primary learned-behavior condition: same trained checkpoint evaluated with reflex support disabled.",
            train_budget="base",
            eval_reflex_scale=0.0,
            checkpoint_source="final",
        ),
        "trained_reflex_annealed": LearningEvidenceConditionSpec(
            name="trained_reflex_annealed",
            description="Named reflex-annealed training regime evaluated as a self-sufficient no-reflex benchmark.",
            train_budget="base",
            training_regime="reflex_annealed",
            eval_reflex_scale=0.0,
            checkpoint_source="final",
        ),
        "trained_late_finetuning": LearningEvidenceConditionSpec(
            name="trained_late_finetuning",
            description="Named late no-reflex fine-tuning regime evaluated as the experiment-of-record self-sufficient benchmark.",
            train_budget="base",
            training_regime="late_finetuning",
            eval_reflex_scale=0.0,
            checkpoint_source="final",
        ),
        "trained_distilled": LearningEvidenceConditionSpec(
            name="trained_distilled",
            description="Modular student trained with offline distillation only, evaluated without reflex support.",
            train_budget="base",
            training_regime=distilled_only_regime,
            eval_reflex_scale=0.0,
            checkpoint_source="final",
            supports_architectures=("modular",),
        ),
        "trained_distilled_finetuned": LearningEvidenceConditionSpec(
            name="trained_distilled_finetuned",
            description="Modular student trained with offline distillation followed by no-reflex RL fine-tuning, evaluated without reflex support.",
            train_budget="base",
            training_regime=distillation_regime,
            eval_reflex_scale=0.0,
            checkpoint_source="final",
            supports_architectures=("modular",),
        ),
        "random_init": LearningEvidenceConditionSpec(
            name="random_init",
            description="Same architecture and configuration, but untrained and evaluated from random initialization.",
            train_budget="none",
            checkpoint_source="initial",
        ),
        "reflex_only": LearningEvidenceConditionSpec(
            name="reflex_only",
            description="Untrained modular architecture running only the local reflex path.",
            policy_mode="reflex_only",
            train_budget="none",
            checkpoint_source="initial",
            supports_architectures=("modular",),
        ),
        "freeze_half_budget": LearningEvidenceConditionSpec(
            name="freeze_half_budget",
            description="Trains for half the budget, freezes weights, and consumes the rest without learning.",
            train_budget="freeze_half",
            checkpoint_source="frozen_half_budget",
        ),
        "trained_long_budget": LearningEvidenceConditionSpec(
            name="trained_long_budget",
            description="Same configuration, but trained with the long-profile budget.",
            train_budget="long",
            checkpoint_source="final",
        ),
    }


def canonical_learning_evidence_condition_names() -> tuple[str, ...]:
    """
    List the canonical learning-evidence condition names in registry order.
    
    Returns:
    	tuple[str, ...]: Tuple of condition name strings in the order they appear in the canonical registry.
    """
    return tuple(canonical_learning_evidence_conditions().keys())


def resolve_learning_evidence_conditions(
    names: Sequence[str] | None,
) -> list[LearningEvidenceConditionSpec]:
    """
    Resolve a sequence of canonical learning-evidence condition names into their spec objects.
    
    Parameters:
        names (Sequence[str] | None): Sequence of canonical condition names to resolve, or
            `None` to select all canonical conditions in their canonical order.
    
    Returns:
        list[LearningEvidenceConditionSpec]: The corresponding `LearningEvidenceConditionSpec`
        objects ordered to match `names` (or the canonical ordering when `names` is `None`).
    
    Raises:
        ValueError: If any requested name is not a recognized canonical condition.
    """
    registry = canonical_learning_evidence_conditions()
    requested_names = (
        list(canonical_learning_evidence_condition_names())
        if names is None
        else [str(name) for name in names]
    )
    missing = [name for name in requested_names if name not in registry]
    if missing:
        raise ValueError(
            f"Invalid learning_evidence conditions: {missing}."
        )
    return [registry[name] for name in requested_names]
