from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass(frozen=True)
class LearningEvidenceConditionSpec:
    name: str
    description: str
    policy_mode: str = "normal"
    train_budget: str = "base"
    eval_reflex_scale: float | None = None
    checkpoint_source: str = "final"
    supports_architectures: tuple[str, ...] = ("modular", "monolithic")


def canonical_learning_evidence_conditions() -> Dict[str, LearningEvidenceConditionSpec]:
    """
    Return the canonical registry of learning-evidence condition specifications.
    
    Each mapping key is a canonical condition name and each value is a
    LearningEvidenceConditionSpec describing training/evaluation configuration
    (policy mode, training budget, evaluation reflex scaling, checkpoint source,
    and supported architectures). The registry includes the canonical conditions
    used by the system such as "trained_final", "trained_without_reflex_support",
    "random_init", "reflex_only", "freeze_half_budget", and "trained_long_budget".
    
    Returns:
        conditions (Dict[str, LearningEvidenceConditionSpec]): Mapping from canonical
        condition name to its specification.
    """
    return {
        "trained_final": LearningEvidenceConditionSpec(
            name="trained_final",
            description="Checkpoint final treinado com o orçamento base e avaliação normal.",
            train_budget="base",
            checkpoint_source="final",
        ),
        "trained_without_reflex_support": LearningEvidenceConditionSpec(
            name="trained_without_reflex_support",
            description="Mesmo checkpoint treinado, mas avaliado com suporte reflexo desligado.",
            train_budget="base",
            eval_reflex_scale=0.0,
            checkpoint_source="final",
        ),
        "random_init": LearningEvidenceConditionSpec(
            name="random_init",
            description="Mesma arquitetura/configuração, sem treino, avaliada a partir da inicialização aleatória.",
            train_budget="none",
            checkpoint_source="initial",
        ),
        "reflex_only": LearningEvidenceConditionSpec(
            name="reflex_only",
            description="Arquitetura modular sem treino, executando apenas o caminho reflexo local.",
            policy_mode="reflex_only",
            train_budget="none",
            checkpoint_source="initial",
            supports_architectures=("modular",),
        ),
        "freeze_half_budget": LearningEvidenceConditionSpec(
            name="freeze_half_budget",
            description="Treina metade do orçamento, congela pesos e consome o restante sem aprendizado.",
            train_budget="freeze_half",
            checkpoint_source="frozen_half_budget",
        ),
        "trained_long_budget": LearningEvidenceConditionSpec(
            name="trained_long_budget",
            description="Mesma configuração, mas treinada com o orçamento do perfil longo.",
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
            f"Condições de learning_evidence inválidas: {missing}."
        )
    return [registry[name] for name in requested_names]
