from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class AnnealingSchedule(str, Enum):
    NONE = "none"
    LINEAR = "linear"
    COSINE = "cosine"


@dataclass(frozen=True)
class TrainingRegimeSpec:
    """Canonical training protocol with validated annealing and fine-tuning invariants."""

    name: str
    annealing_schedule: AnnealingSchedule | str
    anneal_target_scale: float
    anneal_warmup_fraction: float
    finetuning_episodes: int
    finetuning_reflex_scale: float
    loss_override_penalty_weight: float
    loss_dominance_penalty_weight: float

    def __post_init__(self) -> None:
        """Normalize fields and raise ValueError for invalid schedule/fine-tuning combinations."""
        name = str(self.name).strip()
        if not name:
            raise ValueError("Training regime name must be non-empty.")
        object.__setattr__(self, "name", name)

        try:
            schedule = AnnealingSchedule(self.annealing_schedule)
        except ValueError as exc:
            available = ", ".join(item.value for item in AnnealingSchedule)
            raise ValueError(
                f"Invalid annealing_schedule. Available schedules: {available}."
            ) from exc
        object.__setattr__(self, "annealing_schedule", schedule)

        target_scale = self._finite_non_negative(
            self.anneal_target_scale,
            "anneal_target_scale",
        )
        warmup_fraction = self._finite_non_negative(
            self.anneal_warmup_fraction,
            "anneal_warmup_fraction",
        )
        if warmup_fraction >= 1.0:
            raise ValueError("anneal_warmup_fraction must be less than 1.0.")
        raw_finetuning_episodes = self.finetuning_episodes
        if isinstance(raw_finetuning_episodes, bool):
            raise ValueError("finetuning_episodes must be an integer.")
        if isinstance(raw_finetuning_episodes, int):
            finetuning_episodes = raw_finetuning_episodes
        elif isinstance(raw_finetuning_episodes, float):
            if (
                not math.isfinite(raw_finetuning_episodes)
                or not raw_finetuning_episodes.is_integer()
            ):
                raise ValueError("finetuning_episodes must be an integer.")
            finetuning_episodes = int(raw_finetuning_episodes)
        else:
            raise ValueError("finetuning_episodes must be an integer.")
        if finetuning_episodes < 0:
            raise ValueError("finetuning_episodes must be non-negative.")
        finetuning_reflex_scale = self._finite_non_negative(
            self.finetuning_reflex_scale,
            "finetuning_reflex_scale",
        )
        override_penalty = self._finite_non_negative(
            self.loss_override_penalty_weight,
            "loss_override_penalty_weight",
        )
        dominance_penalty = self._finite_non_negative(
            self.loss_dominance_penalty_weight,
            "loss_dominance_penalty_weight",
        )

        if schedule is AnnealingSchedule.NONE and warmup_fraction != 0.0:
            raise ValueError(
                "anneal_warmup_fraction requires an active annealing schedule."
            )
        if finetuning_episodes > 0:
            if schedule is AnnealingSchedule.NONE:
                raise ValueError("finetuning requires an active annealing schedule.")
            if not math.isclose(
                target_scale,
                finetuning_reflex_scale,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "finetuning requires annealing to complete at the "
                    "finetuning_reflex_scale."
                )

        object.__setattr__(self, "anneal_target_scale", target_scale)
        object.__setattr__(self, "anneal_warmup_fraction", warmup_fraction)
        object.__setattr__(self, "finetuning_episodes", finetuning_episodes)
        object.__setattr__(self, "finetuning_reflex_scale", finetuning_reflex_scale)
        object.__setattr__(self, "loss_override_penalty_weight", override_penalty)
        object.__setattr__(self, "loss_dominance_penalty_weight", dominance_penalty)

    @staticmethod
    def _finite_non_negative(value: float, field_name: str) -> float:
        """
        Validate and convert a value to a finite, non-negative float.
        
        Parameters:
            value (float | int | str): A value that can be converted to float.
            field_name (str): Human-readable field name used in error messages.
        
        Returns:
            numeric (float): The validated float value, guaranteed to be finite and >= 0.0.
        
        Raises:
            ValueError: If the value is not numeric, not finite, or negative.
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a number.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite.")
        if numeric < 0.0:
            raise ValueError(f"{field_name} must be non-negative.")
        return numeric

    def to_summary(self) -> dict[str, object]:
        """
        Return a primitive-typed summary of the training regime suitable for serialization.
        
        The returned dictionary contains the regime's canonical values converted to basic Python types:
        - "name": regime name string
        - "annealing_schedule": schedule string value ("none", "linear", or "cosine")
        - "anneal_target_scale": float
        - "anneal_warmup_fraction": float
        - "finetuning_episodes": int
        - "finetuning_reflex_scale": float
        - "loss_override_penalty_weight": float
        - "loss_dominance_penalty_weight": float
        
        Returns:
            dict[str, object]: A mapping of field names to primitive values for JSON-like serialization.
        """
        return {
            "name": self.name,
            "annealing_schedule": self.annealing_schedule.value,
            "anneal_target_scale": float(self.anneal_target_scale),
            "anneal_warmup_fraction": float(self.anneal_warmup_fraction),
            "finetuning_episodes": int(self.finetuning_episodes),
            "finetuning_reflex_scale": float(self.finetuning_reflex_scale),
            "loss_override_penalty_weight": float(
                self.loss_override_penalty_weight
            ),
            "loss_dominance_penalty_weight": float(
                self.loss_dominance_penalty_weight
            ),
        }


TRAINING_REGIMES: dict[str, TrainingRegimeSpec] = {
    "baseline": TrainingRegimeSpec(
        name="baseline",
        annealing_schedule=AnnealingSchedule.NONE,
        anneal_target_scale=1.0,
        anneal_warmup_fraction=0.0,
        finetuning_episodes=0,
        finetuning_reflex_scale=0.0,
        loss_override_penalty_weight=0.0,
        loss_dominance_penalty_weight=0.0,
    ),
    "reflex_annealed": TrainingRegimeSpec(
        name="reflex_annealed",
        annealing_schedule=AnnealingSchedule.LINEAR,
        anneal_target_scale=0.0,
        anneal_warmup_fraction=0.0,
        finetuning_episodes=0,
        finetuning_reflex_scale=0.0,
        loss_override_penalty_weight=0.0,
        loss_dominance_penalty_weight=0.0,
    ),
    "late_finetuning": TrainingRegimeSpec(
        name="late_finetuning",
        annealing_schedule=AnnealingSchedule.LINEAR,
        anneal_target_scale=0.0,
        anneal_warmup_fraction=0.0,
        finetuning_episodes=10,
        finetuning_reflex_scale=0.0,
        loss_override_penalty_weight=0.0,
        loss_dominance_penalty_weight=0.0,
    ),
    "distillation": TrainingRegimeSpec(
        name="distillation",
        annealing_schedule=AnnealingSchedule.LINEAR,
        anneal_target_scale=0.0,
        anneal_warmup_fraction=0.0,
        finetuning_episodes=0,
        finetuning_reflex_scale=0.0,
        loss_override_penalty_weight=0.0,
        loss_dominance_penalty_weight=0.0,
    ),
}

_UNIMPLEMENTED_TRAINING_REGIME_NAMES = frozenset({"distillation"})


def canonical_training_regime_names() -> tuple[str, ...]:
    """
    Return implemented training regime names in their canonical declaration order.
    """
    return tuple(
        name
        for name in TRAINING_REGIMES
        if name not in _UNIMPLEMENTED_TRAINING_REGIME_NAMES
    )


def resolve_training_regime(name: str) -> TrainingRegimeSpec:
    """Return a canonical regime spec, reserving distillation and rejecting unknown names."""
    regime_name = str(name)
    if regime_name in _UNIMPLEMENTED_TRAINING_REGIME_NAMES:
        raise NotImplementedError(
            "The 'distillation' training regime is reserved for scaffolded-policy "
            "to no-reflex-policy distillation and is not implemented yet."
        )
    try:
        return TRAINING_REGIMES[regime_name]
    except KeyError as exc:
        available = ", ".join(repr(item) for item in canonical_training_regime_names())
        raise ValueError(
            f"Invalid training regime {regime_name!r}. Available regimes: {available}."
        ) from exc
