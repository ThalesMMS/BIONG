from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np


def _copy_observation_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).copy()
    if isinstance(value, Mapping):
        return {
            str(key): _copy_observation_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_observation_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_observation_value(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True)
class DistillationLossConfig:
    final_policy_weight: float = 1.0
    action_center_weight: float = 0.5
    valence_weight: float = 0.25
    proposal_weight: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "final_policy_weight",
            "action_center_weight",
            "valence_weight",
            "proposal_weight",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
            object.__setattr__(self, field_name, value)

    def to_summary(self) -> dict[str, float]:
        return {
            "final_policy_weight": float(self.final_policy_weight),
            "action_center_weight": float(self.action_center_weight),
            "valence_weight": float(self.valence_weight),
            "proposal_weight": float(self.proposal_weight),
        }


@dataclass(frozen=True)
class DistillationConfig:
    teacher_checkpoint: str | Path
    dataset_episodes: int | None = None
    distillation_epochs: int = 1
    temperature: float = 1.0
    module_lr: float | None = None
    motor_lr: float | None = None
    arbitration_lr: float | None = None
    shuffle: bool = False
    match_local_proposals: bool = False
    loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self) -> None:
        teacher_checkpoint = str(self.teacher_checkpoint)
        if not teacher_checkpoint.strip():
            raise ValueError("teacher_checkpoint must be a non-empty path.")
        object.__setattr__(self, "teacher_checkpoint", teacher_checkpoint)
        if self.dataset_episodes is not None and int(self.dataset_episodes) < 0:
            raise ValueError("dataset_episodes must be non-negative when provided.")
        object.__setattr__(
            self,
            "dataset_episodes",
            None if self.dataset_episodes is None else int(self.dataset_episodes),
        )
        if int(self.distillation_epochs) <= 0:
            raise ValueError("distillation_epochs must be positive.")
        object.__setattr__(
            self,
            "distillation_epochs",
            int(self.distillation_epochs),
        )
        temperature = float(self.temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        object.__setattr__(self, "temperature", temperature)
        for field_name in ("module_lr", "motor_lr", "arbitration_lr"):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = float(value)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive when provided.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "shuffle", bool(self.shuffle))
        object.__setattr__(
            self,
            "match_local_proposals",
            bool(self.match_local_proposals),
        )

    def to_summary(self) -> dict[str, object]:
        return {
            "teacher_checkpoint": str(self.teacher_checkpoint),
            "dataset_episodes": self.dataset_episodes,
            "distillation_epochs": int(self.distillation_epochs),
            "temperature": float(self.temperature),
            "module_lr": self.module_lr,
            "motor_lr": self.motor_lr,
            "arbitration_lr": self.arbitration_lr,
            "shuffle": bool(self.shuffle),
            "match_local_proposals": bool(self.match_local_proposals),
            "loss": self.loss.to_summary(),
        }


@dataclass
class DistillationSample:
    episode: int
    step: int
    observation: Dict[str, object]
    teacher_policy: np.ndarray
    teacher_total_logits: np.ndarray
    teacher_action_center_policy: np.ndarray
    teacher_action_center_logits: np.ndarray
    teacher_action_intent_idx: int
    teacher_valence_probs: np.ndarray
    teacher_valence_logits: np.ndarray
    teacher_module_policies: Dict[str, np.ndarray] = field(default_factory=dict)


class DistillationDataset:
    def __init__(
        self,
        *,
        teacher_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._samples: List[DistillationSample] = []
        self.teacher_metadata = (
            {
                str(key): deepcopy(value)
                for key, value in teacher_metadata.items()
            }
            if teacher_metadata is not None
            else {}
        )

    def add_sample(
        self,
        *,
        episode: int,
        step: int,
        observation: Mapping[str, object],
        teacher_policy: Sequence[float],
        teacher_total_logits: Sequence[float],
        teacher_action_center_policy: Sequence[float] | None = None,
        teacher_action_center_logits: Sequence[float] | None = None,
        teacher_action_intent_idx: int = 0,
        teacher_valence_probs: Sequence[float] | None = None,
        teacher_valence_logits: Sequence[float] | None = None,
        teacher_module_policies: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._samples.append(
            DistillationSample(
                episode=int(episode),
                step=int(step),
                observation={
                    str(key): _copy_observation_value(value)
                    for key, value in observation.items()
                },
                teacher_policy=np.asarray(teacher_policy, dtype=float).copy(),
                teacher_total_logits=np.asarray(
                    teacher_total_logits,
                    dtype=float,
                ).copy(),
                teacher_action_center_policy=np.asarray(
                    teacher_action_center_policy
                    if teacher_action_center_policy is not None
                    else teacher_policy,
                    dtype=float,
                ).copy(),
                teacher_action_center_logits=np.asarray(
                    teacher_action_center_logits
                    if teacher_action_center_logits is not None
                    else teacher_total_logits,
                    dtype=float,
                ).copy(),
                teacher_action_intent_idx=int(teacher_action_intent_idx),
                teacher_valence_probs=np.asarray(
                    teacher_valence_probs if teacher_valence_probs is not None else (),
                    dtype=float,
                ).copy(),
                teacher_valence_logits=np.asarray(
                    teacher_valence_logits if teacher_valence_logits is not None else (),
                    dtype=float,
                ).copy(),
                teacher_module_policies={
                    str(name): np.asarray(policy, dtype=float).copy()
                    for name, policy in (
                        teacher_module_policies.items()
                        if teacher_module_policies is not None
                        else ()
                    )
                },
            )
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[DistillationSample]:
        return iter(self._samples)

    def iter_samples(
        self,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterable[DistillationSample]:
        if not shuffle or len(self._samples) <= 1:
            return iter(self._samples)
        generator = rng if rng is not None else np.random.default_rng()
        order = generator.permutation(len(self._samples))
        return (self._samples[int(index)] for index in order)

    def to_summary(self) -> dict[str, object]:
        episode_ids = sorted({sample.episode for sample in self._samples})
        proposal_target_count = sum(
            1 for sample in self._samples if sample.teacher_module_policies
        )
        return {
            "episodes": len(episode_ids),
            "episode_ids": episode_ids,
            "samples": len(self._samples),
            "teacher_metadata": deepcopy(self.teacher_metadata),
            "targets": {
                "final_policy": len(self._samples),
                "action_center": len(self._samples),
                "valence": sum(
                    1
                    for sample in self._samples
                    if sample.teacher_valence_probs.size > 0
                ),
                "local_proposals": proposal_target_count,
            },
        }
