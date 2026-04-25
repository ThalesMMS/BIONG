from __future__ import annotations

from .budget import SimulationRuntimeBudgetMixin
from .distillation import SimulationDistillationMixin
from .histories import SimulationTrainingHistoriesMixin
from .checkpoint_selection import SimulationCheckpointSelectionTrainingMixin
from .loop import SimulationTrainingLoopMixin


class SimulationTrainingMixin(
    SimulationTrainingLoopMixin,
    SimulationCheckpointSelectionTrainingMixin,
    SimulationTrainingHistoriesMixin,
    SimulationDistillationMixin,
    SimulationRuntimeBudgetMixin,
):
    pass


__all__ = ["SimulationTrainingMixin"]
