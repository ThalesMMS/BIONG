from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..direct_policy_affordances import (
    AFFORDANCE_GEOMETRY_TARGET_NAMES,
    AFFORDANCE_SHELTER_COLUMN_NAMES,
    AFFORDANCE_SHELTER_POSITION_NAMES,
    AFFORDANCE_SHELTER_ROLE_NAMES,
)
from ..direct_policy_options import OPTION_NAMES, OPTION_TO_INDEX
from ..b_series import B_SERIES_POLICY_NAME, B_SEMANTIC_ACTIONS
from ..distillation.dataset import (
    DistillationConfig,
    DistillationDataset,
    DistillationSample,
    DistillationLossConfig,
)
from ..interfaces import ACTION_CONTEXT_INTERFACE, ACTION_TO_INDEX
from ..modules import ModuleResult
from ..phase import PHASE_LABELS, PHASE_TO_INDEX
from ..nn import one_hot, softmax
from ..nn_utils import cross_entropy_loss, kl_divergence

from .types import BrainStep
