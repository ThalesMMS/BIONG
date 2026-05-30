from __future__ import annotations

from .learning_shared import *
from .learning_continuation import _BrainLearningContinuationMixin
from .learning_distillation import _BrainLearningDistillationMixin
from .learning_learn import _BrainLearningLearnMixin


class BrainLearningMixin(_BrainLearningContinuationMixin, _BrainLearningDistillationMixin, _BrainLearningLearnMixin):
    CONTINUATION_MARGIN = 1.0
    POST_REST_SEQUENCE_REPLAY_PASSES = 6
    POST_REST_SEQUENCE_REPLAY_LR_SCALE = 1.0
