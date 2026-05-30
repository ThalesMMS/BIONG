from __future__ import annotations

from .runtime_shared import *
from .runtime_act_payload import _BrainRuntimeActPayloadMixin
from .runtime_part1 import _BrainRuntimePart1Mixin
from .runtime_part2 import _BrainRuntimePart2Mixin
from .runtime_part3 import _BrainRuntimePart3Mixin
from .runtime_part4 import _BrainRuntimePart4Mixin
from .runtime_part5 import _BrainRuntimePart5Mixin
from .runtime_part6 import _BrainRuntimePart6Mixin
from .runtime_part7 import _BrainRuntimePart7Mixin
from .runtime_part8 import _BrainRuntimePart8Mixin
from .runtime_part9 import _BrainRuntimePart9Mixin
from .runtime_part10 import _BrainRuntimePart10Mixin
from .runtime_part11 import _BrainRuntimePart11Mixin


class BrainRuntimeMixin(_BrainRuntimeActPayloadMixin, _BrainRuntimePart1Mixin, _BrainRuntimePart2Mixin, _BrainRuntimePart3Mixin, _BrainRuntimePart4Mixin, _BrainRuntimePart5Mixin, _BrainRuntimePart6Mixin, _BrainRuntimePart7Mixin, _BrainRuntimePart8Mixin, _BrainRuntimePart9Mixin, _BrainRuntimePart10Mixin, _BrainRuntimePart11Mixin):
    pass
