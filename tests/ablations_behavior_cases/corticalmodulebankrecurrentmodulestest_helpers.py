from __future__ import annotations

from .shared import *


class CorticalModuleBankRecurrentModulesTestHelpers:
    def _make_bank(self, recurrent: tuple[str, ...] = ()) -> CorticalModuleBank:
        rng = np.random.default_rng(44)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
            recurrent_modules=recurrent,
        )
