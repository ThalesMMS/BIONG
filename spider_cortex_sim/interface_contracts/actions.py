from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Sequence, Tuple

LOCOMOTION_ACTIONS: Sequence[str] = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "STAY",
    "ORIENT_UP",
    "ORIENT_DOWN",
    "ORIENT_LEFT",
    "ORIENT_RIGHT",
)
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(LOCOMOTION_ACTIONS)}
class _ActionDeltas(dict[str, Tuple[int, int]]):
    def __iter__(self):
        return (
            action
            for action in LOCOMOTION_ACTIONS
            if dict.__contains__(self, action)
        )


ACTION_DELTAS: Dict[str, Tuple[int, int]] = _ActionDeltas({
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
    "STAY": (0, 0),
})


class _OrientHeadings(tuple[str, ...]):
    def __new__(cls, mapping: Mapping[str, Tuple[int, int]]) -> "_OrientHeadings":
        instance = super().__new__(cls, tuple(mapping))
        instance._mapping = dict(mapping)
        return instance

    def __getitem__(self, key: object) -> object:
        if isinstance(key, str):
            return self._mapping[key]
        return super().__getitem__(key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self._mapping == dict(other)
        return super().__eq__(other)

    def get(
        self,
        key: str,
        default: Tuple[int, int] | None = None,
    ) -> Tuple[int, int] | None:
        return self._mapping.get(key, default)

    def items(self):
        return self._mapping.items()

    def values(self):
        return self._mapping.values()


ORIENT_HEADINGS: tuple[str, ...] = _OrientHeadings({
    "ORIENT_UP": (0, -1),
    "ORIENT_DOWN": (0, 1),
    "ORIENT_LEFT": (-1, 0),
    "ORIENT_RIGHT": (1, 0),
})

__all__ = [
    "ACTION_DELTAS",
    "ACTION_TO_INDEX",
    "LOCOMOTION_ACTIONS",
    "ORIENT_HEADINGS",
]
