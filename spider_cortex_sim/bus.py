from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class Message:
    tick: int
    sender: str
    topic: str
    payload: Dict[str, Any]


class MessageBus:
    def __init__(self) -> None:
        self.tick = 0
        self._messages: List[Message] = []
        self._history: List[Message] = []

    def set_tick(self, tick: int) -> None:
        self.tick = tick
        self._messages = []

    def publish(self, sender: str, topic: str, payload: Dict[str, Any]) -> None:
        message = Message(tick=self.tick, sender=sender, topic=topic, payload=payload)
        self._messages.append(message)
        self._history.append(message)

    def messages(self) -> List[Message]:
        return list(self._messages)

    def history(self) -> List[Message]:
        return list(self._history)

    def topic_messages(self, topic: str) -> List[Message]:
        return [m for m in self._messages if m.topic == topic]

    def serialize_current_tick(self) -> List[Dict[str, Any]]:
        return [asdict(m) for m in self._messages]
