from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemorySlot:
    target: tuple[int, int] | None
    age: int


@dataclass
class SpiderState:
    x: int
    y: int
    hunger: float
    fatigue: float
    sleep_debt: float
    health: float
    recent_pain: float
    recent_contact: float
    sleep_phase: str
    rest_streak: int
    last_reward: float
    total_reward: float
    food_eaten: int
    sleep_events: int
    shelter_entries: int
    alert_events: int
    predator_contacts: int
    predator_sightings: int
    predator_escapes: int
    steps_alive: int
    last_action: str
    last_move_dx: int
    last_move_dy: int
    food_memory: MemorySlot
    predator_memory: MemorySlot
    shelter_memory: MemorySlot
    escape_memory: MemorySlot
