from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, TypeAlias

if TYPE_CHECKING:
    from .world import SpiderWorld


# Tick stages are side-effectful pipeline steps. They must:
# 1. Read and write state only through the provided world and context.
# 2. Record runtime events through context.record_event().
# 3. Communicate outcomes via mutation rather than return values.
Stage: TypeAlias = Callable[["SpiderWorld", "TickContext"], None]


@dataclass
class MemorySlot:
    target: tuple[int, int] | None
    age: int


@dataclass
class PerceptTrace:
    target: tuple[int, int] | None
    age: int
    certainty: float


@dataclass
class SpiderState:
    """Physical, physiological, and perception-derived state for one spider.

    Position, body metrics, counters, heading, and last movement describe the
    spider's physical and physiological state. The four MemorySlot fields are
    perception-derived cognitive state: they store TTL-bounded targets sourced
    from local perception, contact events, or movement history.
    """

    # Physical position and body state.
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

    # Episode bookkeeping and learned-behavior counters.
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
    heading_dx: int
    heading_dy: int

    # Perception-derived cognitive state and short percept traces.
    food_memory: MemorySlot
    predator_memory: MemorySlot
    shelter_memory: MemorySlot
    escape_memory: MemorySlot
    food_trace: PerceptTrace
    shelter_trace: PerceptTrace
    predator_trace: PerceptTrace


@dataclass(frozen=True)
class TickSnapshot:
    tick: int
    spider_pos: tuple[int, int]
    lizard_pos: tuple[int, int]
    was_on_shelter: bool
    prev_shelter_role: str
    prev_food_dist: int
    prev_shelter_dist: int
    prev_predator_dist: int
    prev_predator_visible: bool
    night: bool
    rest_streak: int

    def to_payload(self) -> dict[str, object]:
        """
        Return a JSON-serializable dictionary representing the snapshot's fields.
        
        The dictionary contains primitive representations of the snapshot suitable for JSON encoding: integer ticks and distances, two-element integer lists for positions, and booleans for flag fields.
        
        Returns:
            payload (dict[str, object]): Mapping with keys:
                - "tick": integer tick index
                - "spider_pos": [int x, int y] spider position
                - "lizard_pos": [int x, int y] lizard position
                - "was_on_shelter": `true` if the spider was on shelter, `false` otherwise
                - "prev_shelter_role": previous shelter role string
                - "prev_food_dist": integer distance to food from previous tick
                - "prev_shelter_dist": integer distance to shelter from previous tick
                - "prev_predator_dist": integer distance to predator from previous tick
                - "prev_predator_visible": `true` if predator was visible in previous tick, `false` otherwise
                - "night": `true` if it was night, `false` otherwise
                - "rest_streak": integer rest streak count
        """
        return {
            "tick": int(self.tick),
            "spider_pos": [int(self.spider_pos[0]), int(self.spider_pos[1])],
            "lizard_pos": [int(self.lizard_pos[0]), int(self.lizard_pos[1])],
            "was_on_shelter": bool(self.was_on_shelter),
            "prev_shelter_role": self.prev_shelter_role,
            "prev_food_dist": int(self.prev_food_dist),
            "prev_shelter_dist": int(self.prev_shelter_dist),
            "prev_predator_dist": int(self.prev_predator_dist),
            "prev_predator_visible": bool(self.prev_predator_visible),
            "night": bool(self.night),
            "rest_streak": int(self.rest_streak),
        }


@dataclass(frozen=True)
class TickEvent:
    stage: str
    name: str
    payload: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        """
        Serialize the TickEvent into a JSON-serializable dictionary.
        
        Returns:
            dict[str, object]: A dictionary with keys "stage" (event stage), "name" (event name),
            and "payload" (a shallow copy of the event payload).
        """
        return {
            "stage": self.stage,
            "name": self.name,
            "payload": dict(self.payload),
        }


@dataclass
class TickContext:
    action_idx: int
    intended_action: str
    executed_action: str
    motor_noise_applied: bool
    snapshot: TickSnapshot
    reward_components: dict[str, float]
    execution_difficulty: float = 0.0
    execution_components: dict[str, float] = field(default_factory=dict)
    motor_slip_info: dict[str, object] = field(default_factory=dict)
    info: dict[str, object] = field(default_factory=dict)
    event_log: list[TickEvent] = field(default_factory=list)
    moved: bool = False
    terrain_now: str = ""
    predator_threat: bool = False
    interrupted_rest: bool = False
    exposed_at_night: bool = False
    predator_moved: bool = False
    predator_escape: bool = False
    predator_visible_now: bool = False
    predator_contact_applied: bool = False
    fed_this_tick: bool = False
    done: bool = False
    reward: float = 0.0

    def record_event(self, stage: str, name: str, **payload: object) -> None:
        """
        Record an event in the tick's event log.
        
        Parameters:
            stage (str): Phase or category for the event (e.g., "pre_step", "post_step").
            name (str): Short identifier for the event type.
            **payload (object): Additional key/value information attached to the event; keys and values are copied into the stored event's payload.
        """
        self.event_log.append(TickEvent(stage=stage, name=name, payload=dict(payload)))

    def serialized_event_log(self) -> list[dict[str, object]]:
        """
        Produce a JSON-serializable list of payload dictionaries for the tick's recorded events.
        
        Returns:
            list[dict[str, object]]: List of event payload dictionaries produced by each recorded TickEvent's `to_payload()` method.
        """
        return [event.to_payload() for event in self.event_log]


@dataclass
class StageDescriptor:
    name: str
    run: Stage
    mutates: tuple[str, ...]
