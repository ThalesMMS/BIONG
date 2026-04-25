from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .interfaces import ACTION_CONTEXT_INTERFACE, MODULE_INTERFACES, MOTOR_CONTEXT_INTERFACE


def _clamp01(value: float) -> float:
    """
    Clamp a numeric value into the inclusive range [0.0, 1.0].
    
    Returns:
        A float equal to the input value limited to the range 0.0 through 1.0.
    """
    return float(min(1.0, max(0.0, value)))


@dataclass(frozen=True)
class ObservationAdapterPayload:
    interface_name: str
    observation_key: str
    vector: np.ndarray
    signal_names: tuple[str, ...]
    tick: int
    source: str
    freshness_by_signal: dict[str, float]

    def to_summary(self) -> dict[str, object]:
        """
        Create a trace-friendly summary of the payload suitable for logging or serialization.
        
        Returns:
            dict[str, object]: A mapping containing:
                - "interface_name" (str): the interface that produced the observation.
                - "observation_key" (str): the original observation key.
                - "signal_names" (list[str]): list of signal names in the payload.
                - "dim" (int): dimensionality of the stored vector.
                - "tick" (int): the recorded tick value.
                - "source" (str): origin identifier for the observation.
                - "freshness_by_signal" (dict[str, float]): freshness scores by signal name; keys are sorted and values rounded to 6 decimal places.
        """
        return {
            "interface_name": self.interface_name,
            "observation_key": self.observation_key,
            "signal_names": list(self.signal_names),
            "dim": int(self.vector.shape[0]),
            "tick": int(self.tick),
            "source": self.source,
            "freshness_by_signal": {
                name: round(float(value), 6)
                for name, value in sorted(self.freshness_by_signal.items())
            },
        }


def _freshness_by_signal(
    signal_values: Mapping[str, float],
) -> dict[str, float]:
    """
    Compute per-signal freshness scores based on signal name suffix conventions.
    
    Parameters:
        signal_values (Mapping[str, float]): Mapping from signal name to a numeric value used to derive freshness.
    
    Returns:
        dict[str, float]: Mapping from each signal name to its freshness score (a float clamped to the range 0.0 to 1.0). Rules:
            - If the name ends with "_age": freshness = clamp(1.0 - value).
            - If the name ends with "_trace_strength" or "_certainty": freshness = clamp(value).
            - Otherwise: freshness = 1.0.
    """
    freshness: dict[str, float] = {}
    for name, value in signal_values.items():
        numeric = float(value)
        if name.endswith("_age"):
            freshness[name] = _clamp01(1.0 - numeric)
        elif name.endswith("_trace_strength") or name.endswith("_certainty"):
            freshness[name] = _clamp01(numeric)
        else:
            freshness[name] = 1.0
    return freshness


def adapt_observation_contracts(
    observation: Mapping[str, object],
    *,
    tick: int,
    source: str = "world.observe",
) -> dict[str, ObservationAdapterPayload]:
    """
    Builds ObservationAdapterPayload objects for each required interface from a raw observation mapping.
    
    Parameters:
        observation (Mapping[str, object]): Raw observation mapping keyed by interface observation keys.
        tick (int): Integer tick or timestamp associated with this observation.
        source (str): Origin identifier for the observation (default "world.observe").
    
    Returns:
        dict[str, ObservationAdapterPayload]: Mapping from each interface's observation_key to its payload.
    
    Raises:
        KeyError: If any required interface.observation_key is missing from `observation`.
    """
    adapters: dict[str, ObservationAdapterPayload] = {}
    for interface in (*MODULE_INTERFACES, ACTION_CONTEXT_INTERFACE, MOTOR_CONTEXT_INTERFACE):
        if interface.observation_key not in observation:
            raise KeyError(
                f"Observation missing key {interface.observation_key!r} "
                f"for interface {interface.name!r}."
            )
        vector = np.nan_to_num(
            np.asarray(observation[interface.observation_key], dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        signal_values = interface.bind_values(vector)
        adapters[interface.observation_key] = ObservationAdapterPayload(
            interface_name=interface.name,
            observation_key=interface.observation_key,
            vector=vector.copy(),
            signal_names=interface.signal_names,
            tick=int(tick),
            source=source,
            freshness_by_signal=_freshness_by_signal(signal_values),
        )
    return adapters


def observation_vectors_from_adapters(
    adapters: Mapping[str, ObservationAdapterPayload],
) -> dict[str, np.ndarray]:
    """
    Extract a mapping from adapter keys to copies of each payload's observation vector.
    
    Parameters:
        adapters (Mapping[str, ObservationAdapterPayload]): Mapping of adapter keys to payloads.
    
    Returns:
        dict[str, np.ndarray]: A new dict mapping each adapter key to a copy of that payload's `vector` array.
    """
    return {
        key: payload.vector.copy()
        for key, payload in adapters.items()
    }


def adapter_trace_summary(
    adapters: Mapping[str, ObservationAdapterPayload],
) -> dict[str, object]:
    """
    Produce a trace-friendly mapping from each adapter's interface name to its summary.
    
    Parameters:
        adapters (Mapping[str, ObservationAdapterPayload]): Mapping of adapter keys to payloads to summarize.
    
    Returns:
        dict[str, object]: Dictionary where each key is an adapter's `interface_name` and each value is a summary object containing interface/observation identifiers, `signal_names` (list), `dim` (vector length), `tick`, `source`, and `freshness_by_signal`.
    """
    return {
        payload.interface_name: payload.to_summary()
        for payload in adapters.values()
    }


__all__ = [
    "ObservationAdapterPayload",
    "adapt_observation_contracts",
    "adapter_trace_summary",
    "observation_vectors_from_adapters",
]
