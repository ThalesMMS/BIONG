from __future__ import annotations

from .brain.b_series_trace_fields import append_registered_b_series_trace_fields


def append_b_series_trace_fields(item: dict[str, object], decision) -> None:
    append_registered_b_series_trace_fields(item, decision)
