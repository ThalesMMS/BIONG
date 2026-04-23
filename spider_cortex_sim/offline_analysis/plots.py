from __future__ import annotations

from collections.abc import Mapping, Sequence

from .renderers import render_multi_line_chart, render_placeholder_svg
from .tables import CAPACITY_SWEEP_INTERPRETATION_GUIDANCE
from .utils import _coerce_float


def build_capacity_comparison_plot(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    series_by_variant: dict[str, list[dict[str, object]]] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        variant_name = str(item.get("variant") or "")
        if not variant_name:
            continue
        series_by_variant.setdefault(variant_name, []).append(
            {
                "total_trainable": _coerce_float(item.get("total_trainable"), 0.0),
                "scenario_success_rate": _coerce_float(
                    item.get("scenario_success_rate"),
                    0.0,
                ),
            }
        )
    for points in series_by_variant.values():
        points.sort(key=lambda item: item["total_trainable"])
    available = bool(series_by_variant)
    return {
        "available": available,
        "series": series_by_variant,
        "svg": (
            render_multi_line_chart(
                "Capacity vs scenario success rate",
                series_by_variant,
                x_key="total_trainable",
                y_key="scenario_success_rate",
            )
            if available
            else render_placeholder_svg("No data", "No capacity sweep data")
        ),
        "metadata": {
            "x_axis": "total_trainable",
            "y_axis": "scenario_success_rate",
            "interpretation_guidance": CAPACITY_SWEEP_INTERPRETATION_GUIDANCE,
        },
    }


__all__ = ["build_capacity_comparison_plot"]
