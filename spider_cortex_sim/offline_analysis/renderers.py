from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape

from .extractors import (
    _noise_robustness_marginal,
    _noise_robustness_mean,
    _noise_robustness_rate,
)
from .utils import (
    _coerce_float,
    _format_optional_metric,
    _mean,
    _safe_divide,
)


def _chart_bounds(values: Sequence[float]) -> tuple[float, float]:
    """
    Compute Y-axis bounds for plotting from a sequence of numeric values.
    
    If `values` is empty, returns (0.0, 1.0). If all values are effectively equal (within floating-point tolerance), returns the value padded on each side by 0.15 of its magnitude, or by 1.0 when the value is exactly 0.0. If values vary, returns (min - 10% range, max + 10% range).
    
    Parameters:
        values (Sequence[float]): Numeric values to derive bounds from.
    
    Returns:
        tuple[float, float]: (min_bound, max_bound) padded bounds for plotting.
    """
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        pad = 1.0 if minimum == 0.0 else abs(minimum) * 0.15
        return minimum - pad, maximum + pad
    pad = (maximum - minimum) * 0.1
    return minimum - pad, maximum + pad


def render_placeholder_svg(title: str, message: str) -> str:
    """
    Generate a fixed-size placeholder SVG containing a title and a message.
    
    Parameters:
        title (str): Text rendered as the primary line at the top; HTML-escaped.
        message (str): Text rendered as the secondary line below the title; HTML-escaped.
    
    Returns:
        svg (str): An SVG string sized 900x260 with the escaped title and message drawn as text.
    """
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">'
        '<rect x="0" y="0" width="900" height="260" fill="#f8fafc" />'
        f'<text x="40" y="54" font-size="24" font-family="monospace" fill="#0f172a">{escape(title)}</text>'
        f'<text x="40" y="120" font-size="16" font-family="monospace" fill="#475569">{escape(message)}</text>'
        "</svg>"
    )


def render_line_chart(
    title: str,
    series: Sequence[dict[str, object]],
    *,
    x_key: str = "index",
    y_key: str = "reward",
) -> str:
    """
    Render a simple SVG line chart from a sequence of data points.
    
    Parameters:
        title (str): Chart title (will be HTML-escaped and displayed at the top-left).
        series (Sequence[dict[str, object]]): Sequence of point mappings; each point should contain values for the keys specified by `x_key` and `y_key`.
        x_key (str): Key name to read the x value from each point. Defaults to "index".
        y_key (str): Key name to read the y value from each point. Defaults to "reward".
    
    Returns:
        str: SVG markup for the chart. If `series` is empty, returns a fixed-size placeholder SVG indicating "No data available."
    """
    if not series:
        return render_placeholder_svg(title, "No data available.")
    width = 900
    height = 320
    left = 70
    right = 30
    top = 55
    bottom = 45
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_y = [_coerce_float(point.get(y_key)) for point in series]
    min_y, max_y = _chart_bounds(all_y)
    max_x = max(_coerce_float(point.get(x_key), 1.0) for point in series)
    max_x = max(max_x, 1.0)

    def x_coord(value: float) -> float:
        """
        Map an X value to its horizontal SVG coordinate within the plot area.
        
        Parameters:
            value (float): The X value to map; expected to be in the same scale as `max_x`.
        
        Returns:
            float: The horizontal coordinate in SVG pixels corresponding to `value`, where `0` maps to `left` and `max_x` maps to `left + plot_width`.
        """
        return left + (value / max_x) * plot_width

    def y_coord(value: float) -> float:
        """
        Map a Y data value to its vertical SVG coordinate within the plot area.
        
        Parameters:
            value (float): The Y data value to convert to a pixel Y coordinate.
        
        Returns:
            float: The Y coordinate in pixels corresponding to `value`. If `min_y` and `max_y` are effectively equal, returns the vertical midpoint of the plot; otherwise returns a value linearly interpolated between the top and bottom of the plot (higher data values map to smaller pixel Y).
        """
        if math.isclose(max_y, min_y):
            return top + plot_height / 2.0
        normalized = (value - min_y) / (max_y - min_y)
        return top + (1.0 - normalized) * plot_height

    points_attr = " ".join(
        f"{x_coord(_coerce_float(point.get(x_key), 0.0)):.2f},{y_coord(_coerce_float(point.get(y_key), 0.0)):.2f}"
        for point in series
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{left}" y="32" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{points_attr}" />',
    ]
    for point in series:
        px = x_coord(_coerce_float(point.get(x_key), 0.0))
        py = y_coord(_coerce_float(point.get(y_key), 0.0))
        lines.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="#1d4ed8" />'
        )
    for label in (min_y, (min_y + max_y) / 2.0, max_y):
        y = y_coord(label)
        lines.append(
            f'<text x="12" y="{y + 5:.2f}" font-size="12" font-family="monospace" fill="#475569">{label:.2f}</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_multi_line_chart(
    title: str,
    series_by_name: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    x_key: str = "index",
    y_key: str = "reward",
) -> str:
    populated = {
        str(name): list(series)
        for name, series in series_by_name.items()
        if series
    }
    if not populated:
        return render_placeholder_svg(title, "No data available.")
    width = 920
    height = 360
    left = 70
    right = 180
    top = 55
    bottom = 45
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [
        point
        for series in populated.values()
        for point in series
    ]
    all_y = [_coerce_float(point.get(y_key)) for point in all_points]
    min_y, max_y = _chart_bounds(all_y)
    max_x = max(_coerce_float(point.get(x_key), 1.0) for point in all_points)
    max_x = max(max_x, 1.0)

    def x_coord(value: float) -> float:
        return left + (value / max_x) * plot_width

    def y_coord(value: float) -> float:
        if math.isclose(max_y, min_y):
            return top + plot_height / 2.0
        normalized = (value - min_y) / (max_y - min_y)
        return top + (1.0 - normalized) * plot_height

    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{left}" y="32" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
    ]
    for label in (min_y, (min_y + max_y) / 2.0, max_y):
        y = y_coord(label)
        lines.append(
            f'<text x="12" y="{y + 5:.2f}" font-size="12" font-family="monospace" fill="#475569">{label:.2f}</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
    for index, (name, series) in enumerate(sorted(populated.items())):
        color = colors[index % len(colors)]
        points_attr = " ".join(
            f"{x_coord(_coerce_float(point.get(x_key), 0.0)):.2f},{y_coord(_coerce_float(point.get(y_key), 0.0)):.2f}"
            for point in sorted(
                series,
                key=lambda item: _coerce_float(item.get(x_key), 0.0),
            )
        )
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points_attr}" />'
        )
        for point in series:
            px = x_coord(_coerce_float(point.get(x_key), 0.0))
            py = y_coord(_coerce_float(point.get(y_key), 0.0))
            lines.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="{color}" />'
            )
        legend_y = top + 18 + index * 22
        legend_x = left + plot_width + 20
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 26}" y="{legend_y + 4}" font-size="12" font-family="monospace" fill="#334155">{escape(name)}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_bar_chart(
    title: str,
    items: Sequence[Mapping[str, object]],
    *,
    label_key: str,
    value_key: str,
) -> str:
    """
    Render a horizontal bar chart for the given items.
    
    Each item supplies a label and a numeric value; bars are scaled relative to the largest value
    (with a minimum scale floor of 1.0). Non-finite or missing numeric values are treated as 0.
    
    Parameters:
        title (str): Chart title displayed at the top-left of the SVG.
        items (Sequence[Mapping[str, object]]): Sequence of records for each bar.
        label_key (str): Key in each item mapping used for the bar label (displayed at left).
        value_key (str): Key in each item mapping used for the numeric value determining bar length.
    
    Returns:
        str: An SVG document string containing a horizontal bar chart of the provided items.
             If `items` is empty, returns a placeholder SVG with the message "No data available."
    """
    if not items:
        return render_placeholder_svg(title, "No data available.")
    width = 960
    height = max(260, 70 + 34 * len(items))
    left = 250
    right = 30
    top = 55
    bar_height = 22
    gap = 10
    plot_width = width - left - right
    max_value = max(_coerce_float(item.get(value_key), 0.0) for item in items)
    max_value = max(max_value, 1.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="32" y="32" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
    ]
    for index, item in enumerate(items):
        value = _coerce_float(item.get(value_key), 0.0)
        label = str(item.get(label_key) or "")
        y = top + index * (bar_height + gap)
        bar_width = plot_width * _safe_divide(value, max_value)
        lines.append(
            f'<text x="32" y="{y + 16}" font-size="13" font-family="monospace" fill="#334155">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{plot_width}" height="{bar_height}" fill="#e2e8f0" rx="4" />'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="#2563eb" rx="4" />'
        )
        lines.append(
            f'<text x="{left + plot_width + 8}" y="{y + 16}" font-size="12" font-family="monospace" fill="#475569">{value:.2f}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_matrix_heatmap(
    title: str,
    *,
    train_conditions: Sequence[str],
    eval_conditions: Sequence[str],
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    train_marginals: Mapping[str, object],
    eval_marginals: Mapping[str, object],
) -> str:
    """
    Render a train-by-eval heatmap that displays success-rate values in a grid with an extra "mean" row and column.
    
    Each cell shows a formatted success-rate and a background color interpolated from light to dark according to the clamped value (0.0-1.0). The final "mean" column and row are derived from the provided marginals. If either `train_conditions` or `eval_conditions` is empty, returns a small placeholder SVG indicating no data.
    
    Parameters:
        title (str): Chart title rendered at the top-left of the SVG.
        train_conditions (Sequence[str]): Ordered names for matrix rows (training conditions).
        eval_conditions (Sequence[str]): Ordered names for matrix columns (evaluation conditions).
        matrix (Mapping[str, Mapping[str, Mapping[str, object]]]): Nested mapping train_condition -> eval_condition -> cell payload; cell payload is expected to expose a numeric `scenario_success_rate` when present.
        train_marginals (Mapping[str, object]): Per-train-condition marginal values used to compute the "mean" column and the final mean row.
        eval_marginals (Mapping[str, object]): Per-eval-condition marginal values used to compute the "mean" row and the final mean column.
    
    Returns:
        str: The SVG document as a string. If input conditions are empty, returns a placeholder SVG stating no data.
    """
    if not train_conditions or not eval_conditions:
        return render_placeholder_svg(title, "No data available.")
    width = 980
    height = max(320, 130 + 52 * (len(train_conditions) + 1))
    left = 220
    top = 92
    cell_width = max(78, int((width - left - 60) / (len(eval_conditions) + 1)))
    cell_height = 42

    def _cell_color(value: float | None) -> str:
        """
        Map an optional normalized metric to an RGB color string suitable for heatmap cells.
        
        Parameters:
            value (float | None): Metric in the range [0.0, 1.0]; if `None`, a neutral gray is used.
        
        Returns:
            str: A CSS color string. Returns the hex color `#e2e8f0` when `value` is `None`; otherwise returns an `rgb(r,g,b)` string interpolated across a light-to-dark blue gradient (value 0.0 -> `rgb(241,245,249)`, value 1.0 -> `rgb(37,137,223)`).
        """
        if value is None:
            return "#e2e8f0"
        clamped = max(0.0, min(1.0, value))
        red = int(241 - (clamped * 204))
        green = int(245 - (clamped * 108))
        blue = int(249 - (clamped * 26))
        return f"rgb({red},{green},{blue})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="32" y="36" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
        f'<text x="{left - 120}" y="{top - 18}" font-size="13" font-family="monospace" fill="#475569">train \\ eval</text>',
    ]

    headers = [*eval_conditions, "mean"]
    for index, label in enumerate(headers):
        x = left + index * cell_width
        lines.append(
            f'<text x="{x + 8}" y="{top - 18}" font-size="13" font-family="monospace" fill="#334155">{escape(str(label))}</text>'
        )

    for row_index, train_condition in enumerate([*train_conditions, "mean"]):
        y = top + row_index * cell_height
        lines.append(
            f'<text x="32" y="{y + 26}" font-size="13" font-family="monospace" fill="#334155">{escape(str(train_condition))}</text>'
        )
        for col_index, eval_condition in enumerate(headers):
            x = left + col_index * cell_width
            if train_condition == "mean" and eval_condition == "mean":
                value = _noise_robustness_mean(
                    _noise_robustness_marginal(train_marginals, name)
                    for name in train_conditions
                )
            elif train_condition == "mean":
                value = _noise_robustness_marginal(eval_marginals, eval_condition)
            elif eval_condition == "mean":
                value = _noise_robustness_marginal(train_marginals, train_condition)
            else:
                value = _noise_robustness_rate(
                    matrix,
                    train_condition=train_condition,
                    eval_condition=eval_condition,
                )
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 8}" height="{cell_height - 8}" fill="{_cell_color(value)}" stroke="#cbd5e1" stroke-width="1" rx="4" />'
            )
            lines.append(
                f'<text x="{x + 10}" y="{y + 24}" font-size="13" font-family="monospace" fill="#0f172a">{escape(_format_optional_metric(value))}</text>'
            )
    lines.append("</svg>")
    return "".join(lines)
