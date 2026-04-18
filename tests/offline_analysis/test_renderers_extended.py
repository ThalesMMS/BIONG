"""Tests for SVG rendering helpers in spider_cortex_sim.offline_analysis.renderers.

Covers render_placeholder_svg, render_line_chart, render_bar_chart, and _chart_bounds,
which are newly extracted in this PR and only partially tested elsewhere
(test_renderers.py covers only render_matrix_heatmap).
"""
from __future__ import annotations

import unittest

import defusedxml.ElementTree as ET

from spider_cortex_sim.offline_analysis.renderers import (
    _chart_bounds,
    render_bar_chart,
    render_line_chart,
    render_placeholder_svg,
)


class ChartBoundsTest(unittest.TestCase):
    def test_empty_returns_zero_one(self) -> None:
        lo, hi = _chart_bounds([])
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_single_zero_value_pads_by_one(self) -> None:
        lo, hi = _chart_bounds([0.0])
        self.assertAlmostEqual(lo, -1.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_single_nonzero_value_pads_proportionally(self) -> None:
        lo, hi = _chart_bounds([2.0])
        # pad = abs(2.0) * 0.15 = 0.30
        self.assertAlmostEqual(lo, 2.0 - 0.3)
        self.assertAlmostEqual(hi, 2.0 + 0.3)

    def test_distinct_values_pad_by_10_percent(self) -> None:
        lo, hi = _chart_bounds([0.0, 1.0])
        # pad = (1.0 - 0.0) * 0.1 = 0.1
        self.assertAlmostEqual(lo, -0.1)
        self.assertAlmostEqual(hi, 1.1)

    def test_all_same_nonzero_values_pad_proportionally(self) -> None:
        lo, hi = _chart_bounds([5.0, 5.0, 5.0])
        # math.isclose(5.0, 5.0) => True; pad = abs(5.0)*0.15 = 0.75
        self.assertAlmostEqual(lo, 5.0 - 0.75)
        self.assertAlmostEqual(hi, 5.0 + 0.75)

    def test_negative_range(self) -> None:
        lo, hi = _chart_bounds([-3.0, -1.0])
        # pad = (-1.0 - (-3.0))*0.1 = 0.2
        self.assertAlmostEqual(lo, -3.0 - 0.2)
        self.assertAlmostEqual(hi, -1.0 + 0.2)


class RenderPlaceholderSvgTest(unittest.TestCase):
    def test_returns_svg_element(self) -> None:
        svg = render_placeholder_svg("Title", "message text")
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)

    def test_title_appears_in_output(self) -> None:
        svg = render_placeholder_svg("MyTitle", "Some message")
        self.assertIn("MyTitle", svg)

    def test_message_appears_in_output(self) -> None:
        svg = render_placeholder_svg("T", "Missing data here")
        self.assertIn("Missing data here", svg)

    def test_special_chars_in_title_are_escaped(self) -> None:
        svg = render_placeholder_svg("A <b> & C", "msg")
        self.assertIn("&lt;", svg)
        self.assertIn("&amp;", svg)
        self.assertNotIn("A <b> & C", svg)
        self.assertNotIn("<b>", svg)
        self.assertNotIn(" & C", svg)

    def test_special_chars_in_message_are_escaped(self) -> None:
        svg = render_placeholder_svg("T", "Value > 1 & < 0")
        self.assertIn("&gt;", svg)
        self.assertIn("&amp;", svg)
        self.assertNotIn("Value > 1 & < 0", svg)
        self.assertNotIn("Value > 1", svg)
        self.assertNotIn(" & < 0", svg)
        self.assertNotIn("< 0", svg)

    def test_fixed_dimensions(self) -> None:
        svg = render_placeholder_svg("t", "m")
        self.assertIn('width="900"', svg)
        self.assertIn('height="260"', svg)


class RenderLineChartTest(unittest.TestCase):
    def _single_series(self) -> list[dict[str, object]]:
        return [
            {"index": 1, "reward": 0.1},
            {"index": 2, "reward": 0.5},
            {"index": 3, "reward": 0.9},
        ]

    def test_empty_series_returns_placeholder(self) -> None:
        svg = render_line_chart("Empty", [])
        self.assertIn("No data available", svg)

    def test_non_empty_series_returns_svg(self) -> None:
        svg = render_line_chart("Training reward", self._single_series())
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)

    def test_title_appears_in_svg(self) -> None:
        svg = render_line_chart("My chart title", self._single_series())
        self.assertIn("My chart title", svg)

    def test_polyline_element_present(self) -> None:
        svg = render_line_chart("T", self._single_series())
        self.assertIn("<polyline", svg)

    def test_circle_elements_present_for_each_point(self) -> None:
        svg = render_line_chart("T", self._single_series())
        self.assertEqual(svg.count("<circle"), len(self._single_series()))

    def test_title_special_chars_escaped(self) -> None:
        svg = render_line_chart("A & B", self._single_series())
        self.assertIn("&amp;", svg)

    def test_custom_x_and_y_keys(self) -> None:
        series = [{"step": 1, "score": 0.6}, {"step": 2, "score": 0.8}]
        svg = render_line_chart("Custom", series, x_key="step", y_key="score")
        self.assertIn("<polyline", svg)
        self.assertNotIn("No data available", svg)

    def test_single_point_series_renders(self) -> None:
        svg = render_line_chart("Single", [{"index": 1, "reward": 0.5}])
        self.assertIn("<polyline", svg)

    def test_all_same_values_doesnt_crash(self) -> None:
        series = [{"index": i, "reward": 0.5} for i in range(5)]
        svg = render_line_chart("Flat", series)
        self.assertIn("</svg>", svg)


class RenderBarChartTest(unittest.TestCase):
    def _svg_root(self, svg: str) -> ET.Element:
        return ET.fromstring(svg)

    def _rects(self, svg: str) -> list[ET.Element]:
        root = self._svg_root(svg)
        return [
            element
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "rect"
        ]

    def _height(self, svg: str) -> float:
        root = self._svg_root(svg)
        height = root.get("height")
        if height is not None:
            return float(height)
        view_box = root.get("viewBox")
        self.assertIsNotNone(view_box, "height and viewBox missing from svg root")
        return float(str(view_box).split()[3])

    def _sample_items(self) -> list[dict[str, object]]:
        return [
            {"module": "visual_cortex", "score": 0.8},
            {"module": "sensory_cortex", "score": 0.6},
            {"module": "hunger_center", "score": 0.3},
        ]

    def test_empty_items_returns_placeholder(self) -> None:
        svg = render_bar_chart("Empty", [], label_key="module", value_key="score")
        self.assertIn("No data available", svg)

    def test_non_empty_items_returns_svg(self) -> None:
        svg = render_bar_chart("Modules", self._sample_items(), label_key="module", value_key="score")
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)

    def test_title_appears_in_svg(self) -> None:
        svg = render_bar_chart("ModuleChart", self._sample_items(), label_key="module", value_key="score")
        self.assertIn("ModuleChart", svg)

    def test_label_values_appear_in_svg(self) -> None:
        svg = render_bar_chart("T", self._sample_items(), label_key="module", value_key="score")
        self.assertIn("visual_cortex", svg)
        self.assertIn("sensory_cortex", svg)
        self.assertIn("hunger_center", svg)

    def test_rect_element_per_item(self) -> None:
        items = self._sample_items()
        svg = render_bar_chart("T", items, label_key="module", value_key="score")
        filled_rect_count = sum(
            1 for rect in self._rects(svg) if rect.get("fill") == "#2563eb"
        )
        self.assertEqual(filled_rect_count, len(items))

    def test_special_chars_in_label_escaped(self) -> None:
        items = [{"label": "A & B", "val": 0.5}]
        svg = render_bar_chart("T", items, label_key="label", value_key="val")
        self.assertIn("&amp;", svg)

    def test_title_special_chars_escaped(self) -> None:
        svg = render_bar_chart("A <b>", self._sample_items(), label_key="module", value_key="score")
        self.assertIn("&lt;", svg)

    def test_numeric_values_shown_in_svg(self) -> None:
        items = [{"name": "x", "val": 0.75}]
        svg = render_bar_chart("T", items, label_key="name", value_key="val")
        self.assertIn("0.75", svg)

    def test_single_item_renders_without_crash(self) -> None:
        svg = render_bar_chart("Single", [{"n": "only", "v": 1.0}], label_key="n", value_key="v")
        self.assertIn("</svg>", svg)

    def test_zero_values_dont_crash(self) -> None:
        items = [{"n": "zero", "v": 0.0}]
        svg = render_bar_chart("T", items, label_key="n", value_key="v")
        self.assertIn("</svg>", svg)

    def test_height_scales_with_item_count(self) -> None:
        few = [{"n": "a", "v": 0.5}]
        many = [{"n": f"item_{i}", "v": 0.5} for i in range(20)]
        svg_few = render_bar_chart("T", few, label_key="n", value_key="v")
        svg_many = render_bar_chart("T", many, label_key="n", value_key="v")
        height_few = self._height(svg_few)
        height_many = self._height(svg_many)
        self.assertGreater(height_many, height_few)
