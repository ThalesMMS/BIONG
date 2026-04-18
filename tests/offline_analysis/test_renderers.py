from __future__ import annotations

import unittest

from spider_cortex_sim.offline_analysis.renderers import render_matrix_heatmap

class RenderMatrixHeatmapTest(unittest.TestCase):
    """Tests for render_matrix_heatmap (new in this PR)."""

    def test_returns_svg_string(self) -> None:
        svg = render_matrix_heatmap(
            "Test heatmap",
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
            matrix={
                "none": {"none": {"scenario_success_rate": 1.0}, "high": {"scenario_success_rate": 0.5}},
                "low": {"none": {"scenario_success_rate": 0.8}, "high": {"scenario_success_rate": 0.2}},
            },
            train_marginals={"none": 0.75, "low": 0.5},
            eval_marginals={"none": 0.9, "high": 0.35},
        )
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)

    def test_svg_contains_title(self) -> None:
        svg = render_matrix_heatmap(
            "Noise robustness matrix",
            train_conditions=["none"],
            eval_conditions=["none"],
            matrix={"none": {"none": {"scenario_success_rate": 0.5}}},
            train_marginals={"none": 0.5},
            eval_marginals={"none": 0.5},
        )
        self.assertIn("Noise robustness matrix", svg)

    def test_empty_train_conditions_returns_placeholder_svg(self) -> None:
        svg = render_matrix_heatmap(
            "Empty heatmap",
            train_conditions=[],
            eval_conditions=["none"],
            matrix={},
            train_marginals={},
            eval_marginals={},
        )
        self.assertIn("No data available", svg)

    def test_empty_eval_conditions_returns_placeholder_svg(self) -> None:
        svg = render_matrix_heatmap(
            "Empty heatmap",
            train_conditions=["none"],
            eval_conditions=[],
            matrix={},
            train_marginals={},
            eval_marginals={},
        )
        self.assertIn("No data available", svg)

    def test_svg_contains_train_and_eval_condition_labels(self) -> None:
        svg = render_matrix_heatmap(
            "Heatmap",
            train_conditions=["none", "high"],
            eval_conditions=["low", "medium"],
            matrix={
                "none": {
                    "low": {"scenario_success_rate": 0.9},
                    "medium": {"scenario_success_rate": 0.7},
                },
                "high": {
                    "low": {"scenario_success_rate": 0.6},
                    "medium": {"scenario_success_rate": 0.4},
                },
            },
            train_marginals={"none": 0.8, "high": 0.5},
            eval_marginals={"low": 0.75, "medium": 0.55},
        )
        self.assertIn("none", svg)
        self.assertIn("high", svg)
        self.assertIn("low", svg)
        self.assertIn("medium", svg)

    def test_svg_contains_mean_row_and_column(self) -> None:
        svg = render_matrix_heatmap(
            "Heatmap",
            train_conditions=["none"],
            eval_conditions=["none"],
            matrix={"none": {"none": {"scenario_success_rate": 1.0}}},
            train_marginals={"none": 1.0},
            eval_marginals={"none": 1.0},
        )
        self.assertIn("mean", svg)

    def test_svg_includes_cell_values_as_decimals(self) -> None:
        svg = render_matrix_heatmap(
            "Heatmap",
            train_conditions=["none"],
            eval_conditions=["low"],
            matrix={"none": {"low": {"scenario_success_rate": 0.75}}},
            train_marginals={"none": 0.75},
            eval_marginals={"low": 0.75},
        )
        self.assertIn("0.75", svg)

    def test_title_special_chars_are_escaped(self) -> None:
        svg = render_matrix_heatmap(
            "Test <&> title",
            train_conditions=["none"],
            eval_conditions=["none"],
            matrix={"none": {"none": {"scenario_success_rate": 0.5}}},
            train_marginals={"none": 0.5},
            eval_marginals={"none": 0.5},
        )
        self.assertIn("&lt;", svg)
        self.assertIn("&amp;", svg)
