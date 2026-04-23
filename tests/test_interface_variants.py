from __future__ import annotations

import unittest

import numpy as np

from spider_cortex_sim.interface_docs import render_interfaces_markdown
from spider_cortex_sim.interfaces import (
    INTERFACE_VARIANTS,
    MODULE_INTERFACE_BY_NAME,
    OBSERVATION_VIEW_BY_KEY,
    VARIANT_MODULES,
    get_interface_variant,
    get_variant_levels,
    validate_variant_interfaces,
)
from spider_cortex_sim.module_local_tasks import ModuleVariantComparison, evaluate_variant_sufficiency


class InterfaceVariantContractTest(unittest.TestCase):
    EXPECTED_SIGNAL_COUNTS = {
        ("visual_cortex", 1): 5,
        ("visual_cortex", 2): 11,
        ("visual_cortex", 3): 26,
        ("visual_cortex", 4): 32,
        ("sensory_cortex", 1): 5,
        ("sensory_cortex", 2): 8,
        ("sensory_cortex", 3): 11,
        ("sensory_cortex", 4): 12,
        ("hunger_center", 1): 5,
        ("hunger_center", 2): 8,
        ("hunger_center", 3): 13,
        ("hunger_center", 4): 18,
        ("sleep_center", 1): 3,
        ("sleep_center", 2): 4,
        ("sleep_center", 3): 9,
        ("sleep_center", 4): 18,
        ("alert_center", 1): 3,
        ("alert_center", 2): 5,
        ("alert_center", 3): 9,
        ("alert_center", 4): 27,
    }

    def test_all_variants_are_valid_subsets_of_parent_interfaces(self) -> None:
        for (module_name, _level), interface in INTERFACE_VARIANTS.items():
            parent = MODULE_INTERFACE_BY_NAME[module_name]
            self.assertEqual(interface.observation_key, parent.observation_key)
            self.assertEqual(
                interface.signal_names,
                tuple(
                    signal_name
                    for signal_name in parent.signal_names
                    if signal_name in interface.signal_names
                ),
            )
            self.assertTrue(set(interface.signal_names).issubset(parent.signal_names))

    def test_variant_signal_counts_match_expected_contract(self) -> None:
        for key, expected_count in self.EXPECTED_SIGNAL_COUNTS.items():
            module_name, level = key
            self.assertEqual(get_interface_variant(module_name, level).input_dim, expected_count)

    def test_get_interface_variant_returns_registered_interfaces_for_each_level(self) -> None:
        for module_name in VARIANT_MODULES:
            self.assertEqual(get_variant_levels(module_name), (1, 2, 3, 4))
            for level in (1, 2, 3):
                self.assertIs(get_interface_variant(module_name, level), INTERFACE_VARIANTS[(module_name, level)])

    def test_level_four_returns_canonical_interface_identity(self) -> None:
        for module_name in VARIANT_MODULES:
            self.assertIs(get_interface_variant(module_name, 4), MODULE_INTERFACE_BY_NAME[module_name])

    def test_validate_variant_interfaces_passes(self) -> None:
        validate_variant_interfaces()


class InterfaceVariantEvaluationTest(unittest.TestCase):
    def test_evaluate_variant_sufficiency_returns_expected_structure(self) -> None:
        comparison = evaluate_variant_sufficiency(
            "sleep_center",
            rng=np.random.default_rng(42),
        )

        self.assertIsInstance(comparison, ModuleVariantComparison)
        self.assertEqual(comparison.module_name, "sleep_center")
        self.assertEqual([result.level for result in comparison.results], [1, 2, 3, 4])
        self.assertEqual(comparison.report["module_name"], "sleep_center")
        self.assertEqual(comparison.report["levels_evaluated"], [1, 2, 3, 4])
        self.assertIn("minimal_sufficient_level", comparison.report)

    def test_variant_observation_extraction_works_end_to_end(self) -> None:
        canonical_interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        view_type = OBSERVATION_VIEW_BY_KEY[canonical_interface.observation_key]
        full_mapping = {
            signal_name: float(index + 1) / float(len(view_type.field_names()) + 1)
            for index, signal_name in enumerate(view_type.field_names())
        }

        variant_interface = get_interface_variant("alert_center", 2)
        variant_mapping = {
            signal_name: full_mapping[signal_name]
            for signal_name in variant_interface.signal_names
        }
        vector = variant_interface.vector_from_mapping(variant_mapping)

        self.assertEqual(vector.shape, (variant_interface.input_dim,))
        self.assertEqual(len(vector), 5)
        rebound = variant_interface.bind_values(vector)
        self.assertEqual(tuple(rebound.keys()), variant_interface.signal_names)
        for signal_name in variant_interface.signal_names:
            self.assertAlmostEqual(rebound[signal_name], full_mapping[signal_name])


class InterfaceVariantDocsTest(unittest.TestCase):
    def test_render_interfaces_markdown_includes_variant_section_and_note(self) -> None:
        markdown = render_interfaces_markdown()
        self.assertIn("## Interface Variants", markdown)
        self.assertIn("diagnostic sufficiency probes", markdown)
        self.assertIn("not production save/load interfaces", markdown)

    def test_render_interfaces_markdown_groups_variants_by_module(self) -> None:
        markdown = render_interfaces_markdown()
        self.assertIn("### `visual_cortex`", markdown)
        self.assertIn("### `sensory_cortex`", markdown)
        self.assertIn("### `hunger_center`", markdown)
        self.assertIn("### `sleep_center`", markdown)
        self.assertIn("### `alert_center`", markdown)
        self.assertIn("`visual_v1` -> `visual_v2` -> `visual_v3` -> `visual_cortex`", markdown)
        self.assertIn("`sensory_v1` -> `sensory_v2` -> `sensory_v3` -> `sensory_cortex`", markdown)
        self.assertIn("`hunger_v1` -> `hunger_v2` -> `hunger_v3` -> `hunger_center`", markdown)
        self.assertIn("`heading_dx`", markdown)
        self.assertIn("`food_smell_strength`", markdown)
        self.assertIn("`food_smell_strength`", markdown)
        self.assertIn("`shelter_trace_strength`", markdown)
        self.assertIn("`predator_smell_strength`", markdown)
