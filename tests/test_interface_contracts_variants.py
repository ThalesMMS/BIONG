"""Tests for spider_cortex_sim.interface_contracts.variants module.

Covers get_interface_variant, get_variant_levels, is_variant_interface,
validate_variant_interfaces, and the INTERFACE_VARIANTS registry,
introduced in the PR refactor.
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.interface_contracts.module_specs import (
    MODULE_INTERFACE_BY_NAME,
    ModuleInterface,
)
from spider_cortex_sim.interface_contracts.variants import (
    ALERT_CENTER_V1_INTERFACE,
    ALERT_CENTER_V2_INTERFACE,
    ALERT_CENTER_V3_INTERFACE,
    HUNGER_CENTER_V1_INTERFACE,
    HUNGER_CENTER_V2_INTERFACE,
    HUNGER_CENTER_V3_INTERFACE,
    INTERFACE_VARIANTS,
    OBSERVATION_INTERFACE_BY_KEY,
    SENSORY_CORTEX_V1_INTERFACE,
    SENSORY_CORTEX_V2_INTERFACE,
    SENSORY_CORTEX_V3_INTERFACE,
    SLEEP_CENTER_V1_INTERFACE,
    SLEEP_CENTER_V2_INTERFACE,
    SLEEP_CENTER_V3_INTERFACE,
    VISUAL_CORTEX_V1_INTERFACE,
    VISUAL_CORTEX_V2_INTERFACE,
    VISUAL_CORTEX_V3_INTERFACE,
    get_interface_variant,
    get_variant_levels,
    is_variant_interface,
    validate_variant_interfaces,
)

VARIANT_MODULE_NAMES = (
    "visual_cortex",
    "sensory_cortex",
    "hunger_center",
    "sleep_center",
    "alert_center",
)


class InterfaceVariantsRegistryTest(unittest.TestCase):
    """Tests for the INTERFACE_VARIANTS registry dict."""

    def test_registry_is_dict(self) -> None:
        self.assertIsInstance(INTERFACE_VARIANTS, dict)

    def test_registry_is_non_empty(self) -> None:
        self.assertGreater(len(INTERFACE_VARIANTS), 0)

    def test_all_keys_are_tuples_of_module_name_and_level(self) -> None:
        for key in INTERFACE_VARIANTS:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
            module_name, level = key
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(level, int)

    def test_all_values_are_module_interface(self) -> None:
        for key, iface in INTERFACE_VARIANTS.items():
            self.assertIsInstance(iface, ModuleInterface, f"Non-interface at key {key}")

    def test_each_variant_module_has_levels_1_2_3(self) -> None:
        for module_name in VARIANT_MODULE_NAMES:
            for level in (1, 2, 3):
                self.assertIn(
                    (module_name, level),
                    INTERFACE_VARIANTS,
                    f"Missing ({module_name!r}, {level}) in INTERFACE_VARIANTS",
                )


class GetInterfaceVariantTest(unittest.TestCase):
    """Tests for get_interface_variant()."""

    def test_level_1_returns_v1_interface_for_visual(self) -> None:
        iface = get_interface_variant("visual_cortex", 1)
        self.assertIs(iface, VISUAL_CORTEX_V1_INTERFACE)

    def test_level_2_returns_v2_interface_for_visual(self) -> None:
        iface = get_interface_variant("visual_cortex", 2)
        self.assertIs(iface, VISUAL_CORTEX_V2_INTERFACE)

    def test_level_3_returns_v3_interface_for_visual(self) -> None:
        iface = get_interface_variant("visual_cortex", 3)
        self.assertIs(iface, VISUAL_CORTEX_V3_INTERFACE)

    def test_level_4_returns_canonical_interface(self) -> None:
        canonical = MODULE_INTERFACE_BY_NAME["visual_cortex"]
        iface = get_interface_variant("visual_cortex", 4)
        self.assertIs(iface, canonical)

    def test_level_4_returns_canonical_alert_interface(self) -> None:
        canonical = MODULE_INTERFACE_BY_NAME["alert_center"]
        iface = get_interface_variant("alert_center", 4)
        self.assertIs(iface, canonical)

    def test_all_valid_modules_return_v1(self) -> None:
        for module_name in VARIANT_MODULE_NAMES:
            iface = get_interface_variant(module_name, 1)
            self.assertIsInstance(iface, ModuleInterface)

    def test_unknown_module_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_interface_variant("unknown_module", 1)
        self.assertIn("unknown_module", str(ctx.exception))

    def test_invalid_level_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_interface_variant("visual_cortex", 99)
        self.assertIn("99", str(ctx.exception))

    def test_level_0_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_interface_variant("visual_cortex", 0)

    def test_hunger_center_levels_all_return_interfaces(self) -> None:
        for level in (1, 2, 3, 4):
            iface = get_interface_variant("hunger_center", level)
            self.assertIsInstance(iface, ModuleInterface)


class GetVariantLevelsTest(unittest.TestCase):
    """Tests for get_variant_levels()."""

    def test_returns_tuple(self) -> None:
        result = get_variant_levels("visual_cortex")
        self.assertIsInstance(result, tuple)

    def test_includes_level_4_for_visual_cortex(self) -> None:
        levels = get_variant_levels("visual_cortex")
        self.assertIn(4, levels)

    def test_includes_levels_1_2_3_for_alert_center(self) -> None:
        levels = get_variant_levels("alert_center")
        for level in (1, 2, 3):
            self.assertIn(level, levels)

    def test_is_sorted_ascending(self) -> None:
        levels = get_variant_levels("sleep_center")
        self.assertEqual(list(levels), sorted(levels))

    def test_unknown_module_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_variant_levels("nonexistent_module")

    def test_all_variant_modules_return_non_empty_levels(self) -> None:
        for module_name in VARIANT_MODULE_NAMES:
            levels = get_variant_levels(module_name)
            self.assertGreater(len(levels), 0)


class IsVariantInterfaceTest(unittest.TestCase):
    """Tests for is_variant_interface()."""

    def test_v1_interface_is_variant(self) -> None:
        self.assertTrue(is_variant_interface(VISUAL_CORTEX_V1_INTERFACE))

    def test_v2_interface_is_variant(self) -> None:
        self.assertTrue(is_variant_interface(HUNGER_CENTER_V2_INTERFACE))

    def test_v3_interface_is_variant(self) -> None:
        self.assertTrue(is_variant_interface(ALERT_CENTER_V3_INTERFACE))

    def test_canonical_interface_is_not_variant(self) -> None:
        canonical = MODULE_INTERFACE_BY_NAME["visual_cortex"]
        self.assertFalse(is_variant_interface(canonical))

    def test_fresh_module_interface_is_not_variant(self) -> None:
        from spider_cortex_sim.interface_contracts.module_specs import (
            _subset_interface,
        )
        parent = MODULE_INTERFACE_BY_NAME["hunger_center"]
        new_iface = _subset_interface(
            parent,
            name="custom_test_variant",
            signal_names=("hunger",),
            description="test",
        )
        self.assertFalse(is_variant_interface(new_iface))


class ValidateVariantInterfacesTest(unittest.TestCase):
    """Tests for validate_variant_interfaces()."""

    def test_does_not_raise_for_valid_registry(self) -> None:
        # Should complete without raising
        validate_variant_interfaces()

    def test_variant_signal_names_are_subset_of_parent(self) -> None:
        for (module_name, level), iface in INTERFACE_VARIANTS.items():
            parent = MODULE_INTERFACE_BY_NAME[module_name]
            parent_signal_set = set(parent.signal_names)
            for sig_name in iface.signal_names:
                self.assertIn(
                    sig_name,
                    parent_signal_set,
                    f"{iface.name} has signal {sig_name!r} absent from parent {parent.name}",
                )

    def test_variant_observation_key_matches_parent(self) -> None:
        for (module_name, level), iface in INTERFACE_VARIANTS.items():
            parent = MODULE_INTERFACE_BY_NAME[module_name]
            self.assertEqual(
                iface.observation_key,
                parent.observation_key,
                f"{iface.name} observation_key mismatch with parent",
            )

    def test_v2_signals_include_v1_signals(self) -> None:
        """V2 should be a superset of V1 signals (incremental variants)."""
        for module_name in VARIANT_MODULE_NAMES:
            v1 = INTERFACE_VARIANTS[(module_name, 1)]
            v2 = INTERFACE_VARIANTS[(module_name, 2)]
            v1_set = set(v1.signal_names)
            for sig in v1.signal_names:
                self.assertIn(
                    sig, v2.signal_names,
                    f"V2 for {module_name} missing V1 signal {sig!r}",
                )

    def test_v3_signals_include_v2_signals(self) -> None:
        for module_name in VARIANT_MODULE_NAMES:
            v2 = INTERFACE_VARIANTS[(module_name, 2)]
            v3 = INTERFACE_VARIANTS[(module_name, 3)]
            for sig in v2.signal_names:
                self.assertIn(
                    sig, v3.signal_names,
                    f"V3 for {module_name} missing V2 signal {sig!r}",
                )


class ObservationInterfaceByKeyTest(unittest.TestCase):
    """Tests for OBSERVATION_INTERFACE_BY_KEY."""

    def test_is_non_empty_dict(self) -> None:
        self.assertIsInstance(OBSERVATION_INTERFACE_BY_KEY, dict)
        self.assertGreater(len(OBSERVATION_INTERFACE_BY_KEY), 0)

    def test_all_values_are_module_interface(self) -> None:
        for key, iface in OBSERVATION_INTERFACE_BY_KEY.items():
            self.assertIsInstance(iface, ModuleInterface, f"Non-interface at key {key!r}")

    def test_visual_key_maps_to_visual_cortex_or_similar(self) -> None:
        if "visual" in OBSERVATION_INTERFACE_BY_KEY:
            iface = OBSERVATION_INTERFACE_BY_KEY["visual"]
            self.assertIsInstance(iface, ModuleInterface)


if __name__ == "__main__":
    unittest.main()