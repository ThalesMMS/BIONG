"""Tests for spider_cortex_sim.interface_contracts.registry module.

Covers interface_registry, interface_registry_fingerprint, architecture_signature,
_stable_json, and _fingerprint_payload, introduced in the PR refactor.
"""

from __future__ import annotations

import hashlib
import json
import unittest

from spider_cortex_sim.interface_contracts.registry import (
    ARBITRATION_EVIDENCE_INPUT_DIM,
    ARBITRATION_GATE_DIM,
    ARBITRATION_HIDDEN_DIM,
    ARBITRATION_VALENCE_DIM,
    INTERFACE_REGISTRY_SCHEMA_VERSION,
    OBSERVATION_DIMS,
    OBSERVATION_VIEW_BY_KEY,
    _build_observation_view_registry,
    _fingerprint_payload,
    _stable_json,
    architecture_signature,
    interface_registry,
    interface_registry_fingerprint,
)


class StableJsonTest(unittest.TestCase):
    """Tests for _stable_json."""

    def test_returns_string(self) -> None:
        self.assertIsInstance(_stable_json({"key": "value"}), str)

    def test_dict_is_sorted_by_key(self) -> None:
        result = _stable_json({"z": 1, "a": 2, "m": 3})
        parsed = json.loads(result)
        keys = list(parsed.keys())
        self.assertEqual(keys, sorted(keys))

    def test_compact_separators_no_extra_whitespace(self) -> None:
        result = _stable_json({"a": 1, "b": 2})
        self.assertNotIn(": ", result)  # compact separators avoid extra space
        self.assertNotIn(", ", result)

    def test_deterministic_for_same_input(self) -> None:
        payload = {"x": [1, 2, 3], "y": {"nested": True}}
        self.assertEqual(_stable_json(payload), _stable_json(payload))

    def test_different_insertion_order_same_result(self) -> None:
        a = {"z": 99, "a": 1}
        b = {"a": 1, "z": 99}
        self.assertEqual(_stable_json(a), _stable_json(b))


class FingerprintPayloadTest(unittest.TestCase):
    """Tests for _fingerprint_payload."""

    def test_returns_string(self) -> None:
        self.assertIsInstance(_fingerprint_payload({}), str)

    def test_returns_sha256_hex_string(self) -> None:
        result = _fingerprint_payload({"key": "value"})
        self.assertEqual(len(result), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in result))

    def test_is_deterministic(self) -> None:
        payload = {"a": 1, "b": [2, 3]}
        self.assertEqual(_fingerprint_payload(payload), _fingerprint_payload(payload))

    def test_different_payloads_different_fingerprints(self) -> None:
        fp1 = _fingerprint_payload({"value": 1})
        fp2 = _fingerprint_payload({"value": 2})
        self.assertNotEqual(fp1, fp2)

    def test_matches_manual_sha256(self) -> None:
        payload = {"hello": "world"}
        stable = _stable_json(payload)
        expected = hashlib.sha256(stable.encode("utf-8")).hexdigest()
        self.assertEqual(_fingerprint_payload(payload), expected)


class ObservationViewByKeyTest(unittest.TestCase):
    """Tests for _build_observation_view_registry and OBSERVATION_VIEW_BY_KEY."""

    def test_observation_view_by_key_is_non_empty(self) -> None:
        self.assertGreater(len(OBSERVATION_VIEW_BY_KEY), 0)

    def test_all_keys_map_to_observation_view_subclasses(self) -> None:
        from spider_cortex_sim.interface_contracts.observations import ObservationView
        for key, cls in OBSERVATION_VIEW_BY_KEY.items():
            self.assertTrue(
                issubclass(cls, ObservationView),
                f"{key!r} maps to {cls} which is not an ObservationView subclass",
            )

    def test_visual_key_maps_to_visual_observation(self) -> None:
        from spider_cortex_sim.interface_contracts.observations import VisualObservation
        self.assertIn("visual", OBSERVATION_VIEW_BY_KEY)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["visual"], VisualObservation)

    def test_action_context_key_is_present(self) -> None:
        self.assertIn("action_context", OBSERVATION_VIEW_BY_KEY)

    def test_observation_dims_has_non_zero_values(self) -> None:
        for key, dim in OBSERVATION_DIMS.items():
            self.assertGreater(dim, 0, f"Observation dim for {key!r} is zero")


class InterfaceRegistryTest(unittest.TestCase):
    """Tests for interface_registry()."""

    def setUp(self) -> None:
        self._registry = interface_registry()

    def test_returns_dict(self) -> None:
        self.assertIsInstance(self._registry, dict)

    def test_contains_schema_version(self) -> None:
        self.assertIn("schema_version", self._registry)
        self.assertEqual(self._registry["schema_version"], INTERFACE_REGISTRY_SCHEMA_VERSION)

    def test_contains_actions(self) -> None:
        self.assertIn("actions", self._registry)
        self.assertIsInstance(self._registry["actions"], list)

    def test_contains_proposal_interfaces(self) -> None:
        self.assertIn("proposal_interfaces", self._registry)
        self.assertIsInstance(self._registry["proposal_interfaces"], list)

    def test_contains_context_interfaces(self) -> None:
        self.assertIn("context_interfaces", self._registry)
        self.assertIsInstance(self._registry["context_interfaces"], list)

    def test_contains_interfaces_mapping(self) -> None:
        self.assertIn("interfaces", self._registry)
        self.assertIsInstance(self._registry["interfaces"], dict)

    def test_contains_observation_views(self) -> None:
        self.assertIn("observation_views", self._registry)
        self.assertIsInstance(self._registry["observation_views"], dict)

    def test_observation_views_have_field_names(self) -> None:
        for key, view_info in self._registry["observation_views"].items():
            self.assertIn("field_names", view_info, f"Missing field_names for {key!r}")

    def test_is_deterministic(self) -> None:
        registry1 = interface_registry()
        registry2 = interface_registry()
        self.assertEqual(
            json.dumps(registry1, sort_keys=True),
            json.dumps(registry2, sort_keys=True),
        )


class InterfaceRegistryFingerprintTest(unittest.TestCase):
    """Tests for interface_registry_fingerprint()."""

    def test_returns_string(self) -> None:
        self.assertIsInstance(interface_registry_fingerprint(), str)

    def test_is_sha256_hex_digest(self) -> None:
        fp = interface_registry_fingerprint()
        self.assertEqual(len(fp), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in fp))

    def test_is_deterministic(self) -> None:
        fp1 = interface_registry_fingerprint()
        fp2 = interface_registry_fingerprint()
        self.assertEqual(fp1, fp2)

    def test_matches_fingerprint_of_registry(self) -> None:
        fp = interface_registry_fingerprint()
        expected = _fingerprint_payload(interface_registry())
        self.assertEqual(fp, expected)

    def test_accepts_prebuilt_registry(self) -> None:
        registry = interface_registry()
        self.assertEqual(
            interface_registry_fingerprint(registry),
            _fingerprint_payload(registry),
        )


class ArchitectureSignatureTest(unittest.TestCase):
    """Tests for architecture_signature()."""

    def test_modular_signature_contains_expected_keys(self) -> None:
        sig = architecture_signature(proposal_backend="modular")
        expected_keys = [
            "schema_version", "proposal_backend", "registry_fingerprint",
            "actions", "modules", "contexts", "interface_versions",
            "proposal_order", "arbitration_network", "action_center",
            "motor_cortex", "fingerprint",
        ]
        for key in expected_keys:
            self.assertIn(key, sig, f"Missing key: {key}")

    def test_monolithic_signature_contains_expected_keys(self) -> None:
        sig = architecture_signature(proposal_backend="monolithic")
        self.assertIn("arbitration_network", sig)
        self.assertIn("action_center", sig)
        self.assertIn("motor_cortex", sig)
        self.assertNotIn("direct_policy", sig)

    def test_true_monolithic_signature_contains_direct_policy(self) -> None:
        sig = architecture_signature(proposal_backend="true_monolithic")
        self.assertIn("direct_policy", sig)
        self.assertNotIn("arbitration_network", sig)
        self.assertNotIn("action_center", sig)

    def test_invalid_proposal_backend_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            architecture_signature(proposal_backend="unknown_backend")

    def test_fingerprint_is_64_char_hex(self) -> None:
        sig = architecture_signature(proposal_backend="modular")
        fp = sig["fingerprint"]
        self.assertEqual(len(fp), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in fp))

    def test_fingerprint_changes_with_different_backend(self) -> None:
        fp_modular = architecture_signature(proposal_backend="modular")["fingerprint"]
        fp_monolithic = architecture_signature(proposal_backend="monolithic")["fingerprint"]
        self.assertNotEqual(fp_modular, fp_monolithic)

    def test_proposal_order_uses_module_names_for_modular(self) -> None:
        from spider_cortex_sim.interface_contracts.module_specs import MODULE_INTERFACES
        expected_names = [spec.name for spec in MODULE_INTERFACES]
        sig = architecture_signature(proposal_backend="modular")
        self.assertEqual(sig["proposal_order"], expected_names)

    def test_proposal_order_is_monolithic_policy_for_monolithic(self) -> None:
        sig = architecture_signature(proposal_backend="monolithic")
        self.assertEqual(sig["proposal_order"], ["monolithic_policy"])

    def test_proposal_order_is_true_monolithic_policy(self) -> None:
        sig = architecture_signature(proposal_backend="true_monolithic")
        self.assertEqual(sig["proposal_order"], ["true_monolithic_policy"])

    def test_custom_proposal_order_used_when_provided(self) -> None:
        custom_order = ["alert_center", "hunger_center"]
        sig = architecture_signature(
            proposal_backend="modular",
            proposal_order=custom_order,
        )
        self.assertEqual(sig["proposal_order"], custom_order)

    def test_arbitration_network_input_dim_matches_param(self) -> None:
        sig = architecture_signature(
            proposal_backend="modular",
            arbitration_input_dim=48,
        )
        self.assertEqual(sig["arbitration_network"]["input_dim"], 48)

    def test_arbitration_network_hidden_dim_matches_param(self) -> None:
        sig = architecture_signature(
            proposal_backend="modular",
            arbitration_hidden_dim=64,
        )
        self.assertEqual(sig["arbitration_network"]["hidden_dim"], 64)

    def test_arbitration_hidden_dim_is_capacity_trigger(self) -> None:
        sig = architecture_signature(
            proposal_backend="modular",
            arbitration_hidden_dim=48,
        )

        self.assertIn("capacity", sig)
        self.assertEqual(sig["capacity"]["arbitration_hidden_dim"], 48)
        self.assertEqual(sig["arbitration_network"]["hidden_dim"], 48)

    def test_integration_hidden_dim_feeds_integration_components(self) -> None:
        sig = architecture_signature(
            proposal_backend="modular",
            integration_hidden_dim=40,
        )

        self.assertEqual(sig["capacity"]["action_center_hidden_dim"], 40)
        self.assertEqual(sig["capacity"]["arbitration_hidden_dim"], 40)
        self.assertEqual(sig["capacity"]["motor_hidden_dim"], 40)
        self.assertEqual(sig["arbitration_network"]["hidden_dim"], 40)
        self.assertEqual(sig["action_center"]["hidden_dim"], 40)
        self.assertEqual(sig["motor_cortex"]["hidden_dim"], 40)

    def test_capacity_profile_name_in_signature(self) -> None:
        sig = architecture_signature(proposal_backend="modular", capacity_profile_name="large")
        self.assertEqual(sig["capacity_profile_name"], "large")

    def test_capacity_section_absent_when_no_capacity_params(self) -> None:
        sig = architecture_signature(proposal_backend="modular")
        self.assertNotIn("capacity", sig)

    def test_capacity_section_present_when_module_hidden_dims_provided(self) -> None:
        sig = architecture_signature(
            proposal_backend="modular",
            module_hidden_dims={"alert_center": 64},
        )
        self.assertIn("capacity", sig)
        self.assertIn("module_hidden_dims", sig["capacity"])

    def test_module_variants_are_included_in_signature(self) -> None:
        sig = architecture_signature(
            proposal_backend="modular",
            module_variants={"visual_cortex": 3},
        )
        self.assertEqual(
            sig["module_variants"]["visual_cortex"],
            {"level": 3, "interface": "visual_v3", "version": 7},
        )

    def test_module_variants_change_fingerprint(self) -> None:
        canonical = architecture_signature(proposal_backend="modular")
        visual_v3 = architecture_signature(
            proposal_backend="modular",
            module_variants={"visual_cortex": 3},
        )
        self.assertNotEqual(canonical["fingerprint"], visual_v3["fingerprint"])

    def test_learned_arbitration_true_by_default(self) -> None:
        sig = architecture_signature(proposal_backend="modular")
        self.assertTrue(sig["arbitration_network"]["learned"])

    def test_learned_arbitration_false_when_specified(self) -> None:
        sig = architecture_signature(
            proposal_backend="modular", learned_arbitration=False
        )
        self.assertFalse(sig["arbitration_network"]["learned"])


if __name__ == "__main__":
    unittest.main()
