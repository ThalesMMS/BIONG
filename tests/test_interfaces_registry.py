import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import spider_cortex_sim.interface_docs as interface_docs
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interface_docs import escape_markdown_cell, render_interfaces_markdown
from spider_cortex_sim.interfaces import (
    ALL_INTERFACES,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    ACTION_CONTEXT_INTERFACE,
    ALERT_CENTER_V1_INTERFACE,
    ALERT_CENTER_V2_INTERFACE,
    ALERT_CENTER_V3_INTERFACE,
    HUNGER_CENTER_V1_INTERFACE,
    HUNGER_CENTER_V2_INTERFACE,
    HUNGER_CENTER_V3_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    OBSERVATION_VIEW_BY_KEY,
    ORIENT_HEADINGS,
    SENSORY_CORTEX_V1_INTERFACE,
    SENSORY_CORTEX_V2_INTERFACE,
    SENSORY_CORTEX_V3_INTERFACE,
    SLEEP_CENTER_V1_INTERFACE,
    SLEEP_CENTER_V2_INTERFACE,
    SLEEP_CENTER_V3_INTERFACE,
    VISUAL_CORTEX_V1_INTERFACE,
    VISUAL_CORTEX_V2_INTERFACE,
    VISUAL_CORTEX_V3_INTERFACE,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    HomeostasisObservation,
    INTERFACE_VARIANTS,
    ModuleInterface,
    MotorContextObservation,
    ObservationView,
    PerceptionObservation,
    SensoryObservation,
    SignalSpec,
    SleepObservation,
    ThreatObservation,
    VARIANT_INTERFACES,
    VARIANT_MODULES,
    VisualObservation,
    _validate_exact_keys,
    architecture_signature,
    get_interface_variant,
    get_variant_levels,
    interface_registry,
    interface_registry_fingerprint,
    is_variant_interface,
    MODULE_INTERFACE_BY_NAME,
    validate_variant_interfaces,
)

class ModuleInterfaceMethodsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.interface = MODULE_INTERFACES[0]  # visual_cortex

    def _full_mapping(self, interface=None) -> dict:
        iface = interface or self.interface
        return {name: float(i) / iface.input_dim for i, name in enumerate(iface.signal_names)}

    def test_signal_names_matches_input_specs(self) -> None:
        names = self.interface.signal_names
        expected = tuple(s.name for s in self.interface.inputs)
        self.assertEqual(names, expected)

    def test_bind_values_produces_correct_mapping(self) -> None:
        values = list(range(self.interface.input_dim))
        mapping = self.interface.bind_values(values)
        self.assertEqual(tuple(mapping.keys()), self.interface.signal_names)
        for i, name in enumerate(self.interface.signal_names):
            self.assertAlmostEqual(mapping[name], float(i))

    def test_bind_values_accepts_numpy_array(self) -> None:
        arr = np.linspace(0.0, 1.0, self.interface.input_dim)
        mapping = self.interface.bind_values(arr)
        self.assertEqual(len(mapping), self.interface.input_dim)

    def test_bind_values_rejects_wrong_length(self) -> None:
        wrong = np.zeros(self.interface.input_dim + 2)
        with self.assertRaisesRegex(ValueError, self.interface.name):
            self.interface.bind_values(wrong)

    def test_bind_values_rejects_2d_array(self) -> None:
        arr_2d = np.zeros((self.interface.input_dim, 1))
        with self.assertRaisesRegex(ValueError, self.interface.name):
            self.interface.bind_values(arr_2d)

    def test_validate_signal_mapping_passes_for_correct_keys(self) -> None:
        mapping = self._full_mapping()
        self.interface.validate_signal_mapping(mapping)  # should not raise

    def test_validate_signal_mapping_raises_on_missing(self) -> None:
        mapping = self._full_mapping()
        first_key = self.interface.signal_names[0]
        mapping.pop(first_key)
        with self.assertRaisesRegex(ValueError, self.interface.name):
            self.interface.validate_signal_mapping(mapping)

    def test_validate_signal_mapping_raises_on_extra(self) -> None:
        mapping = self._full_mapping()
        mapping["rogue_signal"] = 0.0
        with self.assertRaisesRegex(ValueError, "rogue_signal"):
            self.interface.validate_signal_mapping(mapping)

    def test_vector_from_mapping_preserves_order(self) -> None:
        mapping = self._full_mapping()
        vector = self.interface.vector_from_mapping(mapping)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (self.interface.input_dim,))
        for i, name in enumerate(self.interface.signal_names):
            self.assertAlmostEqual(vector[i], mapping[name])

    def test_vector_from_mapping_dtype_is_float(self) -> None:
        mapping = self._full_mapping()
        vector = self.interface.vector_from_mapping(mapping)
        self.assertEqual(vector.dtype, np.float64)

    def test_action_context_interface_round_trip(self) -> None:
        iface = ACTION_CONTEXT_INTERFACE
        mapping = self._full_mapping(iface)
        vector = iface.vector_from_mapping(mapping)
        recovered = iface.bind_values(vector)
        for name in iface.signal_names:
            self.assertAlmostEqual(recovered[name], mapping[name])

    def test_motor_context_interface_round_trip(self) -> None:
        """
        Ensure mapping → vector → mapping round-trip preserves signal values for MOTOR_CONTEXT_INTERFACE.
        
        Asserts that for every signal in MOTOR_CONTEXT_INTERFACE, converting a full mapping to a vector and binding it back yields values approximately equal to the originals.
        """
        iface = MOTOR_CONTEXT_INTERFACE
        mapping = self._full_mapping(iface)
        vector = iface.vector_from_mapping(mapping)
        recovered = iface.bind_values(vector)
        for name in iface.signal_names:
            self.assertAlmostEqual(recovered[name], mapping[name])

class InterfaceVersionRegressionTest(unittest.TestCase):
    def test_action_space_output_interfaces_bump_versions(self) -> None:
        """
        Assert the interface registry reports the expected version numbers for core interfaces.
        
        Asserts that the current versions equal:
        - visual_cortex: 7
        - sensory_cortex: 2
        - hunger_center: 4
        - sleep_center: 5
        - alert_center: 8
        - perception_center: 1
        - homeostasis_center: 1
        - threat_center: 1
        - action_center_context: 3
        - motor_cortex_context: 5
        """
        versions = {interface.name: interface.version for interface in ALL_INTERFACES}
        self.assertEqual(versions["visual_cortex"], 7)
        self.assertEqual(versions["sensory_cortex"], 2)
        self.assertEqual(versions["hunger_center"], 4)
        self.assertEqual(versions["sleep_center"], 5)
        self.assertEqual(versions["alert_center"], 8)
        self.assertEqual(versions["perception_center"], 1)
        self.assertEqual(versions["homeostasis_center"], 1)
        self.assertEqual(versions["threat_center"], 1)
        self.assertEqual(versions["action_center_context"], 3)
        self.assertEqual(versions["motor_cortex_context"], 5)

    def test_older_registry_version_checkpoint_is_rejected_on_load(self) -> None:
        brain = SpiderBrain(seed=11)
        older_versions = {
            interface.name: interface.version - 1
            for interface in ALL_INTERFACES
            if interface.version > 1
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "brain"
            brain.save(save_path)

            baseline_reloaded = SpiderBrain(seed=999)
            baseline_reloaded.load(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            baseline_metadata = metadata_path.read_text(encoding="utf-8")
            for interface_name, older_version in older_versions.items():
                with self.subTest(interface_name=interface_name):
                    metadata = json.loads(baseline_metadata)
                    metadata["interface_registry"]["interfaces"][interface_name]["version"] = older_version
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

                    reloaded = SpiderBrain(seed=999)
                    with self.assertRaisesRegex(ValueError, "interface registry incompatible"):
                        reloaded.load(save_path)

class ObservationRegistryTest(unittest.TestCase):
    def test_observation_view_by_key_has_all_module_keys(self) -> None:
        for iface in MODULE_INTERFACES:
            self.assertIn(iface.observation_key, OBSERVATION_VIEW_BY_KEY)

    def test_observation_view_classes_declare_matching_observation_key(self) -> None:
        for key, view_cls in OBSERVATION_VIEW_BY_KEY.items():
            self.assertEqual(view_cls.observation_key, key)

    def test_observation_view_by_key_has_action_context(self) -> None:
        self.assertIn("action_context", OBSERVATION_VIEW_BY_KEY)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["action_context"], ActionContextObservation)

    def test_observation_view_by_key_has_motor_context(self) -> None:
        self.assertIn("motor_context", OBSERVATION_VIEW_BY_KEY)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["motor_context"], MotorContextObservation)

    def test_observation_view_by_key_maps_to_correct_types(self) -> None:
        self.assertIs(OBSERVATION_VIEW_BY_KEY["visual"], VisualObservation)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["sensory"], SensoryObservation)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["hunger"], HungerObservation)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["sleep"], SleepObservation)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["alert"], AlertObservation)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["perception"], PerceptionObservation)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["homeostasis"], HomeostasisObservation)
        self.assertIs(OBSERVATION_VIEW_BY_KEY["threat"], ThreatObservation)

    def test_observation_interface_by_key_has_all_module_keys(self) -> None:
        for iface in MODULE_INTERFACES:
            self.assertIn(iface.observation_key, OBSERVATION_INTERFACE_BY_KEY)
            self.assertIs(OBSERVATION_INTERFACE_BY_KEY[iface.observation_key], iface)

    def test_observation_interface_by_key_has_action_context(self) -> None:
        """
        Check that the observation-interface registry contains the "action_context" key and maps it to ACTION_CONTEXT_INTERFACE.
        """
        self.assertIn("action_context", OBSERVATION_INTERFACE_BY_KEY)
        self.assertIs(OBSERVATION_INTERFACE_BY_KEY["action_context"], ACTION_CONTEXT_INTERFACE)

    def test_observation_interface_by_key_has_motor_context(self) -> None:
        self.assertIn("motor_context", OBSERVATION_INTERFACE_BY_KEY)
        self.assertIs(OBSERVATION_INTERFACE_BY_KEY["motor_context"], MOTOR_CONTEXT_INTERFACE)

    def test_observation_view_fields_match_interface_signal_names(self) -> None:
        for key, view_cls in OBSERVATION_VIEW_BY_KEY.items():
            iface = OBSERVATION_INTERFACE_BY_KEY[key]
            self.assertEqual(
                view_cls.field_names(),
                iface.signal_names,
                msg=f"Field name mismatch for key '{key}'",
            )

    def test_architecture_signature_tracks_proposal_order(self) -> None:
        signature = architecture_signature()
        self.assertEqual(
            signature["proposal_order"],
            [iface.name for iface in MODULE_INTERFACES],
        )
        self.assertEqual(
            [slot["module"] for slot in signature["action_center"]["proposal_slots"]],
            [iface.name for iface in MODULE_INTERFACES],
        )

    def test_architecture_signature_rejects_unknown_proposal_order(self) -> None:
        with self.assertRaises(ValueError):
            architecture_signature(proposal_order=["not_a_module"])

    def test_architecture_signature_rejects_duplicate_proposal_order(self) -> None:
        with self.assertRaises(ValueError):
            architecture_signature(
                proposal_order=["hunger_center", "hunger_center"],
            )

    def test_architecture_signature_declares_inter_stage_inputs(self) -> None:
        signature = architecture_signature()
        action_inputs = signature["action_center"]["inter_stage_inputs"]
        motor_inputs = signature["motor_cortex"]["inter_stage_inputs"]
        self.assertEqual(len(action_inputs), len(MODULE_INTERFACES))
        self.assertTrue(all(entry["name"] == "proposal_logits" for entry in action_inputs))
        self.assertTrue(all(entry["size"] == len(signature["actions"]) for entry in action_inputs))
        self.assertEqual(motor_inputs[0]["name"], "intent")
        self.assertEqual(motor_inputs[0]["size"], len(signature["actions"]))

    def test_registry_contains_all_interfaces_with_versions(self) -> None:
        registry = interface_registry()
        self.assertEqual(
            set(registry["interfaces"].keys()),
            {spec.name for spec in (*ALL_INTERFACES, *VARIANT_INTERFACES)},
        )
        for spec in (*ALL_INTERFACES, *VARIANT_INTERFACES):
            entry = registry["interfaces"][spec.name]
            self.assertEqual(entry["version"], spec.version)
            self.assertIn("compatibility", entry)
            self.assertEqual(
                entry["compatibility"]["save_policy"],
                "exact_match_required",
            )

    def test_architecture_signature_exposes_registry_fingerprint_and_versions(self) -> None:
        signature = architecture_signature()
        self.assertEqual(
            signature["registry_fingerprint"],
            interface_registry_fingerprint(),
        )
        for spec in (*ALL_INTERFACES, *VARIANT_INTERFACES):
            self.assertEqual(signature["interface_versions"][spec.name], spec.version)

    def test_architecture_signature_fingerprint_is_stable(self) -> None:
        self.assertEqual(
            architecture_signature()["fingerprint"],
            architecture_signature()["fingerprint"],
        )

    def test_architecture_signature_excludes_human_description_from_payload(self) -> None:
        self.assertNotIn("architecture_description", architecture_signature())

    def test_rendered_interfaces_doc_is_exactly_in_sync(self) -> None:
        docs_path = Path(__file__).resolve().parents[1] / "docs" / "interfaces.md"
        self.assertEqual(
            docs_path.read_text(encoding="utf-8"),
            render_interfaces_markdown(),
        )
