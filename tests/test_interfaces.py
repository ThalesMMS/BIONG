import unittest
from pathlib import Path

import numpy as np

from spider_cortex_sim.interfaces import (
    ALL_INTERFACES,
    ACTION_CONTEXT_INTERFACE,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    OBSERVATION_VIEW_BY_KEY,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    ModuleInterface,
    MotorContextObservation,
    ObservationView,
    SensoryObservation,
    SignalSpec,
    SleepObservation,
    VisualObservation,
    _validate_exact_keys,
    architecture_signature,
    interface_registry,
    interface_registry_fingerprint,
    render_interfaces_markdown,
)


class ValidateExactKeysTest(unittest.TestCase):
    def test_valid_exact_match_raises_nothing(self) -> None:
        _validate_exact_keys("test", ["a", "b", "c"], {"a": 1.0, "b": 2.0, "c": 3.0})

    def test_missing_key_raises_value_error_with_label(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("MyLabel", ["x", "y"], {"x": 0.0})
        self.assertIn("MyLabel", str(ctx.exception))
        self.assertIn("y", str(ctx.exception))

    def test_extra_key_raises_value_error_listing_extras(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("SomeInterface", ["x"], {"x": 0.0, "z": 1.0})
        self.assertIn("SomeInterface", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_both_missing_and_extra_keys_reported(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("Combo", ["a", "b"], {"a": 1.0, "c": 2.0})
        msg = str(ctx.exception)
        self.assertIn("b", msg)
        self.assertIn("c", msg)

    def test_empty_expected_and_empty_values_is_valid(self) -> None:
        _validate_exact_keys("empty", [], {})

    def test_empty_expected_with_extra_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_exact_keys("empty", [], {"x": 1.0})


class ObservationViewTest(unittest.TestCase):
    def _make_visual(self) -> VisualObservation:
        """
        Create a representative VisualObservation instance populated with deterministic example values for use in tests.
        
        The returned observation models a scenario with visible food (with certainty and a nonzero displacement), no visible shelter or predator, a defined heading vector, nonzero food trace strength, and day/night indicators set to daytime. Other numeric fields are set to zero where no stimulus is present.
        
        Returns:
            VisualObservation: A test-ready VisualObservation populated with sample float values.
        """
        return VisualObservation(
            food_visible=1.0,
            food_certainty=0.8,
            food_occluded=0.0,
            food_dx=0.5,
            food_dy=-0.5,
            shelter_visible=0.0,
            shelter_certainty=0.0,
            shelter_occluded=0.0,
            shelter_dx=0.0,
            shelter_dy=0.0,
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_occluded=0.0,
            predator_dx=0.0,
            predator_dy=0.0,
            heading_dx=1.0,
            heading_dy=0.0,
            food_trace_strength=0.3,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.0,
            predator_motion_salience=0.0,
            day=1.0,
            night=0.0,
        )

    def test_field_names_returns_ordered_tuple(self) -> None:
        names = VisualObservation.field_names()
        self.assertIsInstance(names, tuple)
        self.assertEqual(names[0], "food_visible")
        self.assertEqual(names[-1], "night")
        self.assertEqual(len(names), 23)

    def test_as_mapping_returns_float_values(self) -> None:
        view = self._make_visual()
        mapping = view.as_mapping()
        self.assertIsInstance(mapping, dict)
        self.assertEqual(set(mapping.keys()), set(VisualObservation.field_names()))
        for v in mapping.values():
            self.assertIsInstance(v, float)
        self.assertAlmostEqual(mapping["food_visible"], 1.0)
        self.assertAlmostEqual(mapping["food_dx"], 0.5)
        self.assertAlmostEqual(mapping["day"], 1.0)

    def test_from_mapping_round_trips(self) -> None:
        original = self._make_visual()
        mapping = original.as_mapping()
        recovered = VisualObservation.from_mapping(mapping)
        self.assertEqual(original, recovered)

    def test_from_mapping_converts_to_float(self) -> None:
        mapping = {name: i for i, name in enumerate(VisualObservation.field_names())}
        view = VisualObservation.from_mapping(mapping)
        for v in view.as_mapping().values():
            self.assertIsInstance(v, float)

    def test_from_mapping_rejects_missing_field(self) -> None:
        mapping = self._make_visual().as_mapping()
        mapping.pop("food_visible")
        with self.assertRaisesRegex(ValueError, "VisualObservation"):
            VisualObservation.from_mapping(mapping)

    def test_from_mapping_rejects_extra_field(self) -> None:
        mapping = self._make_visual().as_mapping()
        mapping["extra_field"] = 0.0
        with self.assertRaisesRegex(ValueError, "extra_field"):
            VisualObservation.from_mapping(mapping)

    def test_sensory_observation_field_count(self) -> None:
        self.assertEqual(len(SensoryObservation.field_names()), 12)

    def test_hunger_observation_field_count(self) -> None:
        self.assertEqual(len(HungerObservation.field_names()), 16)

    def test_sleep_observation_field_count(self) -> None:
        self.assertEqual(len(SleepObservation.field_names()), 19)

    def test_alert_observation_field_count(self) -> None:
        self.assertEqual(len(AlertObservation.field_names()), 23)

    def test_action_context_observation_field_count(self) -> None:
        """
        Verify that ActionContextObservation declares exactly 16 field names.
        """
        self.assertEqual(len(ActionContextObservation.field_names()), 16)

    def test_action_context_observation_round_trip(self) -> None:
        """
        Verifies that an ActionContextObservation can be converted to a mapping and reconstructed unchanged.
        
        Constructs an ActionContextObservation with representative values, converts it to a mapping via as_mapping(), recreates it with from_mapping(), and asserts the reconstructed object equals the original.
        """
        view = ActionContextObservation(
            hunger=0.3,
            fatigue=0.5,
            health=0.9,
            recent_pain=0.0,
            recent_contact=0.1,
            on_food=0.0,
            on_shelter=1.0,
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_dist=0.8,
            day=0.0,
            night=1.0,
            last_move_dx=0.0,
            last_move_dy=-1.0,
            sleep_debt=0.4,
            shelter_role_level=0.75,
        )
        recovered = ActionContextObservation.from_mapping(view.as_mapping())
        self.assertEqual(view, recovered)

    def test_motor_context_observation_field_count(self) -> None:
        self.assertEqual(len(MotorContextObservation.field_names()), 10)

    def test_motor_context_observation_round_trip(self) -> None:
        """
        Verify MotorContextObservation round-trips through mapping conversion.
        
        Constructs a MotorContextObservation, converts it to a mapping, reconstructs a view
        from that mapping, and asserts the reconstructed view equals the original.
        """
        view = MotorContextObservation(
            on_food=0.0,
            on_shelter=1.0,
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_dist=0.8,
            day=0.0,
            night=1.0,
            last_move_dx=0.0,
            last_move_dy=-1.0,
            shelter_role_level=0.75,
        )
        recovered = MotorContextObservation.from_mapping(view.as_mapping())
        self.assertEqual(view, recovered)


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
        Verifies that converting a mapping to a vector and back preserves values for the MOTOR_CONTEXT_INTERFACE.
        
        Asserts that for each signal name defined by MOTOR_CONTEXT_INTERFACE, binding the vector produced by
        vector_from_mapping returns values approximately equal to the original mapping.
        """
        iface = MOTOR_CONTEXT_INTERFACE
        mapping = self._full_mapping(iface)
        vector = iface.vector_from_mapping(mapping)
        recovered = iface.bind_values(vector)
        for name in iface.signal_names:
            self.assertAlmostEqual(recovered[name], mapping[name])


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
            {spec.name for spec in ALL_INTERFACES},
        )
        for spec in ALL_INTERFACES:
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
        for spec in ALL_INTERFACES:
            self.assertEqual(signature["interface_versions"][spec.name], spec.version)

    def test_architecture_signature_fingerprint_is_stable(self) -> None:
        self.assertEqual(
            architecture_signature()["fingerprint"],
            architecture_signature()["fingerprint"],
        )

    def test_rendered_interfaces_doc_is_exactly_in_sync(self) -> None:
        docs_path = Path(__file__).resolve().parents[1] / "docs" / "interfaces.md"
        self.assertEqual(
            docs_path.read_text(encoding="utf-8"),
            render_interfaces_markdown(),
        )


class ObservationContractTest(unittest.TestCase):
    def _interfaces(self):
        """
        Return the ordered sequence of observation/module interfaces to include in contract tests.
        
        Returns:
            interfaces (list): A list containing all entries from MODULE_INTERFACES followed by ACTION_CONTEXT_INTERFACE and MOTOR_CONTEXT_INTERFACE.
        """
        return [*MODULE_INTERFACES, ACTION_CONTEXT_INTERFACE, MOTOR_CONTEXT_INTERFACE]

    def _sample_mapping(self, interface):
        """
        Create a deterministic mapping from an interface's signal names to scaled float values.
        
        Parameters:
            interface: Observation interface exposing `signal_names` (ordered iterable of str) and `input_dim` (int).
        
        Returns:
            dict[str, float]: Mapping where each signal name maps to (index + 1) / (input_dim + 1) as a float, with `index` being the signal's zero-based position in `interface.signal_names`.
        """
        denominator = interface.input_dim + 1
        return {
            name: float((idx + 1) / denominator)
            for idx, name in enumerate(interface.signal_names)
        }

    def test_round_trip_mapping_vector_and_typed_view_for_all_interfaces(self) -> None:
        """
        Verify that every observation interface round-trips a sample mapping through vector conversion and produces a matching typed view.
        
        For each interface returned by _interfaces(), constructs a deterministic sample mapping, converts it to a vector with vector_from_mapping, rebinds it with bind_values, and asserts that:
        - the rebound mapping's keys equal interface.signal_names in order,
        - each signal value in the rebound mapping matches the original sample (using approximate equality),
        - a typed observation view built from the rebound mapping yields the same mapping via as_mapping().
        """
        for interface in self._interfaces():
            with self.subTest(interface=interface.name):
                expected_mapping = self._sample_mapping(interface)

                vector = interface.vector_from_mapping(expected_mapping)
                rebound_mapping = interface.bind_values(vector)
                view_type = OBSERVATION_VIEW_BY_KEY[interface.observation_key]
                view = view_type.from_mapping(rebound_mapping)

                self.assertEqual(tuple(rebound_mapping.keys()), interface.signal_names)
                for signal_name in interface.signal_names:
                    self.assertAlmostEqual(rebound_mapping[signal_name], expected_mapping[signal_name])
                self.assertEqual(view.as_mapping(), rebound_mapping)

    def test_vector_from_mapping_rejects_missing_signal(self) -> None:
        """
        Verifies that an interface rejects mappings missing a required signal.
        
        Constructs a sample mapping for the first module interface, removes its first signal,
        and asserts that calling `vector_from_mapping` raises a `ValueError` whose message
        includes the interface's name.
        Raises:
            ValueError: if a required signal is missing (message contains the interface name).
        """
        interface = MODULE_INTERFACES[0]
        missing_signal = interface.signal_names[0]
        mapping = self._sample_mapping(interface)
        mapping.pop(missing_signal)

        with self.assertRaisesRegex(ValueError, interface.name):
            interface.vector_from_mapping(mapping)

    def test_vector_from_mapping_rejects_extra_signal(self) -> None:
        interface = MODULE_INTERFACES[0]
        mapping = self._sample_mapping(interface)
        mapping["unexpected_signal"] = 0.5

        with self.assertRaisesRegex(ValueError, "unexpected_signal"):
            interface.vector_from_mapping(mapping)


class SignalDescriptionRegressionTest(unittest.TestCase):
    """Regression tests for interface signal descriptions translated in this PR."""

    def test_all_signal_descriptions_are_non_empty(self) -> None:
        for interface in ALL_INTERFACES:
            for signal in interface.inputs:
                self.assertTrue(
                    signal.description.strip(),
                    f"Signal {signal.name!r} in {interface.name!r} has empty description",
                )

    def test_all_signal_descriptions_are_strings(self) -> None:
        for interface in ALL_INTERFACES:
            for signal in interface.inputs:
                self.assertIsInstance(
                    signal.description,
                    str,
                    f"Signal {signal.name!r} in {interface.name!r} description is not a string",
                )

    def test_visual_cortex_signal_descriptions_are_in_english(self) -> None:
        # Spot-check a few well-known signals to confirm the Portuguese descriptions
        # were replaced with English ones in this PR.
        visual_iface = next(i for i in ALL_INTERFACES if i.name == "visual_cortex")
        by_name = {s.name: s.description for s in visual_iface.inputs}
        # These descriptions must not contain Portuguese keywords from the old text
        for signal_name, description in by_name.items():
            self.assertNotIn(
                "se há",
                description.lower(),
                f"Signal {signal_name!r} still has Portuguese text: {description!r}",
            )
            self.assertNotIn(
                "durante o",
                description.lower(),
                f"Signal {signal_name!r} still has Portuguese text: {description!r}",
            )

    def test_sensory_cortex_signal_descriptions_are_in_english(self) -> None:
        sensory_iface = next(i for i in ALL_INTERFACES if i.name == "sensory_cortex")
        by_name = {s.name: s.description for s in sensory_iface.inputs}
        for signal_name, description in by_name.items():
            self.assertNotIn(
                "causada",
                description.lower(),
                f"Signal {signal_name!r} still has Portuguese text: {description!r}",
            )

    def test_signal_descriptions_do_not_contain_portuguese_common_words(self) -> None:
        portuguese_markers = [
            "está",
            "aranha",
            "predador",
            "abrigo",
            "comida",
            "direção",
            "distância",
            "visível",
            "intensidade",
            "memória",
        ]
        for interface in ALL_INTERFACES:
            for signal in interface.inputs:
                desc_lower = signal.description.lower()
                for marker in portuguese_markers:
                    self.assertNotIn(
                        marker,
                        desc_lower,
                        f"Signal {signal.name!r} in {interface.name!r} still has Portuguese "
                        f"word {marker!r}: {signal.description!r}",
                    )

    def test_fingerprint_matches_current_registry(self) -> None:
        # Regression: ensure the fingerprint in docs/interfaces.md matches the actual registry.
        # This catches future description changes that update the registry but not the docs.
        docs_path = Path(__file__).parent.parent / "docs" / "interfaces.md"
        if not docs_path.exists():
            self.skipTest("docs/interfaces.md not found")
        content = docs_path.read_text(encoding="utf-8")
        expected_fingerprint = interface_registry_fingerprint()
        self.assertIn(
            expected_fingerprint,
            content,
            "docs/interfaces.md fingerprint does not match the current registry fingerprint",
        )


if __name__ == "__main__":
    unittest.main()
