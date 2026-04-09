import unittest

import numpy as np

from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    MotorContextObservation,
    SensoryObservation,
    SleepObservation,
    VisualObservation,
)
from spider_cortex_sim.maps import CLUTTER
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.perception import (
    OBSERVATION_LEAKAGE_AUDIT,
    PerceivedTarget,
    build_action_context_observation,
    build_alert_observation,
    build_hunger_observation,
    build_motor_context_observation,
    observation_leakage_audit,
    build_sensory_observation,
    build_sleep_observation,
    build_visual_observation,
    has_line_of_sight,
    line_cells,
    lizard_detects_spider,
    predator_visible_to_spider,
    serialize_observation_view,
    smell_gradient,
    visible_object,
    visibility_confidence,
    visible_range,
)
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.world import SpiderWorld


class PerceptionBuildersTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Initialize test fixtures for PerceptionBuildersTest.
        
        Creates a SpiderWorld seeded with 7 and resets it, and defines three PerceivedTarget fixtures:
        - `null_percept`: all visibility/certainty/occluded set to 0.0, dx/dy = 0.0, dist = 99.
        - `food_percept`: visible=1.0, certainty=0.9, occluded=0.0, dx=0.5, dy=0.0, dist=3, with a concrete position.
        - `predator_percept`: visible=1.0, certainty=0.8, occluded=0.0, dx=-0.3, dy=0.6, dist=4, with a concrete position.
        """
        self.world = SpiderWorld(seed=7)
        self.world.reset(seed=7)
        self.null_percept = PerceivedTarget(
            visible=0.0,
            certainty=0.0,
            occluded=0.0,
            dx=0.0,
            dy=0.0,
            dist=99,
        )
        self.food_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.9,
            occluded=0.0,
            dx=0.5,
            dy=0.0,
            dist=3,
            position=(8, 5),
        )
        self.predator_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.8,
            occluded=0.0,
            dx=-0.3,
            dy=0.6,
            dist=4,
            position=(3, 7),
        )

    def test_serialize_visual_observation_produces_correct_shape(self) -> None:
        view = VisualObservation(
            food_visible=1.0,
            food_certainty=0.9,
            food_occluded=0.0,
            food_dx=0.5,
            food_dy=0.0,
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
            food_trace_strength=0.1,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.0,
            predator_motion_salience=0.0,
            day=1.0,
            night=0.0,
        )
        vector = serialize_observation_view("visual", view)
        self.assertEqual(vector.shape, (23,))

    def test_serialize_observation_view_values_match_as_mapping(self) -> None:
        view = VisualObservation(
            food_visible=0.0,
            food_certainty=0.0,
            food_occluded=0.0,
            food_dx=0.0,
            food_dy=0.0,
            shelter_visible=1.0,
            shelter_certainty=0.7,
            shelter_occluded=0.0,
            shelter_dx=-0.4,
            shelter_dy=0.2,
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_occluded=0.0,
            predator_dx=0.0,
            predator_dy=0.0,
            heading_dx=0.0,
            heading_dy=1.0,
            food_trace_strength=0.0,
            shelter_trace_strength=0.3,
            predator_trace_strength=0.0,
            predator_motion_salience=0.0,
            day=0.0,
            night=1.0,
        )
        vector = serialize_observation_view("visual", view)
        rebound = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(vector)
        for name, value in view.as_mapping().items():
            self.assertAlmostEqual(rebound[name], value)

    def test_serialize_action_context_produces_correct_shape(self) -> None:
        obs = self.world.observe()
        self.assertEqual(obs["action_context"].shape, (ACTION_CONTEXT_INTERFACE.input_dim,))

    def test_serialize_motor_context_produces_correct_shape(self) -> None:
        obs = self.world.observe()
        self.assertEqual(obs["motor_context"].shape, (MOTOR_CONTEXT_INTERFACE.input_dim,))

    def test_build_visual_observation_fields_match_percepts(self) -> None:
        visual = build_visual_observation(
            food_view=self.food_percept,
            shelter_view=self.null_percept,
            predator_view=self.predator_percept,
            heading_dx=1.0,
            heading_dy=0.0,
            food_trace_strength=0.25,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.5,
            predator_motion_salience_value=0.1,
            day=1.0,
            night=0.0,
        )
        self.assertIsInstance(visual, VisualObservation)
        self.assertAlmostEqual(visual.food_visible, self.food_percept.visible)
        self.assertAlmostEqual(visual.food_certainty, self.food_percept.certainty)
        self.assertAlmostEqual(visual.food_dx, self.food_percept.dx)
        self.assertAlmostEqual(visual.predator_visible, self.predator_percept.visible)
        self.assertAlmostEqual(visual.predator_dx, self.predator_percept.dx)
        self.assertAlmostEqual(visual.heading_dx, 1.0)
        self.assertAlmostEqual(visual.predator_trace_strength, 0.5)

    def test_build_visual_observation_null_percept(self) -> None:
        visual = build_visual_observation(
            food_view=self.null_percept,
            shelter_view=self.null_percept,
            predator_view=self.null_percept,
            heading_dx=0.0,
            heading_dy=1.0,
            food_trace_strength=0.0,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.0,
            predator_motion_salience_value=0.0,
            day=0.5,
            night=0.5,
        )
        self.assertAlmostEqual(visual.food_visible, 0.0)
        self.assertAlmostEqual(visual.shelter_visible, 0.0)
        self.assertAlmostEqual(visual.predator_visible, 0.0)

    def test_build_sensory_observation_maps_state_fields(self) -> None:
        self.world.state.recent_pain = 0.3
        self.world.state.recent_contact = 0.1
        self.world.state.health = 0.85
        self.world.state.hunger = 0.6
        self.world.state.fatigue = 0.4
        sensory = build_sensory_observation(
            self.world,
            food_smell_strength=0.7,
            food_smell_dx=0.5,
            food_smell_dy=-0.2,
            predator_smell_strength=0.1,
            predator_smell_dx=0.0,
            predator_smell_dy=0.0,
            light=0.9,
        )
        self.assertIsInstance(sensory, SensoryObservation)
        self.assertAlmostEqual(sensory.recent_pain, 0.3)
        self.assertAlmostEqual(sensory.health, 0.85)
        self.assertAlmostEqual(sensory.hunger, 0.6)
        self.assertAlmostEqual(sensory.food_smell_strength, 0.7)

    def test_build_sensory_observation_serializes_correctly(self) -> None:
        sensory = build_sensory_observation(
            self.world,
            food_smell_strength=0.0,
            food_smell_dx=0.0,
            food_smell_dy=0.0,
            predator_smell_strength=0.0,
            predator_smell_dx=0.0,
            predator_smell_dy=0.0,
            light=0.5,
        )
        self.assertEqual(serialize_observation_view("sensory", sensory).shape, (12,))

    def test_build_hunger_observation_populates_all_fields(self) -> None:
        self.world.state.hunger = 0.75
        hunger = build_hunger_observation(
            self.world,
            on_food=1.0,
            food_view=self.food_percept,
            food_smell_strength=0.5,
            food_smell_dx=0.3,
            food_smell_dy=-0.1,
            food_trace=(0.4, -0.2, 0.6),
            food_memory=(0.6, -0.2, 0.4),
        )
        self.assertIsInstance(hunger, HungerObservation)
        self.assertAlmostEqual(hunger.hunger, 0.75)
        self.assertAlmostEqual(hunger.food_trace_dx, 0.4)
        self.assertAlmostEqual(hunger.food_memory_dx, 0.6)
        self.assertAlmostEqual(hunger.food_memory_age, 0.4)

    def test_build_hunger_observation_zero_memory(self) -> None:
        hunger = build_hunger_observation(
            self.world,
            on_food=0.0,
            food_view=self.null_percept,
            food_smell_strength=0.0,
            food_smell_dx=0.0,
            food_smell_dy=0.0,
            food_trace=(0.0, 0.0, 0.0),
            food_memory=(0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(hunger.food_memory_age, 1.0)

    def test_observation_leakage_audit_marks_predator_dist_as_high_risk(self) -> None:
        audit = observation_leakage_audit()
        self.assertEqual(audit["predator_dist"]["risk"], "high")
        self.assertEqual(set(audit.keys()), set(OBSERVATION_LEAKAGE_AUDIT.keys()))

    def test_observation_leakage_audit_returns_deep_copy_not_reference(self) -> None:
        audit1 = observation_leakage_audit()
        audit2 = observation_leakage_audit()
        audit1["predator_dist"]["risk"] = "mutated"
        self.assertEqual(audit2["predator_dist"]["risk"], "high")

    def test_observation_leakage_audit_home_vector_is_high_risk_world_derived(self) -> None:
        audit = observation_leakage_audit()
        self.assertEqual(audit["home_vector"]["risk"], "high")
        self.assertEqual(audit["home_vector"]["classification"], "world_derived_navigation_hint")

    def test_observation_leakage_audit_shelter_memory_vector_is_high_risk(self) -> None:
        audit = observation_leakage_audit()
        self.assertEqual(audit["shelter_memory_vector"]["risk"], "high")
        self.assertEqual(audit["shelter_memory_vector"]["classification"], "world_owned_memory")

    def test_observation_leakage_audit_low_risk_entries_use_direct_perception(self) -> None:
        audit = observation_leakage_audit()
        low_risk = [name for name, data in audit.items() if data["risk"] == "low"]
        for name in low_risk:
            self.assertEqual(
                audit[name]["classification"],
                "direct_perception",
                f"Low-risk entry {name!r} expected classification 'direct_perception'",
            )

    def test_observation_leakage_audit_all_entries_have_required_fields(self) -> None:
        audit = observation_leakage_audit()
        required_fields = {"classification", "risk", "modules", "source", "notes"}
        for key, metadata in audit.items():
            for field in required_fields:
                self.assertIn(
                    field,
                    metadata,
                    f"Observation leakage entry {key!r} missing field {field!r}",
                )

    def test_observation_leakage_audit_modules_are_lists(self) -> None:
        audit = observation_leakage_audit()
        for key, metadata in audit.items():
            self.assertIsInstance(
                metadata["modules"],
                list,
                f"Observation leakage entry {key!r} 'modules' should be a list",
            )
            self.assertGreater(
                len(metadata["modules"]),
                0,
                f"Observation leakage entry {key!r} should have at least one module",
            )

    def test_observation_leakage_audit_risk_values_are_valid(self) -> None:
        audit = observation_leakage_audit()
        valid_risk_levels = {"low", "medium", "high"}
        for key, metadata in audit.items():
            self.assertIn(
                metadata["risk"],
                valid_risk_levels,
                f"Entry {key!r} has unexpected risk level {metadata['risk']!r}",
            )

    def test_observation_leakage_audit_predator_memory_vector_is_medium_risk(self) -> None:
        audit = observation_leakage_audit()
        self.assertEqual(audit["predator_memory_vector"]["risk"], "medium")

    def test_build_sleep_observation_maps_state_and_args(self) -> None:
        self.world.state.fatigue = 0.65
        self.world.state.hunger = 0.2
        self.world.state.health = 0.9
        self.world.state.recent_pain = 0.0
        self.world.state.sleep_debt = 0.5
        sleep = build_sleep_observation(
            self.world,
            on_shelter=1.0,
            night=1.0,
            home_dx=-0.3,
            home_dy=0.1,
            home_dist=0.4,
            sleep_phase_level=0.7,
            rest_streak_norm=0.5,
            shelter_role_level=0.8,
            shelter_trace=(0.1, -0.2, 0.55),
            shelter_memory=(0.2, -0.1, 0.3),
        )
        self.assertIsInstance(sleep, SleepObservation)
        self.assertAlmostEqual(sleep.sleep_debt, 0.5)
        self.assertAlmostEqual(sleep.shelter_trace_strength, 0.55)
        self.assertAlmostEqual(sleep.shelter_memory_dx, 0.2)

    def test_build_sleep_observation_serializes_to_correct_shape(self) -> None:
        sleep = build_sleep_observation(
            self.world,
            on_shelter=0.0,
            night=0.0,
            home_dx=0.0,
            home_dy=0.0,
            home_dist=1.0,
            sleep_phase_level=0.0,
            rest_streak_norm=0.0,
            shelter_role_level=0.0,
            shelter_trace=(0.0, 0.0, 0.0),
            shelter_memory=(0.0, 0.0, 1.0),
        )
        self.assertEqual(serialize_observation_view("sleep", sleep).shape, (19,))

    def test_build_alert_observation_maps_percept_and_state(self) -> None:
        self.world.state.recent_pain = 0.2
        self.world.state.recent_contact = 0.0
        alert = build_alert_observation(
            self.world,
            predator_view=self.predator_percept,
            predator_dist_norm=0.35,
            predator_smell_strength=0.1,
            predator_motion_salience_value=0.1,
            home_dx=0.4,
            home_dy=-0.3,
            on_shelter=0.0,
            night=0.0,
            predator_trace=(0.2, -0.4, 0.7),
            predator_memory=(0.7, -0.2, 0.5),
            escape_memory=(0.0, 0.0, 1.0),
        )
        self.assertIsInstance(alert, AlertObservation)
        self.assertAlmostEqual(alert.predator_dist, 0.35)
        self.assertAlmostEqual(alert.predator_trace_strength, 0.7)
        self.assertAlmostEqual(alert.predator_memory_dx, 0.7)

    def test_build_alert_observation_serializes_to_correct_shape(self) -> None:
        alert = build_alert_observation(
            self.world,
            predator_view=self.null_percept,
            predator_dist_norm=1.0,
            predator_smell_strength=0.0,
            predator_motion_salience_value=0.0,
            home_dx=0.0,
            home_dy=0.0,
            on_shelter=0.0,
            night=0.0,
            predator_trace=(0.0, 0.0, 0.0),
            predator_memory=(0.0, 0.0, 1.0),
            escape_memory=(0.0, 0.0, 1.0),
        )
        self.assertEqual(serialize_observation_view("alert", alert).shape, (23,))

    def test_build_action_context_observation_maps_state_fields(self) -> None:
        self.world.state.hunger = 0.4
        self.world.state.fatigue = 0.3
        self.world.state.health = 0.95
        self.world.state.sleep_debt = 0.25
        self.world.state.last_move_dx = 1
        ac = build_action_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=1.0,
            predator_view=self.null_percept,
            predator_dist_norm=0.9,
            day=1.0,
            night=0.0,
            shelter_role_level=0.6,
        )
        self.assertIsInstance(ac, ActionContextObservation)
        self.assertAlmostEqual(ac.last_move_dx, 1.0)
        self.assertAlmostEqual(ac.sleep_debt, 0.25)

    def test_build_action_context_observation_serializes_to_correct_shape(self) -> None:
        ac = build_action_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            predator_dist_norm=1.0,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        self.assertEqual(serialize_observation_view("action_context", ac).shape, (ACTION_CONTEXT_INTERFACE.input_dim,))

    def test_build_motor_context_observation_serializes_to_correct_shape(self) -> None:
        mc = build_motor_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            predator_dist_norm=1.0,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        self.assertEqual(serialize_observation_view("motor_context", mc).shape, (MOTOR_CONTEXT_INTERFACE.input_dim,))

    def test_observe_all_modalities_match_interface_dims(self) -> None:
        obs = self.world.observe()
        for key in ("visual", "sensory", "hunger", "sleep", "alert", "action_context", "motor_context"):
            iface = OBSERVATION_INTERFACE_BY_KEY[key]
            self.assertEqual(obs[key].shape, (iface.input_dim,))

    def test_observe_motor_extra_equals_motor_context(self) -> None:
        obs = self.world.observe()
        np.testing.assert_array_equal(obs["motor_extra"], obs["motor_context"])

    def test_observe_builder_values_consistent_with_state(self) -> None:
        """
        Verifies that the `action_context` observation reflects the world's `hunger` and `fatigue` state fields.
        
        Reconstructs an ActionContextObservation from the serialized `action_context` vector and asserts `hunger` and `fatigue` match the world state within 1e-5.
        """
        self.world.state.hunger = 0.77
        self.world.state.fatigue = 0.33
        obs = self.world.observe()
        ac = ActionContextObservation.from_mapping(
            ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        )
        self.assertAlmostEqual(ac.hunger, 0.77, places=5)
        self.assertAlmostEqual(ac.fatigue, 0.33, places=5)

    def test_observe_visual_day_night_consistent_with_meta(self) -> None:
        obs = self.world.observe()
        visual_mapping = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(obs["visual"])
        self.assertEqual(obs["meta"]["day"], bool(visual_mapping["day"]))
        self.assertEqual(obs["meta"]["night"], bool(visual_mapping["night"]))

    def test_observe_sleep_phase_level_in_sleep_observation(self) -> None:
        obs = self.world.observe()
        sleep_mapping = OBSERVATION_INTERFACE_BY_KEY["sleep"].bind_values(obs["sleep"])
        self.assertAlmostEqual(sleep_mapping["sleep_phase_level"], obs["meta"]["sleep_phase_level"], places=5)

    def test_observe_returns_numpy_arrays_for_all_vector_keys(self) -> None:
        obs = self.world.observe()
        for key in ("visual", "sensory", "hunger", "sleep", "alert", "action_context", "motor_context", "motor_extra"):
            self.assertIsInstance(obs[key], np.ndarray)

    def test_build_visual_observation_round_trips_through_serialization(self) -> None:
        visual = build_visual_observation(
            food_view=self.food_percept,
            shelter_view=self.null_percept,
            predator_view=self.null_percept,
            heading_dx=1.0,
            heading_dy=0.0,
            food_trace_strength=0.3,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.0,
            predator_motion_salience_value=0.0,
            day=1.0,
            night=0.0,
        )
        vector = serialize_observation_view("visual", visual)
        recovered_mapping = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(vector)
        recovered_view = VisualObservation.from_mapping(recovered_mapping)
        self.assertEqual(visual, recovered_view)

    def test_observe_world_meta_exposes_heading_and_percept_traces(self) -> None:
        obs = self.world.observe()
        self.assertIn("heading", obs["meta"])
        self.assertIn("dx", obs["meta"]["heading"])
        self.assertIn("dy", obs["meta"]["heading"])
        self.assertIn("percept_traces", obs["meta"])
        self.assertEqual(
            set(obs["meta"]["percept_traces"].keys()),
            {"food", "shelter", "predator"},
        )
        for key in ("food", "shelter", "predator"):
            trace = obs["meta"]["percept_traces"][key]
            self.assertEqual(
                set(trace.keys()),
                {"target", "age", "certainty", "strength", "dx", "dy", "ttl", "decay"},
            )
        self.assertIn("predator_motion_salience", obs["meta"])


class HeadingAwarePerceptionTest(unittest.TestCase):
    def test_visible_object_respects_forward_fov(self) -> None:
        world = SpiderWorld(seed=101, vision_range=6, lizard_move_interval=999999)
        world.reset(seed=101)
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0

        front = visible_object(world, [(5, 3)], radius=visible_range(world), apply_noise=False)
        back = visible_object(world, [(1, 3)], radius=visible_range(world), apply_noise=False)

        self.assertGreater(front.visible, 0.0)
        self.assertEqual(front.occluded, 0.0)
        self.assertGreater(front.certainty, 0.0)
        self.assertEqual(back.certainty, 0.0)
        self.assertEqual(back.visible, 0.0)

    def test_predator_motion_salience_is_explicit(self) -> None:
        world = SpiderWorld(seed=103, lizard_move_interval=999999)
        world.reset(seed=103)
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0

        world.lizard.x = 1
        world.lizard.y = 3
        world.lizard.mode = "PATROL"
        patrol_obs = world.observe()
        patrol_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(patrol_obs["visual"])
        )
        patrol_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(patrol_obs["alert"])
        )
        self.assertAlmostEqual(patrol_visual.predator_motion_salience, 0.0)
        self.assertAlmostEqual(patrol_alert.predator_motion_salience, 0.0)

        world.lizard.x = 5
        world.lizard.y = 3
        world.lizard.mode = "CHASE"
        chase_obs = world.observe()
        chase_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(chase_obs["visual"])
        )
        chase_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(chase_obs["alert"])
        )
        self.assertGreater(chase_visual.predator_motion_salience, 0.0)
        self.assertGreater(chase_alert.predator_motion_salience, 0.0)


class LineCellsTest(unittest.TestCase):
    def test_line_cells_same_point_returns_empty(self) -> None:
        cells = line_cells((3, 3), (3, 3))
        self.assertEqual(cells, [])

    def test_line_cells_horizontal_right(self) -> None:
        cells = line_cells((0, 0), (3, 0))
        self.assertEqual(cells, [(1, 0), (2, 0)])

    def test_line_cells_horizontal_left(self) -> None:
        cells = line_cells((3, 0), (0, 0))
        self.assertEqual(cells, [(2, 0), (1, 0)])

    def test_line_cells_vertical_down(self) -> None:
        cells = line_cells((0, 0), (0, 3))
        self.assertEqual(cells, [(0, 1), (0, 2)])

    def test_line_cells_adjacent_returns_empty(self) -> None:
        cells = line_cells((2, 2), (3, 2))
        self.assertEqual(cells, [])

    def test_line_cells_diagonal_excludes_endpoints(self) -> None:
        cells = line_cells((0, 0), (3, 3))
        self.assertNotIn((0, 0), cells)
        self.assertNotIn((3, 3), cells)
        self.assertGreater(len(cells), 0)

    def test_line_cells_two_apart_has_one_intermediate(self) -> None:
        cells = line_cells((0, 0), (2, 0))
        self.assertEqual(cells, [(1, 0)])


class VisibleRangeTest(unittest.TestCase):
    def _profile_with_perception_updates(self, **updates: float) -> OperationalProfile:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "perception_test_profile"
        summary["version"] = 11
        summary["perception"].update({name: float(value) for name, value in updates.items()})
        return OperationalProfile.from_summary(summary)

    def test_visible_range_full_during_day(self) -> None:
        """
        Verifies that a world reports its full vision range during daytime.
        
        Sets the world's tick to the start of day, asserts the world is not night, and checks that visible_range(world) equals the configured vision_range (4).
        """
        world = SpiderWorld(seed=1, vision_range=4, lizard_move_interval=999999)
        world.reset(seed=1)
        world.tick = 0
        self.assertFalse(world.is_night())
        self.assertEqual(visible_range(world), 4)

    def test_visible_range_reduced_at_night(self) -> None:
        world = SpiderWorld(seed=1, vision_range=4, day_length=5, night_length=10, lizard_move_interval=999999)
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertLess(visible_range(world), 4)

    def test_visible_range_minimum_two_at_night(self) -> None:
        world = SpiderWorld(seed=1, vision_range=2, day_length=5, night_length=10, lizard_move_interval=999999)
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertEqual(visible_range(world), 2)

    def test_visible_range_uses_operational_profile(self) -> None:
        world = SpiderWorld(
            seed=1,
            vision_range=4,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_perception_updates(
                night_vision_range_penalty=3.0,
                night_vision_min_range=1.0,
            ),
        )
        world.reset(seed=1)
        world.tick = 6
        self.assertEqual(visible_range(world), 1)


class HasLineOfSightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=1, lizard_move_interval=999999)
        self.world.reset(seed=1)

    def test_los_same_cell_is_clear(self) -> None:
        pos = self.world.spider_pos()
        self.assertTrue(has_line_of_sight(self.world, pos, pos))

    def test_los_adjacent_open_cells(self) -> None:
        self.world.state.x, self.world.state.y = 2, 2
        self.assertTrue(has_line_of_sight(self.world, (2, 2), (3, 2)))

    def test_los_blocked_by_blocked_cell(self) -> None:
        from spider_cortex_sim.maps import BLOCKED
        # Find two open cells with a blocked cell in between
        blocked = list(self.world.blocked_cells)
        if blocked:
            bx, by = blocked[0]
            for origin in [(bx - 2, by), (bx - 1, by)]:
                ox, oy = origin
                if (0 <= ox < self.world.width and 0 <= oy < self.world.height
                        and self.world.terrain_at(origin) != BLOCKED):
                    target = (bx + 2, by)
                    if (0 <= target[0] < self.world.width
                            and self.world.terrain_at(target) != BLOCKED):
                        result = has_line_of_sight(self.world, origin, target)
                        self.assertFalse(result)
                        return
        self.skipTest("no suitable blocked-cell geometry found")

    def test_los_outside_to_deep_shelter_is_false(self) -> None:
        """
        Verifies that an outside map cell does not have line of sight into a deep shelter cell.
        
        If the world has no deep shelter cells or no outside cells the test is skipped.
        """
        deep_cells = list(self.world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        deep = deep_cells[0]
        # Find an outside cell
        outside = None
        for x in range(self.world.width):
            for y in range(self.world.height):
                if self.world.shelter_role_at((x, y)) == "outside":
                    outside = (x, y)
                    break
            if outside:
                break
        if outside is None:
            self.skipTest("No outside cells")
        self.assertFalse(has_line_of_sight(self.world, outside, deep))


class SmellGradientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=1, lizard_move_interval=999999)
        self.world.reset(seed=1)

    def test_smell_gradient_returns_zero_when_no_targets(self) -> None:
        self.world.state.x, self.world.state.y = 5, 5
        strength, gx, gy, dist = smell_gradient(self.world, [], radius=5)
        self.assertAlmostEqual(strength, 0.0)
        self.assertAlmostEqual(gx, 0.0)
        self.assertAlmostEqual(gy, 0.0)
        self.assertEqual(dist, 10**9)

    def test_smell_gradient_returns_zero_when_target_out_of_range(self) -> None:
        self.world.state.x, self.world.state.y = 0, 0
        far_target = [(self.world.width - 1, self.world.height - 1)]
        strength, _, _, _ = smell_gradient(self.world, far_target, radius=2)
        self.assertAlmostEqual(strength, 0.0)

    def test_smell_gradient_detects_nearby_target(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        nearby = [(4, 3)]
        strength, _, _, dist = smell_gradient(self.world, nearby, radius=5)
        self.assertGreater(strength, 0.0)
        self.assertEqual(dist, 1)

    def test_smell_gradient_direction_positive_x(self) -> None:
        self.world.state.x, self.world.state.y = 2, 2
        target = [(5, 2)]
        _, gx, gy, _ = smell_gradient(self.world, target, radius=5)
        self.assertGreater(gx, 0.0)
        self.assertAlmostEqual(gy, 0.0, places=5)

    def test_smell_gradient_at_same_position_returns_nonzero_strength(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        same_pos = [(3, 3)]
        strength, _, _, dist = smell_gradient(self.world, same_pos, radius=5)
        self.assertGreater(strength, 0.0)
        self.assertEqual(dist, 0)


class NoiseAwarePerceptionTest(unittest.TestCase):
    def _noise_profile(self) -> NoiseConfig:
        """
        Create a deterministic NoiseConfig tailored for perception tests.
        
        Returns:
            NoiseConfig: A noise profile named "perception_noise_test" with small, fixed jitter and zero-probability randomness:
                - visual: certainty_jitter=0.2, direction_jitter=0.1, dropout_prob=0.0
                - olfactory: strength_jitter=0.2, direction_jitter=0.1
                - motor: action_flip_prob=0.0
                - spawn: uniform_mix=0.0
                - predator: random_choice_prob=0.0
        """
        return NoiseConfig(
            name="perception_noise_test",
            visual={"certainty_jitter": 0.2, "direction_jitter": 0.1, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.2, "direction_jitter": 0.1},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )

    def test_visual_noise_is_reproducible_for_same_seed(self) -> None:
        world_a = SpiderWorld(seed=5, vision_range=4, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        world_b = SpiderWorld(seed=5, vision_range=4, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        for world in (world_a, world_b):
            world.reset(seed=105)
            world.state.x, world.state.y = 2, 2
            world.lizard.x, world.lizard.y = 3, 2
            world.lizard.mode = "CHASE"

        percept_a = predator_visible_to_spider(world_a)
        percept_b = predator_visible_to_spider(world_b)

        self.assertAlmostEqual(percept_a.certainty, percept_b.certainty)
        self.assertAlmostEqual(percept_a.dx, percept_b.dx)
        self.assertAlmostEqual(percept_a.dy, percept_b.dy)
        self.assertEqual(percept_a.visible, percept_b.visible)

    def test_olfactory_noise_is_reproducible_for_same_seed(self) -> None:
        world_a = SpiderWorld(seed=7, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        world_b = SpiderWorld(seed=7, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        for world in (world_a, world_b):
            world.reset(seed=207)
            world.state.x, world.state.y = 3, 3

        smell_a = smell_gradient(world_a, [(5, 3)], radius=6)
        smell_b = smell_gradient(world_b, [(5, 3)], radius=6)

        self.assertEqual(smell_a[3], smell_b[3])
        self.assertAlmostEqual(smell_a[0], smell_b[0])
        self.assertAlmostEqual(smell_a[1], smell_b[1])
        self.assertAlmostEqual(smell_a[2], smell_b[2])

    def test_smell_field_does_not_consume_olfactory_rng(self) -> None:
        world_a = SpiderWorld(seed=8, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        world_b = SpiderWorld(seed=8, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        for world in (world_a, world_b):
            world.reset(seed=208)
            world.state.x, world.state.y = 3, 3

        world_a.smell_field("food")
        smell_a = smell_gradient(world_a, [(5, 3)], radius=6)
        smell_b = smell_gradient(world_b, [(5, 3)], radius=6)

        self.assertEqual(smell_a, smell_b)

    def test_visual_noise_recomputes_visible_after_dropout(self) -> None:
        profile = NoiseConfig(
            name="visual_dropout_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.1, "dropout_prob": 1.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )
        world = SpiderWorld(seed=11, vision_range=4, noise_profile=profile, lizard_move_interval=999999)
        world.reset(seed=211)
        world.state.x, world.state.y = 2, 2
        world.lizard.x, world.lizard.y = 4, 2
        world.lizard.mode = "PATROL"

        percept = predator_visible_to_spider(world)

        self.assertEqual(percept.visible, 0.0)
        self.assertEqual(percept.dx, 0.0)
        self.assertEqual(percept.dy, 0.0)

    def test_noise_profile_does_not_change_observation_shapes(self) -> None:
        world = SpiderWorld(seed=9, noise_profile="high", lizard_move_interval=999999)
        obs = world.reset(seed=9)

        self.assertEqual(obs["visual"].shape, (23,))
        self.assertEqual(obs["sensory"].shape, (12,))
        self.assertEqual(obs["hunger"].shape, (16,))
        self.assertEqual(obs["sleep"].shape, (19,))
        self.assertEqual(obs["alert"].shape, (23,))
        self.assertEqual(obs["action_context"].shape, (16,))
        self.assertEqual(obs["motor_context"].shape, (10,))


class VisibilityConfidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepare a deterministic SpiderWorld test fixture on self.world.
        
        Initializes self.world with seed=1, vision_range=4, and lizard_move_interval=999999, then calls reset(seed=1) so tests start from a reproducible state.
        """
        self.world = SpiderWorld(seed=1, vision_range=4, lizard_move_interval=999999)
        self.world.reset(seed=1)

    def test_visibility_confidence_at_zero_dist_is_near_one(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        conf = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(3, 3),
            dist=0,
            radius=4,
        )
        self.assertGreater(conf, 0.8)

    def test_visibility_confidence_at_max_range_is_low(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        conf = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(7, 3),
            dist=4,
            radius=4,
        )
        self.assertLess(conf, 0.5)

    def test_visibility_confidence_clipped_to_zero_one(self) -> None:
        # Very far away should not go below 0
        self.world.state.x, self.world.state.y = 3, 3
        conf = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(7, 3),
            dist=100,
            radius=2,
        )
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_visibility_confidence_motion_bonus_increases_value(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        conf_no_bonus = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            motion_bonus=0.0,
        )
        conf_with_bonus = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            motion_bonus=0.20,
        )
        self.assertGreater(conf_with_bonus, conf_no_bonus)

    def test_visibility_confidence_uses_operational_profile_penalties(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "visibility_profile"
        summary["version"] = 12
        summary["perception"]["visibility_clutter_penalty"] = 0.0
        world = SpiderWorld(
            seed=1,
            vision_range=4,
            lizard_move_interval=999999,
            operational_profile=OperationalProfile.from_summary(summary),
        )
        world.reset(seed=1)

        clutter_target = None
        for x in range(world.width):
            for y in range(world.height):
                if world.terrain_at((x, y)) == CLUTTER:
                    clutter_target = (x, y)
                    break
            if clutter_target is not None:
                break
        if clutter_target is None:
            self.skipTest("No clutter cell found for visibility test")

        world.state.x, world.state.y = max(0, clutter_target[0] - 1), clutter_target[1]
        dist = world.manhattan(world.spider_pos(), clutter_target)
        no_penalty_conf = visibility_confidence(
            world,
            source=world.spider_pos(),
            target=clutter_target,
            dist=dist,
            radius=4,
        )

        baseline_world = SpiderWorld(seed=1, vision_range=4, lizard_move_interval=999999)
        baseline_world.reset(seed=1)
        baseline_world.state.x, baseline_world.state.y = world.spider_pos()
        baseline_conf = visibility_confidence(
            baseline_world,
            source=baseline_world.spider_pos(),
            target=clutter_target,
            dist=dist,
            radius=4,
        )

        self.assertGreater(no_penalty_conf, baseline_conf)


class LizardDetectsSpiderTest(unittest.TestCase):
    def test_deep_shelter_blocks_lizard_detection(self) -> None:
        world = SpiderWorld(seed=21, lizard_move_interval=999999)
        world.reset(seed=21)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        deep = sorted(deep_cells)[len(deep_cells) // 2]
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        entrance = sorted(entrance_cells)[len(entrance_cells) // 2]
        world.state.x, world.state.y = deep
        world.lizard.x = max(0, entrance[0] - 1)
        world.lizard.y = entrance[1]
        self.assertFalse(lizard_detects_spider(world))

    def test_lizard_detects_exposed_spider(self) -> None:
        world = SpiderWorld(seed=21, lizard_move_interval=999999)
        world.reset(seed=21)
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        entrance = sorted(entrance_cells)[len(entrance_cells) // 2]
        world.state.x, world.state.y = entrance
        world.lizard.x = max(0, entrance[0] - 1)
        world.lizard.y = entrance[1]
        world.lizard_vision_range = 5
        self.assertTrue(lizard_detects_spider(world))

    def test_predator_visible_to_spider_far_away(self) -> None:
        world = SpiderWorld(seed=5, vision_range=4, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 1, 1
        world.lizard.x = world.width - 1
        world.lizard.y = world.height - 1
        percept = predator_visible_to_spider(world)
        self.assertAlmostEqual(percept.visible, 0.0)

    def test_predator_visible_to_spider_very_close(self) -> None:
        world = SpiderWorld(seed=5, vision_range=4, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.lizard.x = 3
        world.lizard.y = 2
        percept = predator_visible_to_spider(world)
        self.assertGreater(percept.certainty, 0.0)


class ObserveWorldMetaTest(unittest.TestCase):
    def test_observe_world_meta_has_required_fields(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        obs = world.observe()
        meta = obs["meta"]
        required = [
            "food_dist", "shelter_dist", "predator_dist",
            "night", "day", "on_shelter", "on_food",
            "predator_visible", "lizard_x", "lizard_y", "lizard_mode",
            "sleep_phase", "sleep_debt", "shelter_role",
            "terrain", "map_template", "reward_profile",
            "vision", "memory_vectors",
        ]
        for field in required:
            self.assertIn(field, meta, f"Missing meta field: {field}")

    def test_observe_world_memory_vectors_all_four_keys(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        obs = world.observe()
        memory_vectors = obs["meta"]["memory_vectors"]
        self.assertIn("food", memory_vectors)
        self.assertIn("predator", memory_vectors)
        self.assertIn("shelter", memory_vectors)
        self.assertIn("escape", memory_vectors)

    def test_observe_world_vision_has_food_shelter_predator(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        obs = world.observe()
        vision = obs["meta"]["vision"]
        self.assertIn("food", vision)
        self.assertIn("shelter", vision)
        self.assertIn("predator", vision)


class OperationalProfilePerceptionIntegrationTest(unittest.TestCase):
    """Additional tests verifying that perception functions use the operational profile correctly."""

    def _make_profile(self, **perception_updates: float) -> OperationalProfile:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "perception_integration_test"
        summary["version"] = 77
        summary["perception"].update({k: float(v) for k, v in perception_updates.items()})
        return OperationalProfile.from_summary(summary)

    def test_world_stores_operational_profile(self) -> None:
        profile = self._make_profile(night_vision_range_penalty=2.0)
        world = SpiderWorld(seed=1, lizard_move_interval=999999, operational_profile=profile)
        self.assertIs(world.operational_profile, profile)

    def test_visible_range_night_penalty_scales_correctly(self) -> None:
        # With penalty=2, vision_range=5, min=1 → night range = max(1, 5-2) = 3
        world = SpiderWorld(
            seed=1,
            vision_range=5,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                night_vision_range_penalty=2.0,
                night_vision_min_range=1.0,
            ),
        )
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertEqual(visible_range(world), 3)

    def test_visible_range_min_range_enforced(self) -> None:
        # With large penalty, min_range clamps result
        world = SpiderWorld(
            seed=1,
            vision_range=3,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                night_vision_range_penalty=10.0,
                night_vision_min_range=2.0,
            ),
        )
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertEqual(visible_range(world), 2)

    def test_visibility_night_penalty_from_profile(self) -> None:
        # Zero night penalty should increase confidence compared to default
        world_no_penalty = SpiderWorld(
            seed=1,
            vision_range=4,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(visibility_night_penalty=0.0),
        )
        world_no_penalty.reset(seed=1)
        world_no_penalty.tick = 6
        world_no_penalty.state.x, world_no_penalty.state.y = 3, 3

        world_default = SpiderWorld(seed=1, vision_range=4, day_length=5, night_length=10, lizard_move_interval=999999)
        world_default.reset(seed=1)
        world_default.tick = 6
        world_default.state.x, world_default.state.y = 3, 3

        conf_no_penalty = visibility_confidence(
            world_no_penalty,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
        )
        conf_default = visibility_confidence(
            world_default,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
        )
        self.assertGreater(conf_no_penalty, conf_default)

    def test_lizard_detection_threshold_from_profile_blocks_detection(self) -> None:
        # Very high detection threshold → lizard can't detect spider even when close
        world = SpiderWorld(
            seed=21,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(lizard_detection_threshold=1.1),
        )
        world.reset(seed=21)
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        entrance = sorted(entrance_cells)[len(entrance_cells) // 2]
        world.state.x, world.state.y = entrance
        world.lizard.x = max(0, entrance[0] - 1)
        world.lizard.y = entrance[1]
        world.lizard_vision_range = 5
        self.assertFalse(lizard_detects_spider(world))

    def test_predator_motion_bonus_from_profile_increases_exported_salience(self) -> None:
        # A higher predator_motion_bonus should yield a stronger exported salience channel when the predator is seen.
        world_high = SpiderWorld(
            seed=5,
            vision_range=4,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(predator_motion_bonus=0.50),
        )
        world_low = SpiderWorld(
            seed=5,
            vision_range=4,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(predator_motion_bonus=0.0),
        )
        for world in (world_high, world_low):
            world.reset(seed=5)
            world.state.x, world.state.y = 2, 2
            world.lizard.x = 3
            world.lizard.y = 2
            world.lizard.mode = "CHASE"
            world.state.heading_dx = 1
            world.state.heading_dy = 0

        high_obs = world_high.observe()
        low_obs = world_low.observe()
        high_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(high_obs["visual"])
        )
        low_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(low_obs["visual"])
        )
        high_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(high_obs["alert"])
        )
        low_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(low_obs["alert"])
        )
        self.assertGreater(high_visual.predator_motion_salience, low_visual.predator_motion_salience)
        self.assertGreater(high_alert.predator_motion_salience, low_alert.predator_motion_salience)

    def test_occluded_certainty_base_from_profile(self) -> None:
        # With a very high occluded_certainty_base, certainty of occluded target should be higher
        world_high = SpiderWorld(
            seed=1,
            vision_range=8,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                occluded_certainty_base=0.90,
                occluded_certainty_min=0.10,
                occluded_certainty_decay_per_step=0.01,
            ),
        )
        world_low = SpiderWorld(
            seed=1,
            vision_range=8,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                occluded_certainty_base=0.10,
                occluded_certainty_min=0.05,
                occluded_certainty_decay_per_step=0.01,
            ),
        )
        # Find an occluded food item by moving spider far away and measuring perceived certainty
        for world in (world_high, world_low):
            world.reset(seed=1)
            world.state.x, world.state.y = 0, 3

        # Use a known occluded line on the current map so the branch is always exercised.
        for world in (world_high, world_low):
            world.food_positions = [(5, 6)]
            world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
            world.state.heading_dx = 1
            world.state.heading_dy = 0

        from spider_cortex_sim.perception import visible_object
        high_percept = visible_object(
            world_high,
            positions=world_high.food_positions,
            origin=world_high.spider_pos(),
            radius=visible_range(world_high),
        )
        low_percept = visible_object(
            world_low,
            positions=world_low.food_positions,
            origin=world_low.spider_pos(),
            radius=visible_range(world_low),
        )
        self.assertIsNotNone(high_percept)
        self.assertIsNotNone(low_percept)
        self.assertEqual(high_percept.visible, 0.0)
        self.assertEqual(low_percept.visible, 0.0)
        self.assertGreater(high_percept.occluded, 0.0)
        self.assertGreater(low_percept.occluded, 0.0)
        self.assertGreater(high_percept.certainty, low_percept.certainty)
