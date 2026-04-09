"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 9
- SpiderBrain.action_center / motor_cortex network types
- BrainStep new fields (action_center_logits, action_center_policy,
  action_intent_idx, motor_override, action_center_input)
- SpiderBrain.estimate_value uses action_center, not motor_cortex alone
- _build_motor_input NaN / inf sanitization
- ActionContextObservation and MotorContextObservation field composition
- ACTION_CONTEXT_INTERFACE / MOTOR_CONTEXT_INTERFACE properties
- OBSERVATION_DIMS updated entries
- build_action_context_observation / build_motor_context_observation
"""

import unittest
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import default_brain_config
from spider_cortex_sim.agent import ArbitrationDecision, BrainStep, SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_DIMS,
    ActionContextObservation,
    MotorContextObservation,
    architecture_signature,
)
from spider_cortex_sim.nn import MotorNetwork, ProposalNetwork
from spider_cortex_sim.perception import (
    PerceivedTarget,
    build_action_context_observation,
    build_motor_context_observation,
)
from spider_cortex_sim.world import SpiderWorld


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_percept() -> PerceivedTarget:
    """Return a PerceivedTarget representing no visible predator."""
    return PerceivedTarget(visible=0.0, certainty=0.0, occluded=0.0, dx=0.0, dy=0.0, dist=10**9)


def _blank_obs() -> dict[str, np.ndarray]:
    """
    Create a zero-filled observation mapping for every module and context interface.
    
    Constructs a dict whose keys are each interface's `observation_key` (for all entries in
    `MODULE_INTERFACES`, plus `ACTION_CONTEXT_INTERFACE` and `MOTOR_CONTEXT_INTERFACE`) and
    whose values are NumPy arrays of zeros sized to the interface's `input_dim`.
    
    Returns:
        dict[str, np.ndarray]: Mapping from observation key to zeroed observation vector.
    """
    obs: dict[str, np.ndarray] = {}
    for iface in MODULE_INTERFACES:
        obs[iface.observation_key] = np.zeros(iface.input_dim, dtype=float)
    obs[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
        ACTION_CONTEXT_INTERFACE.input_dim, dtype=float
    )
    obs[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
        MOTOR_CONTEXT_INTERFACE.input_dim, dtype=float
    )
    return obs


def _module_interface(name: str):
    """
    Retrieve the module interface spec with the given name.
    
    Parameters:
        name (str): The name of the module interface to find.
    
    Returns:
        The matching module interface spec.
    
    Raises:
        StopIteration: If no module interface with the given name exists.
    """
    return next(spec for spec in MODULE_INTERFACES if spec.name == name)


def _vector_for(interface, **updates) -> np.ndarray:
    """
    Create a vector for the given interface by filling unspecified signals with 0.0 and applying provided signal-value updates.
    
    Parameters:
        interface: An interface spec exposing `signal_names` and `vector_from_mapping(mapping)`; `signal_names` defines the ordered signal keys.
        **updates: Signal name → numeric value overrides to apply on top of zeros.
    
    Returns:
        A numpy ndarray produced by `interface.vector_from_mapping(mapping)` where `mapping` contains all signals from `interface.signal_names` with unspecified signals set to 0.0 and specified signals set to the provided values.
    """
    mapping = {name: 0.0 for name in interface.signal_names}
    mapping.update({key: float(value) for key, value in updates.items()})
    return interface.vector_from_mapping(mapping)


# ---------------------------------------------------------------------------
# architecture_signature() tests
# ---------------------------------------------------------------------------

class ArchitectureSignatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sig = architecture_signature()

    def test_top_level_keys(self) -> None:
        """architecture_signature must expose actions, modules, action_center, motor_cortex."""
        self.assertIn("actions", self.sig)
        self.assertIn("modules", self.sig)
        self.assertIn("contexts", self.sig)
        self.assertIn("fingerprint", self.sig)
        self.assertIn("registry_fingerprint", self.sig)
        self.assertIn("action_center", self.sig)
        self.assertIn("motor_cortex", self.sig)

    def test_no_legacy_motor_context_key(self) -> None:
        """The old 'motor_context' top-level key must be absent (replaced by separate sections)."""
        self.assertNotIn("motor_context", self.sig)

    def test_actions_is_locomotion_actions(self) -> None:
        self.assertEqual(self.sig["actions"], list(LOCOMOTION_ACTIONS))

    def test_action_center_observation_key(self) -> None:
        ac = self.sig["action_center"]
        self.assertEqual(ac["observation_key"], "action_context")

    def test_action_center_value_head_is_true(self) -> None:
        ac = self.sig["action_center"]
        self.assertIs(ac["value_head"], True)

    def test_action_center_signature_exposes_priority_gating(self) -> None:
        arbitration = self.sig["action_center"]["arbitration"]
        self.assertEqual(arbitration["strategy"], "priority_gating")
        self.assertEqual(
            arbitration["valences"],
            ["threat", "hunger", "sleep", "exploration"],
        )

    def test_action_center_signature_maps_module_roles(self) -> None:
        module_roles = self.sig["action_center"]["arbitration"]["module_roles"]
        self.assertEqual(module_roles["alert_center"], "threat")
        self.assertEqual(module_roles["hunger_center"], "hunger")
        self.assertEqual(module_roles["sleep_center"], "sleep")
        self.assertNotIn("monolithic_policy", module_roles)

    def test_action_center_inputs_count(self) -> None:
        """action_center must list exactly 16 input signal names."""
        ac = self.sig["action_center"]
        self.assertEqual(len(ac["inputs"]), 16)

    def test_action_center_inputs_match_interface(self) -> None:
        expected = [s.name for s in ACTION_CONTEXT_INTERFACE.inputs]
        self.assertEqual(self.sig["action_center"]["inputs"], expected)

    def test_action_center_outputs_are_locomotion_actions(self) -> None:
        self.assertEqual(self.sig["action_center"]["outputs"], list(LOCOMOTION_ACTIONS))

    def test_action_center_inter_stage_inputs_match_proposal_slots(self) -> None:
        action_center = self.sig["action_center"]
        self.assertEqual(
            [entry["source"] for entry in action_center["inter_stage_inputs"]],
            self.sig["proposal_order"],
        )
        self.assertTrue(
            all(
                entry["signals"] == list(LOCOMOTION_ACTIONS)
                for entry in action_center["inter_stage_inputs"]
            )
        )

    def test_motor_cortex_observation_key(self) -> None:
        mc = self.sig["motor_cortex"]
        self.assertEqual(mc["observation_key"], "motor_context")

    def test_motor_cortex_value_head_is_false(self) -> None:
        mc = self.sig["motor_cortex"]
        self.assertIs(mc["value_head"], False)

    def test_motor_cortex_inputs_count(self) -> None:
        """motor_cortex must list exactly 10 input signal names."""
        mc = self.sig["motor_cortex"]
        self.assertEqual(len(mc["inputs"]), 10)

    def test_motor_cortex_inputs_match_interface(self) -> None:
        expected = [s.name for s in MOTOR_CONTEXT_INTERFACE.inputs]
        self.assertEqual(self.sig["motor_cortex"]["inputs"], expected)

    def test_motor_cortex_outputs_are_locomotion_actions(self) -> None:
        self.assertEqual(self.sig["motor_cortex"]["outputs"], list(LOCOMOTION_ACTIONS))

    def test_motor_cortex_inter_stage_inputs_describe_intent(self) -> None:
        mc = self.sig["motor_cortex"]
        self.assertEqual(
            mc["inter_stage_inputs"],
            [
                {
                    "name": "intent",
                    "source": "action_center",
                    "encoding": "one_hot",
                    "size": len(LOCOMOTION_ACTIONS),
                    "signals": list(LOCOMOTION_ACTIONS),
                }
            ],
        )

    def test_monolithic_signature_uses_single_proposal_source(self) -> None:
        sig = architecture_signature(
            proposal_backend="monolithic",
            proposal_order=[SpiderBrain.MONOLITHIC_POLICY_NAME],
        )
        self.assertEqual(sig["proposal_order"], [SpiderBrain.MONOLITHIC_POLICY_NAME])
        self.assertEqual(
            sig["action_center"]["proposal_slots"],
            [{"module": SpiderBrain.MONOLITHIC_POLICY_NAME, "actions": list(LOCOMOTION_ACTIONS)}],
        )
        self.assertEqual(
            sig["action_center"]["inter_stage_inputs"][0]["source"],
            SpiderBrain.MONOLITHIC_POLICY_NAME,
        )
        self.assertEqual(
            sig["action_center"]["arbitration"]["module_roles"],
            {SpiderBrain.MONOLITHIC_POLICY_NAME: "integrated_policy"},
        )

    def test_signature_is_json_serialisable(self) -> None:
        import json
        serialised = json.dumps(self.sig)
        recovered = json.loads(serialised)
        self.assertEqual(recovered["action_center"]["value_head"], True)
        self.assertEqual(recovered["motor_cortex"]["value_head"], False)


# ---------------------------------------------------------------------------
# Interface dimension / property tests
# ---------------------------------------------------------------------------

class ActionContextInterfaceTest(unittest.TestCase):
    def test_name(self) -> None:
        self.assertEqual(ACTION_CONTEXT_INTERFACE.name, "action_center_context")

    def test_observation_key(self) -> None:
        self.assertEqual(ACTION_CONTEXT_INTERFACE.observation_key, "action_context")

    def test_input_dim(self) -> None:
        self.assertEqual(ACTION_CONTEXT_INTERFACE.input_dim, 16)

    def test_signal_names_ordered(self) -> None:
        names = ACTION_CONTEXT_INTERFACE.signal_names
        self.assertEqual(names[0], "hunger")
        self.assertEqual(names[-1], "shelter_role_level")

    def test_sleep_debt_present_in_signals(self) -> None:
        self.assertIn("sleep_debt", ACTION_CONTEXT_INTERFACE.signal_names)

    def test_recent_pain_present_in_signals(self) -> None:
        self.assertIn("recent_pain", ACTION_CONTEXT_INTERFACE.signal_names)

    def test_role_context(self) -> None:
        self.assertEqual(ACTION_CONTEXT_INTERFACE.role, "context")


class MotorContextInterfaceTest(unittest.TestCase):
    def test_name(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.name, "motor_cortex_context")

    def test_observation_key(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.observation_key, "motor_context")

    def test_input_dim(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.input_dim, 10)

    def test_no_hunger_signal(self) -> None:
        """Motor context must NOT contain hunger (that belongs to action_center)."""
        self.assertNotIn("hunger", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_no_sleep_debt_signal(self) -> None:
        """Motor context must NOT contain sleep_debt (that belongs to action_center)."""
        self.assertNotIn("sleep_debt", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_no_recent_pain_signal(self) -> None:
        self.assertNotIn("recent_pain", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_on_food_present(self) -> None:
        self.assertIn("on_food", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_on_shelter_present(self) -> None:
        self.assertIn("on_shelter", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_predator_visible_present(self) -> None:
        self.assertIn("predator_visible", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_last_move_signals_present(self) -> None:
        self.assertIn("last_move_dx", MOTOR_CONTEXT_INTERFACE.signal_names)
        self.assertIn("last_move_dy", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_role_context(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.role, "context")


class ObservationDimsTest(unittest.TestCase):
    def test_action_context_dim(self) -> None:
        self.assertEqual(OBSERVATION_DIMS["action_context"], 16)

    def test_motor_context_dim(self) -> None:
        self.assertEqual(OBSERVATION_DIMS["motor_context"], 10)

    def test_both_present(self) -> None:
        self.assertIn("action_context", OBSERVATION_DIMS)
        self.assertIn("motor_context", OBSERVATION_DIMS)


# ---------------------------------------------------------------------------
# ActionContextObservation / MotorContextObservation dataclass tests
# ---------------------------------------------------------------------------

class ActionContextObservationCompositionTest(unittest.TestCase):
    """Tests that ActionContextObservation has the correct fields and no motor-only fields."""

    def test_observation_key(self) -> None:
        self.assertEqual(ActionContextObservation.observation_key, "action_context")

    def test_contains_internal_state_fields(self) -> None:
        names = ActionContextObservation.field_names()
        for field in ("hunger", "fatigue", "health", "recent_pain", "recent_contact"):
            with self.subTest(field=field):
                self.assertIn(field, names)

    def test_contains_sleep_debt(self) -> None:
        self.assertIn("sleep_debt", ActionContextObservation.field_names())

    def test_as_mapping_contains_all_16_fields(self) -> None:
        view = ActionContextObservation(
            hunger=0.1, fatigue=0.2, health=0.9, recent_pain=0.0, recent_contact=0.0,
            on_food=0.0, on_shelter=0.0, predator_visible=0.0, predator_certainty=0.0,
            predator_dist=1.0, day=1.0, night=0.0, last_move_dx=0.0, last_move_dy=0.0,
            sleep_debt=0.0, shelter_role_level=0.0,
        )
        self.assertEqual(len(view.as_mapping()), 16)


class MotorContextObservationCompositionTest(unittest.TestCase):
    """Tests that MotorContextObservation has the correct 10 fields."""

    def test_observation_key(self) -> None:
        self.assertEqual(MotorContextObservation.observation_key, "motor_context")

    def test_does_not_contain_hunger(self) -> None:
        self.assertNotIn("hunger", MotorContextObservation.field_names())

    def test_does_not_contain_sleep_debt(self) -> None:
        self.assertNotIn("sleep_debt", MotorContextObservation.field_names())

    def test_does_not_contain_recent_pain(self) -> None:
        self.assertNotIn("recent_pain", MotorContextObservation.field_names())

    def test_contains_shelter_role_level(self) -> None:
        self.assertIn("shelter_role_level", MotorContextObservation.field_names())

    def test_predator_certainty_present(self) -> None:
        self.assertIn("predator_certainty", MotorContextObservation.field_names())

    def test_as_mapping_contains_all_10_fields(self) -> None:
        view = MotorContextObservation(
            on_food=0.0, on_shelter=0.0, predator_visible=0.0, predator_certainty=0.0,
            predator_dist=1.0, day=1.0, night=0.0, last_move_dx=0.0, last_move_dy=0.0,
            shelter_role_level=0.0,
        )
        self.assertEqual(len(view.as_mapping()), 10)


# ---------------------------------------------------------------------------
# build_action_context_observation / build_motor_context_observation tests
# ---------------------------------------------------------------------------

class BuildActionContextObservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=7)
        self.world.reset(seed=7)
        self.null_percept = _null_percept()

    def _build(self, **kwargs) -> ActionContextObservation:
        defaults = dict(
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            predator_dist_norm=1.0,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        defaults.update(kwargs)
        return build_action_context_observation(self.world, **defaults)

    def test_returns_action_context_observation_type(self) -> None:
        result = self._build()
        self.assertIsInstance(result, ActionContextObservation)

    def test_hunger_from_world_state(self) -> None:
        self.world.state.hunger = 0.72
        result = self._build()
        self.assertAlmostEqual(result.hunger, 0.72, places=5)

    def test_fatigue_from_world_state(self) -> None:
        self.world.state.fatigue = 0.55
        result = self._build()
        self.assertAlmostEqual(result.fatigue, 0.55, places=5)

    def test_health_from_world_state(self) -> None:
        self.world.state.health = 0.85
        result = self._build()
        self.assertAlmostEqual(result.health, 0.85, places=5)

    def test_sleep_debt_from_world_state(self) -> None:
        self.world.state.sleep_debt = 0.33
        result = self._build()
        self.assertAlmostEqual(result.sleep_debt, 0.33, places=5)

    def test_last_move_from_world_state(self) -> None:
        self.world.state.last_move_dx = -1
        self.world.state.last_move_dy = 0
        result = self._build()
        self.assertAlmostEqual(result.last_move_dx, -1.0, places=5)
        self.assertAlmostEqual(result.last_move_dy, 0.0, places=5)

    def test_on_food_passed_through(self) -> None:
        result = self._build(on_food=1.0)
        self.assertAlmostEqual(result.on_food, 1.0)

    def test_on_shelter_passed_through(self) -> None:
        result = self._build(on_shelter=1.0)
        self.assertAlmostEqual(result.on_shelter, 1.0)

    def test_predator_visibility_from_percept(self) -> None:
        percept = PerceivedTarget(
            visible=0.95,
            certainty=0.88,
            occluded=0.0,
            dx=1.0,
            dy=0.0,
            dist=3,
            position=(8, 5),
        )
        result = self._build(predator_view=percept)
        self.assertAlmostEqual(result.predator_visible, 0.95, places=5)
        self.assertAlmostEqual(result.predator_certainty, 0.88, places=5)

    def test_predator_dist_passed_through(self) -> None:
        result = self._build(predator_dist_norm=0.25)
        self.assertAlmostEqual(result.predator_dist, 0.25)

    def test_day_night_passed_through(self) -> None:
        result = self._build(day=0.0, night=1.0)
        self.assertAlmostEqual(result.day, 0.0)
        self.assertAlmostEqual(result.night, 1.0)

    def test_shelter_role_level_passed_through(self) -> None:
        result = self._build(shelter_role_level=0.5)
        self.assertAlmostEqual(result.shelter_role_level, 0.5)


class BuildMotorContextObservationTest(unittest.TestCase):
    """Tests that build_motor_context_observation produces a MotorContextObservation
    with only the execution-relevant fields (no hunger, fatigue, sleep_debt)."""

    def setUp(self) -> None:
        self.world = SpiderWorld(seed=11)
        self.world.reset(seed=11)
        self.null_percept = _null_percept()

    def _build(self, **kwargs) -> MotorContextObservation:
        defaults = dict(
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            predator_dist_norm=1.0,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        defaults.update(kwargs)
        return build_motor_context_observation(self.world, **defaults)

    def test_returns_motor_context_observation_type(self) -> None:
        result = self._build()
        self.assertIsInstance(result, MotorContextObservation)

    def test_on_food_passed_through(self) -> None:
        result = self._build(on_food=1.0)
        self.assertAlmostEqual(result.on_food, 1.0)

    def test_on_shelter_passed_through(self) -> None:
        result = self._build(on_shelter=1.0)
        self.assertAlmostEqual(result.on_shelter, 1.0)

    def test_predator_visibility_from_percept(self) -> None:
        percept = PerceivedTarget(
            visible=0.8,
            certainty=0.7,
            occluded=0.0,
            dx=-1.0,
            dy=0.0,
            dist=5,
            position=(1, 5),
        )
        result = self._build(predator_view=percept)
        self.assertAlmostEqual(result.predator_visible, 0.8, places=5)
        self.assertAlmostEqual(result.predator_certainty, 0.7, places=5)

    def test_predator_dist_passed_through(self) -> None:
        result = self._build(predator_dist_norm=0.4)
        self.assertAlmostEqual(result.predator_dist, 0.4)

    def test_last_move_from_world_state(self) -> None:
        self.world.state.last_move_dx = 1
        self.world.state.last_move_dy = -1
        result = self._build()
        self.assertAlmostEqual(result.last_move_dx, 1.0, places=5)
        self.assertAlmostEqual(result.last_move_dy, -1.0, places=5)

    def test_shelter_role_level_passed_through(self) -> None:
        result = self._build(shelter_role_level=0.6)
        self.assertAlmostEqual(result.shelter_role_level, 0.6)

    def test_day_night_passed_through(self) -> None:
        result = self._build(day=0.0, night=1.0)
        self.assertAlmostEqual(result.day, 0.0)
        self.assertAlmostEqual(result.night, 1.0)

    def test_no_hunger_attribute(self) -> None:
        """MotorContextObservation must not expose hunger (belongs to action_center)."""
        result = self._build()
        self.assertFalse(hasattr(result, "hunger"))

    def test_no_sleep_debt_attribute(self) -> None:
        """MotorContextObservation must not expose sleep_debt."""
        result = self._build()
        self.assertFalse(hasattr(result, "sleep_debt"))

    def test_serialised_shape_equals_input_dim(self) -> None:
        from spider_cortex_sim.perception import serialize_observation_view
        result = self._build()
        vec = serialize_observation_view("motor_context", result)
        self.assertEqual(vec.shape, (MOTOR_CONTEXT_INTERFACE.input_dim,))

    def test_world_hunger_does_not_leak_into_motor_context(self) -> None:
        """Changing world hunger must not change any motor context field."""
        self.world.state.hunger = 0.0
        mc_low = self._build()
        self.world.state.hunger = 0.99
        mc_high = self._build()
        np.testing.assert_array_equal(
            np.array(list(mc_low.as_mapping().values())),
            np.array(list(mc_high.as_mapping().values())),
        )


# ---------------------------------------------------------------------------
# SpiderBrain architecture tests
# ---------------------------------------------------------------------------

class SpiderBrainArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Initialize the test fixture by creating a deterministic SpiderBrain instance.
        
        Creates a SpiderBrain with seed=42 and module_dropout=0.0 so tests run with reproducible weights and no module dropout.
        """
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def test_architecture_version(self) -> None:
        self.assertEqual(SpiderBrain.ARCHITECTURE_VERSION, 9)

    def test_action_center_is_motor_network(self) -> None:
        """action_center must be a MotorNetwork (has value head)."""
        self.assertIsInstance(self.brain.action_center, MotorNetwork)

    def test_motor_cortex_is_proposal_network(self) -> None:
        """motor_cortex must be a ProposalNetwork (no value head)."""
        self.assertIsInstance(self.brain.motor_cortex, ProposalNetwork)

    def test_action_center_has_value_head_attributes(self) -> None:
        """MotorNetwork has W2_policy and W2_value; ProposalNetwork does not."""
        self.assertTrue(hasattr(self.brain.action_center, "W2_policy"))
        self.assertTrue(hasattr(self.brain.action_center, "W2_value"))

    def test_motor_cortex_lacks_value_head(self) -> None:
        self.assertFalse(hasattr(self.brain.motor_cortex, "W2_policy"))
        self.assertFalse(hasattr(self.brain.motor_cortex, "W2_value"))

    def test_motor_cortex_has_proposal_attributes(self) -> None:
        self.assertTrue(hasattr(self.brain.motor_cortex, "W2"))
        self.assertTrue(hasattr(self.brain.motor_cortex, "b2"))

    def test_module_names_includes_action_center_before_motor_cortex(self) -> None:
        names = self.brain._module_names()
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)
        ac_idx = names.index("action_center")
        mc_idx = names.index("motor_cortex")
        self.assertLess(ac_idx, mc_idx)

    def test_parameter_norms_includes_action_center(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertIn("action_center", norms)

    def test_parameter_norms_action_center_norm_is_positive(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertGreater(norms["action_center"], 0.0)

    def test_parameter_norms_motor_cortex_still_present(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertIn("motor_cortex", norms)


class SpiderBrainMonolithicArchitectureTest(unittest.TestCase):
    def _config(self):
        from spider_cortex_sim.ablations import BrainAblationConfig
        return BrainAblationConfig(
            name="monolithic_policy",
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
        )

    def test_monolithic_module_names_includes_action_center(self) -> None:
        brain = SpiderBrain(seed=2, config=self._config())
        names = brain._module_names()
        self.assertIn("monolithic_policy", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)

    def test_monolithic_parameter_norms_includes_action_center(self) -> None:
        brain = SpiderBrain(seed=3, config=self._config())
        norms = brain.parameter_norms()
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)


# ---------------------------------------------------------------------------
# BrainStep new fields
# ---------------------------------------------------------------------------

class BrainStepNewFieldsTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepare the test fixture by creating a SpiderBrain and a blank observation.
        
        Sets:
            self.brain: a SpiderBrain instance initialized with seed 99 and module_dropout 0.0.
            self.obs: a blank observation mapping produced by _blank_obs().
        """
        self.brain = SpiderBrain(seed=99, module_dropout=0.0)
        self.obs = _blank_obs()

    def _obs_for_valence(self, valence: str) -> dict[str, np.ndarray]:
        """
        Construct a test observation dictionary whose module and context vectors are tuned to a specified behavioral valence.
        
        This helper produces a mapping from observation keys (e.g., "action_context", "visual", "sensory", "hunger", "sleep", "alert") to numpy arrays representing interface vectors. Each returned vector encodes signal values chosen to exercise the corresponding valence so tests can validate arbitration, proposal, and downstream behaviour.
        
        Parameters:
            valence (str): Target valence to simulate; expected values include "threat", "hunger", "sleep", and "exploration". The chosen valence affects which signals are elevated in the returned observation vectors.
        
        Returns:
            dict[str, np.ndarray]: A mapping from observation names to their serialized numpy vector representations.
        """
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            day=1.0 if valence == "exploration" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
            hunger=0.95 if valence == "hunger" else 0.05,
            fatigue=0.95 if valence == "sleep" else 0.08,
            recent_pain=0.7 if valence == "threat" else 0.0,
            recent_contact=0.8 if valence == "threat" else 0.0,
            on_shelter=1.0 if valence == "sleep" else 0.0,
            predator_visible=0.95 if valence == "threat" else 0.0,
            predator_certainty=0.92 if valence == "threat" else 0.0,
            predator_dist=0.08 if valence == "threat" else 1.0,
            sleep_debt=0.95 if valence == "sleep" else 0.05,
            shelter_role_level=0.9 if valence == "sleep" else 0.0,
        )
        obs["visual"] = _vector_for(
            _module_interface("visual_cortex"),
            food_visible=0.9 if valence == "hunger" else 0.0,
            food_certainty=0.85 if valence == "hunger" else 0.0,
            predator_visible=0.95 if valence == "threat" else 0.0,
            predator_certainty=0.9 if valence == "threat" else 0.0,
            day=1.0 if valence == "exploration" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
        )
        obs["sensory"] = _vector_for(
            _module_interface("sensory_cortex"),
            recent_pain=0.7 if valence == "threat" else 0.0,
            recent_contact=0.8 if valence == "threat" else 0.0,
            hunger=0.95 if valence == "hunger" else 0.05,
            fatigue=0.95 if valence == "sleep" else 0.08,
            food_smell_strength=0.7 if valence == "hunger" else 0.1,
            food_smell_dx=0.4 if valence == "exploration" else 0.0,
            predator_smell_strength=0.85 if valence == "threat" else 0.0,
            light=1.0 if valence == "exploration" else 0.0,
        )
        obs["hunger"] = _vector_for(
            _module_interface("hunger_center"),
            hunger=0.95 if valence == "hunger" else 0.05,
            food_visible=0.95 if valence == "hunger" else 0.0,
            food_certainty=0.9 if valence == "hunger" else 0.0,
            food_smell_strength=0.82 if valence == "hunger" else 0.0,
            food_memory_age=0.1 if valence == "hunger" else 1.0,
        )
        obs["sleep"] = _vector_for(
            _module_interface("sleep_center"),
            fatigue=0.95 if valence == "sleep" else 0.08,
            hunger=0.05,
            on_shelter=1.0 if valence == "sleep" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
            home_dist=0.05 if valence == "sleep" else 1.0,
            sleep_phase_level=0.6 if valence == "sleep" else 0.0,
            sleep_debt=0.95 if valence == "sleep" else 0.05,
            shelter_role_level=0.9 if valence == "sleep" else 0.0,
            shelter_memory_age=0.1 if valence == "sleep" else 1.0,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=0.95 if valence == "threat" else 0.0,
            predator_certainty=0.92 if valence == "threat" else 0.0,
            predator_dist=0.08 if valence == "threat" else 1.0,
            predator_smell_strength=0.9 if valence == "threat" else 0.0,
            recent_pain=0.7 if valence == "threat" else 0.0,
            recent_contact=0.8 if valence == "threat" else 0.0,
            on_shelter=1.0 if valence == "sleep" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
        )
        return obs

    def test_act_returns_brain_step(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step, BrainStep)

    def test_action_center_logits_shape(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertEqual(step.action_center_logits.shape, (len(LOCOMOTION_ACTIONS),))

    def test_action_center_policy_shape(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertEqual(step.action_center_policy.shape, (len(LOCOMOTION_ACTIONS),))

    def test_action_center_policy_sums_to_one(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertAlmostEqual(float(np.sum(step.action_center_policy)), 1.0, places=5)

    def test_action_center_policy_non_negative(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertTrue(np.all(step.action_center_policy >= 0.0))

    def test_action_intent_idx_is_valid(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreaterEqual(step.action_intent_idx, 0)
        self.assertLess(step.action_intent_idx, len(LOCOMOTION_ACTIONS))

    def test_action_idx_is_valid(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))

    def test_motor_override_is_bool(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.motor_override, bool)

    def test_action_center_input_is_not_empty(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreater(step.action_center_input.size, 0)

    def test_motor_input_is_not_empty(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreater(step.motor_input.size, 0)

    def test_motor_correction_logits_shape(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertEqual(step.motor_correction_logits.shape, (len(LOCOMOTION_ACTIONS),))

    def test_final_reflex_override_is_bool(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.final_reflex_override, bool)

    def test_value_is_finite_float(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.value, float)
        self.assertTrue(np.isfinite(step.value))

    def test_action_center_input_has_expected_dim(self) -> None:
        """action_center_input must equal num_modules*action_dim + action_context_dim."""
        step = self.brain.act(self.obs, bus=None, sample=False)
        num_modules = len(self.brain.module_bank.specs)
        action_dim = self.brain.action_dim
        expected_dim = num_modules * action_dim + ACTION_CONTEXT_INTERFACE.input_dim
        self.assertEqual(step.action_center_input.shape, (expected_dim,))

    def test_action_context_nan_inf_are_sanitized_before_binding(self) -> None:
        obs = _blank_obs()
        obs["action_context"] = np.array(
            [np.nan, np.inf, -np.inf] + [0.0] * (ACTION_CONTEXT_INTERFACE.input_dim - 3),
            dtype=float,
        )
        step = self.brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.isfinite(step.action_center_input).all())
        self.assertTrue(all(np.isfinite(v) for v in step.arbitration_decision.valence_scores.values()))

    def test_motor_input_has_expected_dim(self) -> None:
        """motor_input must equal action_dim + motor_context_dim."""
        step = self.brain.act(self.obs, bus=None, sample=False)
        expected_dim = self.brain.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
        self.assertEqual(step.motor_input.shape, (expected_dim,))

    def test_arbitration_decision_is_present(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsNotNone(step.arbitration_decision)

    def test_threat_case_wins_threat_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("threat"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "threat")
        self.assertLess(step.arbitration_decision.module_gates["hunger_center"], 0.5)

    def test_hunger_case_wins_hunger_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("hunger"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "hunger")
        hunger_result = next(result for result in step.module_results if result.name == "hunger_center")
        self.assertEqual(hunger_result.valence_role, "hunger")

    def test_sleep_case_wins_sleep_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("sleep"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "sleep")
        self.assertLess(step.arbitration_decision.module_gates["visual_cortex"], 0.6)

    def test_exploration_case_wins_exploration_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("exploration"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "exploration")
        self.assertGreaterEqual(step.arbitration_decision.module_gates["visual_cortex"], 0.95)


# ---------------------------------------------------------------------------
# _build_motor_input NaN/inf sanitization
# ---------------------------------------------------------------------------

class BuildMotorInputSanitizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=5, module_dropout=0.0)
        self.motor_obs = {
            MOTOR_CONTEXT_INTERFACE.observation_key: np.zeros(
                MOTOR_CONTEXT_INTERFACE.input_dim, dtype=float
            )
        }

    def test_nan_in_intent_becomes_zero(self) -> None:
        intent = np.full(self.brain.action_dim, np.nan)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        # First action_dim values must be 0.0
        np.testing.assert_array_equal(
            result[: self.brain.action_dim], np.zeros(self.brain.action_dim)
        )

    def test_posinf_in_intent_becomes_one(self) -> None:
        intent = np.full(self.brain.action_dim, np.inf)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        np.testing.assert_array_equal(
            result[: self.brain.action_dim], np.ones(self.brain.action_dim)
        )

    def test_neginf_in_intent_becomes_minus_one(self) -> None:
        intent = np.full(self.brain.action_dim, -np.inf)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        np.testing.assert_array_equal(
            result[: self.brain.action_dim], np.full(self.brain.action_dim, -1.0)
        )

    def test_mixed_nan_inf_sanitized(self) -> None:
        intent = np.array([np.nan, np.inf, -np.inf, 0.5, 0.0], dtype=float)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        expected = np.array([0.0, 1.0, -1.0, 0.5, 0.0], dtype=float)
        np.testing.assert_array_equal(result[: self.brain.action_dim], expected)

    def test_valid_intent_passes_through_unchanged(self) -> None:
        intent = np.array([0.2, 0.3, 0.1, 0.4, 0.0], dtype=float)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        np.testing.assert_allclose(result[: self.brain.action_dim], intent)

    def test_output_length_equals_action_dim_plus_motor_context_dim(self) -> None:
        intent = np.zeros(self.brain.action_dim)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        expected_len = self.brain.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
        self.assertEqual(result.shape, (expected_len,))


# ---------------------------------------------------------------------------
# estimate_value tests
# ---------------------------------------------------------------------------

class EstimateValueTest(unittest.TestCase):
    """Tests that estimate_value uses action_center (which has a value head)."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=17, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_estimate_value_returns_finite_float(self) -> None:
        value = self.brain.estimate_value(self.obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_zeroing_motor_cortex_does_not_eliminate_value(self) -> None:
        """Zeroing motor_cortex should not make estimate_value return 0.0
        because estimate_value uses action_center, not motor_cortex."""
        # Zero all motor_cortex parameters
        self.brain.motor_cortex.W1.fill(0.0)
        self.brain.motor_cortex.b1.fill(0.0)
        self.brain.motor_cortex.W_mid.fill(0.0)
        self.brain.motor_cortex.b_mid.fill(0.0)
        self.brain.motor_cortex.W2.fill(0.0)
        self.brain.motor_cortex.b2.fill(0.0)
        # estimate_value should still work through action_center
        value = self.brain.estimate_value(self.obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_zeroing_action_center_weights_affects_value(self) -> None:
        """Zeroing action_center weights and biases should make value 0.0
        since action_center.W2_value @ 0 + 0 = 0."""
        self.brain.action_center.W1.fill(0.0)
        self.brain.action_center.b1.fill(0.0)
        self.brain.action_center.W_mid.fill(0.0)
        self.brain.action_center.b_mid.fill(0.0)
        self.brain.action_center.W2_policy.fill(0.0)
        self.brain.action_center.b2_policy.fill(0.0)
        self.brain.action_center.W2_value.fill(0.0)
        self.brain.action_center.b2_value.fill(0.0)
        value = self.brain.estimate_value(self.obs)
        self.assertAlmostEqual(value, 0.0, places=10)

    def test_estimate_value_matches_act_value(self) -> None:
        """estimate_value and act().value can differ (different cache states),
        but both should be finite."""
        step = self.brain.act(self.obs, bus=None, sample=False)
        value = self.brain.estimate_value(self.obs)
        self.assertTrue(np.isfinite(step.value))
        self.assertTrue(np.isfinite(value))


# ---------------------------------------------------------------------------
# Integration: action_context in observe_world output
# ---------------------------------------------------------------------------

class ObserveWorldActionContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=23)
        self.obs = self.world.reset(seed=23)

    def test_action_context_key_present(self) -> None:
        self.assertIn("action_context", self.obs)

    def test_action_context_is_numpy_array(self) -> None:
        self.assertIsInstance(self.obs["action_context"], np.ndarray)

    def test_action_context_shape(self) -> None:
        self.assertEqual(self.obs["action_context"].shape, (16,))

    def test_motor_context_shape_is_10(self) -> None:
        """After this PR, motor_context must have 10 elements (down from 16)."""
        self.assertEqual(self.obs["motor_context"].shape, (10,))

    def test_action_context_reflects_hunger(self) -> None:
        self.world.state.hunger = 0.66
        obs = self.world.observe()
        mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        self.assertAlmostEqual(mapping["hunger"], 0.66, places=5)

    def test_action_context_has_sleep_debt(self) -> None:
        self.world.state.sleep_debt = 0.44
        obs = self.world.observe()
        mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        self.assertAlmostEqual(mapping["sleep_debt"], 0.44, places=5)

    def test_motor_context_lacks_hunger(self) -> None:
        """motor_context vector must not contain a 'hunger' entry."""
        mapping = MOTOR_CONTEXT_INTERFACE.bind_values(self.obs["motor_context"])
        self.assertNotIn("hunger", mapping)

    def test_motor_context_lacks_sleep_debt(self) -> None:
        mapping = MOTOR_CONTEXT_INTERFACE.bind_values(self.obs["motor_context"])
        self.assertNotIn("sleep_debt", mapping)

    def test_action_context_and_motor_context_share_on_food_value(self) -> None:
        """Both contexts must agree on on_food flag."""
        obs = self.world.observe()
        ac_mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        mc_mapping = MOTOR_CONTEXT_INTERFACE.bind_values(obs["motor_context"])
        self.assertAlmostEqual(ac_mapping["on_food"], mc_mapping["on_food"], places=5)

    def test_action_context_and_motor_context_share_on_shelter_value(self) -> None:
        obs = self.world.observe()
        ac_mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        mc_mapping = MOTOR_CONTEXT_INTERFACE.bind_values(obs["motor_context"])
        self.assertAlmostEqual(ac_mapping["on_shelter"], mc_mapping["on_shelter"], places=5)


# ---------------------------------------------------------------------------
# Boundary / regression tests
# ---------------------------------------------------------------------------

class ActionCenterRegressionTest(unittest.TestCase):
    """Additional boundary and regression cases."""

    def test_brain_step_greedy_and_sample_give_valid_actions(self) -> None:
        brain = SpiderBrain(seed=31, module_dropout=0.0)
        obs = _blank_obs()
        for sample in (True, False):
            with self.subTest(sample=sample):
                step = brain.act(obs, bus=None, sample=sample)
                self.assertIn(step.action_idx, range(len(LOCOMOTION_ACTIONS)))
                self.assertIn(step.action_intent_idx, range(len(LOCOMOTION_ACTIONS)))

    def test_action_center_logits_are_finite(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.all(np.isfinite(step.action_center_logits)))

    def test_motor_correction_logits_are_finite(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.all(np.isfinite(step.motor_correction_logits)))

    def test_action_center_input_is_finite(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.all(np.isfinite(step.action_center_input)))

    def test_action_center_policy_does_not_contain_nan(self) -> None:
        brain = SpiderBrain(seed=77, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertFalse(np.any(np.isnan(step.action_center_policy)))

    def test_motor_override_false_when_no_motor_correction(self) -> None:
        """If motor_cortex weights are zero, correction logits are 0; motor_override
        should be False when action_center intent matches total argmax."""
        brain = SpiderBrain(seed=55, module_dropout=0.0)
        # Zero motor_cortex so it adds no correction
        brain.motor_cortex.W1.fill(0.0)
        brain.motor_cortex.b1.fill(0.0)
        brain.motor_cortex.W_mid.fill(0.0)
        brain.motor_cortex.b_mid.fill(0.0)
        brain.motor_cortex.W2.fill(0.0)
        brain.motor_cortex.b2.fill(0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        # With zero motor correction, action_center_logits == total_logits
        np.testing.assert_allclose(step.action_center_logits, step.total_logits, atol=1e-10)
        self.assertFalse(step.motor_override)

    def test_action_context_observation_key_is_action_context(self) -> None:
        self.assertEqual(ActionContextObservation.observation_key, "action_context")

    def test_motor_context_observation_key_is_motor_context(self) -> None:
        self.assertEqual(MotorContextObservation.observation_key, "motor_context")

    def test_action_dim_is_five(self) -> None:
        """LOCOMOTION_ACTIONS has 5 entries; action_dim on brain must match."""
        brain = SpiderBrain(seed=1, module_dropout=0.0)
        self.assertEqual(brain.action_dim, 5)


# ---------------------------------------------------------------------------
# ValenceScore serialization tests
# ---------------------------------------------------------------------------

class ValenceScoreSerializationTest(unittest.TestCase):
    """Tests for ValenceScore.to_payload()."""

    def _make(self, name="threat", score=0.75, evidence=None):
        from spider_cortex_sim.agent import ValenceScore
        if evidence is None:
            evidence = {"predator_visible": 1.0, "recent_contact": 0.5}
        return ValenceScore(name=name, score=score, evidence=evidence)

    def test_to_payload_has_required_keys(self) -> None:
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make()
        payload = vs.to_payload()
        self.assertIn("name", payload)
        self.assertIn("score", payload)
        self.assertIn("evidence", payload)

    def test_to_payload_name_preserved(self) -> None:
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make(name="hunger")
        self.assertEqual(vs.to_payload()["name"], "hunger")

    def test_to_payload_score_rounded_to_six_decimals(self) -> None:
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make(score=0.123456789)
        payload = vs.to_payload()
        self.assertEqual(payload["score"], round(0.123456789, 6))

    def test_to_payload_evidence_keys_sorted(self) -> None:
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make(evidence={"z_key": 0.1, "a_key": 0.9, "m_key": 0.5})
        payload = vs.to_payload()
        keys = list(payload["evidence"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_evidence_values_rounded(self) -> None:
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make(evidence={"k": 0.999999999})
        payload = vs.to_payload()
        self.assertEqual(payload["evidence"]["k"], round(0.999999999, 6))

    def test_to_payload_empty_evidence(self) -> None:
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make(evidence={})
        payload = vs.to_payload()
        self.assertEqual(payload["evidence"], {})

    def test_to_payload_score_zero(self) -> None:
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make(score=0.0)
        self.assertEqual(vs.to_payload()["score"], 0.0)

    def test_to_payload_is_json_serializable(self) -> None:
        import json
        from spider_cortex_sim.agent import ValenceScore
        vs = self._make()
        # Should not raise
        json.dumps(vs.to_payload())


# ---------------------------------------------------------------------------
# ArbitrationDecision serialization tests
# ---------------------------------------------------------------------------

class ArbitrationDecisionSerializationTest(unittest.TestCase):
    """Tests for ArbitrationDecision.to_payload()."""

    def _make(self, winning_valence="threat", intent_before=0, intent_after=2):
        from spider_cortex_sim.agent import ArbitrationDecision
        return ArbitrationDecision(
            strategy="priority_gating",
            winning_valence=winning_valence,
            valence_scores={"threat": 0.6, "hunger": 0.2, "sleep": 0.1, "exploration": 0.1},
            module_gates={"alert_center": 1.0, "hunger_center": 0.18},
            suppressed_modules=["hunger_center"],
            evidence={
                "threat": {"predator_visible": 1.0},
                "hunger": {"hunger": 0.3},
            },
            intent_before_gating_idx=intent_before,
            intent_after_gating_idx=intent_after,
        )

    def test_to_payload_strategy_preserved(self) -> None:
        d = self._make()
        payload = d.to_payload()
        self.assertEqual(payload["strategy"], "priority_gating")

    def test_to_payload_winning_valence_preserved(self) -> None:
        d = self._make(winning_valence="hunger")
        self.assertEqual(d.to_payload()["winning_valence"], "hunger")

    def test_to_payload_has_all_required_keys(self) -> None:
        payload = self._make().to_payload()
        for key in ("strategy", "winning_valence", "valence_scores", "module_gates",
                    "module_contribution_share", "dominant_module", "dominant_module_share",
                    "effective_module_count", "module_agreement_rate", "module_disagreement_rate",
                    "suppressed_modules", "evidence", "intent_before_gating", "intent_after_gating"):
            self.assertIn(key, payload)

    def test_legacy_constructor_keeps_new_payload_fields_with_defaults(self) -> None:
        payload = self._make().to_payload()
        self.assertEqual(payload["module_contribution_share"], {})
        self.assertEqual(payload["dominant_module"], "")
        self.assertEqual(payload["dominant_module_share"], 0.0)
        self.assertEqual(payload["effective_module_count"], 0.0)
        self.assertEqual(payload["module_agreement_rate"], 0.0)
        self.assertEqual(payload["module_disagreement_rate"], 0.0)

    def test_to_payload_module_contribution_share_sorted(self) -> None:
        from spider_cortex_sim.agent import ArbitrationDecision
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
            module_contribution_share={"visual_cortex": 0.25, "alert_center": 0.75},
        )
        keys = list(d.to_payload()["module_contribution_share"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_new_metrics_rounded(self) -> None:
        from spider_cortex_sim.agent import ArbitrationDecision
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
            module_contribution_share={"alert_center": 0.3333333333},
            dominant_module="alert_center",
            dominant_module_share=0.7777777777,
            effective_module_count=1.6666666666,
            module_agreement_rate=0.8888888888,
            module_disagreement_rate=0.1111111111,
        )
        payload = d.to_payload()
        self.assertEqual(payload["module_contribution_share"]["alert_center"], round(0.3333333333, 6))
        self.assertEqual(payload["dominant_module"], "alert_center")
        self.assertEqual(payload["dominant_module_share"], round(0.7777777777, 6))
        self.assertEqual(payload["effective_module_count"], round(1.6666666666, 6))
        self.assertEqual(payload["module_agreement_rate"], round(0.8888888888, 6))
        self.assertEqual(payload["module_disagreement_rate"], round(0.1111111111, 6))

    def test_to_payload_intent_before_gating_is_action_name(self) -> None:
        d = self._make(intent_before=0)
        payload = d.to_payload()
        self.assertIn(payload["intent_before_gating"], LOCOMOTION_ACTIONS)

    def test_to_payload_intent_after_gating_is_action_name(self) -> None:
        d = self._make(intent_after=2)
        payload = d.to_payload()
        self.assertIn(payload["intent_after_gating"], LOCOMOTION_ACTIONS)

    def test_to_payload_intent_names_differ_when_indices_differ(self) -> None:
        d = self._make(intent_before=0, intent_after=1)
        payload = d.to_payload()
        self.assertNotEqual(payload["intent_before_gating"], payload["intent_after_gating"])

    def test_to_payload_intent_names_same_when_indices_same(self) -> None:
        d = self._make(intent_before=3, intent_after=3)
        payload = d.to_payload()
        self.assertEqual(payload["intent_before_gating"], payload["intent_after_gating"])

    def test_to_payload_valence_scores_sorted(self) -> None:
        d = self._make()
        payload = d.to_payload()
        keys = list(payload["valence_scores"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_module_gates_sorted(self) -> None:
        d = self._make()
        payload = d.to_payload()
        keys = list(payload["module_gates"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_suppressed_modules_is_list(self) -> None:
        d = self._make()
        self.assertIsInstance(d.to_payload()["suppressed_modules"], list)

    def test_to_payload_evidence_inner_keys_sorted(self) -> None:
        from spider_cortex_sim.agent import ArbitrationDecision
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"threat": {"z": 1.0, "a": 0.5}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        evidence_threat = d.to_payload()["evidence"]["threat"]
        keys = list(evidence_threat.keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_is_json_serializable(self) -> None:
        import json
        json.dumps(self._make().to_payload())

    def test_to_payload_valence_scores_rounded(self) -> None:
        from spider_cortex_sim.agent import ArbitrationDecision
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={"threat": 0.1111111111},
            module_gates={},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        self.assertEqual(payload["valence_scores"]["threat"], round(0.1111111111, 6))


# ---------------------------------------------------------------------------
# SpiderBrain._clamp_unit tests
# ---------------------------------------------------------------------------

class ClampUnitTest(unittest.TestCase):
    """Tests for SpiderBrain._clamp_unit."""

    def test_value_below_zero_clamped_to_zero(self) -> None:
        self.assertEqual(SpiderBrain._clamp_unit(-0.5), 0.0)

    def test_value_above_one_clamped_to_one(self) -> None:
        self.assertEqual(SpiderBrain._clamp_unit(1.5), 1.0)

    def test_exact_zero_unchanged(self) -> None:
        self.assertEqual(SpiderBrain._clamp_unit(0.0), 0.0)

    def test_exact_one_unchanged(self) -> None:
        self.assertEqual(SpiderBrain._clamp_unit(1.0), 1.0)

    def test_midrange_value_unchanged(self) -> None:
        self.assertAlmostEqual(SpiderBrain._clamp_unit(0.5), 0.5)

    def test_returns_float(self) -> None:
        result = SpiderBrain._clamp_unit(0.3)
        self.assertIsInstance(result, float)

    def test_large_negative_clamped_to_zero(self) -> None:
        self.assertEqual(SpiderBrain._clamp_unit(-100.0), 0.0)

    def test_large_positive_clamped_to_one(self) -> None:
        self.assertEqual(SpiderBrain._clamp_unit(100.0), 1.0)

    def test_very_small_positive_preserved(self) -> None:
        result = SpiderBrain._clamp_unit(1e-10)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)


# ---------------------------------------------------------------------------
# SpiderBrain._bound_observation tests
# ---------------------------------------------------------------------------

class BoundObservationTest(unittest.TestCase):
    """Tests for SpiderBrain._bound_observation."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=1, module_dropout=0.0)

    def test_valid_interface_returns_dict(self) -> None:
        obs = _blank_obs()
        result = self.brain._bound_observation("visual_cortex", obs)
        self.assertIsInstance(result, dict)

    def test_valid_interface_keys_are_strings(self) -> None:
        obs = _blank_obs()
        result = self.brain._bound_observation("hunger_center", obs)
        for key in result:
            self.assertIsInstance(key, str)

    def test_valid_interface_values_are_floats(self) -> None:
        obs = _blank_obs()
        result = self.brain._bound_observation("alert_center", obs)
        for value in result.values():
            self.assertIsInstance(value, float)

    def test_unknown_interface_raises_key_error(self) -> None:
        obs = _blank_obs()
        with self.assertRaises(KeyError):
            self.brain._bound_observation("nonexistent_interface", obs)

    def test_all_module_interfaces_bindable(self) -> None:
        obs = _blank_obs()
        for iface in MODULE_INTERFACES:
            result = self.brain._bound_observation(iface.name, obs)
            self.assertIsInstance(result, dict)
            self.assertGreater(len(result), 0)

    def test_nonzero_values_propagated(self) -> None:
        iface = _module_interface("hunger_center")
        obs = _blank_obs()
        obs[iface.observation_key] = _vector_for(iface, hunger=0.88)
        result = self.brain._bound_observation("hunger_center", obs)
        self.assertIn("hunger", result)
        self.assertAlmostEqual(result["hunger"], 0.88, places=5)


# ---------------------------------------------------------------------------
# SpiderBrain._compute_arbitration tests
# ---------------------------------------------------------------------------

class ComputeArbitrationTest(unittest.TestCase):
    """Tests for SpiderBrain._compute_arbitration."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=3, module_dropout=0.0)
        self.obs = _blank_obs()

    def _get_arbitration(self, obs=None):
        if obs is None:
            obs = self.obs
        step = self.brain.act(obs, bus=None, sample=False)
        return step.arbitration_decision

    def test_returns_arbitration_decision(self) -> None:
        from spider_cortex_sim.agent import ArbitrationDecision
        arb = self._get_arbitration()
        self.assertIsInstance(arb, ArbitrationDecision)

    def test_strategy_is_priority_gating(self) -> None:
        arb = self._get_arbitration()
        self.assertEqual(arb.strategy, "priority_gating")

    def test_valence_scores_sum_to_one(self) -> None:
        arb = self._get_arbitration()
        total = sum(arb.valence_scores.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_valence_scores_all_non_negative(self) -> None:
        arb = self._get_arbitration()
        for score in arb.valence_scores.values():
            self.assertGreaterEqual(score, 0.0)

    def test_winning_valence_is_in_valence_order(self) -> None:
        arb = self._get_arbitration()
        self.assertIn(arb.winning_valence, SpiderBrain.VALENCE_ORDER)

    def test_module_gates_contains_all_active_modules(self) -> None:
        arb = self._get_arbitration()
        for iface in MODULE_INTERFACES:
            self.assertIn(iface.name, arb.module_gates)

    def test_module_gate_values_in_unit_range(self) -> None:
        arb = self._get_arbitration()
        for weight in arb.module_gates.values():
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)

    def test_suppressed_modules_is_list(self) -> None:
        arb = self._get_arbitration()
        self.assertIsInstance(arb.suppressed_modules, list)

    def test_evidence_contains_four_valences(self) -> None:
        arb = self._get_arbitration()
        for valence in SpiderBrain.VALENCE_ORDER:
            self.assertIn(valence, arb.evidence)

    def test_intent_indices_are_valid_action_indices(self) -> None:
        arb = self._get_arbitration()
        self.assertIn(arb.intent_before_gating_idx, range(len(LOCOMOTION_ACTIONS)))
        self.assertIn(arb.intent_after_gating_idx, range(len(LOCOMOTION_ACTIONS)))

    def test_blank_obs_defaults_exploration_wins_or_has_nonzero_score(self) -> None:
        # With blank (all-zero) observation, exploration should have score > 0
        # because residual_drive is high when threat/hunger/sleep are zero
        arb = self._get_arbitration()
        self.assertGreater(arb.valence_scores["exploration"], 0.0)

    def test_suppressed_modules_are_subset_of_module_gate_keys(self) -> None:
        arb = self._get_arbitration()
        for name in arb.suppressed_modules:
            self.assertIn(name, arb.module_gates)

    def test_suppressed_module_gates_are_less_than_one(self) -> None:
        arb = self._get_arbitration()
        for name in arb.suppressed_modules:
            self.assertLess(arb.module_gates[name], 0.999)

    def test_threat_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("predator_visible", "predator_certainty", "predator_proximity",
                    "recent_contact", "recent_pain", "predator_smell_strength"):
            self.assertIn(key, arb.evidence["threat"])

    def test_hunger_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("hunger", "on_food", "food_visible", "food_certainty",
                    "food_smell_strength", "food_memory_freshness"):
            self.assertIn(key, arb.evidence["hunger"])

    def test_sleep_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("fatigue", "sleep_debt", "night", "on_shelter",
                    "shelter_role_level", "home_pressure"):
            self.assertIn(key, arb.evidence["sleep"])

    def test_exploration_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("safety_margin", "residual_drive", "day", "off_shelter",
                    "visual_openness", "food_smell_directionality"):
            self.assertIn(key, arb.evidence["exploration"])


# ---------------------------------------------------------------------------
# SpiderBrain._apply_priority_gating tests
# ---------------------------------------------------------------------------

class ApplyPriorityGatingTest(unittest.TestCase):
    """Tests for SpiderBrain._apply_priority_gating."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=11, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_apply_priority_gating_sets_gate_weight(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.gate_weight)
            self.assertIsInstance(result.gate_weight, float)

    def test_apply_priority_gating_sets_gated_logits(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.gated_logits)
            self.assertEqual(result.gated_logits.shape, result.probs.shape)

    def test_apply_priority_gating_sets_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.valence_role)
            self.assertIsInstance(result.valence_role, str)

    def test_alert_center_has_threat_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        alert = next(r for r in step.module_results if r.name == "alert_center")
        self.assertEqual(alert.valence_role, "threat")

    def test_sleep_center_has_sleep_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        sleep = next(r for r in step.module_results if r.name == "sleep_center")
        self.assertEqual(sleep.valence_role, "sleep")

    def test_visual_cortex_has_support_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        visual = next(r for r in step.module_results if r.name == "visual_cortex")
        self.assertEqual(visual.valence_role, "support")

    def test_probs_sum_to_one_after_gating(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertAlmostEqual(float(np.sum(result.probs)), 1.0, places=5)

    def test_gate_weight_one_for_primary_module_under_threat(self) -> None:
        # When threat wins, alert_center should have gate_weight = 1.0
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            predator_visible=1.0,
            predator_certainty=1.0,
            predator_dist=0.0,
            recent_contact=1.0,
            recent_pain=0.8,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_smell_strength=1.0,
        )
        step = self.brain.act(obs, bus=None, sample=False)
        if step.arbitration_decision.winning_valence == "threat":
            alert = next(r for r in step.module_results if r.name == "alert_center")
            self.assertAlmostEqual(alert.gate_weight, 1.0, places=5)


# ---------------------------------------------------------------------------
# PRIORITY_GATING_WEIGHTS constant tests
# ---------------------------------------------------------------------------

class PriorityGatingWeightsTest(unittest.TestCase):
    """Tests for SpiderBrain.PRIORITY_GATING_WEIGHTS table correctness."""

    def test_all_four_valences_present(self) -> None:
        for valence in SpiderBrain.VALENCE_ORDER:
            self.assertIn(valence, SpiderBrain.PRIORITY_GATING_WEIGHTS)

    def test_each_valence_has_all_module_entries(self) -> None:
        expected_modules = {
            "alert_center", "hunger_center", "sleep_center",
            "visual_cortex", "sensory_cortex", SpiderBrain.MONOLITHIC_POLICY_NAME,
        }
        for valence, weights in SpiderBrain.PRIORITY_GATING_WEIGHTS.items():
            self.assertEqual(set(weights.keys()), expected_modules,
                             f"Missing modules for valence '{valence}'")

    def test_threat_primary_module_has_weight_one(self) -> None:
        self.assertAlmostEqual(
            SpiderBrain.PRIORITY_GATING_WEIGHTS["threat"]["alert_center"], 1.0
        )

    def test_threat_suppresses_hunger_center(self) -> None:
        w = SpiderBrain.PRIORITY_GATING_WEIGHTS["threat"]["hunger_center"]
        self.assertLess(w, 0.5)

    def test_hunger_primary_module_has_weight_one(self) -> None:
        self.assertAlmostEqual(
            SpiderBrain.PRIORITY_GATING_WEIGHTS["hunger"]["hunger_center"], 1.0
        )

    def test_sleep_primary_module_has_weight_one(self) -> None:
        self.assertAlmostEqual(
            SpiderBrain.PRIORITY_GATING_WEIGHTS["sleep"]["sleep_center"], 1.0
        )

    def test_exploration_visual_cortex_high_weight(self) -> None:
        w = SpiderBrain.PRIORITY_GATING_WEIGHTS["exploration"]["visual_cortex"]
        self.assertGreater(w, 0.9)

    def test_all_gate_weights_in_unit_range(self) -> None:
        for valence, weights in SpiderBrain.PRIORITY_GATING_WEIGHTS.items():
            for module, weight in weights.items():
                self.assertGreaterEqual(weight, 0.0,
                    f"Gate weight for {valence}/{module} is negative")
                self.assertLessEqual(weight, 1.0,
                    f"Gate weight for {valence}/{module} exceeds 1.0")

    def test_monolithic_policy_always_weight_one(self) -> None:
        for valence in SpiderBrain.VALENCE_ORDER:
            self.assertAlmostEqual(
                SpiderBrain.PRIORITY_GATING_WEIGHTS[valence][SpiderBrain.MONOLITHIC_POLICY_NAME],
                1.0,
            )

    def test_sleep_suppresses_hunger_center(self) -> None:
        w = SpiderBrain.PRIORITY_GATING_WEIGHTS["sleep"]["hunger_center"]
        self.assertLess(w, 0.5)

    def test_missing_module_gate_weight_raises_clear_error(self) -> None:
        incomplete_weights = {
            valence: dict(weights)
            for valence, weights in SpiderBrain.PRIORITY_GATING_WEIGHTS.items()
        }
        del incomplete_weights["threat"]["alert_center"]
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            predator_visible=1.0,
            predator_certainty=1.0,
            predator_dist=0.0,
            recent_contact=1.0,
            recent_pain=0.8,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_smell_strength=1.0,
        )
        with patch.object(SpiderBrain, "PRIORITY_GATING_WEIGHTS", incomplete_weights):
            brain = SpiderBrain(seed=5, module_dropout=0.0)
            with self.assertRaisesRegex(
                ValueError,
                "Priority gating weights missing module 'alert_center'",
            ):
                brain.act(obs, bus=None, sample=False)


# ---------------------------------------------------------------------------
# MODULE_VALENCE_ROLES constant tests
# ---------------------------------------------------------------------------

class ModuleValenceRolesTest(unittest.TestCase):
    """Tests for SpiderBrain.MODULE_VALENCE_ROLES mapping."""

    def test_alert_center_maps_to_threat(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["alert_center"], "threat")

    def test_hunger_center_maps_to_hunger(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["hunger_center"], "hunger")

    def test_sleep_center_maps_to_sleep(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["sleep_center"], "sleep")

    def test_visual_cortex_maps_to_support(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["visual_cortex"], "support")

    def test_sensory_cortex_maps_to_support(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["sensory_cortex"], "support")

    def test_monolithic_policy_maps_to_integrated_policy(self) -> None:
        self.assertEqual(
            SpiderBrain.MODULE_VALENCE_ROLES[SpiderBrain.MONOLITHIC_POLICY_NAME],
            "integrated_policy",
        )

    def test_missing_module_falls_back_to_support_via_get(self) -> None:
        role = SpiderBrain.MODULE_VALENCE_ROLES.get("unknown_module", "support")
        self.assertEqual(role, "support")


# ---------------------------------------------------------------------------
# ModuleResult new field defaults tests
# ---------------------------------------------------------------------------

class ModuleResultNewFieldsDefaultsTest(unittest.TestCase):
    """Tests for the new ModuleResult fields (valence_role, gate_weight, gated_logits)."""

    def _make_result(self):
        from spider_cortex_sim.modules import ModuleResult
        logits = np.zeros(5, dtype=float)
        probs = np.ones(5, dtype=float) / 5.0
        return ModuleResult(
            interface=None,
            name="test_module",
            observation_key="test",
            observation=np.zeros(3),
            logits=logits,
            probs=probs,
            active=True,
        )

    def test_valence_role_default_is_support(self) -> None:
        result = self._make_result()
        self.assertEqual(result.valence_role, "support")

    def test_gate_weight_default_is_one(self) -> None:
        result = self._make_result()
        self.assertEqual(result.gate_weight, 1.0)

    def test_gated_logits_default_is_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.gated_logits)

    def test_all_module_results_after_act_have_non_none_gated_logits(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.gated_logits)

    def test_all_module_results_have_valence_role_string(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for result in step.module_results:
            self.assertIsInstance(result.valence_role, str)

    def test_all_module_results_have_gate_weight_float(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for result in step.module_results:
            self.assertIsInstance(result.gate_weight, float)


# ---------------------------------------------------------------------------
# Valence normalization edge case: all-zero raw scores
# ---------------------------------------------------------------------------

class ValenceNormalizationEdgeCaseTest(unittest.TestCase):
    """Tests for the normalization logic in _compute_arbitration when raw scores are near zero."""

    def test_exploration_gets_full_score_when_total_is_near_zero(self) -> None:
        # When all raw evidence is zero and total is effectively 0,
        # the code should default exploration score to 1.0.
        #
        # To trigger the total <= 1e-8 condition in _compute_arbitration,
        # we need all raw valence scores (threat, hunger, sleep, exploration) to be near zero.
        # This is impossible with normal observations because exploration_raw depends on
        # residual_drive and safety_margin, which are 1.0 when other scores are 0.
        #
        # We use monkeypatching to force _clamp_unit to return 0.0 for all calculations,
        # which makes all raw scores evaluate to 0.0, triggering the fallback branch.

        import unittest.mock

        brain = SpiderBrain(seed=99, module_dropout=0.0)
        obs = _blank_obs()

        # Get module results from blank observation
        module_results = brain._proposal_results(obs, store_cache=False, training=False)

        # Monkeypatch _clamp_unit to always return 0.0, forcing all raw scores to 0
        with unittest.mock.patch.object(brain, '_clamp_unit', return_value=0.0):
            arb = brain._compute_arbitration(module_results, obs)

        # Assert the fallback behavior: exploration gets score 1.0
        self.assertAlmostEqual(arb.valence_scores["exploration"], 1.0, places=5)
        self.assertAlmostEqual(arb.valence_scores["threat"], 0.0, places=5)
        self.assertAlmostEqual(arb.valence_scores["hunger"], 0.0, places=5)
        self.assertAlmostEqual(arb.valence_scores["sleep"], 0.0, places=5)

        # Scores must sum to 1
        total = sum(arb.valence_scores.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_valence_scores_are_non_negative_for_blank_obs(self) -> None:
        brain = SpiderBrain(seed=99, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for valence, score in step.arbitration_decision.valence_scores.items():
            self.assertGreaterEqual(score, 0.0, f"Score for {valence} is negative")


# ---------------------------------------------------------------------------
# Arbitration decision fields in BrainStep for reflex_only mode
# ---------------------------------------------------------------------------

class ReflexOnlyModeArbitrationTest(unittest.TestCase):
    """Tests that reflex_only mode does not expose arbitration from prior act calls."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=22, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_reflex_only_mode_also_has_arbitration_decision(self) -> None:
        # reflex_only still runs arbitration since it's computed before the mode branch
        step = self.brain.act(self.obs, bus=None, sample=False, policy_mode="reflex_only")
        # Test that the act call succeeds and returns a BrainStep
        self.assertIsInstance(step, BrainStep)
        # Test that arbitration_decision is computed for reflex_only path
        self.assertIsNotNone(step.arbitration_decision)
        self.assertIsInstance(step.arbitration_decision, ArbitrationDecision)

    def test_invalid_policy_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.brain.act(self.obs, bus=None, sample=False, policy_mode="invalid_mode")


if __name__ == "__main__":
    unittest.main()
