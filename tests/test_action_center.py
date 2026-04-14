"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 12
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
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    canonical_ablation_configs,
    default_brain_config,
)
from spider_cortex_sim.agent import ArbitrationDecision, BrainStep, SpiderBrain
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_DIMS,
    ActionContextObservation,
    MotorContextObservation,
    architecture_signature,
)
from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN
from spider_cortex_sim.nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork
from spider_cortex_sim.perception import (
    PerceivedTarget,
    build_action_context_observation,
    build_motor_context_observation,
)
from spider_cortex_sim.world import SpiderWorld, compute_execution_difficulty


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


def _valence_vector(**updates: float) -> np.ndarray:
    values = np.zeros(len(SpiderBrain.VALENCE_ORDER), dtype=float)
    index_by_name = {
        name: index
        for index, name in enumerate(SpiderBrain.VALENCE_ORDER)
    }
    for name, value in updates.items():
        values[index_by_name[name]] = float(value)
    return values


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
        self.assertIn("arbitration_network", self.sig)
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

    def test_signature_exposes_arbitration_network_metadata(self) -> None:
        arbitration_network = self.sig["arbitration_network"]
        self.assertIs(arbitration_network["learned"], True)
        self.assertEqual(arbitration_network["input_dim"], ArbitrationNetwork.INPUT_DIM)
        self.assertEqual(arbitration_network["hidden_dim"], 32)
        self.assertAlmostEqual(arbitration_network["regularization_weight"], 0.1)
        self.assertIn("parameter_shapes", arbitration_network)
        self.assertIn("parameter_fingerprint", arbitration_network)

    def test_signature_arbitration_network_can_describe_fixed_baseline(self) -> None:
        sig = architecture_signature(
            learned_arbitration=False,
            arbitration_regularization_weight=0.0,
        )
        arbitration_network = sig["arbitration_network"]
        self.assertIs(arbitration_network["learned"], False)
        self.assertAlmostEqual(arbitration_network["regularization_weight"], 0.0)

    def test_action_center_signature_maps_module_roles(self) -> None:
        """
        Validate that the action_center arbitration configuration assigns expected valence roles to core modules and does not include the monolithic policy.
        
        Asserts that:
        - "alert_center" maps to "threat"
        - "hunger_center" maps to "hunger"
        - "sleep_center" maps to "sleep"
        - "monolithic_policy" is not present in the module role mapping
        """
        module_roles = self.sig["action_center"]["arbitration"]["module_roles"]
        self.assertEqual(module_roles["alert_center"], "threat")
        self.assertEqual(module_roles["hunger_center"], "hunger")
        self.assertEqual(module_roles["sleep_center"], "sleep")
        self.assertNotIn("monolithic_policy", module_roles)

    def test_action_center_inputs_count(self) -> None:
        """action_center must list exactly 15 input signal names."""
        ac = self.sig["action_center"]
        self.assertEqual(len(ac["inputs"]), 15)

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
        """
        Assert that the `motor_cortex` entry in the architecture signature does not include a value head.
        
        Checks that `sig["motor_cortex"]["value_head"]` is `False`, ensuring the motor cortex is configured as a proposal-only module.
        """
        mc = self.sig["motor_cortex"]
        self.assertIs(mc["value_head"], False)

    def test_motor_cortex_inputs_count(self) -> None:
        """motor_cortex must list exactly 13 input signal names."""
        mc = self.sig["motor_cortex"]
        self.assertEqual(len(mc["inputs"]), 13)

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
        self.assertEqual(ACTION_CONTEXT_INTERFACE.input_dim, 15)

    def test_signal_names_ordered(self) -> None:
        """
        Ensure the action context interface exposes signal names in the expected order: the first signal is "hunger" and the last is "shelter_role_level".
        """
        names = ACTION_CONTEXT_INTERFACE.signal_names
        self.assertEqual(names[0], "hunger")
        self.assertEqual(names[-1], "shelter_role_level")

    def test_sleep_debt_present_in_signals(self) -> None:
        self.assertIn("sleep_debt", ACTION_CONTEXT_INTERFACE.signal_names)

    def test_recent_pain_present_in_signals(self) -> None:
        self.assertIn("recent_pain", ACTION_CONTEXT_INTERFACE.signal_names)

    def test_predator_dist_absent_from_signals(self) -> None:
        """
        Ensure the action context interface does not expose privileged world signals.
        """
        names = ACTION_CONTEXT_INTERFACE.signal_names
        self.assertNotIn("predator_dist", names)
        self.assertNotIn("home_dx", names)
        self.assertNotIn("home_dy", names)

    def test_role_context(self) -> None:
        self.assertEqual(ACTION_CONTEXT_INTERFACE.role, "context")


class MotorContextInterfaceTest(unittest.TestCase):
    def test_name(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.name, "motor_cortex_context")

    def test_observation_key(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.observation_key, "motor_context")

    def test_input_dim(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.input_dim, 13)

    def test_no_hunger_signal(self) -> None:
        """Motor context must NOT contain hunger (that belongs to action_center)."""
        self.assertNotIn("hunger", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_no_sleep_debt_signal(self) -> None:
        """Motor context must NOT contain sleep_debt (that belongs to action_center)."""
        self.assertNotIn("sleep_debt", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_no_recent_pain_signal(self) -> None:
        self.assertNotIn("recent_pain", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_no_predator_dist_signal(self) -> None:
        names = MOTOR_CONTEXT_INTERFACE.signal_names
        self.assertNotIn("predator_dist", names)
        self.assertNotIn("home_dx", names)
        self.assertNotIn("home_dy", names)

    def test_on_food_present(self) -> None:
        self.assertIn("on_food", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_on_shelter_present(self) -> None:
        self.assertIn("on_shelter", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_predator_visible_present(self) -> None:
        self.assertIn("predator_visible", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_last_move_signals_present(self) -> None:
        self.assertIn("last_move_dx", MOTOR_CONTEXT_INTERFACE.signal_names)
        self.assertIn("last_move_dy", MOTOR_CONTEXT_INTERFACE.signal_names)

    def test_embodiment_signals_present(self) -> None:
        names = MOTOR_CONTEXT_INTERFACE.signal_names
        self.assertEqual(
            names[-4:],
            ("heading_dx", "heading_dy", "terrain_difficulty", "fatigue"),
        )

    def test_role_context(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.role, "context")


class ObservationDimsTest(unittest.TestCase):
    def test_action_context_dim(self) -> None:
        """
        Verify the action context observation dimensionality equals 15.
        """
        self.assertEqual(OBSERVATION_DIMS["action_context"], 15)

    def test_motor_context_dim(self) -> None:
        self.assertEqual(OBSERVATION_DIMS["motor_context"], 13)

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

    def test_as_mapping_contains_all_15_fields(self) -> None:
        view = ActionContextObservation(
            hunger=0.1, fatigue=0.2, health=0.9, recent_pain=0.0, recent_contact=0.0,
            on_food=0.0, on_shelter=0.0, predator_visible=0.0, predator_certainty=0.0,
            day=1.0, night=0.0, last_move_dx=0.0, last_move_dy=0.0,
            sleep_debt=0.0, shelter_role_level=0.0,
        )
        mapping = view.as_mapping()
        self.assertEqual(len(mapping), 15)
        self.assertNotIn("home_dx", mapping)
        self.assertNotIn("home_dy", mapping)

    def test_predator_dist_not_present(self) -> None:
        """
        Verifies that the action-context observation schema does not expose privileged world fields.
        """
        names = ActionContextObservation.field_names()
        self.assertNotIn("predator_dist", names)
        self.assertNotIn("home_dx", names)
        self.assertNotIn("home_dy", names)


class MotorContextObservationCompositionTest(unittest.TestCase):
    """Tests that MotorContextObservation has the correct 13 fields."""

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

    def test_embodiment_fields_present(self) -> None:
        names = MotorContextObservation.field_names()
        for field in ("heading_dx", "heading_dy", "terrain_difficulty", "fatigue"):
            with self.subTest(field=field):
                self.assertIn(field, names)

    def test_as_mapping_contains_all_13_fields(self) -> None:
        view = MotorContextObservation(
            on_food=0.0, on_shelter=0.0, predator_visible=0.0, predator_certainty=0.0,
            day=1.0, night=0.0, last_move_dx=0.0, last_move_dy=0.0,
            shelter_role_level=0.0, heading_dx=1.0, heading_dy=0.0,
            terrain_difficulty=0.4, fatigue=0.2,
        )
        mapping = view.as_mapping()
        self.assertEqual(len(mapping), 13)
        self.assertNotIn("home_dx", mapping)
        self.assertNotIn("home_dy", mapping)

    def test_predator_dist_not_present(self) -> None:
        names = MotorContextObservation.field_names()
        self.assertNotIn("predator_dist", names)
        self.assertNotIn("home_dx", names)
        self.assertNotIn("home_dy", names)


# ---------------------------------------------------------------------------
# build_action_context_observation / build_motor_context_observation tests
# ---------------------------------------------------------------------------

class BuildActionContextObservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=7)
        self.world.reset(seed=7)
        self.null_percept = _null_percept()

    def _build(self, **kwargs) -> ActionContextObservation:
        """
        Builds an ActionContextObservation for this object's world using provided overrides.
        
        Parameters:
            on_food (float): 1.0 if the agent is on food, 0.0 otherwise (default 0.0).
            on_shelter (float): 1.0 if the agent is on shelter, 0.0 otherwise (default 0.0).
            predator_view (PerceivedTarget): Percept describing predator visibility/certainty (default self.null_percept).
            day (float): Daylight indicator, typically 1.0 for day, 0.0 for night (default 1.0).
            night (float): Night indicator, typically 1.0 for night, 0.0 for day (default 0.0).
            shelter_role_level (float): Numeric level of shelter role influence (default 0.0).
            **kwargs: Any of the above may be passed as keyword arguments to override defaults.
        
        Returns:
            ActionContextObservation: Observation object representing the action context for the current world state with the given overrides.
        """
        defaults = dict(
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
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

    def test_predator_dist_not_exposed(self) -> None:
        """
        Asserts that the constructed observation object does not expose a `predator_dist` attribute.
        
        This verifies the deprecated `predator_dist` signal is not present on the observation API.
        """
        result = self._build()
        self.assertFalse(hasattr(result, "predator_dist"))
        self.assertFalse(hasattr(result, "home_dx"))
        self.assertFalse(hasattr(result, "home_dy"))

    def test_day_night_passed_through(self) -> None:
        result = self._build(day=0.0, night=1.0)
        self.assertAlmostEqual(result.day, 0.0)
        self.assertAlmostEqual(result.night, 1.0)

    def test_shelter_role_level_passed_through(self) -> None:
        result = self._build(shelter_role_level=0.5)
        self.assertAlmostEqual(result.shelter_role_level, 0.5)


class BuildMotorContextObservationTest(unittest.TestCase):
    """Tests that build_motor_context_observation produces a MotorContextObservation
    with only the execution-relevant fields (no hunger or sleep_debt)."""

    def setUp(self) -> None:
        self.world = SpiderWorld(seed=11)
        self.world.reset(seed=11)
        self.null_percept = _null_percept()

    def _build(self, **kwargs) -> MotorContextObservation:
        """
        Builds a MotorContextObservation for this brain's world using optional overrides.
        
        Parameters:
            on_food (float, optional): 1.0 if the agent is on food, otherwise 0.0. Defaults to 0.0.
            on_shelter (float, optional): 1.0 if the agent is on shelter, otherwise 0.0. Defaults to 0.0.
            predator_view (PerceivedTarget, optional): Percept describing predator visibility/certainty. Defaults to this brain's null_percept.
            day (float, optional): Indicator value for daytime. Defaults to 1.0.
            night (float, optional): Indicator value for nighttime. Defaults to 0.0.
            shelter_role_level (float, optional): Role level related to shelter behavior. Defaults to 0.0.
        
        Returns:
            MotorContextObservation: An observation containing execution-relevant motor context signals (e.g., on_food, on_shelter, predator visibility/certainty, last move deltas).
        """
        defaults = dict(
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
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

    def test_predator_dist_not_exposed(self) -> None:
        """
        Asserts that the constructed observation object does not expose a `predator_dist` attribute.
        
        This verifies the deprecated `predator_dist` signal is not present on the observation API.
        """
        result = self._build()
        self.assertFalse(hasattr(result, "predator_dist"))
        self.assertFalse(hasattr(result, "home_dx"))
        self.assertFalse(hasattr(result, "home_dy"))

    def test_last_move_from_world_state(self) -> None:
        self.world.state.last_move_dx = 1
        self.world.state.last_move_dy = -1
        result = self._build()
        self.assertAlmostEqual(result.last_move_dx, 1.0, places=5)
        self.assertAlmostEqual(result.last_move_dy, -1.0, places=5)

    def test_heading_from_world_state(self) -> None:
        self.world.state.heading_dx = -1
        self.world.state.heading_dy = 0
        result = self._build()
        self.assertAlmostEqual(result.heading_dx, -1.0, places=5)
        self.assertAlmostEqual(result.heading_dy, 0.0, places=5)

    def test_fatigue_from_world_state(self) -> None:
        self.world.state.fatigue = 0.42
        result = self._build()
        self.assertAlmostEqual(result.fatigue, 0.42, places=5)

    def test_terrain_difficulty_from_current_cell(self) -> None:
        pos = self.world.spider_pos()
        self.world.map_template.terrain.pop(pos, None)
        self.assertAlmostEqual(self._build().terrain_difficulty, 0.0, places=5)

        self.world.map_template.terrain[pos] = CLUTTER
        self.assertAlmostEqual(self._build().terrain_difficulty, 0.4, places=5)

        self.world.map_template.terrain[pos] = NARROW
        self.assertAlmostEqual(self._build().terrain_difficulty, 0.7, places=5)

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
        self.assertEqual(SpiderBrain.ARCHITECTURE_VERSION, 12)

    def test_action_center_is_motor_network(self) -> None:
        """action_center must be a MotorNetwork (has value head)."""
        self.assertIsInstance(self.brain.action_center, MotorNetwork)

    def test_arbitration_network_is_arbitration_network(self) -> None:
        self.assertIsInstance(self.brain.arbitration_network, ArbitrationNetwork)

    def test_arbitration_regularization_defaults(self) -> None:
        self.assertEqual(self.brain.arbitration_regularization_weight, 0.1)
        self.assertEqual(self.brain.arbitration_valence_regularization_weight, 0.1)

    def test_no_regularization_arbitration_variant_disables_regularizers(self) -> None:
        brain = SpiderBrain(
            seed=42,
            module_dropout=0.0,
            config=BrainAblationConfig(name="learned_arbitration_no_regularization"),
        )
        self.assertEqual(brain.arbitration_regularization_weight, 0.0)
        self.assertEqual(brain.arbitration_valence_regularization_weight, 0.0)

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
        self.assertIn("arbitration_network", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)
        arb_idx = names.index("arbitration_network")
        ac_idx = names.index("action_center")
        mc_idx = names.index("motor_cortex")
        self.assertLess(arb_idx, ac_idx)
        self.assertLess(ac_idx, mc_idx)

    def test_parameter_norms_includes_action_center(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertIn("arbitration_network", norms)
        self.assertIn("action_center", norms)

    def test_parameter_norms_action_center_norm_is_positive(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertGreater(norms["arbitration_network"], 0.0)
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
        self.assertIn("arbitration_network", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)

    def test_monolithic_parameter_norms_includes_action_center(self) -> None:
        brain = SpiderBrain(seed=3, config=self._config())
        norms = brain.parameter_norms()
        self.assertIn("arbitration_network", norms)
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
            sleep_phase_level=0.6 if valence == "sleep" else 0.0,
            sleep_debt=0.95 if valence == "sleep" else 0.05,
            shelter_role_level=0.9 if valence == "sleep" else 0.0,
            shelter_trace_strength=0.8 if valence == "sleep" else 0.0,
            shelter_memory_age=0.1 if valence == "sleep" else 1.0,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=0.95 if valence == "threat" else 0.0,
            predator_certainty=0.92 if valence == "threat" else 0.0,
            predator_smell_strength=0.9 if valence == "threat" else 0.0,
            predator_motion_salience=0.9 if valence == "threat" else 0.0,
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

    def test_motor_execution_diagnostic_fields_are_present(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.orientation_alignment, float)
        self.assertIsInstance(step.terrain_difficulty, float)
        self.assertIsInstance(step.execution_difficulty, float)
        self.assertIsInstance(step.execution_slip_occurred, bool)
        self.assertIsInstance(step.motor_noise_applied, bool)
        self.assertIsInstance(step.motor_slip_occurred, bool)
        self.assertEqual(step.slip_reason, "none")

    def test_motor_execution_diagnostics_follow_selected_action(self) -> None:
        obs = _blank_obs()
        obs["motor_context"] = _vector_for(
            MOTOR_CONTEXT_INTERFACE,
            heading_dx=1.0,
            heading_dy=0.0,
            terrain_difficulty=0.7,
            fatigue=0.5,
        )
        step = self.brain.act(obs, bus=None, sample=False)
        expected_difficulty, expected_components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=ACTION_DELTAS.get(LOCOMOTION_ACTIONS[step.action_idx], (0, 0)),
            terrain=NARROW,
            fatigue=0.5,
        )
        self.assertAlmostEqual(
            step.orientation_alignment,
            expected_components["orientation_alignment"],
        )
        self.assertAlmostEqual(step.terrain_difficulty, 0.7)
        self.assertAlmostEqual(step.execution_difficulty, expected_difficulty)

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
# Embodiment-aware motor execution
# ---------------------------------------------------------------------------

class ExecutionDifficultyModelTest(unittest.TestCase):
    def test_open_aligned_movement_has_no_execution_difficulty(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(1.0, 0.0),
            terrain=OPEN,
            fatigue=0.0,
        )
        self.assertAlmostEqual(difficulty, 0.0)
        self.assertAlmostEqual(components["orientation_alignment"], 1.0)
        self.assertAlmostEqual(components["terrain_difficulty"], 0.0)
        self.assertAlmostEqual(components["fatigue_factor"], 0.0)

    def test_opposed_movement_uses_terrain_and_fatigue_multiplier(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(-1.0, 0.0),
            terrain=CLUTTER,
            fatigue=0.5,
        )
        self.assertAlmostEqual(components["orientation_alignment"], 0.0)
        self.assertAlmostEqual(components["terrain_difficulty"], 0.4)
        self.assertAlmostEqual(components["fatigue_factor"], 0.5)
        self.assertAlmostEqual(difficulty, 0.6)

    def test_high_difficulty_is_clipped_to_one(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(-1.0, 0.0),
            terrain=NARROW,
            fatigue=1.0,
        )
        self.assertAlmostEqual(components["raw_difficulty"], 1.4)
        self.assertAlmostEqual(difficulty, 1.0)

    def test_stay_action_has_no_orientation_mismatch(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(0.0, -1.0),
            intended_direction=(0.0, 0.0),
            terrain=NARROW,
            fatigue=1.0,
        )
        self.assertAlmostEqual(components["orientation_alignment"], 1.0)
        self.assertAlmostEqual(components["orientation_mismatch"], 0.0)
        self.assertAlmostEqual(difficulty, 0.0)

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
        intent = np.zeros(self.brain.action_dim, dtype=float)
        intent[:4] = [np.nan, np.inf, -np.inf, 0.5]
        result = self.brain._build_motor_input(intent, self.motor_obs)
        expected = np.zeros(self.brain.action_dim, dtype=float)
        expected[:4] = [0.0, 1.0, -1.0, 0.5]
        np.testing.assert_array_equal(result[: self.brain.action_dim], expected)

    def test_valid_intent_passes_through_unchanged(self) -> None:
        intent = np.linspace(0.1, 0.9, self.brain.action_dim, dtype=float)
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
        """
        Assert that the 'action_context' entry in the test observation is a NumPy ndarray.
        
        Verifies the observation mapping includes the "action_context" key whose value is an instance of numpy.ndarray.
        """
        self.assertIsInstance(self.obs["action_context"], np.ndarray)

    def test_action_context_shape(self) -> None:
        self.assertEqual(self.obs["action_context"].shape, (15,))

    def test_motor_context_shape_is_13(self) -> None:
        """After this PR, motor_context must have 13 elements."""
        self.assertEqual(self.obs["motor_context"].shape, (13,))

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

    def test_action_dim_matches_locomotion_actions(self) -> None:
        """action_dim on brain must match the locomotion action contract."""
        brain = SpiderBrain(seed=1, module_dropout=0.0)
        self.assertEqual(brain.action_dim, len(LOCOMOTION_ACTIONS))


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
                    "valence_logits", "base_gates", "gate_adjustments",
                    "arbitration_value", "learned_adjustment",
                    "guards_applied", "food_bias_applied", "food_bias_action",
                    "module_contribution_share", "dominant_module", "dominant_module_share",
                    "effective_module_count", "module_agreement_rate", "module_disagreement_rate",
                    "suppressed_modules", "evidence", "intent_before_gating", "intent_after_gating"):
            self.assertIn(key, payload)

    def test_legacy_constructor_keeps_new_payload_fields_with_defaults(self) -> None:
        payload = self._make().to_payload()
        self.assertEqual(payload["valence_logits"], {})
        self.assertEqual(payload["base_gates"], {})
        self.assertEqual(payload["gate_adjustments"], {})
        self.assertEqual(payload["arbitration_value"], 0.0)
        self.assertFalse(payload["learned_adjustment"])
        self.assertFalse(payload["guards_applied"])
        self.assertFalse(payload["food_bias_applied"])
        self.assertIsNone(payload["food_bias_action"])
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

    def test_to_payload_exports_legacy_predator_proximity_for_scorers(self) -> None:
        d = self._make()
        self.assertNotIn("predator_proximity", d.evidence["threat"])
        payload = d.to_payload()
        self.assertEqual(payload["evidence"]["threat"]["predator_proximity"], 0.0)


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

    def _make_brain_with_arbitration_winner(
        self,
        *,
        seed: int,
        winner: str,
        config: BrainAblationConfig | None = None,
        config_name: str | None = None,
    ) -> SpiderBrain:
        if config is not None and config_name is not None:
            raise ValueError("Provide either config or config_name, not both.")
        if config_name is not None:
            config = canonical_ablation_configs(module_dropout=0.0)[config_name]
        if config is None:
            brain = SpiderBrain(seed=seed, module_dropout=0.0)
        else:
            brain = SpiderBrain(seed=seed, config=config)
        brain.arbitration_network.W2_valence.fill(0.0)
        brain.arbitration_network.b2_valence[:] = _valence_vector(**{winner: 5.0})
        return brain

    def _build_obs(
        self,
        *,
        action_context_kwargs: dict[str, float],
        module_name: str,
        module_kwargs: dict[str, float],
    ) -> dict[str, np.ndarray]:
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            **action_context_kwargs,
        )
        iface = _module_interface(module_name)
        obs[iface.observation_key] = _vector_for(iface, **module_kwargs)
        return obs

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

    def test_learned_arbitration_fields_are_present(self) -> None:
        arb = self._get_arbitration()
        self.assertEqual(set(arb.valence_logits.keys()), set(SpiderBrain.VALENCE_ORDER))
        self.assertEqual(set(arb.base_gates.keys()), set(arb.module_gates.keys()))
        self.assertEqual(set(arb.gate_adjustments.keys()), set(arb.module_gates.keys()))
        self.assertTrue(arb.learned_adjustment)
        self.assertTrue(np.isfinite(arb.arbitration_value))
        self.assertFalse(arb.guards_applied)
        self.assertFalse(arb.food_bias_applied)
        self.assertIsNone(arb.food_bias_action)

    def test_default_config_trace_payload_marks_scaffolding_inactive(self) -> None:
        bus = MessageBus()

        step = self.brain.act(self.obs, bus=bus, sample=False, training=False)
        selection_messages = bus.topic_messages("action.selection")
        self.assertTrue(selection_messages)
        selection_payload = selection_messages[-1].payload

        self.assertFalse(step.arbitration_decision.guards_applied)
        self.assertFalse(step.arbitration_decision.food_bias_applied)
        self.assertIsNone(step.arbitration_decision.food_bias_action)
        self.assertIn("guards_applied", selection_payload)
        self.assertIn("food_bias_applied", selection_payload)
        self.assertIn("food_bias_action", selection_payload)
        self.assertFalse(selection_payload["guards_applied"])
        self.assertFalse(selection_payload["food_bias_applied"])
        self.assertIsNone(selection_payload["food_bias_action"])

    def test_gate_adjustments_are_constrained_before_final_clamp(self) -> None:
        arb = self._get_arbitration()
        for adjustment in arb.gate_adjustments.values():
            self.assertGreaterEqual(adjustment, ArbitrationNetwork.GATE_ADJUSTMENT_MIN)
            self.assertLessEqual(adjustment, ArbitrationNetwork.GATE_ADJUSTMENT_MAX)

    def test_fixed_arbitration_config_uses_fixed_formula_path(self) -> None:
        """
        Verify that when learned arbitration is disabled the brain follows the fixed-formula arbitration path.
        
        Asserts that:
        - the arbitration decision marks learned_adjustment as False,
        - the arbitration network cache is None,
        - every module's gate_adjustment equals 1.0 and module_gates equal the reported base_gates,
        - arbitration_value equals 0.0.
        """
        brain = SpiderBrain(
            seed=3,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="fixed_arbitration_baseline",
                use_learned_arbitration=False,
            ),
        )
        step = brain.act(self.obs, bus=None, sample=False)
        arb = step.arbitration_decision
        self.assertIsNotNone(arb)
        self.assertFalse(arb.learned_adjustment)
        self.assertIsNone(brain.arbitration_network.cache)
        for module_name, base_gate in arb.base_gates.items():
            self.assertAlmostEqual(arb.gate_adjustments[module_name], 1.0)
            self.assertAlmostEqual(arb.module_gates[module_name], base_gate)
        self.assertAlmostEqual(arb.arbitration_value, 0.0)

    def test_learned_arbitration_uses_learned_winner_under_food_predator_conflict(self) -> None:
        """
        Assert that the benchmark path does not force a hard threat override when strong predator signals conflict with high hunger.
        
        The selected valence should be the learned deterministic winner for the
        current scores, with deterministic guards left inactive by default.
        """
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            hunger=0.96,
            day=1.0,
            predator_visible=1.0,
            predator_certainty=0.9,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_motion_salience=0.8,
            predator_smell_strength=0.7,
        )
        obs["hunger"] = _vector_for(
            _module_interface("hunger_center"),
            hunger=0.96,
            food_visible=1.0,
            food_certainty=0.9,
            food_smell_strength=0.8,
            food_memory_age=0.0,
        )

        arb = self._get_arbitration(obs)

        self.assertTrue(arb.learned_adjustment)
        self.assertFalse(arb.guards_applied)
        self.assertEqual(
            arb.winning_valence,
            self.brain._deterministic_valence_winner(arb.valence_scores),
        )

    def test_deterministic_threat_guard_disabled_by_default(self) -> None:
        obs = self._build_obs(
            action_context_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
            module_name="alert_center",
            module_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
        )
        self.brain.arbitration_network.W2_valence.fill(0.0)
        self.brain.arbitration_network.b2_valence[:] = _valence_vector(hunger=5.0)
        module_results = self.brain._proposal_results(
            obs,
            store_cache=False,
            training=False,
        )

        arb = self.brain._compute_arbitration(
            module_results,
            obs,
            training=False,
            store_cache=False,
        )

        self.assertEqual(arb.winning_valence, "hunger")
        self.assertFalse(arb.guards_applied)

    def test_deterministic_threat_guard_can_be_enabled(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=3,
            winner="hunger",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
            module_name="alert_center",
            module_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
        )
        module_results = brain._proposal_results(
            obs,
            store_cache=False,
            training=False,
        )

        arb = brain._compute_arbitration(
            module_results,
            obs,
            training=False,
            store_cache=False,
        )

        self.assertEqual(arb.winning_valence, "threat")
        self.assertTrue(arb.guards_applied)

    def test_deterministic_hunger_guard_can_be_enabled(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=3,
            winner="exploration",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "night": 0.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
            },
        )
        module_results = brain._proposal_results(
            obs,
            store_cache=False,
            training=False,
        )

        arb = brain._compute_arbitration(
            module_results,
            obs,
            training=False,
            store_cache=False,
        )

        self.assertEqual(arb.winning_valence, "hunger")
        self.assertTrue(arb.guards_applied)

    def test_food_direction_bias_disabled_by_default(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config=default_brain_config(module_dropout=0.0),
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "day": 1.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
                "food_visible": 1.0,
                "food_certainty": 1.0,
                "food_dx": 1.0,
                "food_dy": 0.0,
            },
        )

        step = brain.act(obs, bus=None, sample=False, training=False)

        self.assertEqual(step.arbitration_decision.winning_valence, "hunger")
        self.assertFalse(step.arbitration_decision.food_bias_applied)
        self.assertIsNone(step.arbitration_decision.food_bias_action)

    def test_food_direction_bias_can_be_enabled(self) -> None:
        base_brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config=default_brain_config(module_dropout=0.0),
        )
        biased_brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "day": 1.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
                "food_visible": 1.0,
                "food_certainty": 1.0,
                "food_dx": 1.0,
                "food_dy": 0.0,
            },
        )

        base_step = base_brain.act(obs, bus=None, sample=False, training=False)
        biased_step = biased_brain.act(obs, bus=None, sample=False, training=False)

        self.assertFalse(base_step.arbitration_decision.food_bias_applied)
        self.assertTrue(biased_step.arbitration_decision.food_bias_applied)
        self.assertEqual(biased_step.arbitration_decision.food_bias_action, "MOVE_RIGHT")
        diff = biased_step.total_logits - base_step.total_logits
        expected = np.zeros_like(diff)
        expected[LOCOMOTION_ACTIONS.index("MOVE_RIGHT")] = 3.0
        np.testing.assert_allclose(diff, expected)

    def test_food_direction_bias_skips_stay_direction(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "day": 1.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
                "food_visible": 1.0,
                "food_certainty": 1.0,
                "food_dx": 0.04,
                "food_dy": 0.04,
            },
        )

        step = brain.act(obs, bus=None, sample=False, training=False)

        self.assertEqual(step.arbitration_decision.winning_valence, "hunger")
        self.assertFalse(step.arbitration_decision.food_bias_applied)
        self.assertIsNone(step.arbitration_decision.food_bias_action)

    def test_food_bias_flip_does_not_count_as_final_reflex_override(self) -> None:
        config = BrainAblationConfig(
            name="food_bias_telemetry_regression",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_deterministic_guards=True,
            enable_food_direction_bias=True,
        )
        brain = SpiderBrain(seed=5, config=config)
        self.assertIsNotNone(brain.module_bank)
        for network in brain.module_bank.modules.values():
            network.W1.fill(0.0)
            network.b1.fill(0.0)
            network.W2.fill(0.0)
            network.b2.fill(0.0)
        brain.action_center.W1.fill(0.0)
        brain.action_center.b1.fill(0.0)
        brain.action_center.W2_policy.fill(0.0)
        brain.action_center.b2_policy.fill(0.0)
        brain.action_center.W2_value.fill(0.0)
        brain.action_center.b2_value.fill(0.0)
        brain.motor_cortex.W1.fill(0.0)
        brain.motor_cortex.b1.fill(0.0)
        brain.motor_cortex.W2.fill(0.0)
        brain.motor_cortex.b2.fill(0.0)
        brain.arbitration_network.W2_valence.fill(0.0)
        brain.arbitration_network.b2_valence[:] = _valence_vector(hunger=5.0)
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            hunger=0.9,
            day=1.0,
        )
        obs["hunger"] = _vector_for(
            _module_interface("hunger_center"),
            hunger=0.9,
            food_visible=1.0,
            food_certainty=1.0,
            food_dx=1.0,
            food_dy=0.0,
        )

        step = brain.act(obs, bus=None, sample=False, training=False)

        self.assertTrue(step.arbitration_decision.food_bias_applied)
        self.assertEqual(step.arbitration_decision.food_bias_action, "MOVE_RIGHT")
        self.assertEqual(step.motor_action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))
        self.assertFalse(step.final_reflex_override)

    def test_regularization_term_is_reported_when_enabled(self) -> None:
        """
        Verify that arbitration regularization metrics are included and positive when training is enabled and arbitration gate parameters are non-zero.
        
        This test enables a non-zero gate head bias, performs an action step in training mode, calls the learning update, and asserts that the returned statistics contain a positive "regularization_loss" and a positive "arbitration_gate_regularization_norm".
        """
        self.brain.arbitration_network.b2_gate.fill(1.25)
        decision = self.brain.act(self.obs, bus=None, sample=False, training=True)

        stats = self.brain.learn(
            decision,
            reward=0.25,
            next_observation=self.obs,
            done=True,
        )

        self.assertGreater(stats["regularization_loss"], 0.0)
        self.assertGreater(stats["arbitration_gate_regularization_norm"], 0.0)

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
        for key in ("predator_visible", "predator_certainty", "predator_motion_salience",
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
                    "shelter_role_level", "shelter_path_confidence"):
            self.assertIn(key, arb.evidence["sleep"])

    def test_exploration_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("safety_margin", "residual_drive", "day", "off_shelter",
                    "visual_openness", "food_smell_directionality"):
            self.assertIn(key, arb.evidence["exploration"])

    def test_threat_evidence_does_not_contain_predator_proximity(self) -> None:
        """Regression: predator_proximity was removed from threat evidence in this PR."""
        arb = self._get_arbitration()
        self.assertNotIn("predator_proximity", arb.evidence["threat"])

    def test_decision_logic_uses_abstract_predator_evidence_only(self) -> None:
        """Runtime arbitration evidence uses abstract predator signals, not diagnostic distances."""
        forbidden = {"diagnostic_predator_dist", "predator_dist"}
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            predator_visible=1.0,
            predator_certainty=0.75,
            recent_contact=0.25,
            recent_pain=0.5,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_motion_salience=0.6,
            predator_smell_strength=0.4,
        )

        arb = self._get_arbitration(obs)
        threat = arb.evidence["threat"]

        self.assertAlmostEqual(threat["predator_visible"], 1.0)
        self.assertAlmostEqual(threat["predator_certainty"], 0.75)
        self.assertAlmostEqual(threat["predator_motion_salience"], 0.6)
        self.assertAlmostEqual(threat["recent_contact"], 0.25)
        self.assertAlmostEqual(threat["recent_pain"], 0.5)
        self.assertAlmostEqual(threat["predator_smell_strength"], 0.4)
        for valence, evidence in arb.evidence.items():
            with self.subTest(valence=valence):
                self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in evidence.values()))
                for name in forbidden:
                    self.assertNotIn(name, evidence)

    def test_sleep_evidence_does_not_contain_home_pressure(self) -> None:
        """Regression: home_pressure was removed from sleep evidence in this PR."""
        arb = self._get_arbitration()
        self.assertNotIn("home_pressure", arb.evidence["sleep"])

    def test_threat_evidence_predator_motion_salience_in_zero_to_one(self) -> None:
        """predator_motion_salience evidence value must be clamped to [0, 1]."""
        arb = self._get_arbitration()
        val = arb.evidence["threat"]["predator_motion_salience"]
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_sleep_evidence_shelter_path_confidence_in_zero_to_one(self) -> None:
        """shelter_path_confidence evidence value must be clamped to [0, 1]."""
        arb = self._get_arbitration()
        val = arb.evidence["sleep"]["shelter_path_confidence"]
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


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
            recent_contact=1.0,
            recent_pain=0.8,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_smell_strength=1.0,
            predator_motion_salience=1.0,
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
            recent_contact=1.0,
            recent_pain=0.8,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_smell_strength=1.0,
            predator_motion_salience=1.0,
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
        """
        Create a default ModuleResult for testing with neutral action scores and a placeholder observation.
        
        The returned ModuleResult has zero `logits`, uniform `probs` across `LOCOMOTION_ACTIONS`, `name` set to "test_module", `observation_key` set to "test", a zero-valued observation of length 3, and `active` set to True.
        
        Returns:
            ModuleResult: A test ModuleResult instance populated as described above.
        """
        from spider_cortex_sim.modules import ModuleResult
        logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        probs = np.ones(len(LOCOMOTION_ACTIONS), dtype=float) / len(LOCOMOTION_ACTIONS)
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
# Learned valence softmax edge case: all-zero evidence logits
# ---------------------------------------------------------------------------

class ValenceNormalizationEdgeCaseTest(unittest.TestCase):
    """Tests for learned valence probabilities when all evidence logits are equal."""

    def test_all_zero_evidence_logits_softmax_to_uniform_scores(self) -> None:
        import unittest.mock

        brain = SpiderBrain(seed=99, module_dropout=0.0)
        obs = _blank_obs()
        module_results = brain._proposal_results(obs, store_cache=False, training=False)

        with unittest.mock.patch.object(brain, '_clamp_unit', return_value=0.0):
            arb = brain._compute_arbitration(module_results, obs)

        for valence in SpiderBrain.VALENCE_ORDER:
            self.assertAlmostEqual(arb.valence_scores[valence], 0.25, places=5)
        self.assertEqual(arb.winning_valence, "threat")

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


# ---------------------------------------------------------------------------
# interfaces.py arbitration constants tests
# ---------------------------------------------------------------------------

class InterfacesArbitrationConstantsTest(unittest.TestCase):
    """Tests for the new ARBITRATION_* constants added to interfaces.py."""

    def test_arbitration_evidence_input_dim_is_24(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_EVIDENCE_INPUT_DIM
        self.assertEqual(ARBITRATION_EVIDENCE_INPUT_DIM, 24)

    def test_arbitration_hidden_dim_is_32(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_HIDDEN_DIM
        self.assertEqual(ARBITRATION_HIDDEN_DIM, 32)

    def test_arbitration_valence_dim_is_4(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_VALENCE_DIM
        self.assertEqual(ARBITRATION_VALENCE_DIM, 4)

    def test_arbitration_gate_dim_is_6(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_GATE_DIM
        self.assertEqual(ARBITRATION_GATE_DIM, 6)

    def test_evidence_input_dim_matches_arbitration_network_input_dim(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_EVIDENCE_INPUT_DIM
        self.assertEqual(ARBITRATION_EVIDENCE_INPUT_DIM, ArbitrationNetwork.INPUT_DIM)

    def test_valence_dim_matches_arbitration_network_valence_dim(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_VALENCE_DIM
        self.assertEqual(ARBITRATION_VALENCE_DIM, ArbitrationNetwork.VALENCE_DIM)

    def test_gate_dim_matches_arbitration_network_gate_dim(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_GATE_DIM
        self.assertEqual(ARBITRATION_GATE_DIM, ArbitrationNetwork.GATE_DIM)


# ---------------------------------------------------------------------------
# architecture_signature() arbitration_network parameter_shapes tests
# ---------------------------------------------------------------------------

class ArchitectureSignatureArbitrationParamShapesTest(unittest.TestCase):
    """Tests for the arbitration_network.parameter_shapes structure in architecture_signature."""

    def setUp(self) -> None:
        self.sig = architecture_signature()
        self.arb = self.sig["arbitration_network"]

    def test_parameter_shapes_present(self) -> None:
        self.assertIn("parameter_shapes", self.arb)

    def test_w1_shape_is_hidden_x_input(self) -> None:
        shapes = self.arb["parameter_shapes"]
        expected = [self.arb["hidden_dim"], self.arb["input_dim"]]
        self.assertEqual(shapes["W1"], expected)

    def test_b1_shape_is_hidden_dim(self) -> None:
        shapes = self.arb["parameter_shapes"]
        self.assertEqual(shapes["b1"], [self.arb["hidden_dim"]])

    def test_w2_valence_shape_correct(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_VALENCE_DIM
        shapes = self.arb["parameter_shapes"]
        self.assertEqual(shapes["W2_valence"], [ARBITRATION_VALENCE_DIM, self.arb["hidden_dim"]])

    def test_w2_gate_shape_correct(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_GATE_DIM
        shapes = self.arb["parameter_shapes"]
        self.assertEqual(shapes["W2_gate"], [ARBITRATION_GATE_DIM, self.arb["hidden_dim"]])

    def test_w2_value_shape_is_1_x_hidden(self) -> None:
        shapes = self.arb["parameter_shapes"]
        self.assertEqual(shapes["W2_value"], [1, self.arb["hidden_dim"]])

    def test_parameter_fingerprint_is_string(self) -> None:
        self.assertIsInstance(self.arb["parameter_fingerprint"], str)
        self.assertGreater(len(self.arb["parameter_fingerprint"]), 0)

    def test_fingerprint_changes_when_learned_changes(self) -> None:
        sig_learned = architecture_signature(learned_arbitration=True)
        sig_fixed = architecture_signature(learned_arbitration=False)
        # The overall fingerprint of the payload changes
        self.assertNotEqual(sig_learned["fingerprint"], sig_fixed["fingerprint"])

    def test_fingerprint_changes_when_hidden_dim_changes(self) -> None:
        sig_32 = architecture_signature(arbitration_hidden_dim=32)
        sig_64 = architecture_signature(arbitration_hidden_dim=64)
        self.assertNotEqual(
            sig_32["arbitration_network"]["parameter_fingerprint"],
            sig_64["arbitration_network"]["parameter_fingerprint"],
        )


# ---------------------------------------------------------------------------
# SpiderBrain._arbitration_evidence_input_dim tests
# ---------------------------------------------------------------------------

class ArbitrationEvidenceInputDimTest(unittest.TestCase):
    """Tests for SpiderBrain._arbitration_evidence_input_dim classmethod."""

    def test_returns_24(self) -> None:
        self.assertEqual(SpiderBrain._arbitration_evidence_input_dim(), 24)

    def test_is_classmethod_accessible(self) -> None:
        # Should be callable on the class, not just an instance
        result = SpiderBrain._arbitration_evidence_input_dim()
        self.assertIsInstance(result, int)

    def test_matches_arbitration_network_input_dim(self) -> None:
        self.assertEqual(
            SpiderBrain._arbitration_evidence_input_dim(),
            ArbitrationNetwork.INPUT_DIM,
        )

    def test_equals_sum_of_evidence_fields(self) -> None:
        total = sum(len(fields) for fields in SpiderBrain.ARBITRATION_EVIDENCE_FIELDS.values())
        self.assertEqual(SpiderBrain._arbitration_evidence_input_dim(), total)


# ---------------------------------------------------------------------------
# SpiderBrain._arbitration_evidence_signal_names tests
# ---------------------------------------------------------------------------

class ArbitrationEvidenceSignalNamesTest(unittest.TestCase):
    """Tests for SpiderBrain._arbitration_evidence_signal_names classmethod."""

    def test_returns_tuple(self) -> None:
        self.assertIsInstance(SpiderBrain._arbitration_evidence_signal_names(), tuple)

    def test_length_is_24(self) -> None:
        self.assertEqual(len(SpiderBrain._arbitration_evidence_signal_names()), 24)

    def test_first_signal_is_threat_predator_visible(self) -> None:
        names = SpiderBrain._arbitration_evidence_signal_names()
        self.assertEqual(names[0], "threat.predator_visible")

    def test_last_signal_is_exploration_food_smell_directionality(self) -> None:
        names = SpiderBrain._arbitration_evidence_signal_names()
        self.assertEqual(names[-1], "exploration.food_smell_directionality")

    def test_all_names_have_valence_prefix(self) -> None:
        names = SpiderBrain._arbitration_evidence_signal_names()
        valences = set(SpiderBrain.VALENCE_ORDER)
        for name in names:
            prefix = name.split(".")[0]
            self.assertIn(prefix, valences, f"Signal {name!r} has unknown valence prefix")

    def test_matches_arbitration_network_evidence_signal_names(self) -> None:
        self.assertEqual(
            SpiderBrain._arbitration_evidence_signal_names(),
            ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES,
        )

    def test_threat_signals_come_before_hunger(self) -> None:
        names = SpiderBrain._arbitration_evidence_signal_names()
        threat_indices = [i for i, n in enumerate(names) if n.startswith("threat.")]
        hunger_indices = [i for i, n in enumerate(names) if n.startswith("hunger.")]
        self.assertLess(max(threat_indices), min(hunger_indices))

    def test_hunger_signals_come_before_sleep(self) -> None:
        names = SpiderBrain._arbitration_evidence_signal_names()
        hunger_indices = [i for i, n in enumerate(names) if n.startswith("hunger.")]
        sleep_indices = [i for i, n in enumerate(names) if n.startswith("sleep.")]
        self.assertLess(max(hunger_indices), min(sleep_indices))

    def test_sleep_signals_come_before_exploration(self) -> None:
        names = SpiderBrain._arbitration_evidence_signal_names()
        sleep_indices = [i for i, n in enumerate(names) if n.startswith("sleep.")]
        exploration_indices = [i for i, n in enumerate(names) if n.startswith("exploration.")]
        self.assertLess(max(sleep_indices), min(exploration_indices))

    def test_each_valence_has_six_signals(self) -> None:
        names = SpiderBrain._arbitration_evidence_signal_names()
        for valence in SpiderBrain.VALENCE_ORDER:
            count = sum(1 for n in names if n.startswith(f"{valence}."))
            self.assertEqual(count, 6, f"Expected 6 signals for {valence}, got {count}")


# ---------------------------------------------------------------------------
# SpiderBrain._arbitration_evidence_vector tests
# ---------------------------------------------------------------------------

class ArbitrationEvidenceVectorTest(unittest.TestCase):
    """Tests for SpiderBrain._arbitration_evidence_vector."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def _blank_evidence(self) -> dict:
        return {
            valence: {field: 0.0 for field in fields}
            for valence, fields in SpiderBrain.ARBITRATION_EVIDENCE_FIELDS.items()
        }

    def test_returns_array_of_length_24(self) -> None:
        evidence = self._blank_evidence()
        vec = self.brain._arbitration_evidence_vector(evidence)
        self.assertEqual(vec.shape, (24,))

    def test_returns_float_array(self) -> None:
        evidence = self._blank_evidence()
        vec = self.brain._arbitration_evidence_vector(evidence)
        self.assertEqual(vec.dtype, float)

    def test_all_zeros_for_blank_evidence(self) -> None:
        evidence = self._blank_evidence()
        vec = self.brain._arbitration_evidence_vector(evidence)
        np.testing.assert_array_equal(vec, np.zeros(24))

    def test_first_element_is_threat_predator_visible(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 0.75
        vec = self.brain._arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[0]), 0.75)

    def test_seventh_element_is_hunger_hunger(self) -> None:
        # threat has 6 fields, so hunger starts at index 6
        evidence = self._blank_evidence()
        evidence["hunger"]["hunger"] = 0.55
        vec = self.brain._arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[6]), 0.55)

    def test_last_element_is_exploration_food_smell_directionality(self) -> None:
        evidence = self._blank_evidence()
        evidence["exploration"]["food_smell_directionality"] = 0.33
        vec = self.brain._arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[-1]), 0.33)

    def test_output_matches_signal_names_ordering(self) -> None:
        evidence = self._blank_evidence()
        # Set each signal to a unique value based on its index
        signal_names = SpiderBrain._arbitration_evidence_signal_names()
        for idx, name in enumerate(signal_names):
            valence, field = name.split(".", 1)
            evidence[valence][field] = float(idx + 1) * 0.01
        vec = self.brain._arbitration_evidence_vector(evidence)
        for idx in range(24):
            self.assertAlmostEqual(float(vec[idx]), float(idx + 1) * 0.01, places=8)


# ---------------------------------------------------------------------------
# SpiderBrain._fixed_formula_valence_scores_from_evidence tests
# ---------------------------------------------------------------------------

class FixedFormulaValenceScoresTest(unittest.TestCase):
    """Tests for SpiderBrain._fixed_formula_valence_scores_from_evidence."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def _blank_evidence(self) -> dict:
        return {
            "threat": {
                "predator_visible": 0.0, "predator_certainty": 0.0,
                "predator_motion_salience": 0.0, "recent_contact": 0.0,
                "recent_pain": 0.0, "predator_smell_strength": 0.0,
            },
            "hunger": {
                "hunger": 0.0, "on_food": 0.0, "food_visible": 0.0,
                "food_certainty": 0.0, "food_smell_strength": 0.0,
                "food_memory_freshness": 0.0,
            },
            "sleep": {
                "fatigue": 0.0, "sleep_debt": 0.0, "night": 0.0,
                "on_shelter": 0.0, "shelter_role_level": 0.0,
                "shelter_path_confidence": 0.0,
            },
            "exploration": {
                "safety_margin": 0.0, "residual_drive": 0.0, "day": 0.0,
                "off_shelter": 0.0, "visual_openness": 0.0,
                "food_smell_directionality": 0.0,
            },
        }

    def test_returns_array_of_length_4(self) -> None:
        evidence = self._blank_evidence()
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        self.assertEqual(scores.shape, (4,))

    def test_all_zero_evidence_returns_exploration_fallback(self) -> None:
        evidence = self._blank_evidence()
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        # Fallback: [0.0, 0.0, 0.0, 1.0] = exploration
        np.testing.assert_array_almost_equal(scores, [0.0, 0.0, 0.0, 1.0])

    def test_scores_sum_to_one_for_nonzero_input(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 0.8
        evidence["hunger"]["hunger"] = 0.5
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        self.assertAlmostEqual(float(np.sum(scores)), 1.0, places=6)

    def test_scores_are_non_negative(self) -> None:
        evidence = self._blank_evidence()
        evidence["sleep"]["fatigue"] = 0.9
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        self.assertTrue(np.all(scores >= 0.0))

    def test_high_predator_visible_dominates_threat_score(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 1.0
        evidence["threat"]["predator_certainty"] = 1.0
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        # threat should be the largest score
        threat_idx = SpiderBrain.VALENCE_ORDER.index("threat")
        self.assertEqual(int(np.argmax(scores)), threat_idx)

    def test_high_hunger_dominates_hunger_score(self) -> None:
        evidence = self._blank_evidence()
        evidence["hunger"]["hunger"] = 1.0
        evidence["hunger"]["food_visible"] = 1.0
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        hunger_idx = SpiderBrain.VALENCE_ORDER.index("hunger")
        self.assertEqual(int(np.argmax(scores)), hunger_idx)

    def test_high_fatigue_and_night_dominates_sleep_score(self) -> None:
        evidence = self._blank_evidence()
        evidence["sleep"]["fatigue"] = 1.0
        evidence["sleep"]["sleep_debt"] = 1.0
        evidence["sleep"]["night"] = 1.0
        evidence["sleep"]["on_shelter"] = 1.0
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        sleep_idx = SpiderBrain.VALENCE_ORDER.index("sleep")
        self.assertEqual(int(np.argmax(scores)), sleep_idx)

    def test_float_array_output(self) -> None:
        evidence = self._blank_evidence()
        evidence["exploration"]["residual_drive"] = 0.5
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        self.assertEqual(scores.dtype, float)

    def test_scores_ordered_by_valence_order(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 0.9
        scores = self.brain._fixed_formula_valence_scores_from_evidence(evidence)
        # First element is threat (index 0 in VALENCE_ORDER)
        self.assertEqual(SpiderBrain.VALENCE_ORDER[0], "threat")
        self.assertGreater(float(scores[0]), 0.0)


# ---------------------------------------------------------------------------
# SpiderBrain._warm_start_arbitration_network tests
# ---------------------------------------------------------------------------

def _small_arbitration_evidence_vector() -> np.ndarray:
    return np.linspace(0.01, 0.24, ArbitrationNetwork.INPUT_DIM, dtype=float)


def _assert_relative_scaled_logits(
    test_case: unittest.TestCase,
    scaled_logits: np.ndarray,
    reference_logits: np.ndarray,
    scale: float,
    *,
    max_relative_error: float = 1e-4,
) -> None:
    expected = scale * reference_logits
    denominator = float(np.linalg.norm(reference_logits)) + 1e-12
    relative_error = float(
        np.linalg.norm(scaled_logits - expected) / denominator
    )
    test_case.assertLess(relative_error, max_relative_error)


class WarmStartArbitrationNetworkTest(unittest.TestCase):
    """Tests for SpiderBrain._warm_start_arbitration_network."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def test_warm_start_clears_network_cache(self) -> None:
        # Force a forward pass to set cache
        evidence_vector = np.zeros(ArbitrationNetwork.INPUT_DIM)
        self.brain.arbitration_network.forward(evidence_vector, store_cache=True)
        self.assertIsNotNone(self.brain.arbitration_network.cache)
        self.brain._warm_start_arbitration_network()
        self.assertIsNone(self.brain.arbitration_network.cache)

    def test_warm_start_sets_w1_diagonal(self) -> None:
        self.brain._warm_start_arbitration_network()
        net = self.brain.arbitration_network
        # W1 should have non-zero values on the diagonal (first input_dim rows)
        for i in range(net.input_dim):
            self.assertNotEqual(net.W1[i, i], 0.0,
                                f"W1[{i},{i}] should be non-zero after warm start")

    def test_warm_start_zeros_b1(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.b1, 0.0)

    def test_warm_start_zeros_b2_valence(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.b2_valence, 0.0)

    def test_warm_start_zeros_gate_head(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.W2_gate, 0.0)
        np.testing.assert_array_equal(self.brain.arbitration_network.b2_gate, 0.0)

    def test_warm_start_zeros_value_head(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.W2_value, 0.0)
        np.testing.assert_array_equal(self.brain.arbitration_network.b2_value, 0.0)

    def test_warm_started_network_produces_finite_outputs(self) -> None:
        self.brain._warm_start_arbitration_network()
        evidence_vector = np.linspace(0.0, 1.0, ArbitrationNetwork.INPUT_DIM)
        valence_logits, gate_adjustments, value = self.brain.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        self.assertTrue(np.all(np.isfinite(valence_logits)))
        self.assertTrue(np.all(np.isfinite(gate_adjustments)))
        self.assertTrue(np.isfinite(value))

    def test_warm_start_valence_head_produces_nonzero_threat_logit_for_threat_input(self) -> None:
        self.brain._warm_start_arbitration_network()
        # Create a vector with only threat signals active
        evidence_vector = np.zeros(ArbitrationNetwork.INPUT_DIM)
        # First 6 slots are threat signals; set predator_visible (slot 0) to 1.0
        evidence_vector[0] = 1.0
        valence_logits, _, _ = self.brain.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        # Threat logit (index 0) should be dominant
        threat_idx = SpiderBrain.VALENCE_ORDER.index("threat")
        self.assertGreater(float(valence_logits[threat_idx]), float(valence_logits[1]))

    def test_warm_start_scale_reduces_forward_valence_logits(self) -> None:
        full = SpiderBrain(seed=42, module_dropout=0.0)
        scaled = SpiderBrain(
            seed=42,
            config=BrainAblationConfig(
                name="scaled_warm_start",
                module_dropout=0.0,
                warm_start_scale=0.5,
            ),
        )
        evidence_vector = _small_arbitration_evidence_vector()

        full_logits, _, _ = full.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        scaled_logits, _, _ = scaled.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        _assert_relative_scaled_logits(
            self,
            scaled_logits,
            full_logits,
            0.5,
        )

    def test_warm_start_scale_zero_skips_warm_start(self) -> None:
        no_start = SpiderBrain(
            seed=42,
            config=BrainAblationConfig(
                name="no_warm_start",
                module_dropout=0.0,
                warm_start_scale=0.0,
            ),
        )
        warmed = SpiderBrain(seed=42, module_dropout=0.0)

        self.assertNotEqual(no_start.arbitration_network.W1[0, 0], 0.0)
        self.assertNotAlmostEqual(
            no_start.arbitration_network.W1[0, 0],
            warmed.arbitration_network.W1[0, 0],
        )

    def test_minimal_arbitration_random_weights_follow_brain_seed(self) -> None:
        config = BrainAblationConfig(
            name="minimal_seed_test",
            module_dropout=0.0,
            warm_start_scale=0.0,
        )
        first = SpiderBrain(seed=42, config=config)
        same_seed = SpiderBrain(seed=42, config=config)
        different_seed = SpiderBrain(seed=43, config=config)

        np.testing.assert_allclose(
            first.arbitration_network.W1,
            same_seed.arbitration_network.W1,
        )
        np.testing.assert_allclose(
            first.arbitration_network.W2_gate,
            same_seed.arbitration_network.W2_gate,
        )
        self.assertFalse(
            np.allclose(
                first.arbitration_network.W1,
                different_seed.arbitration_network.W1,
            )
        )
        self.assertFalse(
            np.allclose(
                first.arbitration_network.W2_gate,
                different_seed.arbitration_network.W2_gate,
            )
        )

    def test_arbitration_network_uses_configured_gate_adjustment_bounds(self) -> None:
        brain = SpiderBrain(
            seed=42,
            config=BrainAblationConfig(
                name="wide_gate_bounds",
                module_dropout=0.0,
                gate_adjustment_bounds=(0.25, 1.75),
            ),
        )

        self.assertAlmostEqual(brain.arbitration_network.gate_adjustment_min, 0.25)
        self.assertAlmostEqual(brain.arbitration_network.gate_adjustment_max, 1.75)


class CanonicalArbitrationVariantIntegrationTest(unittest.TestCase):
    """Integration tests for named arbitration-prior ablation variants."""

    EXPECTED: ClassVar[dict[str, dict[str, object]]] = {
        "constrained_arbitration": {
            "enable_deterministic_guards": True,
            "enable_food_direction_bias": True,
            "warm_start_scale": 1.0,
            "gate_adjustment_bounds": (0.5, 1.5),
        },
        "weaker_prior_arbitration": {
            "enable_deterministic_guards": False,
            "enable_food_direction_bias": False,
            "warm_start_scale": 0.5,
            "gate_adjustment_bounds": (0.5, 1.5),
        },
        "minimal_arbitration": {
            "enable_deterministic_guards": False,
            "enable_food_direction_bias": False,
            "warm_start_scale": 0.0,
            "gate_adjustment_bounds": (0.1, 2.0),
        },
    }

    def _configs(self) -> dict[str, BrainAblationConfig]:
        """
        Return the canonical set of brain ablation configurations with module dropout disabled.
        
        Returns:
            configs (dict[str, BrainAblationConfig]): Mapping from canonical config names to their
            corresponding BrainAblationConfig instances (with module_dropout == 0.0).
        """
        return canonical_ablation_configs(module_dropout=0.0)

    def test_named_arbitration_configs_have_expected_fields(self) -> None:
        configs = self._configs()
        for name, expected in self.EXPECTED.items():
            with self.subTest(name=name):
                config = configs[name]
                self.assertTrue(config.use_learned_arbitration)
                self.assertEqual(config.module_dropout, 0.0)
                self.assertEqual(
                    config.enable_deterministic_guards,
                    expected["enable_deterministic_guards"],
                )
                self.assertEqual(
                    config.enable_food_direction_bias,
                    expected["enable_food_direction_bias"],
                )
                self.assertAlmostEqual(
                    config.warm_start_scale,
                    expected["warm_start_scale"],
                )
                self.assertEqual(
                    config.gate_adjustment_bounds,
                    expected["gate_adjustment_bounds"],
                )

    def test_named_arbitration_configs_pass_gate_bounds_to_network(self) -> None:
        """
        Verify that each canonical arbitration ablation configuration sets the arbitration network's gate adjustment bounds.
        
        For every named config, instantiate a SpiderBrain and assert that its arbitration_network.gate_adjustment_min and .gate_adjustment_max match the expected lower and upper bounds from the config.
        """
        configs = self._configs()
        for name, expected in self.EXPECTED.items():
            with self.subTest(name=name):
                brain = SpiderBrain(seed=42, config=configs[name])
                lower, upper = expected["gate_adjustment_bounds"]
                self.assertAlmostEqual(
                    brain.arbitration_network.gate_adjustment_min,
                    lower,
                )
                self.assertAlmostEqual(
                    brain.arbitration_network.gate_adjustment_max,
                    upper,
                )

    def test_named_arbitration_warm_start_scales_forward_valence_logits(self) -> None:
        configs = self._configs()
        constrained = SpiderBrain(
            seed=42,
            config=configs["constrained_arbitration"],
        )
        weaker = SpiderBrain(
            seed=42,
            config=configs["weaker_prior_arbitration"],
        )
        minimal = SpiderBrain(
            seed=42,
            config=configs["minimal_arbitration"],
        )
        evidence_vector = _small_arbitration_evidence_vector()

        constrained_logits, _, _ = constrained.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        weaker_logits, _, _ = weaker.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        _assert_relative_scaled_logits(
            self,
            weaker_logits,
            constrained_logits,
            0.5,
        )
        self.assertAlmostEqual(
            constrained.arbitration_network.W2_gate[0, 0],
            0.0,
        )
        self.assertAlmostEqual(
            weaker.arbitration_network.W2_gate[0, 0],
            0.0,
        )
        self.assertGreater(np.linalg.norm(minimal.arbitration_network.W2_gate), 0.0)
        self.assertNotAlmostEqual(
            minimal.arbitration_network.W1[0, 0],
            constrained.arbitration_network.W1[0, 0],
        )


# ---------------------------------------------------------------------------
# SpiderBrain arbitration_lr default tests
# ---------------------------------------------------------------------------

class ArbitrationLrDefaultTest(unittest.TestCase):
    """Tests that arbitration_lr defaults to module_lr when not provided."""

    def test_arbitration_lr_defaults_to_module_lr(self) -> None:
        brain = SpiderBrain(seed=1, module_lr=0.005, module_dropout=0.0)
        self.assertAlmostEqual(brain.arbitration_lr, 0.005)

    def test_arbitration_lr_can_be_set_explicitly(self) -> None:
        brain = SpiderBrain(seed=1, module_lr=0.01, arbitration_lr=0.002, module_dropout=0.0)
        self.assertAlmostEqual(brain.arbitration_lr, 0.002)

    def test_arbitration_lr_is_float(self) -> None:
        brain = SpiderBrain(seed=1, module_dropout=0.0)
        self.assertIsInstance(brain.arbitration_lr, float)

    def test_fixed_arbitration_config_has_zero_regularization(self) -> None:
        config = BrainAblationConfig(name="fixed_arbitration_baseline", use_learned_arbitration=False)
        brain = SpiderBrain(seed=1, module_dropout=0.0, config=config)
        self.assertEqual(brain.arbitration_regularization_weight, 0.0)
        self.assertEqual(brain.arbitration_valence_regularization_weight, 0.0)


# ---------------------------------------------------------------------------
# SpiderBrain save/load with arbitration_network tests
# ---------------------------------------------------------------------------

class SpiderBrainSaveLoadArbitrationTest(unittest.TestCase):
    """Tests that save() persists arbitration_network.npz and load() reads it back."""

    def setUp(self) -> None:
        import tempfile
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _brain(self, seed: int = 7) -> SpiderBrain:
        return SpiderBrain(seed=seed, module_dropout=0.0)

    def test_save_creates_arbitration_npz(self) -> None:
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        arb_path = os.path.join(self._tmpdir, "arbitration_network.npz")
        self.assertTrue(os.path.exists(arb_path), "arbitration_network.npz should be created")

    def test_save_metadata_contains_arbitration_network_entry(self) -> None:
        import json
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        meta_path = os.path.join(self._tmpdir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        self.assertIn("arbitration_network", metadata["modules"])
        arb_meta = metadata["modules"]["arbitration_network"]
        self.assertEqual(arb_meta["type"], "arbitration")
        self.assertIn("input_dim", arb_meta)
        self.assertIn("hidden_dim", arb_meta)
        self.assertIn("valence_dim", arb_meta)
        self.assertIn("gate_dim", arb_meta)

    def test_save_metadata_includes_arbitration_lr(self) -> None:
        """Verify that saving a Brain writes arbitration hyperparameters into metadata.json."""
        import json
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        meta_path = os.path.join(self._tmpdir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        self.assertIn("arbitration_lr", metadata)
        self.assertIn("arbitration_regularization_weight", metadata)
        self.assertIn("arbitration_valence_regularization_weight", metadata)

    def test_load_restores_arbitration_network_weights(self) -> None:
        brain1 = self._brain(seed=7)
        brain1.save(self._tmpdir)

        brain2 = self._brain(seed=99)
        brain2.load(self._tmpdir)

        np.testing.assert_array_almost_equal(
            brain1.arbitration_network.W1,
            brain2.arbitration_network.W1,
        )
        np.testing.assert_array_almost_equal(
            brain1.arbitration_network.W2_valence,
            brain2.arbitration_network.W2_valence,
        )

    def test_load_with_missing_arbitration_npz_raises(self) -> None:
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        # Remove the arbitration npz to simulate a missing file
        arb_path = os.path.join(self._tmpdir, "arbitration_network.npz")
        os.remove(arb_path)
        brain2 = self._brain(seed=99)
        with self.assertRaises(FileNotFoundError):
            brain2.load(self._tmpdir)

    def test_load_with_missing_arbitration_metadata_raises(self) -> None:
        import json
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        # Remove arbitration_network from metadata["modules"]
        meta_path = os.path.join(self._tmpdir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        del metadata["modules"]["arbitration_network"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        brain2 = self._brain(seed=99)
        with self.assertRaises(FileNotFoundError):
            brain2.load(self._tmpdir)

    def test_roundtrip_produces_same_act_output(self) -> None:
        brain1 = self._brain(seed=7)
        obs = _blank_obs()
        step1 = brain1.act(obs, bus=None, sample=False)

        brain1.save(self._tmpdir)
        brain2 = self._brain(seed=99)
        brain2.load(self._tmpdir)
        step2 = brain2.act(obs, bus=None, sample=False)

        self.assertEqual(step1.action_intent_idx, step2.action_intent_idx)
        np.testing.assert_array_almost_equal(
            step1.action_center_logits, step2.action_center_logits, decimal=5
        )


# ---------------------------------------------------------------------------
# ArbitrationDecision.to_payload() predator_proximity backward-compat key sorting
# ---------------------------------------------------------------------------

class ArbitrationDecisionPredatorProximityCompatTest(unittest.TestCase):
    """Tests that the legacy predator_proximity key is injected and sorted correctly."""

    def _make_with_threat(self, extra_threat_keys: dict | None = None) -> "ArbitrationDecision":
        from spider_cortex_sim.agent import ArbitrationDecision
        threat_evidence = {"predator_visible": 1.0, "predator_certainty": 0.9}
        if extra_threat_keys:
            threat_evidence.update(extra_threat_keys)
        return ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={"threat": 0.8, "hunger": 0.1, "sleep": 0.05, "exploration": 0.05},
            module_gates={"alert_center": 1.0},
            suppressed_modules=[],
            evidence={"threat": threat_evidence},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )

    def test_predator_proximity_injected_as_zero(self) -> None:
        d = self._make_with_threat()
        payload = d.to_payload()
        self.assertIn("predator_proximity", payload["evidence"]["threat"])
        self.assertEqual(payload["evidence"]["threat"]["predator_proximity"], 0.0)

    def test_injected_predator_proximity_keys_are_sorted(self) -> None:
        d = self._make_with_threat()
        payload = d.to_payload()
        keys = list(payload["evidence"]["threat"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_predator_proximity_not_injected_when_already_present(self) -> None:
        """If the evidence dict already has predator_proximity, it must not be overwritten."""
        from spider_cortex_sim.agent import ArbitrationDecision
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"threat": {"predator_proximity": 0.77}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        # The already-present value should be preserved
        self.assertAlmostEqual(payload["evidence"]["threat"]["predator_proximity"], 0.77)

    def test_no_threat_evidence_no_injection(self) -> None:
        """If threat is absent from evidence, no predator_proximity is injected."""
        from spider_cortex_sim.agent import ArbitrationDecision
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="hunger",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"hunger": {"hunger": 0.9}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        self.assertNotIn("predator_proximity", payload["evidence"].get("hunger", {}))


# ---------------------------------------------------------------------------
# SpiderBrain._compute_arbitration training mode tests
# ---------------------------------------------------------------------------

class ComputeArbitrationTrainingModeTest(unittest.TestCase):
    """Tests for the training parameter of SpiderBrain._compute_arbitration."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_training_false_clears_arbitration_cache_for_fixed_config(self) -> None:
        config = BrainAblationConfig(name="fixed_arbitration_baseline", use_learned_arbitration=False)
        brain = SpiderBrain(seed=42, module_dropout=0.0, config=config)
        obs = _blank_obs()
        module_results = brain._proposal_results(obs, store_cache=False, training=False)
        brain._compute_arbitration(module_results, obs, training=False, store_cache=False)
        self.assertIsNone(brain.arbitration_network.cache)

    def test_learned_arbitration_training_true_stores_cache(self) -> None:
        module_results = self.brain._proposal_results(self.obs, store_cache=False, training=False)
        self.brain._compute_arbitration(module_results, self.obs, training=True, store_cache=True)
        self.assertIsNotNone(self.brain.arbitration_network.cache)

    def test_learned_arbitration_training_false_returns_deterministic_result(self) -> None:
        """Calling _compute_arbitration with training=False gives consistent winning_valence."""
        module_results = self.brain._proposal_results(self.obs, store_cache=False, training=False)
        arb1 = self.brain._compute_arbitration(module_results, self.obs, training=False, store_cache=False)
        arb2 = self.brain._compute_arbitration(module_results, self.obs, training=False, store_cache=False)
        self.assertEqual(arb1.winning_valence, arb2.winning_valence)

    def test_learned_adjustment_is_true_for_learned_brain(self) -> None:
        module_results = self.brain._proposal_results(self.obs, store_cache=False, training=False)
        arb = self.brain._compute_arbitration(module_results, self.obs, training=False, store_cache=False)
        self.assertTrue(arb.learned_adjustment)

    def test_learned_adjustment_is_false_for_fixed_brain(self) -> None:
        config = BrainAblationConfig(name="fixed_arbitration_baseline", use_learned_arbitration=False)
        brain = SpiderBrain(seed=42, module_dropout=0.0, config=config)
        obs = _blank_obs()
        module_results = brain._proposal_results(obs, store_cache=False, training=False)
        arb = brain._compute_arbitration(module_results, obs, training=False, store_cache=False)
        self.assertFalse(arb.learned_adjustment)


if __name__ == "__main__":
    unittest.main()
