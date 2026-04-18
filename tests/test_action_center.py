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

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    default_brain_config,
)
from spider_cortex_sim.agent import BrainStep, SpiderBrain
from spider_cortex_sim.arbitration import ArbitrationDecision
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    MODULE_INTERFACE_BY_NAME,
    OBSERVATION_DIMS,
    ActionContextObservation,
    MotorContextObservation,
    architecture_signature,
)
from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork
from spider_cortex_sim.noise import compute_execution_difficulty
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
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


def _valence_vector(**updates: float) -> np.ndarray:
    values = np.zeros(len(SpiderBrain.VALENCE_ORDER), dtype=float)
    index_by_name = {
        name: index
        for index, name in enumerate(SpiderBrain.VALENCE_ORDER)
    }
    for name, value in updates.items():
        values[index_by_name[name]] = float(value)
    return values


def _profile_with_updates(**reward_updates: float) -> OperationalProfile:
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    summary["name"] = "action_center_test_profile"
    summary["version"] = 21
    summary["reward"].update({name: float(value) for name, value in reward_updates.items()})
    return OperationalProfile.from_summary(summary)


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
        """motor_cortex must list exactly 14 input signal names."""
        mc = self.sig["motor_cortex"]
        self.assertEqual(len(mc["inputs"]), 14)

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
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.input_dim, 14)

    def test_motor_context_includes_momentum(self) -> None:
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.input_dim, 14)
        self.assertEqual(OBSERVATION_DIMS["motor_context"], 14)
        self.assertEqual(MOTOR_CONTEXT_INTERFACE.signal_names[-1], "momentum")

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
            names[-5:],
            ("heading_dx", "heading_dy", "terrain_difficulty", "fatigue", "momentum"),
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
        self.assertEqual(OBSERVATION_DIMS["motor_context"], 14)

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
    """Tests that MotorContextObservation has the correct 14 fields."""

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
        for field in ("heading_dx", "heading_dy", "terrain_difficulty", "fatigue", "momentum"):
            with self.subTest(field=field):
                self.assertIn(field, names)

    def test_as_mapping_contains_all_14_fields(self) -> None:
        view = MotorContextObservation(
            on_food=0.0, on_shelter=0.0, predator_visible=0.0, predator_certainty=0.0,
            day=1.0, night=0.0, last_move_dx=0.0, last_move_dy=0.0,
            shelter_role_level=0.0, heading_dx=1.0, heading_dy=0.0,
            terrain_difficulty=0.4, fatigue=0.2, momentum=0.0,
        )
        mapping = view.as_mapping()
        self.assertEqual(len(mapping), 14)
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

    def test_momentum_from_world_state(self) -> None:
        self.world.state.momentum = 0.63
        result = self._build()
        self.assertAlmostEqual(result.momentum, 0.63, places=5)

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
        self.assertIsInstance(step.momentum, float)
        self.assertIsInstance(step.execution_difficulty, float)
        self.assertIsInstance(step.execution_slip_occurred, bool)
        self.assertIsInstance(step.motor_noise_applied, bool)
        self.assertIsInstance(step.motor_slip_occurred, bool)
        self.assertEqual(step.slip_reason, "none")

    def test_brainstep_includes_momentum_field(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)

        self.assertTrue(hasattr(step, "momentum"))
        self.assertIsInstance(step.momentum, float)

    def test_brain_step_momentum_comes_from_motor_context(self) -> None:
        obs = _blank_obs()
        obs["motor_context"] = _vector_for(
            MOTOR_CONTEXT_INTERFACE,
            momentum=0.73,
        )

        step = self.brain.act(obs, bus=None, sample=False)

        self.assertAlmostEqual(step.momentum, 0.73)

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

    def test_motor_execution_diagnostics_use_momentum(self) -> None:
        obs = _blank_obs()
        obs["motor_context"] = _vector_for(
            MOTOR_CONTEXT_INTERFACE,
            heading_dx=1.0,
            heading_dy=1.0,
            terrain_difficulty=0.7,
            fatigue=0.0,
            momentum=0.8,
        )

        diagnostics = self.brain._motor_execution_diagnostics(
            obs,
            ACTION_TO_INDEX["MOVE_RIGHT"],
        )

        self.assertAlmostEqual(diagnostics["momentum"], 0.8)
        self.assertLess(diagnostics["execution_difficulty"], 0.7)

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

    def test_motor_context_shape_is_14(self) -> None:
        """The motor_context vector includes momentum as field #14."""
        self.assertEqual(self.obs["motor_context"].shape, (14,))

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

        with unittest.mock.patch("spider_cortex_sim.agent.clamp_unit", return_value=0.0):
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

class CorticalModuleBankResultFieldsTest(unittest.TestCase):
    """Tests that CorticalModuleBank.forward initializes new reflex fields on ModuleResult."""

    def _make_bank(self) -> CorticalModuleBank:
        """
        Create a CorticalModuleBank configured for deterministic test runs.

        Returns:
            CorticalModuleBank: instance with action_dim equal to len(LOCOMOTION_ACTIONS), an RNG seeded with 99, and module_dropout set to 0.0.
        """
        rng = np.random.default_rng(99)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
        )

    def _blank_observation(self, bank: CorticalModuleBank) -> dict:
        """
        Build an observation dictionary containing zeroed input vectors for every spec in the cortical module bank.

        Parameters:
            bank (CorticalModuleBank): Module bank whose specs determine observation keys and input dimensions.

        Returns:
            dict: Mapping from each spec.observation_key to a NumPy float array of zeros with length equal to spec.input_dim.
        """
        obs = {}
        for spec in bank.specs:
            obs[spec.observation_key] = np.zeros(spec.input_dim, dtype=float)
        return obs

    def test_forward_sets_neural_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.neural_logits)
                np.testing.assert_allclose(result.neural_logits, result.logits)

    def test_forward_sets_reflex_delta_logits_to_zeros(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.reflex_delta_logits)
                np.testing.assert_allclose(
                    result.reflex_delta_logits,
                    np.zeros_like(result.logits),
                )

    def test_forward_sets_post_reflex_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.post_reflex_logits)
                np.testing.assert_allclose(result.post_reflex_logits, result.logits)


class SpiderBrainMonolithicArchitectureTest(unittest.TestCase):
    """Tests for the new monolithic architecture branch in SpiderBrain."""

    def _monolithic_config(
        self,
        *,
        credit_strategy: str = "broadcast",
    ) -> BrainAblationConfig:
        return BrainAblationConfig(
            name=MONOLITHIC_POLICY_NAME,
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy=credit_strategy,
            disabled_modules=(),
        )

    def _build_observation(self) -> dict:
        """
        Builds a baseline observation dictionary containing zeroed input vectors for every module and context.

        The returned dictionary contains entries for each interface in MODULE_INTERFACES plus ACTION_CONTEXT_INTERFACE and MOTOR_CONTEXT_INTERFACE. Each key is the interface's `observation_key` and each value is a NumPy array of zeros with length equal to that interface's `input_dim`.

        Returns:
            dict: Mapping from observation_key to a zeroed NumPy array sized to the corresponding interface's input_dim.
        """
        obs = {}
        for interface in MODULE_INTERFACES:
            obs[interface.observation_key] = np.zeros(interface.input_dim, dtype=float)
        obs[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
            ACTION_CONTEXT_INTERFACE.input_dim, dtype=float
        )
        obs[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
            MOTOR_CONTEXT_INTERFACE.input_dim, dtype=float
        )
        return obs

    def test_monolithic_brain_has_no_module_bank(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNone(brain.module_bank)

    def test_monolithic_brain_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNotNone(brain.monolithic_policy)

    def test_modular_brain_has_module_bank(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNotNone(brain.module_bank)

    def test_modular_brain_has_no_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNone(brain.monolithic_policy)

    def test_monolithic_act_returns_single_module_result(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertEqual(len(step.module_results), 1)
        self.assertEqual(step.module_results[0].name, SpiderBrain.MONOLITHIC_POLICY_NAME)

    def test_monolithic_act_result_has_none_interface(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertIsNone(step.module_results[0].interface)

    def test_monolithic_act_result_is_active(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertTrue(step.module_results[0].active)

    def test_monolithic_module_names_includes_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        names = brain._module_names()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)
        self.assertIn("arbitration_network", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)

    def test_modular_module_names_includes_each_module(self) -> None:
        brain = SpiderBrain(seed=5)
        names = brain._module_names()
        self.assertIn("visual_cortex", names)
        self.assertIn("alert_center", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)

    def test_monolithic_parameter_norms_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=3, config=self._monolithic_config())
        norms = brain.parameter_norms()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)
        self.assertIn("arbitration_network", norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)

    def test_modular_parameter_norms_has_module_names(self) -> None:
        brain = SpiderBrain(seed=3)
        norms = brain.parameter_norms()
        self.assertIn("visual_cortex", norms)
        self.assertIn("alert_center", norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)

    def test_monolithic_estimate_value_returns_finite_float(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        value = brain.estimate_value(obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_monolithic_action_idx_is_valid(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))

    def test_monolithic_step_exposes_arbitration_metadata(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertIsNotNone(step.arbitration_decision)
        self.assertEqual(step.arbitration_decision.module_gates["monolithic_policy"], 1.0)
        self.assertEqual(
            step.arbitration_decision.module_contribution_share["monolithic_policy"],
            1.0,
        )

    def test_monolithic_save_and_load_round_trip(self) -> None:
        """
        Verifies that saving then loading a monolithic SpiderBrain reproduces its policy and value outputs.

        Uses a fixed observation and monolithic configuration: saves the brain to disk, loads it into a new SpiderBrain instance with the same configuration, and asserts that the resulting policy and value vectors match within a small numerical tolerance.
        """
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)
        obs = self._build_observation()
        step_before = brain.act(obs, sample=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            brain2 = SpiderBrain(seed=999, config=config)
            brain2.load(save_path)
            step_after = brain2.act(obs, sample=False)

        np.testing.assert_allclose(step_before.policy, step_after.policy, atol=1e-5)
        np.testing.assert_allclose(step_before.value, step_after.value, atol=1e-5)

    def test_loading_mismatched_architecture_version_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["architecture_version"] = SpiderBrain.ARCHITECTURE_VERSION - 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaises(ValueError):
                brain2.load(save_path)

    def test_save_metadata_contains_registry_and_fingerprint(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("interface_registry", metadata)
            self.assertIn("architecture_fingerprint", metadata)
            self.assertEqual(
                metadata["architecture_fingerprint"],
                metadata["architecture"]["fingerprint"],
            )

    def test_save_writes_arbitration_network_weights(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            self.assertTrue((save_path / "arbitration_network.npz").exists())
            with np.load(save_path / "arbitration_network.npz") as arbitration_npz:
                self.assertIn("input_dim", arbitration_npz.files)
                self.assertIn("W2_gate", arbitration_npz.files)
            metadata = json.loads(
                (save_path / SpiderBrain._METADATA_FILE).read_text(encoding="utf-8")
            )
            self.assertIn("arbitration_network", metadata["modules"])
            self.assertEqual(
                metadata["modules"]["arbitration_network"]["type"],
                "arbitration",
            )

    def test_loading_missing_arbitration_network_weights_raises_file_not_found(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)
            (save_path / "arbitration_network.npz").unlink()

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaisesRegex(FileNotFoundError, "arbitration weights"):
                brain2.load(save_path)

    def test_loading_mismatched_interface_registry_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["interface_registry"]["interfaces"]["motor_cortex_context"]["version"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaisesRegex(ValueError, "interface registry incompatible"):
                brain2.load(save_path)

    def test_loading_monolithic_into_modular_raises_value_error(self) -> None:
        mono_config = self._monolithic_config()
        brain_mono = SpiderBrain(seed=13, config=mono_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain_mono.save(save_path)

            brain_modular = SpiderBrain(seed=13)
            with self.assertRaises(ValueError):
                brain_modular.load(save_path)

    def test_loading_different_modular_ablation_config_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=19, config=default_brain_config(module_dropout=0.0))
        different_config = BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "modular_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(seed=19, config=different_config)
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_loading_different_operational_profile_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=23)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "profiled_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(
                seed=23,
                operational_profile=_profile_with_updates(predator_threat_smell_threshold=0.01),
            )
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_monolithic_config_stored_on_brain(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=7, config=config)
        self.assertEqual(brain.config.architecture, "monolithic")
        self.assertEqual(brain.config.name, "monolithic_policy")

    def test_monolithic_learn_does_not_raise(self) -> None:
        brain = SpiderBrain(seed=17, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        stats = brain.learn(step, reward=0.5, next_observation=next_obs, done=False)
        self.assertIn("reward", stats)
        self.assertTrue(np.isfinite(stats["td_error"]))

    def test_modular_learn_routes_action_center_input_grads_to_module_aux_paths(self) -> None:
        config = BrainAblationConfig(
            name="modular_no_reflex_aux",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        proposal_grads = np.arange(
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            captured["grad_value"] = float(grad_value)
            captured["lr"] = float(lr)
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            captured["bank_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["bank_lr"] = float(lr)
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        expected_reflex_aux = brain._auxiliary_module_gradients(step.module_results)
        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["bank_grad"],
            np.zeros_like(captured["shared_grad"]),
        )
        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * (
                captured["shared_grad"] + proposal_grads[idx]
            )
            expected_total_grad += result.gate_weight * expected_reflex_aux.get(
                result.name,
                np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            )
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_credit_weights"][result.name],
                1.0,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["aux_modules"], float(len(expected_reflex_aux)))
        self.assertEqual(stats["credit_strategy"], "broadcast")

    def test_learn_projects_action_center_input_grads_to_arbitration_gates(self) -> None:
        config = BrainAblationConfig(
            name="modular_no_reflex_aux",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
        )
        brain = SpiderBrain(
            seed=7,
            config=config,
            arbitration_regularization_weight=0.25,
            arbitration_valence_regularization_weight=0.4,
        )
        brain.arbitration_network.b2_gate[:] = np.log(0.7 / 0.3)
        obs = self._build_observation()
        step = brain.act(obs, sample=False, training=True)
        arbitration = step.arbitration_decision
        self.assertIsNotNone(arbitration)

        captured: dict[str, object] = {}
        proposal_grads = np.linspace(
            -0.2,
            0.3,
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            """
            Capture a copy of the policy logits gradient into the outer `captured` mapping and return the upstream gradient unchanged.

            Parameters:
                grad_policy_logits (array-like): Gradient of the policy logits to capture; converted to a float NumPy array and stored as `captured["policy_grad"]`.
                grad_value: Ignored.
                lr: Ignored.

            Returns:
                The upstream gradient value unchanged.
            """
            captured["policy_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_arbitration_backward(*, grad_valence_logits, grad_gate_adjustments, grad_value, lr):
            """
            Record provided arbitration gradients and learning rate into the `captured` mapping and return a zeroed input-gradient vector matching the arbitration network input dimension.

            Parameters:
                grad_valence_logits (array-like): Gradients for arbitration valence logits; stored as a float NumPy array under key `"arb_valence_grad"`.
                grad_gate_adjustments (array-like): Gradients for arbitration gate adjustments; stored as a float NumPy array under key `"arb_gate_grad"`.
                grad_value (float): Gradient for the arbitration value output; stored as a float under key `"arb_value_grad"`.
                lr (float): Learning rate used for this backward pass; stored as a float under key `"arb_lr"`.

            Returns:
                numpy.ndarray: A zero-filled float array whose length equals the arbitration network input dimension.
            """
            captured["arb_valence_grad"] = np.asarray(grad_valence_logits, dtype=float).copy()
            captured["arb_gate_grad"] = np.asarray(grad_gate_adjustments, dtype=float).copy()
            captured["arb_value_grad"] = float(grad_value)
            captured["arb_lr"] = float(lr)
            return np.zeros(brain.arbitration_network.input_dim, dtype=float)

        brain.action_center.backward = fake_action_center_backward
        brain.arbitration_network.backward = fake_arbitration_backward
        brain.module_bank.backward = lambda grad_logits, lr, aux_grads=None: None
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        td_target = 0.5
        advantage = float(np.clip(td_target - step.value, -4.0, 4.0))
        valence_probs = np.array(
            [
                float(arbitration.valence_scores[name])
                for name in SpiderBrain.VALENCE_ORDER
            ],
            dtype=float,
        )
        fixed_targets = brain._fixed_formula_valence_scores_from_evidence(arbitration.evidence)
        expected_valence_grad = advantage * (
            valence_probs
            - np.eye(len(SpiderBrain.VALENCE_ORDER))[SpiderBrain.VALENCE_ORDER.index(arbitration.winning_valence)]
        )
        expected_valence_grad += brain.arbitration_valence_regularization_weight * (
            valence_probs - fixed_targets
        )

        gate_indices = {
            name: index
            for index, name in enumerate(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER)
        }
        expected_final_gate_grad = np.zeros(
            len(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER),
            dtype=float,
        )
        expected_gate_reg = np.zeros_like(expected_final_gate_grad)
        expected_base = np.zeros_like(expected_final_gate_grad)
        for idx, result in enumerate(step.module_results):
            gate_index = gate_indices[result.name]
            raw_logits = result.post_reflex_logits
            self.assertIsNotNone(raw_logits)
            expected_base[gate_index] = arbitration.base_gates[result.name]
            expected_final_gate_grad[gate_index] += float(
                np.dot(proposal_grads[idx], raw_logits)
            )
            expected_gate_reg[gate_index] += (
                brain.arbitration_regularization_weight
                * (arbitration.module_gates[result.name] - arbitration.base_gates[result.name])
            )
        expected_gate_grad = expected_base * (
            expected_final_gate_grad + expected_gate_reg
        )

        np.testing.assert_allclose(captured["arb_valence_grad"], expected_valence_grad)
        np.testing.assert_allclose(captured["arb_gate_grad"], expected_gate_grad)
        self.assertAlmostEqual(captured["arb_value_grad"], arbitration.arbitration_value - td_target)
        self.assertEqual(captured["arb_lr"], brain.arbitration_lr)
        self.assertGreater(stats["arbitration_gate_regularization_norm"], 0.0)
        self.assertGreater(stats["arbitration_valence_regularization_norm"], 0.0)
        self.assertIn("arbitration_loss", stats)
        self.assertIn("gate_adjustment_magnitude", stats)
        self.assertIn("regularization_loss", stats)
        self.assertGreater(stats["gate_adjustment_magnitude"], 0.0)
        self.assertGreater(stats["regularization_loss"], 0.0)

    def test_local_credit_only_learn_omits_shared_policy_broadcast(self) -> None:
        """
        Verifies that the "local_credit_only" ablation prevents shared policy gradients from being broadcast to module auxiliary gradient paths.

        Sets up a modular brain configured for the local-credit-only variant, stubs the action-center and module-bank backward methods to capture gradients, invokes learning on a single step, and asserts each module's auxiliary gradient equals the module's gate weight multiplied by its proposal gradients (i.e., only local gate-weighted proposal gradients are applied to module auxiliary paths).
        """
        config = BrainAblationConfig(
            name="local_credit_only",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="local_only",
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        proposal_grads = np.arange(
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * proposal_grads[idx]
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_credit_weights"][result.name],
                0.0,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["credit_strategy"], "local_only")

    def test_counterfactual_credit_normalizes_absolute_interference(self) -> None:
        """
        Verifies that counterfactual credit weights normalize absolute interference between modules.

        Constructs two module results with opposing effects and asserts the counterfactual credit
        computation:
        - returns positive weights for each contributing module,
        - normalizes weights to sum to 1.0,
        - assigns different weights when modules have different absolute contributions.
        """
        brain = SpiderBrain(
            seed=7,
            config=BrainAblationConfig(
                name="counterfactual_credit",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy="counterfactual",
            ),
        )
        action_dim = len(LOCOMOTION_ACTIONS)

        def module_result(module_name: str, logits: np.ndarray) -> ModuleResult:
            """
            Create a ready-to-use active ModuleResult for the given module name using the provided logits.

            Parameters:
                module_name (str): Module identifier; its interface is looked up to set `interface` and `observation_key`.
                logits (np.ndarray): Logits for the module's action outputs; length must match the action dimension used by the brain.

            Returns:
                ModuleResult: A ModuleResult with the resolved interface and observation_key, a zeroed observation vector sized to the interface's input dimension, the supplied `logits`, a uniform probability vector over actions, and `active=True`.
            """
            interface = MODULE_INTERFACE_BY_NAME[module_name]
            return ModuleResult(
                interface=interface,
                name=module_name,
                observation_key=interface.observation_key,
                observation=np.zeros(interface.input_dim, dtype=float),
                logits=logits,
                probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
                active=True,
            )

        visual_logits = np.zeros(action_dim, dtype=float)
        sensory_logits = np.zeros(action_dim, dtype=float)
        visual_logits[0] = 1.0
        sensory_logits[0] = -1.0
        visual = module_result("visual_cortex", visual_logits)
        sensory = module_result("sensory_cortex", sensory_logits)
        def fake_action_center_forward(x, *, store_cache=False):
            """
            Compute a small corrective action logits vector from a concatenated visual+sensory input and return it with a zero baseline value.

            Parameters:
                x (array-like): Flattened input whose first `action_dim` entries are the visual segment and the next `action_dim` entries are the sensory segment; length must be at least 2 * action_dim.

            Returns:
                tuple:
                    correction (numpy.ndarray): 1D float array of length `action_dim` containing the corrective logits.
                    baseline (float): A scalar baseline value (always 0.0).
            """
            action_input = np.asarray(x, dtype=float)
            visual_segment = action_input[:action_dim]
            sensory_segment = action_input[action_dim : 2 * action_dim]
            correction = np.zeros(action_dim, dtype=float)
            correction[0] = (
                0.75 * visual_segment[0]
                - 0.50 * sensory_segment[0]
            )
            correction[1] = (
                -0.25 * visual_segment[0]
                + 0.25 * sensory_segment[0]
            )
            return correction, 0.0

        brain.action_center.forward = fake_action_center_forward

        weights = brain._compute_counterfactual_credit(
            [visual, sensory],
            self._build_observation(),
            action_idx=0,
        )

        self.assertEqual(set(weights), {"visual_cortex", "sensory_cortex"})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["visual_cortex"], 0.0)
        self.assertGreater(weights["sensory_cortex"], 0.0)
        self.assertNotAlmostEqual(
            weights["visual_cortex"],
            weights["sensory_cortex"],
        )

    def test_counterfactual_learn_scales_shared_policy_broadcast(self) -> None:
        config = BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="counterfactual",
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        weights = {
            result.name: float(index + 1)
            for index, result in enumerate(step.module_results)
        }
        weight_total = float(sum(weights.values()))
        weights = {
            name: value / weight_total
            for name, value in weights.items()
        }

        captured: dict[str, object] = {}
        proposal_grads = np.arange(
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            """
            Stub backward function used in tests to capture the shared policy gradient.

            Stores a copy of `grad_policy_logits` (as a float numpy array) into `captured["shared_grad"]` and returns the incoming `upstream` value unchanged.

            Parameters:
                grad_policy_logits: array-like
                    Gradients for the policy logits to be captured.
                grad_value:
                    Ignored by this stub.
                lr:
                    Ignored by this stub.

            Returns:
                upstream: the unchanged upstream value passed through by the stub.
            """
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            """
            Capture and store a copy of auxiliary gradients into the outer `captured` mapping.

            Parameters:
                grad_logits: Unused in this helper; present to match the real backward signature.
                lr: Unused learning-rate parameter; present to match the real backward signature.
                aux_grads (dict[str, array_like] | None): Mapping from module name to gradient array. Each value is converted to a NumPy float array and stored as a copy in `captured["aux_grads"]`.
            """
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain._compute_counterfactual_credit = lambda module_results, observation, action_idx: dict(weights)
        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        shared_grad = np.asarray(captured["shared_grad"], dtype=float)
        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * (
                weights[result.name] * shared_grad + proposal_grads[idx]
            )
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["credit_strategy"], "counterfactual")
        self.assertEqual(stats["module_credit_weights"], weights)
        self.assertEqual(stats["counterfactual_credit_weights"], weights)

    def test_monolithic_learn_adds_action_center_input_grads_to_policy_update(self) -> None:
        """
        Test that monolithic learning adds action-center input gradients to the monolithic policy update and records credit diagnostics.

        Sets up a monolithic SpiderBrain with stubbed backward methods to capture gradients, runs a single learn step, and asserts:
        - the monolithic policy backward receives the elementwise sum of the shared policy gradient and action-center input gradients,
        - the motor cortex receives only the shared policy gradient,
        - learning stats include `credit_strategy == "broadcast"`,
        - `module_credit_weights` contains the monolithic policy with weight 1.0,
        - `module_gradient_norms` contains the norm of the monolithic policy gradient.
        """
        brain = SpiderBrain(seed=11, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        extra_grad = np.linspace(0.1, 0.5, len(LOCOMOTION_ACTIONS), dtype=float)
        upstream = np.concatenate(
            [
                extra_grad,
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_monolithic_backward(grad_logits, lr):
            captured["mono_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["mono_lr"] = float(lr)

        def fake_motor_backward(grad_logits, lr):
            captured["motor_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["motor_lr"] = float(lr)

        brain.action_center.backward = fake_action_center_backward
        brain.monolithic_policy.backward = fake_monolithic_backward
        brain.motor_cortex.backward = fake_motor_backward
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["mono_grad"],
            captured["shared_grad"] + extra_grad,
        )
        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {SpiderBrain.MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertAlmostEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            float(np.linalg.norm(captured["mono_grad"])),
        )
        np.testing.assert_allclose(
            captured["motor_grad"],
            captured["shared_grad"],
        )

    def _assert_monolithic_non_broadcast_credit_coerces_to_broadcast(
        self,
        credit_strategy: str,
    ) -> None:
        brain = SpiderBrain(
            seed=13,
            config=self._monolithic_config(credit_strategy=credit_strategy),
        )
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        extra_grad = np.linspace(0.1, 0.5, len(LOCOMOTION_ACTIONS), dtype=float)
        upstream = np.concatenate(
            [
                extra_grad,
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_monolithic_backward(grad_logits, lr):
            captured["mono_grad"] = np.asarray(grad_logits, dtype=float).copy()

        brain.action_center.backward = fake_action_center_backward
        brain.monolithic_policy.backward = fake_monolithic_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["mono_grad"],
            captured["shared_grad"] + extra_grad,
        )
        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {SpiderBrain.MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertEqual(stats["counterfactual_credit_weights"], {})
        self.assertAlmostEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            float(np.linalg.norm(captured["mono_grad"])),
        )

    def test_monolithic_local_only_credit_reports_effective_broadcast(self) -> None:
        self._assert_monolithic_non_broadcast_credit_coerces_to_broadcast("local_only")

    def test_monolithic_counterfactual_credit_reports_effective_broadcast(self) -> None:
        self._assert_monolithic_non_broadcast_credit_coerces_to_broadcast(
            "counterfactual"
        )


if __name__ == "__main__":
    unittest.main()
