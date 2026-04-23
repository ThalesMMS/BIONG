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

from tests.fixtures.action_center import (
    _null_percept,
    _blank_obs,
    _module_interface,
    _vector_for,
    _valence_vector,
    _profile_with_updates,
)

class BuildActionInputCompatibilityTest(unittest.TestCase):
    def test_build_action_input_falls_back_when_gated_logits_missing(self) -> None:
        brain = SpiderBrain(seed=11, module_dropout=0.0)
        observation = _blank_obs()

        compat_result = type("CompatResult", (), {})()
        compat_result.logits = np.full(len(LOCOMOTION_ACTIONS), 0.25)
        gated_result = type("GatedResult", (), {})()
        gated_result.logits = np.full(len(LOCOMOTION_ACTIONS), -1.0)
        gated_result.gated_logits = np.full(len(LOCOMOTION_ACTIONS), 0.75)

        action_input = brain._build_action_input(
            [compat_result, gated_result],
            observation,
        )

        np.testing.assert_allclose(
            action_input[: len(LOCOMOTION_ACTIONS)],
            compat_result.logits,
        )
        np.testing.assert_allclose(
            action_input[len(LOCOMOTION_ACTIONS) : 2 * len(LOCOMOTION_ACTIONS)],
            gated_result.gated_logits,
        )

    def test_monolithic_proposal_forwards_training_flag(self) -> None:
        config = BrainAblationConfig(
            name="monolithic_test",
            architecture="monolithic",
            module_dropout=0.0,
        )
        brain = SpiderBrain(seed=12, module_dropout=0.0, config=config)
        calls = []

        def fake_forward(x, *, store_cache=True, training=False):
            calls.append((bool(store_cache), bool(training), x.shape))
            return np.zeros(brain.action_dim, dtype=float)

        with patch.object(brain.monolithic_policy, "forward", side_effect=fake_forward):
            results = brain._proposal_results(
                _blank_obs(),
                store_cache=False,
                training=True,
            )

        expected_shape = (sum(spec.input_dim for spec in MODULE_INTERFACES),)
        self.assertEqual(calls, [(False, True, expected_shape)])
        self.assertEqual(results[0].name, SpiderBrain.MONOLITHIC_POLICY_NAME)

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

    def test_arbitration_gate_dim_is_9(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_GATE_DIM
        self.assertEqual(ARBITRATION_GATE_DIM, 9)

    def test_evidence_input_dim_matches_arbitration_network_input_dim(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_EVIDENCE_INPUT_DIM
        self.assertEqual(ARBITRATION_EVIDENCE_INPUT_DIM, ArbitrationNetwork.INPUT_DIM)

    def test_valence_dim_matches_arbitration_network_valence_dim(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_VALENCE_DIM
        self.assertEqual(ARBITRATION_VALENCE_DIM, ArbitrationNetwork.VALENCE_DIM)

    def test_gate_dim_matches_arbitration_network_gate_dim(self) -> None:
        from spider_cortex_sim.interfaces import ARBITRATION_GATE_DIM
        self.assertEqual(ARBITRATION_GATE_DIM, ArbitrationNetwork.GATE_DIM)

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
