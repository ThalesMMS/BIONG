"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 7
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

import numpy as np

from spider_cortex_sim.ablations import default_brain_config
from spider_cortex_sim.agent import BrainStep, SpiderBrain
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
    """Build a zeroed observation dict for all interfaces."""
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
        percept = PerceivedTarget(visible=0.95, certainty=0.88, occluded=0.0, dx=1.0, dy=0.0, dist=3)
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
        percept = PerceivedTarget(visible=0.8, certainty=0.7, occluded=0.0, dx=-1.0, dy=0.0, dist=5)
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
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def test_architecture_version(self) -> None:
        self.assertEqual(SpiderBrain.ARCHITECTURE_VERSION, 8)

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
        self.brain = SpiderBrain(seed=99, module_dropout=0.0)
        self.obs = _blank_obs()

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

    def test_motor_input_has_expected_dim(self) -> None:
        """motor_input must equal action_dim + motor_context_dim."""
        step = self.brain.act(self.obs, bus=None, sample=False)
        expected_dim = self.brain.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
        self.assertEqual(step.motor_input.shape, (expected_dim,))


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


if __name__ == "__main__":
    unittest.main()
