from __future__ import annotations

import unittest

from spider_cortex_sim.metrics import EpisodeStats
from spider_cortex_sim.scenarios import get_scenario
from spider_cortex_sim.scenarios.trace import (
    _clamp01,
    _float_or_none,
    _food_signal_strength,
    _hunger_valence_rate,
    _int_or_none,
    _mapping_predator_visible,
    _memory_vector_freshness,
    _observation_food_distances,
    _payload_float,
    _payload_text,
    _predator_visible_flag,
    _state_alive,
    _state_food_distance,
    _state_position,
    _trace_action_selection_payloads,
    _trace_any_mode,
    _trace_any_sleep_phase,
    _extract_exposed_day_trace_metrics,
    _trace_death_tick,
    _trace_dominant_predator_types,
    _trace_escape_seen,
    _trace_food_distances,
    _trace_food_signal_strengths,
    _trace_post_observation_food_distances,
    _trace_predator_memory_seen,
    _trace_predator_visible,
    _trace_pre_observation_food_distances,
    _trace_shelter_cells,
    _trace_shelter_exit,
    _trace_states,
    _trace_tick,
)
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES, SpiderWorld

from tests.fixtures.scenario_trace_builders import (
    _make_episode_stats,
    _make_action_selection_trace_item,
    _make_food_deprivation_trace_item,
    _make_open_field_trace_item,
    _make_exposed_day_trace_item,
    _make_corridor_trace_item,
    _trace_positions_for_scenario,
    _open_field_trace_positions,
    _exposed_day_trace_positions,
    _corridor_trace_positions,
    _open_field_failure_base_metrics,
    _make_behavior_check_result,
    _make_behavioral_episode_score,
    _make_corridor_episode_stats,
)

class TraceHelpersTest(unittest.TestCase):
    """Tests for _trace_states, _trace_any_mode, _trace_any_sleep_phase,
    _trace_predator_memory_seen, _trace_escape_seen."""

    def test_trace_states_extracts_dict_states(self) -> None:
        trace = [
            {"tick": 0, "state": {"lizard_mode": "PATROL"}},
            {"tick": 1, "state": {"lizard_mode": "CHASE"}},
            {"tick": 2, "no_state_key": True},
        ]
        states = _trace_states(trace)
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0]["lizard_mode"], "PATROL")
        self.assertEqual(states[1]["lizard_mode"], "CHASE")

    def test_trace_states_skips_non_dict_state(self) -> None:
        trace = [
            {"state": "not_a_dict"},
            {"state": {"lizard_mode": "PATROL"}},
        ]
        states = _trace_states(trace)
        self.assertEqual(len(states), 1)

    def test_trace_states_empty_trace(self) -> None:
        self.assertEqual(_trace_states([]), [])

    def test_trace_any_mode_found(self) -> None:
        trace = [
            {"state": {"lizard_mode": "PATROL"}},
            {"state": {"lizard_mode": "CHASE"}},
        ]
        self.assertTrue(_trace_any_mode(trace, "CHASE"))

    def test_trace_any_mode_not_found(self) -> None:
        trace = [{"state": {"lizard_mode": "PATROL"}}]
        self.assertFalse(_trace_any_mode(trace, "RECOVER"))

    def test_trace_any_mode_empty_trace(self) -> None:
        self.assertFalse(_trace_any_mode([], "PATROL"))

    def test_trace_any_sleep_phase_found(self) -> None:
        trace = [
            {"state": {"sleep_phase": "AWAKE"}},
            {"state": {"sleep_phase": "DEEP_SLEEP"}},
        ]
        self.assertTrue(_trace_any_sleep_phase(trace, "DEEP_SLEEP"))

    def test_trace_any_sleep_phase_not_found(self) -> None:
        trace = [{"state": {"sleep_phase": "AWAKE"}}]
        self.assertFalse(_trace_any_sleep_phase(trace, "DEEP_SLEEP"))

    def test_trace_any_sleep_phase_empty(self) -> None:
        self.assertFalse(_trace_any_sleep_phase([], "DEEP_SLEEP"))

    def test_trace_predator_memory_seen_found(self) -> None:
        trace = [
            {"state": {"predator_memory": {"target": None}}},
            {"state": {"predator_memory": {"target": (3, 4)}}},
        ]
        self.assertTrue(_trace_predator_memory_seen(trace))

    def test_trace_predator_memory_seen_not_found_when_all_none(self) -> None:
        trace = [{"state": {"predator_memory": {"target": None}}}]
        self.assertFalse(_trace_predator_memory_seen(trace))

    def test_trace_predator_memory_seen_not_found_empty(self) -> None:
        self.assertFalse(_trace_predator_memory_seen([]))

    def test_trace_predator_memory_seen_non_dict_predator_memory(self) -> None:
        trace = [{"state": {"predator_memory": "not_a_dict"}}]
        self.assertFalse(_trace_predator_memory_seen(trace))

    def test_trace_escape_seen_true(self) -> None:
        trace = [
            {"predator_escape": False},
            {"predator_escape": True},
        ]
        self.assertTrue(_trace_escape_seen(trace))

    def test_trace_escape_seen_false(self) -> None:
        trace = [{"predator_escape": False}, {"predator_escape": False}]
        self.assertFalse(_trace_escape_seen(trace))

    def test_trace_escape_seen_missing_key(self) -> None:
        trace = [{"tick": 0}, {"tick": 1}]
        self.assertFalse(_trace_escape_seen(trace))

    def test_trace_escape_seen_empty(self) -> None:
        self.assertFalse(_trace_escape_seen([]))

    def test_trace_action_selection_payloads_extracts_action_center_messages(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {"selected_action": "MOVE_LEFT"},
                    },
                ]
            }
        ]
        payloads = _trace_action_selection_payloads(trace)
        self.assertEqual(payloads, [{"winning_valence": "threat"}])

    def test_payload_float_and_text_helpers(self) -> None:
        payload = {"winning_valence": "sleep", "module_gates": {"visual_cortex": 0.48}}
        self.assertEqual(_payload_text(payload, "winning_valence"), "sleep")
        self.assertAlmostEqual(_payload_float(payload, "module_gates", "visual_cortex"), 0.48)
        self.assertIsNone(_payload_text(payload, "missing"))
        self.assertIsNone(_payload_float(payload, "module_gates", "missing"))

class PredatorVisibleFlagTest(unittest.TestCase):
    """Unit tests for _predator_visible_flag type-coercion logic."""

    def test_bool_true_returns_true(self) -> None:
        self.assertIs(_predator_visible_flag(True), True)

    def test_bool_false_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(False), False)

    def test_numeric_at_threshold_returns_true(self) -> None:
        self.assertIs(_predator_visible_flag(1.0), True)
        self.assertIs(_predator_visible_flag(0.5), True)

    def test_numeric_below_threshold_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(0.49), False)
        self.assertIs(_predator_visible_flag(0.1), False)

    def test_zero_numeric_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(0.0), False)
        self.assertIs(_predator_visible_flag(0), False)

    def test_negative_numeric_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(-1.0), False)

    def test_string_true_tokens_return_true(self) -> None:
        for token in ("true", "yes", "visible"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), True)

    def test_string_true_tokens_case_insensitive(self) -> None:
        for token in ("TRUE", "YES", "VISIBLE", "True", "Yes", "Visible"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), True)

    def test_string_false_tokens_return_false(self) -> None:
        for token in ("false", "no", "hidden", "none"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), False)

    def test_string_false_tokens_case_insensitive(self) -> None:
        for token in ("FALSE", "NO", "HIDDEN", "NONE", "False", "No", "Hidden", "None"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), False)

    def test_string_with_whitespace_stripped(self) -> None:
        self.assertIs(_predator_visible_flag("  true  "), True)
        self.assertIs(_predator_visible_flag("  false  "), False)

    def test_unknown_string_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag("unknown"))
        self.assertIsNone(_predator_visible_flag("maybe"))
        self.assertIsNone(_predator_visible_flag(""))

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag(None))

    def test_list_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag([]))

    def test_dict_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag({}))

    def test_numeric_string_zero_returns_false(self) -> None:
        # "0" converts to float 0.0 via _float_or_none and returns False.
        self.assertIs(_predator_visible_flag("0"), False)

    def test_numeric_string_one_returns_true(self) -> None:
        # "1" converts to float 1.0 via _float_or_none and returns True.
        self.assertIs(_predator_visible_flag("1"), True)

    def test_numeric_string_below_threshold_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag("0.49"), False)

class MappingPredatorVisibleTest(unittest.TestCase):
    """Unit tests for _mapping_predator_visible nested mapping inspection."""

    def test_empty_mapping_returns_false(self) -> None:
        self.assertFalse(_mapping_predator_visible({}))

    def test_predator_visible_true(self) -> None:
        self.assertTrue(_mapping_predator_visible({"predator_visible": True}))

    def test_predator_visible_false_returns_false(self) -> None:
        self.assertFalse(_mapping_predator_visible({"predator_visible": False}))

    def test_prev_predator_visible_true(self) -> None:
        self.assertTrue(_mapping_predator_visible({"prev_predator_visible": True}))

    def test_prev_predator_visible_false_returns_false(self) -> None:
        self.assertFalse(_mapping_predator_visible({"prev_predator_visible": False}))

    def test_any_visible_flag_returns_true(self) -> None:
        self.assertTrue(
            _mapping_predator_visible({"predator_visible": False, "prev_predator_visible": True})
        )

    def test_meta_nested_predator_visible(self) -> None:
        source = {"meta": {"predator_visible": True}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_meta_non_mapping_is_ignored(self) -> None:
        self.assertFalse(_mapping_predator_visible({"meta": "string_meta"}))

    def test_vision_predator_visible_key(self) -> None:
        source = {"vision": {"predator": {"visible": True}}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_vision_predator_certainty_key(self) -> None:
        for certainty in (0.5, 1.0):
            with self.subTest(certainty=certainty):
                source = {"vision": {"predator": {"certainty": certainty}}}
                self.assertTrue(_mapping_predator_visible(source))

    def test_vision_predator_certainty_zero_is_false(self) -> None:
        for certainty in (0.0, 0.49):
            with self.subTest(certainty=certainty):
                source = {"vision": {"predator": {"certainty": certainty}}}
                self.assertFalse(_mapping_predator_visible(source))

    def test_meta_predator_visible_bool_is_honored(self) -> None:
        self.assertTrue(_mapping_predator_visible({"meta": {"predator_visible": True}}))

    def test_vision_non_mapping_predator_ignored(self) -> None:
        source = {"vision": {"predator": "patrol"}}
        self.assertFalse(_mapping_predator_visible(source))

    def test_vision_non_mapping_ignored(self) -> None:
        source = {"vision": "something"}
        self.assertFalse(_mapping_predator_visible(source))

    def test_evidence_threat_predator_visible(self) -> None:
        source = {"evidence": {"threat": {"predator_visible": 1.0}}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_evidence_threat_predator_not_visible(self) -> None:
        source = {"evidence": {"threat": {"predator_visible": 0.0}}}
        self.assertFalse(_mapping_predator_visible(source))

    def test_evidence_non_mapping_ignored(self) -> None:
        source = {"evidence": "some_string"}
        self.assertFalse(_mapping_predator_visible(source))

    def test_deeply_nested_via_evidence_threat(self) -> None:
        # evidence.threat.meta.predator_visible should also resolve
        source = {"evidence": {"threat": {"meta": {"predator_visible": True}}}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_unrecognized_keys_only_returns_false(self) -> None:
        source = {"food_visible": True, "hunger": 0.9}
        self.assertFalse(_mapping_predator_visible(source))

class TracePredatorVisibleTest(unittest.TestCase):
    """Unit tests for _trace_predator_visible trace-item inspection."""

    def test_empty_item_returns_false(self) -> None:
        self.assertFalse(_trace_predator_visible({}))

    def test_direct_predator_visible_true(self) -> None:
        self.assertTrue(_trace_predator_visible({"predator_visible": True}))

    def test_state_with_predator_visible(self) -> None:
        item = {"state": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_observation_with_predator_visible(self) -> None:
        item = {"observation": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_next_observation_with_predator_visible(self) -> None:
        item = {"next_observation": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_debug_with_predator_visible(self) -> None:
        item = {"debug": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_snapshot_with_predator_visible(self) -> None:
        item = {
            "event_log": [
                {
                    "stage": "pre_tick",
                    "name": "snapshot",
                    "payload": {"meta": {"predator_visible": True}},
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_snapshot_non_mapping_payload_ignored(self) -> None:
        item = {
            "event_log": [
                {
                    "stage": "pre_tick",
                    "name": "snapshot",
                    "payload": "not_a_mapping",
                }
            ]
        }
        self.assertFalse(_trace_predator_visible(item))

    def test_messages_payload_with_predator_visible(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"predator_visible": True},
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_messages_payload_meta_nested(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"predator_visible": True}},
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_non_mapping_message_ignored(self) -> None:
        item = {"messages": ["not_a_mapping"]}
        self.assertFalse(_trace_predator_visible(item))

    def test_message_without_payload_ignored(self) -> None:
        item = {"messages": [{"sender": "environment", "topic": "observation"}]}
        self.assertFalse(_trace_predator_visible(item))

    def test_evidence_threat_in_message_payload(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "action_center",
                    "topic": "action.selection",
                    "payload": {
                        "evidence": {"threat": {"predator_visible": 1.0}},
                    },
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_all_channels_absent_returns_false(self) -> None:
        item = {
            "state": {"x": 5, "y": 3, "health": 1.0},
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 6}},
                }
            ],
        }
        self.assertFalse(_trace_predator_visible(item))

    def test_state_non_mapping_ignored(self) -> None:
        item = {"state": "not_a_mapping"}
        self.assertFalse(_trace_predator_visible(item))

class PayloadFloatEdgeCasesTest(unittest.TestCase):
    """Comprehensive edge case tests for _payload_float."""

    def test_missing_top_level_key_returns_none(self) -> None:
        payload = {"other_key": 1.0}
        self.assertIsNone(_payload_float(payload, "missing_key"))

    def test_missing_nested_key_returns_none(self) -> None:
        payload = {"outer": {"inner": 0.5}}
        self.assertIsNone(_payload_float(payload, "outer", "missing"))

    def test_none_value_returns_none(self) -> None:
        payload = {"key": None}
        self.assertIsNone(_payload_float(payload, "key"))

    def test_non_numeric_string_returns_none(self) -> None:
        payload = {"key": "not_a_number"}
        self.assertIsNone(_payload_float(payload, "key"))

    def test_valid_int_returns_float(self) -> None:
        payload = {"key": 3}
        result = _payload_float(payload, "key")
        self.assertAlmostEqual(result, 3.0)
        self.assertIsInstance(result, float)

    def test_valid_float_returned(self) -> None:
        payload = {"key": 0.75}
        self.assertAlmostEqual(_payload_float(payload, "key"), 0.75)

    def test_empty_path_returns_none(self) -> None:
        # No path arguments -> current is the dict itself, float({}) raises TypeError
        payload = {"key": 1.0}
        self.assertIsNone(_payload_float(payload))

    def test_intermediate_not_dict_returns_none(self) -> None:
        payload = {"outer": "string_not_dict"}
        self.assertIsNone(_payload_float(payload, "outer", "inner"))

    def test_deep_nested_path(self) -> None:
        payload = {"a": {"b": {"c": 0.42}}}
        self.assertAlmostEqual(_payload_float(payload, "a", "b", "c"), 0.42)

    def test_bool_true_converts_to_one(self) -> None:
        payload = {"flag": True}
        self.assertAlmostEqual(_payload_float(payload, "flag"), 1.0)

    def test_bool_false_converts_to_zero(self) -> None:
        payload = {"flag": False}
        self.assertAlmostEqual(_payload_float(payload, "flag"), 0.0)

    def test_zero_float_value(self) -> None:
        payload = {"key": 0.0}
        self.assertEqual(_payload_float(payload, "key"), 0.0)

    def test_list_as_value_returns_none(self) -> None:
        payload = {"key": [1, 2, 3]}
        self.assertIsNone(_payload_float(payload, "key"))

class PayloadTextEdgeCasesTest(unittest.TestCase):
    """Comprehensive edge case tests for _payload_text."""

    def test_missing_top_level_key_returns_none(self) -> None:
        payload = {"other_key": "value"}
        self.assertIsNone(_payload_text(payload, "missing_key"))

    def test_none_value_returns_none(self) -> None:
        payload = {"key": None}
        self.assertIsNone(_payload_text(payload, "key"))

    def test_string_value_returned_as_is(self) -> None:
        payload = {"key": "threat"}
        self.assertEqual(_payload_text(payload, "key"), "threat")

    def test_int_value_converted_to_string(self) -> None:
        payload = {"key": 42}
        self.assertEqual(_payload_text(payload, "key"), "42")

    def test_float_value_converted_to_string(self) -> None:
        payload = {"key": 3.14}
        result = _payload_text(payload, "key")
        self.assertIsInstance(result, str)
        self.assertIn("3.14", result)

    def test_nested_path_retrieval(self) -> None:
        payload = {"outer": {"inner": "sleep"}}
        self.assertEqual(_payload_text(payload, "outer", "inner"), "sleep")

    def test_intermediate_not_dict_returns_none(self) -> None:
        payload = {"outer": "flat_string"}
        self.assertIsNone(_payload_text(payload, "outer", "inner"))

    def test_bool_true_converted_to_string(self) -> None:
        payload = {"key": True}
        self.assertEqual(_payload_text(payload, "key"), "True")

    def test_empty_path_returns_string_of_dict(self) -> None:
        # No path -> current is the full dict, str(dict) is returned
        payload = {"key": "val"}
        result = _payload_text(payload)
        self.assertIsInstance(result, str)
        self.assertIn("key", result)

    def test_missing_nested_key_returns_none(self) -> None:
        payload = {"outer": {"other": "value"}}
        self.assertIsNone(_payload_text(payload, "outer", "missing"))

class TraceActionSelectionPayloadsEdgeCasesTest(unittest.TestCase):
    """Edge cases for _trace_action_selection_payloads."""

    def test_empty_trace_returns_empty_list(self) -> None:
        self.assertEqual(_trace_action_selection_payloads([]), [])

    def test_item_without_messages_key_skipped(self) -> None:
        trace = [{"state": {"tick": 1}}]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_messages_not_a_list_skipped(self) -> None:
        trace = [{"messages": "not_a_list"}]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_non_dict_message_skipped(self) -> None:
        trace = [{"messages": ["not_a_dict", 42]}]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_wrong_sender_skipped(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "motor_cortex",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    }
                ]
            }
        ]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_wrong_topic_skipped(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.execution",
                        "payload": {"selected_action": "MOVE_UP"},
                    }
                ]
            }
        ]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_non_dict_payload_skipped(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": "not_a_dict",
                    }
                ]
            }
        ]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_multiple_matching_messages_all_returned(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    },
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "sleep"},
                    },
                ]
            }
        ]
        payloads = _trace_action_selection_payloads(trace)
        self.assertEqual(len(payloads), 2)

    def test_across_multiple_trace_items(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    }
                ]
            },
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "hunger"},
                    }
                ]
            },
        ]
        payloads = _trace_action_selection_payloads(trace)
        self.assertEqual(len(payloads), 2)
        valences = [p["winning_valence"] for p in payloads]
        self.assertIn("threat", valences)
        self.assertIn("hunger", valences)
