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

class FloatOrNoneTest(unittest.TestCase):
    """Tests for _float_or_none conversion helper."""

    def test_integer_value_converts(self) -> None:
        self.assertEqual(_float_or_none(3), 3.0)

    def test_float_value_passes_through(self) -> None:
        self.assertAlmostEqual(_float_or_none(2.5), 2.5)

    def test_numeric_string_converts(self) -> None:
        self.assertAlmostEqual(_float_or_none("3.14"), 3.14)

    def test_integer_string_converts(self) -> None:
        self.assertEqual(_float_or_none("42"), 42.0)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_float_or_none(None))

    def test_non_numeric_string_returns_none(self) -> None:
        self.assertIsNone(_float_or_none("abc"))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_float_or_none(""))

    def test_list_returns_none(self) -> None:
        self.assertIsNone(_float_or_none([1.0]))

    def test_dict_returns_none(self) -> None:
        self.assertIsNone(_float_or_none({"x": 1}))

    def test_zero_converts(self) -> None:
        self.assertEqual(_float_or_none(0), 0.0)

    def test_negative_value_converts(self) -> None:
        self.assertAlmostEqual(_float_or_none(-7.5), -7.5)

    def test_bool_true_converts_to_one(self) -> None:
        # Python bool is a subclass of int; float(True) == 1.0
        self.assertEqual(_float_or_none(True), 1.0)

    def test_bool_false_converts_to_zero(self) -> None:
        self.assertEqual(_float_or_none(False), 0.0)

    def test_return_type_is_float(self) -> None:
        result = _float_or_none(5)
        self.assertIsInstance(result, float)

class IntOrNoneTest(unittest.TestCase):
    """Tests for _int_or_none conversion helper."""

    def test_integer_value_passes_through(self) -> None:
        self.assertEqual(_int_or_none(7), 7)

    def test_integer_string_converts(self) -> None:
        self.assertEqual(_int_or_none("42"), 42)

    def test_negative_string_converts(self) -> None:
        self.assertEqual(_int_or_none("-3"), -3)

    def test_float_truncates_to_int(self) -> None:
        self.assertEqual(_int_or_none(3.9), 3)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_int_or_none(None))

    def test_non_numeric_string_returns_none(self) -> None:
        self.assertIsNone(_int_or_none("xyz"))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_int_or_none(""))

    def test_list_returns_none(self) -> None:
        self.assertIsNone(_int_or_none([1]))

    def test_float_string_returns_none(self) -> None:
        # int("3.5") raises ValueError
        self.assertIsNone(_int_or_none("3.5"))

    def test_zero_converts(self) -> None:
        self.assertEqual(_int_or_none(0), 0)

    def test_bool_true_converts_to_one(self) -> None:
        self.assertEqual(_int_or_none(True), 1)

    def test_return_type_is_int(self) -> None:
        result = _int_or_none(10)
        self.assertIsInstance(result, int)

class StatePositionTest(unittest.TestCase):
    """Tests for _state_position extraction from state mappings."""

    def test_extracts_x_y_keys(self) -> None:
        self.assertEqual(_state_position({"x": 3, "y": 5}), (3, 5))

    def test_x_y_as_strings(self) -> None:
        self.assertEqual(_state_position({"x": "2", "y": "4"}), (2, 4))

    def test_spider_pos_list(self) -> None:
        self.assertEqual(_state_position({"spider_pos": [6, 7]}), (6, 7))

    def test_spider_pos_tuple(self) -> None:
        self.assertEqual(_state_position({"spider_pos": (1, 2)}), (1, 2))

    def test_x_y_preferred_over_spider_pos(self) -> None:
        self.assertEqual(
            _state_position({"x": 10, "y": 11, "spider_pos": [0, 0]}), (10, 11)
        )

    def test_missing_x_falls_back_to_spider_pos(self) -> None:
        self.assertEqual(_state_position({"y": 5, "spider_pos": [3, 4]}), (3, 4))

    def test_missing_y_falls_back_to_spider_pos(self) -> None:
        self.assertEqual(_state_position({"x": 2, "spider_pos": [8, 9]}), (8, 9))

    def test_empty_dict_returns_none(self) -> None:
        self.assertIsNone(_state_position({}))

    def test_non_numeric_x_falls_back_to_spider_pos(self) -> None:
        self.assertEqual(
            _state_position({"x": "bad", "y": 5, "spider_pos": [1, 2]}), (1, 2)
        )

    def test_spider_pos_too_short_returns_none(self) -> None:
        self.assertIsNone(_state_position({"spider_pos": [5]}))

    def test_spider_pos_non_numeric_returns_none(self) -> None:
        self.assertIsNone(_state_position({"spider_pos": ["a", "b"]}))

    def test_spider_pos_none_returns_none(self) -> None:
        self.assertIsNone(_state_position({"spider_pos": None}))

    def test_zero_coordinates_extracted(self) -> None:
        self.assertEqual(_state_position({"x": 0, "y": 0}), (0, 0))

class TraceTickTest(unittest.TestCase):
    """Tests for _trace_tick priority and fallback logic."""

    def test_item_tick_takes_priority(self) -> None:
        item = {"tick": 5}
        state = {"tick": 99}
        self.assertEqual(_trace_tick(item, state, fallback=0), 5)

    def test_state_tick_used_when_item_has_none(self) -> None:
        item = {}
        state = {"tick": 7}
        self.assertEqual(_trace_tick(item, state, fallback=0), 7)

    def test_fallback_used_when_both_missing(self) -> None:
        self.assertEqual(_trace_tick({}, {}, fallback=42), 42)

    def test_item_tick_string_coerced(self) -> None:
        item = {"tick": "3"}
        self.assertEqual(_trace_tick(item, {}, fallback=0), 3)

    def test_state_tick_string_coerced(self) -> None:
        state = {"tick": "8"}
        self.assertEqual(_trace_tick({}, state, fallback=0), 8)

    def test_item_tick_non_numeric_falls_to_state(self) -> None:
        item = {"tick": "nan_value"}
        state = {"tick": 12}
        self.assertEqual(_trace_tick(item, state, fallback=0), 12)

    def test_both_non_numeric_returns_fallback(self) -> None:
        item = {"tick": "bad"}
        state = {"tick": "also_bad"}
        self.assertEqual(_trace_tick(item, state, fallback=99), 99)

    def test_item_tick_zero_is_valid(self) -> None:
        item = {"tick": 0}
        state = {"tick": 5}
        self.assertEqual(_trace_tick(item, state, fallback=99), 0)

class StateAliveTest(unittest.TestCase):
    """Tests for _state_alive liveness extraction."""

    def test_explicit_true_alive(self) -> None:
        self.assertTrue(_state_alive({"alive": True}))

    def test_explicit_false_alive(self) -> None:
        self.assertFalse(_state_alive({"alive": False}))

    def test_health_above_zero_is_alive(self) -> None:
        self.assertTrue(_state_alive({"health": 0.5}))

    def test_health_at_zero_is_dead(self) -> None:
        self.assertFalse(_state_alive({"health": 0.0}))

    def test_health_below_zero_is_dead(self) -> None:
        self.assertFalse(_state_alive({"health": -1.0}))

    def test_health_one_is_alive(self) -> None:
        self.assertTrue(_state_alive({"health": 1.0}))

    def test_empty_state_returns_none(self) -> None:
        self.assertIsNone(_state_alive({}))

    def test_no_relevant_keys_returns_none(self) -> None:
        self.assertIsNone(_state_alive({"x": 3, "y": 4}))

    def test_non_bool_alive_ignored_falls_to_health(self) -> None:
        # alive key is not a bool, should fall through to health
        self.assertTrue(_state_alive({"alive": "yes", "health": 0.8}))

    def test_non_bool_alive_no_health_returns_none(self) -> None:
        self.assertIsNone(_state_alive({"alive": 1}))

    def test_bool_alive_takes_priority_over_health(self) -> None:
        # explicit alive=False should override health > 0
        self.assertFalse(_state_alive({"alive": False, "health": 0.9}))

class StateFoodDistanceTest(unittest.TestCase):
    """Tests for _state_food_distance key priority and extraction."""

    def test_food_dist_key_used_first(self) -> None:
        state = {"food_dist": 5.0, "food_distance": 3.0, "nearest_food_dist": 1.0}
        self.assertEqual(_state_food_distance(state), 5.0)

    def test_food_distance_used_if_food_dist_missing(self) -> None:
        state = {"food_distance": 4.0, "nearest_food_dist": 2.0}
        self.assertEqual(_state_food_distance(state), 4.0)

    def test_nearest_food_dist_used_as_last_resort(self) -> None:
        state = {"nearest_food_dist": 7.5}
        self.assertEqual(_state_food_distance(state), 7.5)

    def test_all_keys_missing_returns_none(self) -> None:
        self.assertIsNone(_state_food_distance({"x": 3}))

    def test_empty_state_returns_none(self) -> None:
        self.assertIsNone(_state_food_distance({}))

    def test_non_numeric_value_skipped(self) -> None:
        state = {"food_dist": "far", "food_distance": 6.0}
        self.assertEqual(_state_food_distance(state), 6.0)

    def test_none_value_skipped(self) -> None:
        state = {"food_dist": None, "nearest_food_dist": 3.0}
        self.assertEqual(_state_food_distance(state), 3.0)

    def test_zero_distance_returned(self) -> None:
        self.assertEqual(_state_food_distance({"food_dist": 0.0}), 0.0)

class TraceDeathTickTest(unittest.TestCase):
    """Tests for _trace_death_tick first-death detection."""

    def test_empty_trace_returns_none(self) -> None:
        self.assertIsNone(_trace_death_tick([]))

    def test_always_alive_returns_none(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 1, "state": {"health": 0.8}},
        ]
        self.assertIsNone(_trace_death_tick(trace))

    def test_dead_from_start_returns_tick_zero(self) -> None:
        trace = [{"tick": 0, "state": {"health": 0.0}}]
        self.assertEqual(_trace_death_tick(trace), 0)

    def test_death_transition_detected(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 1, "state": {"health": 0.5}},
            {"tick": 2, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 2)

    def test_death_tick_from_item_used(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 5, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 5)

    def test_death_via_explicit_alive_false(self) -> None:
        trace = [
            {"tick": 0, "state": {"alive": True}},
            {"tick": 3, "state": {"alive": False}},
        ]
        self.assertEqual(_trace_death_tick(trace), 3)

    def test_items_without_state_skipped(self) -> None:
        trace = [
            {"tick": 0},
            {"tick": 1, "state": {"health": 1.0}},
            {"tick": 2, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 2)

    def test_states_without_alive_or_health_skipped(self) -> None:
        trace = [
            {"tick": 0, "state": {"x": 3}},
            {"tick": 1, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 1)

    def test_first_death_returned_not_subsequent(self) -> None:
        # Once dead stays dead; should return first tick
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 3, "state": {"health": 0.0}},
            {"tick": 5, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 3)

    def test_health_exactly_zero_counts_as_dead(self) -> None:
        trace = [{"tick": 0, "state": {"health": 0.0}}]
        self.assertEqual(_trace_death_tick(trace), 0)

    def test_health_epsilon_above_zero_counts_as_alive(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1e-9}},
        ]
        self.assertIsNone(_trace_death_tick(trace))

class Clamp01Test(unittest.TestCase):
    """Tests for the _clamp01 utility introduced in this PR."""

    def test_value_below_zero_returns_zero(self) -> None:
        self.assertEqual(_clamp01(-1.0), 0.0)

    def test_value_above_one_returns_one(self) -> None:
        self.assertEqual(_clamp01(2.5), 1.0)

    def test_exact_zero_returns_zero(self) -> None:
        self.assertEqual(_clamp01(0.0), 0.0)

    def test_exact_one_returns_one(self) -> None:
        self.assertEqual(_clamp01(1.0), 1.0)

    def test_midrange_value_unchanged(self) -> None:
        self.assertAlmostEqual(_clamp01(0.5), 0.5)

    def test_small_positive_value_unchanged(self) -> None:
        self.assertAlmostEqual(_clamp01(0.01), 0.01)

    def test_value_just_below_zero_returns_zero(self) -> None:
        self.assertEqual(_clamp01(-0.0001), 0.0)

    def test_value_just_above_one_returns_one(self) -> None:
        self.assertEqual(_clamp01(1.0001), 1.0)

    def test_returns_float(self) -> None:
        result = _clamp01(0.7)
        self.assertIsInstance(result, float)

    def test_negative_large_value_returns_zero(self) -> None:
        self.assertEqual(_clamp01(-1000.0), 0.0)

class TraceDominantPredatorTypesTest(unittest.TestCase):
    """Tests for _trace_dominant_predator_types from the new trace module."""

    def _make_meta_item(self, label: str) -> dict:
        return {"observation": {"meta": {"dominant_predator_type_label": label}}}

    def test_empty_trace_returns_empty_set(self) -> None:
        result = _trace_dominant_predator_types([])
        self.assertEqual(result, set())

    def test_visual_label_detected(self) -> None:
        trace = [self._make_meta_item("visual")]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("visual", result)

    def test_olfactory_label_detected(self) -> None:
        trace = [self._make_meta_item("olfactory")]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("olfactory", result)

    def test_both_labels_detected_across_items(self) -> None:
        trace = [
            self._make_meta_item("visual"),
            self._make_meta_item("olfactory"),
        ]
        result = _trace_dominant_predator_types(trace)
        self.assertEqual(result, {"visual", "olfactory"})

    def test_unknown_label_ignored(self) -> None:
        trace = [self._make_meta_item("thermal")]
        result = _trace_dominant_predator_types(trace)
        self.assertEqual(result, set())

    def test_case_normalized_to_lowercase(self) -> None:
        trace = [self._make_meta_item("VISUAL")]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("visual", result)

    def test_none_label_ignored(self) -> None:
        trace = [{"observation": {"meta": {"dominant_predator_type_label": None}}}]
        result = _trace_dominant_predator_types(trace)
        self.assertEqual(result, set())

    def test_message_meta_also_scanned(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "environment",
                        "topic": "observation",
                        "payload": {"meta": {"dominant_predator_type_label": "olfactory"}},
                    }
                ]
            }
        ]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("olfactory", result)

    def test_returns_set_type(self) -> None:
        result = _trace_dominant_predator_types([])
        self.assertIsInstance(result, set)
