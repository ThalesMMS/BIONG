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

class ObservationFoodDistancesTest(unittest.TestCase):
    """Tests for _observation_food_distances extraction from trace items."""

    def test_observation_meta_food_dist(self) -> None:
        item = {"observation": {"meta": {"food_dist": 4.0}}}
        self.assertEqual(_observation_food_distances(item), [4.0])

    def test_next_observation_meta_food_dist(self) -> None:
        item = {"next_observation": {"meta": {"food_dist": 6.0}}}
        self.assertEqual(_observation_food_distances(item), [6.0])

    def test_both_observation_and_next_observation(self) -> None:
        item = {
            "observation": {"meta": {"food_dist": 5.0}},
            "next_observation": {"meta": {"food_dist": 3.0}},
        }
        self.assertEqual(_observation_food_distances(item), [5.0, 3.0])

    def test_environment_observation_message(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 8.0}},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [8.0])

    def test_non_environment_sender_ignored(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "action_center",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 3.0}},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [])

    def test_non_observation_topic_ignored(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "action.selection",
                    "payload": {"meta": {"food_dist": 3.0}},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [])

    def test_message_without_meta_ignored(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"other_key": 5.0},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [])

    def test_empty_item_returns_empty(self) -> None:
        self.assertEqual(_observation_food_distances({}), [])

    def test_observation_without_meta_skipped(self) -> None:
        item = {"observation": {"no_meta_here": 1.0}}
        self.assertEqual(_observation_food_distances(item), [])

    def test_combined_observations_and_messages(self) -> None:
        item = {
            "observation": {"meta": {"food_dist": 10.0}},
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 9.0}},
                }
            ],
        }
        distances = _observation_food_distances(item)
        self.assertIn(10.0, distances)
        self.assertIn(9.0, distances)

    def test_non_numeric_food_dist_skipped(self) -> None:
        item = {"observation": {"meta": {"food_dist": "far"}}}
        self.assertEqual(_observation_food_distances(item), [])

    def test_pre_and_post_helpers_keep_environment_messages_post_step(self) -> None:
        item = {
            "observation": {"meta": {"food_dist": 10.0}},
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 9.0}},
                }
            ],
        }
        self.assertEqual(_trace_pre_observation_food_distances(item), [10.0])
        self.assertEqual(_trace_post_observation_food_distances(item), [9.0])

class TraceFoodDistancesTest(unittest.TestCase):
    """Tests for _trace_food_distances aggregation over a full trace."""

    def test_empty_trace_returns_empty(self) -> None:
        self.assertEqual(_trace_food_distances([]), [])

    def test_single_item_observation_distance(self) -> None:
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        distances = _trace_food_distances(trace)
        self.assertIn(5.0, distances)

    def test_state_food_dist_included(self) -> None:
        trace = [{"state": {"food_dist": 7.0}}]
        distances = _trace_food_distances(trace)
        self.assertIn(7.0, distances)

    def test_delta_synthesizes_missing_post_observation_distance(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "distance_deltas": {"food": 2.0},
            }
        ]
        distances = _trace_food_distances(trace)
        # Should include original (10.0) and delta-adjusted (10.0 - 2.0 = 8.0)
        self.assertIn(10.0, distances)
        self.assertIn(8.0, distances)

    def test_environment_observation_counts_as_explicit_post_distance(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "distance_deltas": {"food": 2.0},
                "messages": [
                    {
                        "sender": "environment",
                        "topic": "observation",
                        "payload": {"meta": {"food_dist": 9.0}},
                    }
                ],
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [10.0, 9.0])
        self.assertNotIn(7.0, distances)
        self.assertNotIn(8.0, distances)

    def test_delta_does_not_adjust_explicit_next_observation(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "next_observation": {"meta": {"food_dist": 8.0}},
                "distance_deltas": {"food": 2.0},
            }
        ]
        self.assertEqual(_trace_food_distances(trace), [10.0, 8.0])

    def test_delta_does_not_synthesize_when_state_distance_is_present(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "state": {"food_dist": 8.0},
                "distance_deltas": {"food": 2.0},
            }
        ]
        self.assertEqual(_trace_food_distances(trace), [10.0, 8.0])

    def test_delta_can_be_read_from_info_payload(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "info": {"distance_deltas": {"food": 2.0}},
            }
        ]
        self.assertEqual(_trace_food_distances(trace), [10.0, 8.0])

    def test_delta_zero_preserves_unique_distance_once(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "distance_deltas": {"food": 0.0},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [5.0])

    def test_exposed_day_final_distance_uses_raw_trace_endpoint(self) -> None:
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
            {"observation": {"meta": {"food_dist": 5.0}}},
        ]
        self.assertEqual(_trace_food_distances(trace), [5.0, 4.0])
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 5.0)

    def test_no_delta_key_skips_delta_addition(self) -> None:
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        distances = _trace_food_distances(trace)
        self.assertEqual(len(distances), 1)
        self.assertEqual(distances[0], 5.0)

    def test_multiple_items_all_collected(self) -> None:
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"observation": {"meta": {"food_dist": 6.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(sorted(distances), [4.0, 6.0, 8.0])

    def test_non_food_delta_key_ignored(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "distance_deltas": {"shelter": 3.0},
            }
        ]
        distances = _trace_food_distances(trace)
        # No food delta, so no extra adjusted entry
        self.assertEqual(distances, [5.0])

    def test_item_without_state_does_not_crash(self) -> None:
        trace = [{"tick": 0}]
        self.assertEqual(_trace_food_distances(trace), [])

class TraceShelterCellsTest(unittest.TestCase):
    """Tests for _trace_shelter_cells map-template building."""

    def test_valid_central_burrow_template_returns_nonempty_set(self) -> None:
        state = {"map_template": "central_burrow"}
        cells = _trace_shelter_cells(state)
        self.assertGreater(len(cells), 0)
        self.assertIsInstance(cells, set)

    def test_all_cells_are_tuples_of_two_ints(self) -> None:
        state = {"map_template": "central_burrow"}
        cells = _trace_shelter_cells(state)
        for cell in cells:
            self.assertIsInstance(cell, tuple)
            self.assertEqual(len(cell), 2)
            self.assertIsInstance(cell[0], int)
            self.assertIsInstance(cell[1], int)

    def test_missing_map_template_returns_empty_set(self) -> None:
        self.assertEqual(_trace_shelter_cells({}), set())

    def test_non_string_map_template_returns_empty_set(self) -> None:
        self.assertEqual(_trace_shelter_cells({"map_template": 42}), set())

    def test_unknown_template_name_returns_empty_set(self) -> None:
        self.assertEqual(_trace_shelter_cells({"map_template": "no_such_map"}), set())

    def test_width_height_fallback_used_when_absent(self) -> None:
        state_with = {"map_template": "central_burrow", "width": 14, "height": 14}
        state_without = {"map_template": "central_burrow"}
        cells_with = _trace_shelter_cells(state_with)
        cells_without = _trace_shelter_cells(state_without)
        # Both should produce non-empty shelter sets
        self.assertGreater(len(cells_with), 0)
        self.assertGreater(len(cells_without), 0)

    def test_non_numeric_width_falls_back_to_default(self) -> None:
        state = {"map_template": "central_burrow", "width": "bad"}
        cells = _trace_shelter_cells(state)
        self.assertGreater(len(cells), 0)

class TraceShelterExitTest(unittest.TestCase):
    """Tests for _trace_shelter_exit shelter-departure detection."""

    def test_empty_trace_returns_no_exit(self) -> None:
        escaped, tick = _trace_shelter_exit([])
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_no_state_in_items_returns_no_exit(self) -> None:
        trace = [{"tick": 0}, {"tick": 1}]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_shelter_role_outside_detected(self) -> None:
        trace = [
            {"tick": 0, "state": {"shelter_role": "inside"}},
            {"tick": 3, "state": {"shelter_role": "outside"}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 3)

    def test_always_inside_returns_no_exit(self) -> None:
        trace = [
            {"tick": 0, "state": {"shelter_role": "inside"}},
            {"tick": 1, "state": {"shelter_role": "inside"}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_position_outside_shelter_detected(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        shelter = world.shelter_cells
        outside = next(
            cell
            for cell in world.map_template.traversable_cells
            if cell not in shelter
        )
        deep = sorted(world.shelter_deep_cells)[0]
        trace = [
            {
                "tick": 0,
                "state": {
                    "x": deep[0],
                    "y": deep[1],
                    "map_template": "central_burrow",
                },
            },
            {
                "tick": 2,
                "state": {
                    "x": outside[0],
                    "y": outside[1],
                    "map_template": "central_burrow",
                },
            },
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 2)

    def test_position_inside_shelter_no_exit(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        deep = sorted(world.shelter_deep_cells)[0]
        trace = [
            {
                "tick": 0,
                "state": {
                    "x": deep[0],
                    "y": deep[1],
                    "map_template": "central_burrow",
                },
            },
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_tick_from_item_used_for_exit(self) -> None:
        trace = [
            {"tick": 10, "state": {"shelter_role": "outside"}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 10)

    def test_tick_from_state_used_when_item_has_none(self) -> None:
        trace = [
            {"state": {"shelter_role": "outside", "tick": 7}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 7)

class HungerValenceRateTest(unittest.TestCase):
    """Tests for _hunger_valence_rate fraction computation."""

    def test_empty_trace_returns_zero(self) -> None:
        self.assertEqual(_hunger_valence_rate([]), 0.0)

    def test_no_action_selection_messages_returns_zero(self) -> None:
        trace = [{"tick": 0, "state": {}}]
        self.assertEqual(_hunger_valence_rate(trace), 0.0)

    def test_all_hunger_returns_one(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 1.0)

    def test_no_hunger_returns_zero(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
            _make_action_selection_trace_item({"winning_valence": "exploration"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 0.0)

    def test_mixed_half_and_half(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 0.5)

    def test_one_of_three_hunger(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 1 / 3)

    def test_return_type_is_float(self) -> None:
        trace = [_make_action_selection_trace_item({"winning_valence": "hunger"})]
        result = _hunger_valence_rate(trace)
        self.assertIsInstance(result, float)

    def test_missing_winning_valence_counts_as_non_hunger(self) -> None:
        trace = [
            _make_action_selection_trace_item({}),
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 0.5)

class MemoryVectorFreshnessTest(unittest.TestCase):
    """Tests for _memory_vector_freshness introduced in this PR."""

    def test_non_mapping_returns_zero(self) -> None:
        self.assertEqual(_memory_vector_freshness(None), 0.0)
        self.assertEqual(_memory_vector_freshness(42), 0.0)
        self.assertEqual(_memory_vector_freshness("string"), 0.0)
        self.assertEqual(_memory_vector_freshness([1, 2]), 0.0)

    def test_zero_displacement_returns_zero(self) -> None:
        self.assertEqual(_memory_vector_freshness({"dx": 0.0, "dy": 0.0, "age": 0.1}), 0.0)

    def test_zero_dx_nonzero_dy_uses_displacement(self) -> None:
        result = _memory_vector_freshness({"dx": 0.0, "dy": 1.0, "age": 0.2})
        self.assertAlmostEqual(result, 0.8)

    def test_missing_age_returns_zero(self) -> None:
        self.assertEqual(_memory_vector_freshness({"dx": 1.0, "dy": 0.0}), 0.0)

    def test_age_zero_returns_one(self) -> None:
        result = _memory_vector_freshness({"dx": 0.5, "dy": 0.5, "age": 0.0})
        self.assertAlmostEqual(result, 1.0)

    def test_age_one_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 0.5, "dy": 0.5, "age": 1.0})
        self.assertAlmostEqual(result, 0.0)

    def test_age_in_range_returns_one_minus_age(self) -> None:
        result = _memory_vector_freshness({"dx": 0.5, "dy": 0.5, "age": 0.3})
        self.assertAlmostEqual(result, 0.7)

    def test_age_above_one_with_no_ttl_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 5.0})
        self.assertEqual(result, 0.0)

    def test_age_above_one_with_ttl_zero_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 5.0, "ttl": 0.0})
        self.assertEqual(result, 0.0)

    def test_age_above_one_with_valid_ttl(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 4.0, "ttl": 8.0})
        self.assertAlmostEqual(result, 0.5)

    def test_age_greater_than_ttl_clamped_to_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 12.0, "ttl": 8.0})
        self.assertEqual(result, 0.0)

    def test_negative_ttl_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 3.0, "ttl": -1.0})
        self.assertEqual(result, 0.0)

class FoodSignalStrengthTest(unittest.TestCase):
    """Tests for _food_signal_strength introduced in this PR."""

    def test_empty_mapping_returns_zero(self) -> None:
        self.assertEqual(_food_signal_strength({}), 0.0)

    def test_food_visible_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_visible": 0.6}), 0.6)

    def test_food_certainty_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_certainty": 0.4}), 0.4)

    def test_food_smell_strength_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_smell_strength": 0.7}), 0.7)

    def test_food_trace_strength_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_trace_strength": 0.55}), 0.55)

    def test_food_memory_freshness_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_memory_freshness": 0.3}), 0.3)

    def test_value_above_one_is_clamped(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_visible": 2.5}), 1.0)

    def test_value_below_zero_is_clamped(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_visible": -0.5}), 0.0)

    def test_returns_max_of_multiple_direct_keys(self) -> None:
        source = {"food_visible": 0.3, "food_smell_strength": 0.8, "food_certainty": 0.5}
        self.assertAlmostEqual(_food_signal_strength(source), 0.8)

    def test_evidence_hunger_nesting(self) -> None:
        source = {"evidence": {"hunger": {"food_visible": 0.65}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.65)

    def test_evidence_visual_nesting(self) -> None:
        source = {"evidence": {"visual": {"food_certainty": 0.45}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.45)

    def test_evidence_sensory_nesting(self) -> None:
        source = {"evidence": {"sensory": {"food_smell_strength": 0.72}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.72)

    def test_meta_nesting(self) -> None:
        source = {"meta": {"food_visible": 0.55}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.55)

    def test_vision_food_visible(self) -> None:
        source = {"vision": {"food": {"visible": 0.88}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.88)

    def test_vision_food_certainty(self) -> None:
        source = {"vision": {"food": {"certainty": 0.77}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.77)

    def test_percept_traces_food_strength(self) -> None:
        source = {"percept_traces": {"food": {"strength": 0.6}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.6)

    def test_memory_vectors_food_freshness(self) -> None:
        source = {"memory_vectors": {"food": {"dx": 1.0, "dy": 0.0, "age": 0.4}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.6)

    def test_food_trace_state_key(self) -> None:
        source = {"food_trace": {"strength": 0.5, "dx": 1.0, "dy": 0.0}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.5)

    def test_food_memory_with_target_age_ttl(self) -> None:
        source = {"food_memory": {"target": (5, 3), "age": 3.0, "ttl": 12.0}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.75)

    def test_food_memory_without_target_ignored(self) -> None:
        source = {"food_memory": {"target": None, "age": 1.0, "ttl": 12.0}}
        self.assertEqual(_food_signal_strength(source), 0.0)

    def test_max_across_all_nested_sources(self) -> None:
        source = {
            "food_visible": 0.2,
            "meta": {"food_smell_strength": 0.9},
            "evidence": {"hunger": {"food_certainty": 0.5}},
        }
        self.assertAlmostEqual(_food_signal_strength(source), 0.9)

    def test_vision_food_not_mapping_ignored(self) -> None:
        source = {"vision": {"food": "not_a_mapping"}}
        self.assertEqual(_food_signal_strength(source), 0.0)

    def test_percept_traces_food_not_mapping_ignored(self) -> None:
        source = {"percept_traces": {"food": "absent"}}
        self.assertEqual(_food_signal_strength(source), 0.0)

class TraceFoodSignalStrengthsTest(unittest.TestCase):
    """Tests for _trace_food_signal_strengths introduced in this PR."""

    def test_empty_trace_returns_empty_list(self) -> None:
        self.assertEqual(_trace_food_signal_strengths([]), [])

    def test_single_item_with_no_signal_returns_zero(self) -> None:
        result = _trace_food_signal_strengths([{"tick": 0}])
        self.assertEqual(result, [0.0])

    def test_signal_from_state(self) -> None:
        item = {"state": {"food_visible": 0.7}}
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.7)

    def test_signal_from_observation_key(self) -> None:
        item = {"observation": {"food_certainty": 0.6}}
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.6)

    def test_signal_from_next_observation_key(self) -> None:
        item = {"next_observation": {"food_smell_strength": 0.5}}
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.5)

    def test_signal_from_message_payload(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"food_visible": 0.8},
                }
            ]
        }
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.8)

    def test_per_tick_maximum_across_sources(self) -> None:
        item = {
            "state": {"food_visible": 0.3},
            "observation": {"food_smell_strength": 0.9},
        }
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.9)

    def test_multiple_items_aligned_to_trace(self) -> None:
        items = [
            {"state": {"food_visible": 0.5}},
            {"state": {"food_visible": 0.0}},
            {"state": {"food_visible": 0.8}},
        ]
        result = _trace_food_signal_strengths(items)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.8)

    def test_message_with_non_mapping_payload_ignored(self) -> None:
        item = {
            "messages": [
                {"sender": "env", "topic": "obs", "payload": "not_a_mapping"},
            ]
        }
        result = _trace_food_signal_strengths([item])
        self.assertEqual(result[0], 0.0)

    def test_messages_not_a_list_ignored(self) -> None:
        item = {"messages": "invalid"}
        result = _trace_food_signal_strengths([item])
        self.assertEqual(result[0], 0.0)

    def test_full_open_field_trace_item_signal(self) -> None:
        item = _make_open_field_trace_item(
            tick=0,
            pos=(4, 5),
            health=1.0,
            food_dist=6,
            food_signal_strength=0.6,
        )
        result = _trace_food_signal_strengths([item])
        self.assertGreater(result[0], 0.0)

    def test_full_open_field_trace_item_no_signal(self) -> None:
        item = _make_open_field_trace_item(
            tick=0,
            pos=(4, 5),
            health=1.0,
            food_dist=6,
            food_signal_strength=0.0,
        )
        result = _trace_food_signal_strengths([item])
        self.assertEqual(result[0], 0.0)

class TraceFoodDistancesDeduplicationTest(unittest.TestCase):
    """Tests for the new deduplication behavior in _trace_food_distances (dict.fromkeys)."""

    def test_exact_duplicates_removed(self) -> None:
        """Same distance appearing in observation and state should appear only once."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "state": {"food_dist": 5.0},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances.count(5.0), 1)

    def test_distinct_values_all_kept(self) -> None:
        """Different distances are preserved in order."""
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"observation": {"meta": {"food_dist": 6.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [8.0, 6.0, 4.0])

    def test_repeated_value_across_items_deduplicated(self) -> None:
        """Same distance seen in multiple items appears only at first occurrence."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
            {"observation": {"meta": {"food_dist": 5.0}}},  # duplicate of first
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [5.0, 4.0])
        self.assertEqual(distances.count(5.0), 1)

    def test_insertion_order_preserved(self) -> None:
        """First-seen order is preserved after deduplication."""
        trace = [
            {"observation": {"meta": {"food_dist": 10.0}}},
            {"observation": {"meta": {"food_dist": 7.0}}},
            {"observation": {"meta": {"food_dist": 10.0}}},
            {"observation": {"meta": {"food_dist": 3.0}}},
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [10.0, 7.0, 3.0])

    def test_synthesized_distance_same_as_explicit_deduplicated(self) -> None:
        """Synthesized post-distance equal to observation is deduplicated."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "distance_deltas": {"food": 0.0},  # synthesized post = 5.0 - 0.0 = 5.0
            }
        ]
        distances = _trace_food_distances(trace)
        # 5.0 (observation) + synthesized 5.0 -> deduplicated to [5.0]
        self.assertEqual(distances, [5.0])

    def test_empty_trace_returns_empty_list(self) -> None:
        self.assertEqual(_trace_food_distances([]), [])

    def test_return_type_is_list(self) -> None:
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        result = _trace_food_distances(trace)
        self.assertIsInstance(result, list)

    def test_observation_and_next_observation_same_value_deduplicated(self) -> None:
        """Same distance in observation and next_observation deduplicated."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 6.0}},
                "next_observation": {"meta": {"food_dist": 6.0}},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances.count(6.0), 1)

    def test_regression_observation_state_two_unique_distances(self) -> None:
        """Observation and state with different values both appear."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 8.0}},
                "state": {"food_dist": 7.0},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertIn(8.0, distances)
        self.assertIn(7.0, distances)

class ExtractExposedDayTraceMetricsFinalDistanceTest(unittest.TestCase):
    """Tests for final_distance_to_food in _extract_exposed_day_trace_metrics.

    In the new implementation, final_distance_to_food uses reversed(trace) to find
    the last trace item that has food distance data, not _trace_food_distances[-1].
    """

    def test_final_distance_is_raw_last_item_value(self) -> None:
        """final_distance_to_food is from the last trace item, not the last unique value."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
            {"observation": {"meta": {"food_dist": 5.0}}},  # last item, same as first
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        # The last item has food_dist=5.0, so that's the final distance
        self.assertEqual(metrics["final_distance_to_food"], 5.0)

    def test_final_distance_different_from_deduplicated_last(self) -> None:
        """Demonstrates that final_distance_to_food differs from _trace_food_distances[-1]."""
        trace = [
            {"observation": {"meta": {"food_dist": 10.0}}},
            {"observation": {"meta": {"food_dist": 7.0}}},
            {"observation": {"meta": {"food_dist": 10.0}}},  # repeat, dedup would give 7.0 as last
        ]
        food_distances = _trace_food_distances(trace)
        # Dedup gives [10.0, 7.0] - so [-1] would be 7.0
        self.assertEqual(food_distances[-1], 7.0)
        # But final_distance_to_food uses the actual last item with data: 10.0
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 10.0)

    def test_final_distance_is_none_when_no_food_data(self) -> None:
        """Returns None when no food distance info exists in any trace item."""
        trace = [{"tick": 0}, {"tick": 1}]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertIsNone(metrics["final_distance_to_food"])

    def test_final_distance_from_single_item(self) -> None:
        """Single trace item with food_dist returns that distance."""
        trace = [{"observation": {"meta": {"food_dist": 9.0}}}]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 9.0)

    def test_final_distance_uses_last_item_with_food_data(self) -> None:
        """Uses last item that has food data even if some items lack it."""
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"tick": 1},  # no food data
            {"tick": 2},  # no food data
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 8.0)

    def test_final_distance_from_next_observation(self) -> None:
        """final_distance_to_food can come from next_observation meta."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"next_observation": {"meta": {"food_dist": 3.0}}},
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 3.0)

    def test_final_distance_from_state(self) -> None:
        """final_distance_to_food can come from state field."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"state": {"food_dist": 2.0}},
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 2.0)

    def test_other_metrics_keys_present(self) -> None:
        """All expected metric keys are present in the result."""
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        metrics = _extract_exposed_day_trace_metrics(trace)
        for key in ("shelter_exit_tick", "left_shelter", "peak_food_progress",
                    "predator_visible_ticks", "final_distance_to_food"):
            self.assertIn(key, metrics)

    def test_monotonically_decreasing_distances_final_is_minimum(self) -> None:
        """For strictly decreasing distances, final is the minimum."""
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"observation": {"meta": {"food_dist": 6.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 4.0)
