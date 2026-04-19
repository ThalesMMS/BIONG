import unittest
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import canonical_ablation_configs
from spider_cortex_sim.comparison import (
    build_ablation_deltas,
    build_learning_evidence_deltas,
    build_learning_evidence_summary,
    build_predator_type_specialization_summary,
    build_reward_audit,
    build_reward_audit_comparison,
    austere_survival_gate_passed,
    compare_ablation_suite,
    compare_behavior_suite,
    compare_configurations,
    compare_learning_evidence,
    compare_noise_robustness,
    compare_reward_profiles,
    compare_training_regimes,
    condition_compact_summary,
    profile_comparison_metrics,
)
from spider_cortex_sim.curriculum import CURRICULUM_FOCUS_SCENARIOS
from spider_cortex_sim.learning_evidence import LearningEvidenceConditionSpec
from spider_cortex_sim.noise import RobustnessMatrixSpec
from spider_cortex_sim.reward import SCENARIO_AUSTERE_REQUIREMENTS, SHAPING_GAP_POLICY
from spider_cortex_sim.simulation import SpiderSimulation

class BuildRewardAuditTest(unittest.TestCase):
    """Tests for comparison.build_reward_audit and related helpers."""

    def test_build_reward_audit_returns_required_top_level_keys(self) -> None:
        audit = build_reward_audit()
        expected_keys = {
            "current_profile", "minimal_profile", "reward_components",
            "observation_signals", "memory_signals", "reward_profiles", "notes",
        }
        self.assertTrue(expected_keys.issubset(set(audit.keys())))

    def test_build_reward_audit_current_profile_recorded(self) -> None:
        audit = build_reward_audit(current_profile="classic")
        self.assertEqual(audit["current_profile"], "classic")

    def test_build_reward_audit_none_current_profile(self) -> None:
        audit = build_reward_audit(current_profile=None)
        self.assertIsNone(audit["current_profile"])

    def test_build_reward_audit_minimal_profile_is_austere(self) -> None:
        audit = build_reward_audit()
        self.assertEqual(audit["minimal_profile"], "austere")

    def test_build_reward_audit_reward_profiles_contains_all_known_profiles(self) -> None:
        from spider_cortex_sim.reward import REWARD_PROFILES
        audit = build_reward_audit()
        for profile_name in REWARD_PROFILES:
            self.assertIn(
                profile_name,
                audit["reward_profiles"],
                f"Profile {profile_name!r} missing from reward_audit['reward_profiles']",
            )

    def test_build_reward_audit_observation_signals_contains_predator_dist(self) -> None:
        audit = build_reward_audit()
        self.assertIn("predator_dist", audit["observation_signals"])

    def test_build_reward_audit_memory_signals_contains_shelter_memory(self) -> None:
        audit = build_reward_audit()
        self.assertIn("shelter_memory", audit["memory_signals"])

    def test_build_reward_audit_with_comparison_payload_includes_comparison(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.6,
                        "episode_success_rate": 0.55,
                    }
                },
            }
        }
        audit = build_reward_audit(comparison_payload=payload)
        self.assertIn("comparison", audit)

    def test_build_reward_audit_without_comparison_payload_no_comparison_key(self) -> None:
        audit = build_reward_audit()
        self.assertNotIn("comparison", audit)

    def test_build_reward_audit_notes_is_non_empty_list(self) -> None:
        audit = build_reward_audit()
        self.assertIsInstance(audit["notes"], list)
        self.assertGreater(len(audit["notes"]), 0)

    def test_build_reward_audit_reward_components_contains_food_progress(self) -> None:
        audit = build_reward_audit()
        self.assertIn("food_progress", audit["reward_components"])

class BuildRewardAuditComparisonTest(unittest.TestCase):
    """Tests for comparison.build_reward_audit_comparison."""

    def _payload_with_austere_suite(self) -> dict[str, object]:
        """
        Builds a test payload containing two reward profiles, `classic` and `austere`, each with summary metrics and per-scenario suite results.

        The returned dictionary has the top-level key `"reward_profiles"` mapping profile names to objects with:
        - `summary`: contains `scenario_success_rate` and `episode_success_rate`.
        - `suite`: maps scenario names to `{ "success_rate": float, "episodes": int }` entries.

        Returns:
            payload (dict[str, object]): A payload suitable for testing reward-audit comparison logic, including `classic` and `austere` profiles with differing per-scenario outcomes.
        """
        return {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.9,
                        "episode_success_rate": 0.9,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        "open_field_foraging": {"success_rate": 1.0, "episodes": 2},
                    },
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.5,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        "open_field_foraging": {"success_rate": 0.0, "episodes": 2},
                    },
                },
            }
        }

    def test_returns_none_for_none_input(self) -> None:
        result = build_reward_audit_comparison(None)
        self.assertIsNone(result)

    def test_returns_none_for_non_dict_input(self) -> None:
        result = build_reward_audit_comparison("not_a_dict")
        self.assertIsNone(result)

    def test_returns_none_when_reward_profiles_missing(self) -> None:
        result = build_reward_audit_comparison({"other_key": {}})
        self.assertIsNone(result)

    def test_returns_none_when_reward_profiles_empty(self) -> None:
        result = build_reward_audit_comparison({"reward_profiles": {}})
        self.assertIsNone(result)

    def test_returns_none_when_reward_profiles_not_dict(self) -> None:
        result = build_reward_audit_comparison({"reward_profiles": "bad"})
        self.assertIsNone(result)

    def test_minimal_profile_is_none_when_austere_absent(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.7,
                        "episode_success_rate": 0.6,
                    }
                }
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertIsNotNone(result)
        self.assertIsNone(result["minimal_profile"])
        self.assertEqual(result["deltas_vs_minimal"], {})

    def test_minimal_profile_is_austere_when_present(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertEqual(result["minimal_profile"], "austere")

    def test_deltas_vs_minimal_computed_for_all_profiles(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertIn("classic", result["deltas_vs_minimal"])
        self.assertIn("austere", result["deltas_vs_minimal"])

    def test_delta_keys_are_correct(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        delta_keys = set(result["deltas_vs_minimal"]["classic"].keys())
        self.assertEqual(
            delta_keys,
            {"scenario_success_rate_delta", "episode_success_rate_delta", "mean_reward_delta"},
        )

    def test_austere_delta_vs_itself_is_zero(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        austere_delta = result["deltas_vs_minimal"]["austere"]
        self.assertAlmostEqual(austere_delta["scenario_success_rate_delta"], 0.0)
        self.assertAlmostEqual(austere_delta["episode_success_rate_delta"], 0.0)
        self.assertAlmostEqual(austere_delta["mean_reward_delta"], 0.0)

    def test_classic_delta_vs_austere_is_positive_when_classic_has_higher_rate(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        classic_delta = result["deltas_vs_minimal"]["classic"]
        self.assertAlmostEqual(classic_delta["scenario_success_rate_delta"], 0.3)
        self.assertAlmostEqual(classic_delta["episode_success_rate_delta"], 0.3)

    def test_result_profiles_are_sorted(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {"summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.75}},
                "austere": {"summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.45}},
                "ecological": {"summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.65}},
            }
        }
        result = build_reward_audit_comparison(payload)
        profile_keys = list(result["profiles"].keys())
        self.assertEqual(profile_keys, sorted(profile_keys))

    def test_result_has_notes_list(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {"summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.75}},
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertIsInstance(result["notes"], list)
        self.assertGreater(len(result["notes"]), 0)

    def test_behavior_survival_field_present_with_austere_profile(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertIn("behavior_survival", result)
        self.assertTrue(result["behavior_survival"]["available"])
        self.assertIn("austere_survival_summary", result)
        self.assertIn("gap_policy_check", result)
        self.assertIn("shaping_dependent_behaviors", result)

    def test_behavior_survival_can_be_derived_from_episode_details(self) -> None:
        result = build_reward_audit_comparison(
            {
                "reward_profiles": {
                    "classic": {
                        "summary": {
                            "scenario_success_rate": 1.0,
                            "episode_success_rate": 1.0,
                        },
                        "episodes_detail": [
                            {
                                "scenario": "night_rest",
                                "alive": True,
                                "total_reward": 1.0,
                            }
                        ],
                    },
                    "austere": {
                        "summary": {
                            "scenario_success_rate": 1.0,
                            "episode_success_rate": 1.0,
                        },
                        "episodes_detail": [
                            {
                                "scenario": "night_rest",
                                "alive": True,
                                "total_reward": 0.5,
                            }
                        ],
                    },
                }
            }
        )
        self.assertTrue(result["behavior_survival"]["available"])
        self.assertEqual(
            result["behavior_survival"]["scenarios"]["night_rest"][
                "austere_success_rate"
            ],
            1.0,
        )

    def test_austere_survival_summary_reports_gate_counts(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        summary = result["austere_survival_summary"]
        self.assertTrue(summary["available"])
        self.assertAlmostEqual(summary["overall_survival_rate"], 0.5)
        self.assertEqual(summary["gate_pass_count"], 1)
        self.assertEqual(summary["gate_fail_count"], 0)
        self.assertEqual(summary["observed_gate_count"], 1)
        self.assertFalse(summary["gate_coverage_complete"])

    def test_gap_policy_check_is_promoted_in_comparison(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertTrue(result["gap_policy_check"]["violations"])
        self.assertEqual(
            result["austere_survival_summary"]["gap_policy_violations"],
            result["gap_policy_check"]["violations"],
        )

    def test_shaping_dependent_behaviors_reports_scenario_gaps(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        dependent = result["shaping_dependent_behaviors"]
        self.assertEqual(len(dependent), 1)
        self.assertEqual(dependent[0]["scenario"], "open_field_foraging")
        self.assertEqual(dependent[0]["profile"], "classic")
        self.assertAlmostEqual(dependent[0]["success_rate_delta"], 1.0)

    def test_behavior_survival_flag_true_above_threshold(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        survival = result["behavior_survival"]
        self.assertTrue(survival["scenarios"]["night_rest"]["survives"])
        self.assertFalse(survival["scenarios"]["open_field_foraging"]["survives"])

    def test_behavior_survival_rate_summary_is_computed(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertAlmostEqual(result["behavior_survival"]["survival_rate"], 0.5)
        self.assertAlmostEqual(result["survival_rate"], 0.5)

    def test_behavior_survival_reports_austere_scenario_success(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        survival = result["behavior_survival"]

        self.assertTrue(survival["available"])
        self.assertEqual(survival["scenario_count"], 2)
        self.assertEqual(survival["surviving_scenario_count"], 1)
        self.assertAlmostEqual(survival["survival_rate"], 0.5)
        self.assertAlmostEqual(result["survival_rate"], 0.5)
        self.assertTrue(survival["scenarios"]["night_rest"]["survives"])
        self.assertFalse(survival["scenarios"]["open_field_foraging"]["survives"])

    def test_behavior_survival_threshold_is_minimal_shaping_survival_threshold(self) -> None:
        from spider_cortex_sim.reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertAlmostEqual(
            result["behavior_survival"]["survival_threshold"],
            MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )

    def test_behavior_survival_minimal_profile_set_correctly(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertEqual(result["behavior_survival"]["minimal_profile"], "austere")

    def test_behavior_survival_not_available_when_no_austere_profile(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 0.8, "episodes": 2},
                    },
                },
                "ecological": {
                    "summary": {
                        "scenario_success_rate": 0.7,
                        "episode_success_rate": 0.65,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 0.7, "episodes": 2},
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertFalse(result["behavior_survival"]["available"])
        self.assertAlmostEqual(result["behavior_survival"]["survival_rate"], 0.0)

    def test_behavior_survival_not_available_when_austere_has_no_suite(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 0.8, "episodes": 2},
                    },
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    },
                    # no "suite" key
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertFalse(result["behavior_survival"]["available"])

    def test_behavior_survival_scenario_exactly_at_threshold_survives(self) -> None:
        from spider_cortex_sim.reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.8},
                    "suite": {"night_rest": {"success_rate": 0.8, "episodes": 2}},
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                        "episode_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                    },
                    "suite": {
                        "night_rest": {
                            "success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                            "episodes": 2,
                        }
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        survival = result["behavior_survival"]
        self.assertTrue(survival["scenarios"]["night_rest"]["survives"])

    def test_behavior_survival_scenario_just_below_threshold_does_not_survive(self) -> None:
        from spider_cortex_sim.reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.8},
                    "suite": {"night_rest": {"success_rate": 0.8, "episodes": 2}},
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD - 0.01,
                        "episode_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD - 0.01,
                    },
                    "suite": {
                        "night_rest": {
                            "success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD - 0.01,
                            "episodes": 2,
                        }
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        survival = result["behavior_survival"]
        self.assertFalse(survival["scenarios"]["night_rest"]["survives"])

    def test_behavior_survival_episodes_count_captured(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        survival = result["behavior_survival"]
        self.assertEqual(survival["scenarios"]["night_rest"]["episodes"], 2)
        self.assertEqual(survival["scenarios"]["open_field_foraging"]["episodes"], 2)

    def test_survival_rate_top_level_matches_behavior_survival_rate(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertAlmostEqual(
            result["survival_rate"],
            result["behavior_survival"]["survival_rate"],
        )

    def test_behavior_survival_scenario_count_uses_austere_suite_only(self) -> None:
        # Classic has an additional scenario not in austere suite
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.9},
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        "extra_scenario": {"success_rate": 1.0, "episodes": 2},
                    },
                },
                "austere": {
                    "summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.5},
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        # extra_scenario present in classic but missing from austere suite
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        survival = result["behavior_survival"]
        self.assertEqual(survival["scenario_count"], 1)
        self.assertIn("night_rest", survival["scenarios"])
        self.assertNotIn("extra_scenario", survival["scenarios"])

    def test_notes_contains_behavior_survival_note(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        notes_text = " ".join(result["notes"])
        self.assertIn("behavior_survival", notes_text)

    def test_behavior_survival_zero_scenarios_when_austere_suite_is_empty_dict(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.8},
                    "suite": {"night_rest": {"success_rate": 0.8, "episodes": 2}},
                },
                "austere": {
                    "summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.5},
                    "suite": {},  # empty suite
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertFalse(result["behavior_survival"]["available"])
