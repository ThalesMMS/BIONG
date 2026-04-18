import unittest

from spider_cortex_sim.curriculum import (
    CURRICULUM_FOCUS_SCENARIOS,
    CURRICULUM_PROFILE_NAMES,
    SUBSKILL_CHECK_MAPPINGS,
    CurriculumPhaseDefinition,
    PromotionCheckCriteria,
    evaluate_promotion_check_specs,
    regime_row_metadata_from_summary,
    resolve_curriculum_phase_budgets,
    resolve_curriculum_profile,
)


class CurriculumModuleTest(unittest.TestCase):
    # ---------------------------------------------------------------------------
    # Curriculum constants
    # ---------------------------------------------------------------------------

    def test_curriculum_profile_names_contains_required_values(self) -> None:
        self.assertIn("none", CURRICULUM_PROFILE_NAMES)
        self.assertIn("ecological_v1", CURRICULUM_PROFILE_NAMES)
        self.assertIn("ecological_v2", CURRICULUM_PROFILE_NAMES)

    def test_curriculum_focus_scenarios_are_expected_four(self) -> None:
        expected = {
            "open_field_foraging",
            "corridor_gauntlet",
            "exposed_day_foraging",
            "food_deprivation",
        }
        self.assertEqual(set(CURRICULUM_FOCUS_SCENARIOS), expected)

    def test_subskill_check_mappings_include_required_subskills(self) -> None:
        required_subskills = {
            "shelter_exit",
            "food_approach",
            "predator_response",
            "corridor_navigation",
            "hunger_commitment",
        }
        self.assertTrue(required_subskills.issubset(set(SUBSKILL_CHECK_MAPPINGS)))
        for criteria in SUBSKILL_CHECK_MAPPINGS.values():
            self.assertIsInstance(criteria, tuple)
            self.assertTrue(criteria)
            self.assertTrue(
                all(isinstance(item, PromotionCheckCriteria) for item in criteria)
            )

    def test_subskill_check_mappings_store_named_checks(self) -> None:
        check_names = {
            name: {criteria.check_name for criteria in criteria_list}
            for name, criteria_list in SUBSKILL_CHECK_MAPPINGS.items()
        }
        expected_checks = {
            "shelter_exit": {"commits_to_foraging"},
            "food_approach": {
                "approaches_food",
                "made_food_progress",
                "day_food_progress",
            },
            "predator_response": {"predator_detected", "predator_reacted"},
            "corridor_navigation": {
                "corridor_survives",
                "corridor_food_progress",
            },
            "hunger_commitment": {"hunger_reduced", "survives_deprivation"},
        }
        for subskill, required_names in expected_checks.items():
            with self.subTest(subskill=subskill):
                self.assertIn(subskill, check_names)
                self.assertTrue(required_names.issubset(check_names[subskill]))

    # ---------------------------------------------------------------------------
    # PromotionCheckCriteria dataclass
    # ---------------------------------------------------------------------------

    def test_promotion_check_criteria_stores_fields_and_default_aggregation(self) -> None:
        criteria = PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="hunger_reduced",
            required_pass_rate=0.75,
        )
        self.assertEqual(criteria.scenario, "food_deprivation")
        self.assertEqual(criteria.check_name, "hunger_reduced")
        self.assertEqual(criteria.required_pass_rate, 0.75)
        self.assertEqual(criteria.aggregation, "all")

    def test_promotion_check_criteria_is_frozen(self) -> None:
        criteria = PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="hunger_reduced",
            required_pass_rate=1.0,
        )
        with self.assertRaises((AttributeError, TypeError)):
            criteria.check_name = "modified"  # type: ignore[misc]

    # ---------------------------------------------------------------------------
    # CurriculumPhaseDefinition dataclass
    # ---------------------------------------------------------------------------

    def test_curriculum_phase_definition_stores_fields(self) -> None:
        phase = CurriculumPhaseDefinition(
            name="phase_test",
            training_scenarios=("scenario_a",),
            promotion_scenarios=("scenario_a",),
            success_threshold=0.75,
            max_episodes=10,
            min_episodes=5,
        )
        self.assertEqual(phase.name, "phase_test")
        self.assertEqual(phase.training_scenarios, ("scenario_a",))
        self.assertEqual(phase.success_threshold, 0.75)
        self.assertEqual(phase.max_episodes, 10)
        self.assertEqual(phase.min_episodes, 5)
        self.assertEqual(phase.skill_name, "")
        self.assertEqual(phase.promotion_check_specs, ())

    def test_curriculum_phase_definition_stores_subskill_and_promotion_checks(self) -> None:
        criteria = PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="hunger_reduced",
            required_pass_rate=1.0,
        )
        phase = CurriculumPhaseDefinition(
            name="phase_hunger",
            training_scenarios=("food_deprivation",),
            promotion_scenarios=("food_deprivation",),
            success_threshold=0.5,
            max_episodes=10,
            min_episodes=5,
            skill_name="hunger_commitment",
            promotion_check_specs=(criteria,),
        )
        self.assertEqual(phase.skill_name, "hunger_commitment")
        self.assertEqual(phase.promotion_check_specs, (criteria,))

    def test_curriculum_phase_definition_is_frozen(self) -> None:
        phase = CurriculumPhaseDefinition(
            name="p",
            training_scenarios=("s",),
            promotion_scenarios=("s",),
            success_threshold=1.0,
            max_episodes=4,
            min_episodes=2,
        )
        with self.assertRaises((AttributeError, TypeError)):
            phase.name = "modified"  # type: ignore[misc]

    # ---------------------------------------------------------------------------
    # resolve_curriculum_phase_budgets
    # ---------------------------------------------------------------------------

    def test_resolve_curriculum_phase_budgets_zero_returns_all_zeros(self) -> None:
        result = resolve_curriculum_phase_budgets(0)
        self.assertEqual(result, [0, 0, 0, 0])

    def test_resolve_curriculum_phase_budgets_negative_treated_as_zero(self) -> None:
        result = resolve_curriculum_phase_budgets(-5)
        self.assertEqual(result, [0, 0, 0, 0])

    def test_resolve_curriculum_phase_budgets_returns_four_phases(self) -> None:
        result = resolve_curriculum_phase_budgets(12)
        self.assertEqual(len(result), 4)

    def test_resolve_curriculum_phase_budgets_sum_equals_total(self) -> None:
        for total in (1, 6, 12, 24, 60, 100):
            with self.subTest(total=total):
                result = resolve_curriculum_phase_budgets(total)
                self.assertEqual(sum(result), total)

    def test_resolve_curriculum_phase_budgets_single_episode_distributes(self) -> None:
        result = resolve_curriculum_phase_budgets(1)
        self.assertEqual(sum(result), 1)
        # At least the first non-zero phase gets the episode
        self.assertTrue(any(b > 0 for b in result))

    def test_resolve_curriculum_phase_budgets_all_nonnegative(self) -> None:
        for total in (0, 1, 3, 6, 7, 12, 50):
            with self.subTest(total=total):
                result = resolve_curriculum_phase_budgets(total)
                self.assertTrue(all(b >= 0 for b in result))

    def test_resolve_curriculum_phase_budgets_last_phase_gets_remainder(self) -> None:
        result = resolve_curriculum_phase_budgets(12)
        self.assertEqual(result, [2, 2, 4, 4])

    # ---------------------------------------------------------------------------
    # resolve_curriculum_profile
    # ---------------------------------------------------------------------------

    def test_resolve_curriculum_profile_none_returns_empty_list(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="none", total_episodes=12
        )
        self.assertEqual(phases, [])

    def test_resolve_curriculum_profile_invalid_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_curriculum_profile(
                curriculum_profile="bad_profile", total_episodes=12
            )

    def test_resolve_curriculum_profile_ecological_v1_returns_four_phases(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        self.assertEqual(len(phases), 4)

    def test_resolve_curriculum_profile_ecological_v1_all_are_phase_definitions(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        for phase in phases:
            self.assertIsInstance(phase, CurriculumPhaseDefinition)

    def test_resolve_curriculum_profile_ecological_v1_phase_names(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        expected_names = [
            "phase_1_night_rest_predator_edge",
            "phase_2_entrance_ambush_shelter_blockade",
            "phase_3_open_field_exposed_day",
            "phase_4_corridor_food_deprivation",
        ]
        actual_names = [p.name for p in phases]
        self.assertEqual(actual_names, expected_names)

    def test_resolve_curriculum_profile_ecological_v1_skill_names(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        self.assertEqual(
            [phase.skill_name for phase in phases],
            [
                "predator_response",
                "shelter_exit",
                "food_approach",
                "corridor_gauntlet+food_deprivation",
            ],
        )

    def test_resolve_curriculum_profile_ecological_v1_thresholds(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        # Early phases require perfect score, later phases require half
        self.assertEqual(phases[0].success_threshold, 1.0)
        self.assertEqual(phases[1].success_threshold, 1.0)
        self.assertEqual(phases[2].success_threshold, 0.5)
        self.assertEqual(phases[3].success_threshold, 0.5)

    def test_resolve_curriculum_profile_ecological_v1_min_not_exceed_max(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        for phase in phases:
            self.assertLessEqual(phase.min_episodes, phase.max_episodes)

    def test_resolve_curriculum_profile_ecological_v1_budgets_sum_equals_total(self) -> None:
        total = 12
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=total
        )
        self.assertEqual(sum(p.max_episodes for p in phases), total)

    def test_resolve_curriculum_profile_phase3_scenarios_include_focus(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        phase3_scenarios = set(phases[2].training_scenarios)
        self.assertIn("open_field_foraging", phase3_scenarios)
        self.assertIn("exposed_day_foraging", phase3_scenarios)

    def test_resolve_curriculum_profile_phase4_scenarios_include_focus(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v1", total_episodes=12
        )
        phase4_scenarios = set(phases[3].training_scenarios)
        self.assertIn("corridor_gauntlet", phase4_scenarios)
        self.assertIn("food_deprivation", phase4_scenarios)

    def test_resolve_curriculum_profile_ecological_v2_returns_four_phases(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v2", total_episodes=12
        )
        self.assertEqual(len(phases), 4)

    def test_resolve_curriculum_profile_ecological_v2_phase_names(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v2", total_episodes=12
        )
        self.assertEqual(
            [phase.name for phase in phases],
            [
                "phase_1_shelter_safety_predator_awareness",
                "phase_2_shelter_exit_commitment",
                "phase_3_food_approach_under_exposure",
                "phase_4_corridor_navigation_hunger_survival",
            ],
        )

    def test_resolve_curriculum_profile_ecological_v2_has_explicit_checks(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v2", total_episodes=12
        )
        for phase in phases:
            self.assertTrue(phase.skill_name)
            self.assertTrue(phase.promotion_check_specs)
            self.assertTrue(
                all(
                    isinstance(spec, PromotionCheckCriteria)
                    for spec in phase.promotion_check_specs
                )
            )
        self.assertEqual(
            phases[3].skill_name,
            "corridor_navigation+hunger_commitment",
        )

    def test_resolve_curriculum_profile_ecological_v2_allows_scenario_split(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v2", total_episodes=12
        )
        self.assertNotEqual(phases[0].training_scenarios, phases[0].promotion_scenarios)
        self.assertNotEqual(phases[1].training_scenarios, phases[1].promotion_scenarios)
        self.assertEqual(phases[0].promotion_scenarios, ("predator_edge",))
        self.assertEqual(phases[1].promotion_scenarios, ("food_deprivation",))

    def test_resolve_curriculum_profile_ecological_v2_uses_subskill_mappings(self) -> None:
        phases = resolve_curriculum_profile(
            curriculum_profile="ecological_v2", total_episodes=12
        )
        self.assertEqual(
            phases[0].promotion_check_specs,
            tuple(SUBSKILL_CHECK_MAPPINGS["predator_response"]),
        )
        self.assertEqual(
            phases[1].promotion_check_specs,
            tuple(SUBSKILL_CHECK_MAPPINGS["shelter_exit"]),
        )
        self.assertEqual(
            phases[2].promotion_check_specs,
            tuple(SUBSKILL_CHECK_MAPPINGS["food_approach"]),
        )
        self.assertEqual(
            phases[3].promotion_check_specs,
            (
                *SUBSKILL_CHECK_MAPPINGS["corridor_navigation"],
                *SUBSKILL_CHECK_MAPPINGS["hunger_commitment"],
            ),
        )

    # ---------------------------------------------------------------------------
    # Promotion-check criteria evaluation
    # ---------------------------------------------------------------------------

    def test_evaluate_promotion_check_specs_uses_check_pass_rates(self) -> None:
        specs = (
            PromotionCheckCriteria(
                scenario="food_deprivation",
                check_name="hunger_reduced",
                required_pass_rate=0.5,
            ),
        )
        payload = {
            "suite": {
                "food_deprivation": {
                    "checks": {
                        "hunger_reduced": {"pass_rate": 1.0},
                    }
                }
            }
        }
        results, passed, reason = evaluate_promotion_check_specs(
            payload, specs
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "all_checks_passed")
        hunger_result = results["food_deprivation"]["hunger_reduced"]
        self.assertEqual(hunger_result["scenario"], "food_deprivation")
        self.assertEqual(hunger_result["pass_rate"], 1.0)
        self.assertEqual(hunger_result["required"], 0.5)
        self.assertTrue(hunger_result["passed"])

    def test_evaluate_promotion_check_specs_reports_first_failed_check(self) -> None:
        specs = (
            PromotionCheckCriteria(
                scenario="predator_edge",
                check_name="predator_detected",
                required_pass_rate=1.0,
            ),
            PromotionCheckCriteria(
                scenario="predator_edge",
                check_name="predator_reacted",
                required_pass_rate=1.0,
            ),
        )
        payload = {
            "suite": {
                "predator_edge": {
                    "checks": {
                        "predator_detected": {"pass_rate": 1.0},
                        "predator_reacted": {"pass_rate": 0.0},
                    }
                }
            }
        }
        results, passed, reason = evaluate_promotion_check_specs(
            payload, specs
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "check_failed:predator_reacted")
        predator_results = results["predator_edge"]
        self.assertTrue(predator_results["predator_detected"]["passed"])
        self.assertFalse(predator_results["predator_reacted"]["passed"])

    def test_evaluate_promotion_check_specs_any_promotes_on_one_passing_check(self) -> None:
        specs = (
            PromotionCheckCriteria(
                scenario="predator_edge",
                check_name="predator_detected",
                required_pass_rate=1.0,
                aggregation="any",
            ),
            PromotionCheckCriteria(
                scenario="predator_edge",
                check_name="predator_reacted",
                required_pass_rate=1.0,
                aggregation="any",
            ),
        )
        payload = {
            "suite": {
                "predator_edge": {
                    "checks": {
                        "predator_detected": {"pass_rate": 0.0},
                        "predator_reacted": {"pass_rate": 1.0},
                    }
                }
            }
        }
        results, passed, reason = evaluate_promotion_check_specs(
            payload, specs
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "any_check_passed")
        predator_results = results["predator_edge"]
        self.assertFalse(predator_results["predator_detected"]["passed"])
        self.assertTrue(predator_results["predator_reacted"]["passed"])

    def test_evaluate_promotion_check_specs_rejects_invalid_aggregation(self) -> None:
        specs = (
            PromotionCheckCriteria(
                scenario="food_deprivation",
                check_name="hunger_reduced",
                required_pass_rate=1.0,
                aggregation="bad",
            ),
        )
        with self.assertRaises(ValueError):
            evaluate_promotion_check_specs(
                {"suite": {"food_deprivation": {"checks": {}}}},
                specs,
            )

    def test_evaluate_promotion_check_specs_rejects_mixed_aggregations(self) -> None:
        specs = (
            PromotionCheckCriteria(
                scenario="predator_edge",
                check_name="predator_detected",
                required_pass_rate=1.0,
                aggregation="all",
            ),
            PromotionCheckCriteria(
                scenario="predator_edge",
                check_name="predator_reacted",
                required_pass_rate=1.0,
                aggregation="any",
            ),
        )
        payload = {
            "suite": {
                "predator_edge": {
                    "checks": {
                        "predator_detected": {"pass_rate": 1.0},
                        "predator_reacted": {"pass_rate": 1.0},
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            evaluate_promotion_check_specs(payload, specs)

    def test_evaluate_promotion_check_specs_rejects_duplicate_specs(self) -> None:
        specs = (
            PromotionCheckCriteria(
                scenario="food_deprivation",
                check_name="hunger_reduced",
                required_pass_rate=0.5,
            ),
            PromotionCheckCriteria(
                scenario="food_deprivation",
                check_name="hunger_reduced",
                required_pass_rate=1.0,
            ),
        )
        payload = {
            "suite": {
                "food_deprivation": {
                    "checks": {
                        "hunger_reduced": {"pass_rate": 1.0},
                    }
                }
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate promotion check spec",
        ):
            evaluate_promotion_check_specs(payload, specs)

    def test_evaluate_promotion_check_specs_rejects_required_rate_out_of_bounds(
        self,
    ) -> None:
        payload = {
            "suite": {
                "food_deprivation": {
                    "checks": {
                        "hunger_reduced": {"pass_rate": 1.0},
                    }
                }
            }
        }
        for required_pass_rate in (-0.1, 1.1):
            with self.subTest(required_pass_rate=required_pass_rate):
                specs = (
                    PromotionCheckCriteria(
                        scenario="food_deprivation",
                        check_name="hunger_reduced",
                        required_pass_rate=required_pass_rate,
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "food_deprivation.*hunger_reduced",
                ):
                    evaluate_promotion_check_specs(payload, specs)

    def test_evaluate_promotion_check_specs_empty_specs_returns_clear_reason(self) -> None:
        results, passed, reason = evaluate_promotion_check_specs(
            {"suite": {}},
            (),
        )
        self.assertEqual(results, {})
        self.assertFalse(passed)
        self.assertEqual(reason, "no_checks_specified")

    def test_evaluate_promotion_check_specs_fails_missing_check(self) -> None:
        specs = (
            PromotionCheckCriteria(
                scenario="food_deprivation",
                check_name="hunger_reduced",
                required_pass_rate=1.0,
            ),
        )
        payload = {"suite": {"food_deprivation": {"checks": {}}}}
        results, passed, reason = evaluate_promotion_check_specs(
            payload, specs
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "check_failed:hunger_reduced")
        hunger_result = results["food_deprivation"]["hunger_reduced"]
        self.assertEqual(hunger_result["pass_rate"], 0.0)
        self.assertFalse(hunger_result["passed"])

    # ---------------------------------------------------------------------------
    # regime_row_metadata_from_summary
    # ---------------------------------------------------------------------------

    def test_regime_row_metadata_flat_no_curriculum(self) -> None:
        training_regime = {"mode": "flat", "curriculum_profile": "none"}
        result = regime_row_metadata_from_summary(training_regime, None)
        self.assertEqual(result["training_regime"], "flat")
        self.assertEqual(result["curriculum_profile"], "none")
        self.assertEqual(result["curriculum_phase"], "")
        self.assertEqual(result["curriculum_skill"], "")
        self.assertEqual(result["curriculum_phase_status"], "")
        self.assertEqual(result["curriculum_promotion_reason"], "")

    def test_regime_row_metadata_curriculum_with_phases(self) -> None:
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        curriculum_summary = {
            "executed_training_episodes": 6,
            "phases": [
                {
                    "name": "phase_1_night_rest_predator_edge",
                    "skill_name": "predator_response",
                    "episodes_executed": 3,
                    "status": "promoted",
                    "promotion_reason": "threshold_fallback",
                },
                {
                    "name": "phase_4_corridor_food_deprivation",
                    "skill_name": "corridor_gauntlet+food_deprivation",
                    "episodes_executed": 3,
                    "status": "max_budget_exhausted",
                    "promotion_reason": "check_failed:corridor_food_progress",
                },
            ]
        }
        result = regime_row_metadata_from_summary(
            training_regime, curriculum_summary
        )
        self.assertEqual(result["training_regime"], "curriculum")
        self.assertEqual(result["curriculum_profile"], "ecological_v1")
        # Should use the last phase
        self.assertEqual(result["curriculum_phase"], "phase_4_corridor_food_deprivation")
        self.assertEqual(
            result["curriculum_skill"],
            "corridor_gauntlet+food_deprivation",
        )
        self.assertEqual(result["curriculum_phase_status"], "max_budget_exhausted")
        self.assertEqual(
            result["curriculum_promotion_reason"],
            "check_failed:corridor_food_progress",
        )

    def test_regime_row_metadata_zero_executed_curriculum_omits_phase(self) -> None:
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        curriculum_summary = {
            "executed_training_episodes": 0,
            "phases": [
                {
                    "name": "phase_4_corridor_food_deprivation",
                    "skill_name": "corridor_gauntlet+food_deprivation",
                    "status": "not_started",
                    "promotion_reason": "threshold_fallback",
                },
            ],
        }
        result = regime_row_metadata_from_summary(
            training_regime, curriculum_summary
        )
        self.assertEqual(result["curriculum_profile"], "ecological_v1")
        self.assertEqual(result["curriculum_phase"], "")
        self.assertEqual(result["curriculum_skill"], "")
        self.assertEqual(result["curriculum_phase_status"], "")
        self.assertEqual(result["curriculum_promotion_reason"], "")

    def test_regime_row_metadata_no_executed_phase_omits_phase(self) -> None:
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        curriculum_summary = {
            "executed_training_episodes": 2,
            "phases": [
                {
                    "name": "phase_1_night_rest_predator_edge",
                    "skill_name": "predator_response",
                    "episodes_executed": 0,
                    "status": "not_started",
                    "promotion_reason": "threshold_fallback",
                },
            ],
        }
        result = regime_row_metadata_from_summary(
            training_regime, curriculum_summary
        )
        self.assertEqual(result["curriculum_phase"], "")
        self.assertEqual(result["curriculum_skill"], "")
        self.assertEqual(result["curriculum_phase_status"], "")
        self.assertEqual(result["curriculum_promotion_reason"], "")

    def test_regime_row_metadata_malformed_phase_episode_count_is_ignored(self) -> None:
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        curriculum_summary = {
            "executed_training_episodes": 2,
            "phases": [
                {
                    "name": "phase_1_night_rest_predator_edge",
                    "skill_name": "predator_response",
                    "episodes_executed": 1,
                    "status": "promoted",
                    "promotion_reason": "threshold_fallback",
                },
                {
                    "name": "phase_4_corridor_food_deprivation",
                    "skill_name": "corridor_gauntlet+food_deprivation",
                    "episodes_executed": "malformed",
                    "status": "max_budget_exhausted",
                    "promotion_reason": "check_failed:corridor_food_progress",
                },
            ],
        }

        result = regime_row_metadata_from_summary(
            training_regime, curriculum_summary
        )

        self.assertEqual(result["curriculum_phase"], "phase_1_night_rest_predator_edge")
        self.assertEqual(result["curriculum_skill"], "predator_response")

    def test_regime_row_metadata_single_malformed_episode_count_omits_phase(self) -> None:
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        curriculum_summary = {
            "executed_training_episodes": 2,
            "phases": [
                {
                    "name": "phase_4_corridor_food_deprivation",
                    "skill_name": "corridor_gauntlet+food_deprivation",
                    "episodes_executed": "bad",
                    "status": "max_budget_exhausted",
                    "promotion_reason": "check_failed:corridor_food_progress",
                },
            ],
        }

        result = regime_row_metadata_from_summary(
            training_regime, curriculum_summary
        )

        self.assertEqual(result["curriculum_phase"], "")
        self.assertEqual(result["curriculum_skill"], "")
        self.assertEqual(result["curriculum_phase_status"], "")
        self.assertEqual(result["curriculum_promotion_reason"], "")

    def test_regime_row_metadata_none_curriculum_summary(self) -> None:
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        result = regime_row_metadata_from_summary(training_regime, None)
        self.assertEqual(result["curriculum_phase"], "")
        self.assertEqual(result["curriculum_skill"], "")
        self.assertEqual(result["curriculum_phase_status"], "")
        self.assertEqual(result["curriculum_promotion_reason"], "")

    def test_regime_row_metadata_empty_phases_list(self) -> None:
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        curriculum_summary = {"phases": []}
        result = regime_row_metadata_from_summary(
            training_regime, curriculum_summary
        )
        self.assertEqual(result["curriculum_phase"], "")
        self.assertEqual(result["curriculum_skill"], "")
        self.assertEqual(result["curriculum_phase_status"], "")
        self.assertEqual(result["curriculum_promotion_reason"], "")

    def test_regime_row_metadata_single_phase_in_curriculum(self) -> None:
        """
        Verify that when a curriculum-mode training regime has a single executed phase, the returned row metadata reflects that phase's identifying and promotion fields.
        
        Asserts that with one phase whose episodes_executed equals executed_training_episodes, the resulting mapping contains the phase `name` in `curriculum_phase`, the `skill_name` in `curriculum_skill`, and preserves `status` and `promotion_reason` in `curriculum_phase_status` and `curriculum_promotion_reason`, respectively.
        """
        training_regime = {"mode": "curriculum", "curriculum_profile": "ecological_v1"}
        curriculum_summary = {
            "executed_training_episodes": 3,
            "phases": [
                {
                    "name": "phase_1_night_rest_predator_edge",
                    "skill_name": "predator_response",
                    "episodes_executed": 3,
                    "status": "promoted",
                    "promotion_reason": "threshold_fallback",
                },
            ]
        }
        result = regime_row_metadata_from_summary(
            training_regime, curriculum_summary
        )
        self.assertEqual(result["curriculum_phase"], "phase_1_night_rest_predator_edge")
        self.assertEqual(result["curriculum_skill"], "predator_response")
        self.assertEqual(result["curriculum_phase_status"], "promoted")
        self.assertEqual(result["curriculum_promotion_reason"], "threshold_fallback")

    def test_regime_row_metadata_missing_keys_use_defaults(self) -> None:
        # training_regime with no recognized keys
        result = regime_row_metadata_from_summary({}, None)
        self.assertEqual(result["training_regime"], "flat")
        self.assertEqual(result["curriculum_profile"], "none")


if __name__ == "__main__":
    unittest.main()
