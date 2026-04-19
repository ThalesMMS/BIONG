"""Focused metrics and behavior-evaluation tests."""

from __future__ import annotations

from collections.abc import Mapping
import unittest

from spider_cortex_sim.ablations import PROPOSAL_SOURCE_NAMES
from spider_cortex_sim.metrics import (
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    REFLEX_MODULE_NAMES,
    _aggregate_values,
    _mean_like,
    _mean_map,
    aggregate_behavior_scores,
    aggregate_episode_stats,
    build_behavior_check,
    build_behavior_score,
    flatten_behavior_rows,
    summarize_behavior_suite,
)
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES
from tests.fixtures.shared_builders import (
    make_behavior_check_result,
    make_episode_stats,
)

class AggregateEpisodeStatsReflexTest(unittest.TestCase):
    """Tests for aggregate_episode_stats reflex metrics (new in this PR)."""

    def _make_episode_stats(
        self,
        *,
        reflex_usage_rate: float = 0.0,
        final_reflex_override_rate: float = 0.0,
        mean_reflex_dominance: float = 0.0,
        module_reflex_usage_rates: dict | None = None,
        module_reflex_override_rates: dict | None = None,
        module_reflex_dominance: dict | None = None,
        module_contribution_share: dict | None = None,
        dominant_module: str = "",
        dominant_module_share: float = 0.0,
        effective_module_count: float = 0.0,
        module_agreement_rate: float = 0.0,
        module_disagreement_rate: float = 0.0,
        mean_module_credit_weights: dict | None = None,
        module_gradient_norm_means: dict | None = None,
        motor_slip_rate: float = 0.0,
        mean_orientation_alignment: float = 0.0,
        mean_terrain_difficulty: float = 0.0,
        terrain_slip_rates: dict | None = None,
        night_role_ticks: dict | None = None,
        night_role_distribution: dict | None = None,
    ) -> EpisodeStats:
        """
        Create a deterministic EpisodeStats object populated with reflex-, module-, and terrain-related metrics for testing.
        
        Parameters:
            reflex_usage_rate (float): Overall fraction of timesteps where reflexes were used.
            final_reflex_override_rate (float): Fraction of final actions that were reflex overrides.
            mean_reflex_dominance (float): Average dominance score of reflex control.
            module_reflex_usage_rates (dict | None): Per-module reflex usage rates; keys override defaults for REFLEX_MODULE_NAMES.
            module_reflex_override_rates (dict | None): Per-module reflex override rates; keys override defaults for REFLEX_MODULE_NAMES.
            module_reflex_dominance (dict | None): Per-module reflex dominance scores; keys override defaults for REFLEX_MODULE_NAMES.
            module_contribution_share (dict | None): Per-proposal-source contribution shares; keys override defaults for PROPOSAL_SOURCE_NAMES.
            dominant_module (str): Name of the dominant module for the episode.
            dominant_module_share (float): Share (0.0-1.0) of the dominant module's contribution.
            effective_module_count (float): Effective number of modules contributing.
            module_agreement_rate (float): Fraction of time modules agreed on proposals.
            module_disagreement_rate (float): Fraction of time modules disagreed.
            mean_module_credit_weights (dict | None): Per-proposal-source mean credit weights; keys override defaults for PROPOSAL_SOURCE_NAMES.
            module_gradient_norm_means (dict | None): Per-proposal-source mean gradient norms; keys override defaults for PROPOSAL_SOURCE_NAMES.
            motor_slip_rate (float): Mean rate of motor slip events.
            mean_orientation_alignment (float): Mean orientation alignment score.
            mean_terrain_difficulty (float): Mean terrain difficulty experienced.
            terrain_slip_rates (dict | None): Per-terrain-type slip rates.
        
        Returns:
            EpisodeStats: An EpisodeStats instance with the provided metric values applied; unspecified per-module/per-source keys are filled with zeros for completeness.
        """
        module_rates = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_overrides = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_dominance = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_contribution = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        module_credit = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        module_gradients = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        if module_reflex_usage_rates:
            module_rates.update(module_reflex_usage_rates)
        if module_reflex_override_rates:
            module_overrides.update(module_reflex_override_rates)
        if module_reflex_dominance:
            module_dominance.update(module_reflex_dominance)
        if module_contribution_share:
            module_contribution.update(module_contribution_share)
        if mean_module_credit_weights:
            module_credit.update(mean_module_credit_weights)
        if module_gradient_norm_means:
            module_gradients.update(module_gradient_norm_means)
        return EpisodeStats(
            episode=0,
            seed=7,
            training=False,
            scenario=None,
            total_reward=0.0,
            steps=10,
            food_eaten=0,
            sleep_events=0,
            shelter_entries=0,
            alert_events=0,
            predator_contacts=0,
            predator_sightings=0,
            predator_escapes=0,
            night_ticks=0,
            night_shelter_ticks=0,
            night_still_ticks=0,
            night_role_ticks=dict(
                night_role_ticks
                if night_role_ticks is not None
                else {"outside": 0, "entrance": 0, "inside": 0, "deep": 0}
            ),
            night_shelter_occupancy_rate=0.0,
            night_stillness_rate=0.0,
            night_role_distribution=dict(
                night_role_distribution
                if night_role_distribution is not None
                else {"outside": 0.0, "entrance": 0.0, "inside": 0.0, "deep": 0.0}
            ),
            predator_response_events=0,
            mean_predator_response_latency=0.0,
            mean_sleep_debt=0.0,
            food_distance_delta=0.0,
            shelter_distance_delta=0.0,
            final_hunger=0.0,
            final_fatigue=0.0,
            final_sleep_debt=0.0,
            final_health=1.0,
            alive=True,
            reward_component_totals={},
            predator_state_ticks={"PATROL": 10},
            predator_mode_transitions=0,
            dominant_predator_state="PATROL",
            reflex_usage_rate=reflex_usage_rate,
            final_reflex_override_rate=final_reflex_override_rate,
            mean_reflex_dominance=mean_reflex_dominance,
            module_reflex_usage_rates=module_rates,
            module_reflex_override_rates=module_overrides,
            module_reflex_dominance=module_dominance,
            module_contribution_share=module_contribution,
            dominant_module=dominant_module,
            dominant_module_share=dominant_module_share,
            effective_module_count=effective_module_count,
            module_agreement_rate=module_agreement_rate,
            module_disagreement_rate=module_disagreement_rate,
            mean_module_credit_weights=module_credit,
            module_gradient_norm_means=module_gradients,
            motor_slip_rate=motor_slip_rate,
            mean_orientation_alignment=mean_orientation_alignment,
            mean_terrain_difficulty=mean_terrain_difficulty,
            terrain_slip_rates=dict(terrain_slip_rates or {}),
        )

    def test_aggregate_empty_history_returns_zeros(self) -> None:
        result = aggregate_episode_stats([])
        self.assertAlmostEqual(result["mean_reflex_usage_rate"], 0.0)
        self.assertAlmostEqual(result["mean_final_reflex_override_rate"], 0.0)
        self.assertAlmostEqual(result["mean_reflex_dominance"], 0.0)
        self.assertAlmostEqual(result["mean_motor_slip_rate"], 0.0)
        self.assertAlmostEqual(result["mean_orientation_alignment"], 0.0)
        self.assertAlmostEqual(result["mean_terrain_difficulty"], 0.0)
        self.assertEqual(result["mean_terrain_slip_rates"], {})
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, result["mean_module_credit_weights"])
            self.assertIn(name, result["module_gradient_norm_means"])

    def test_aggregate_tolerates_missing_night_role_keys(self) -> None:
        result = aggregate_episode_stats(
            [
                self._make_episode_stats(
                    night_role_ticks={"outside": 2},
                    night_role_distribution={"outside": 1.0},
                )
            ]
        )

        self.assertEqual(result["mean_night_role_ticks"]["deep"], 0)
        self.assertAlmostEqual(result["mean_night_role_distribution"]["deep"], 0.0)

    def test_aggregate_includes_mean_reflex_usage_rate(self) -> None:
        history = [
            self._make_episode_stats(reflex_usage_rate=0.4),
            self._make_episode_stats(reflex_usage_rate=0.6),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_reflex_usage_rate"], 0.5)

    def test_aggregate_includes_motor_execution_metrics(self) -> None:
        history = [
            self._make_episode_stats(
                motor_slip_rate=0.2,
                mean_orientation_alignment=0.8,
                mean_terrain_difficulty=0.1,
                terrain_slip_rates={"open": 0.0, "narrow": 0.4},
            ),
            self._make_episode_stats(
                motor_slip_rate=0.6,
                mean_orientation_alignment=0.4,
                mean_terrain_difficulty=0.5,
                terrain_slip_rates={"narrow": 0.8},
            ),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_motor_slip_rate"], 0.4)
        self.assertAlmostEqual(result["mean_orientation_alignment"], 0.6)
        self.assertAlmostEqual(result["mean_terrain_difficulty"], 0.3)
        self.assertAlmostEqual(result["mean_terrain_slip_rates"]["open"], 0.0)
        self.assertAlmostEqual(result["mean_terrain_slip_rates"]["narrow"], 0.6)

    def test_aggregate_includes_mean_final_reflex_override_rate(self) -> None:
        history = [
            self._make_episode_stats(final_reflex_override_rate=0.2),
            self._make_episode_stats(final_reflex_override_rate=0.4),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_final_reflex_override_rate"], 0.3)

    def test_aggregate_includes_mean_reflex_dominance(self) -> None:
        history = [
            self._make_episode_stats(mean_reflex_dominance=0.1),
            self._make_episode_stats(mean_reflex_dominance=0.3),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_reflex_dominance"], 0.2)

    def test_aggregate_includes_mean_module_reflex_usage_rate(self) -> None:
        history = [
            self._make_episode_stats(module_reflex_usage_rates={"alert_center": 0.4}),
            self._make_episode_stats(module_reflex_usage_rates={"alert_center": 0.8}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_reflex_usage_rate", result)
        self.assertAlmostEqual(result["mean_module_reflex_usage_rate"]["alert_center"], 0.6)

    def test_aggregate_includes_mean_module_reflex_override_rate(self) -> None:
        history = [
            self._make_episode_stats(module_reflex_override_rates={"alert_center": 0.2}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_reflex_override_rate", result)
        self.assertAlmostEqual(result["mean_module_reflex_override_rate"]["alert_center"], 0.2)

    def test_aggregate_includes_mean_module_reflex_dominance(self) -> None:
        history = [
            self._make_episode_stats(module_reflex_dominance={"alert_center": 0.35}),
            self._make_episode_stats(module_reflex_dominance={"alert_center": 0.65}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_reflex_dominance", result)
        self.assertAlmostEqual(result["mean_module_reflex_dominance"]["alert_center"], 0.5)

    def test_aggregate_module_rate_keys_include_all_module_names(self) -> None:
        history = [self._make_episode_stats()]
        result = aggregate_episode_stats(history)
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, result["mean_module_reflex_usage_rate"])
            self.assertIn(name, result["mean_module_reflex_override_rate"])
            self.assertIn(name, result["mean_module_reflex_dominance"])

    def test_aggregate_includes_mean_module_contribution_share(self) -> None:
        history = [
            self._make_episode_stats(module_contribution_share={"alert_center": 0.2}),
            self._make_episode_stats(module_contribution_share={"alert_center": 0.6}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_contribution_share", result)
        self.assertAlmostEqual(
            result["mean_module_contribution_share"]["alert_center"],
            0.4,
        )

    def test_aggregate_includes_mean_module_credit_weights(self) -> None:
        history = [
            self._make_episode_stats(mean_module_credit_weights={"alert_center": 0.25}),
            self._make_episode_stats(mean_module_credit_weights={"alert_center": 0.75}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_credit_weights", result)
        self.assertAlmostEqual(
            result["mean_module_credit_weights"]["alert_center"],
            0.5,
        )

    def test_aggregate_includes_module_gradient_norm_means(self) -> None:
        history = [
            self._make_episode_stats(module_gradient_norm_means={"alert_center": 1.5}),
            self._make_episode_stats(module_gradient_norm_means={"alert_center": 2.5}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("module_gradient_norm_means", result)
        self.assertAlmostEqual(
            result["module_gradient_norm_means"]["alert_center"],
            2.0,
        )

    def test_aggregate_includes_dominant_module_distribution(self) -> None:
        history = [
            self._make_episode_stats(dominant_module="alert_center"),
            self._make_episode_stats(dominant_module="alert_center"),
            self._make_episode_stats(dominant_module="hunger_center"),
        ]
        result = aggregate_episode_stats(history)
        self.assertEqual(result["dominant_module"], "alert_center")
        self.assertAlmostEqual(
            result["dominant_module_distribution"]["alert_center"],
            2 / 3,
        )

    def test_aggregate_includes_mean_independence_scalars(self) -> None:
        history = [
            self._make_episode_stats(
                dominant_module_share=0.7,
                effective_module_count=1.5,
                module_agreement_rate=0.8,
                module_disagreement_rate=0.2,
            ),
            self._make_episode_stats(
                dominant_module_share=0.5,
                effective_module_count=2.0,
                module_agreement_rate=0.6,
                module_disagreement_rate=0.4,
            ),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_dominant_module_share"], 0.6)
        self.assertAlmostEqual(result["mean_effective_module_count"], 1.75)
        self.assertAlmostEqual(result["mean_module_agreement_rate"], 0.7)
        self.assertAlmostEqual(result["mean_module_disagreement_rate"], 0.3)

class AggregateEpisodeStatsLearningMetricsTest(unittest.TestCase):
    """Tests for aggregate_episode_stats learning credit weight/norm aggregation."""

    def _make_minimal_episode_stats(
        self,
        mean_module_credit_weights: Mapping[str, float] | None = None,
        module_gradient_norm_means: Mapping[str, float] | None = None,
    ) -> EpisodeStats:
        """
        Construct a minimal EpisodeStats populated with sensible defaults, allowing override of learning-related per-module credit weights and gradient-norm means.
        
        Parameters:
            mean_module_credit_weights (Mapping[str, float] | None): Optional mapping of proposal-source names to credit weights; any keys provided will override the corresponding defaults (defaults include all PROPOSAL_SOURCE_NAMES set to 0.0).
            module_gradient_norm_means (Mapping[str, float] | None): Optional mapping of proposal-source names to mean gradient norms; any keys provided will override the corresponding defaults (defaults include all PROPOSAL_SOURCE_NAMES set to 0.0).
        
        Returns:
            EpisodeStats: An EpisodeStats instance with default values for all fields except the provided learning metrics which are applied to `mean_module_credit_weights` and `module_gradient_norm_means`.
        """
        credit = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        gradients = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        if mean_module_credit_weights:
            credit.update(mean_module_credit_weights)
        if module_gradient_norm_means:
            gradients.update(module_gradient_norm_means)
        return EpisodeStats(
            episode=0,
            seed=1,
            training=False,
            scenario=None,
            total_reward=0.0,
            steps=10,
            food_eaten=0,
            sleep_events=0,
            shelter_entries=0,
            alert_events=0,
            predator_contacts=0,
            predator_sightings=0,
            predator_escapes=0,
            night_ticks=0,
            night_shelter_ticks=0,
            night_still_ticks=0,
            night_role_ticks={"outside": 0, "entrance": 0, "inside": 0, "deep": 0},
            night_shelter_occupancy_rate=0.0,
            night_stillness_rate=0.0,
            night_role_distribution={"outside": 0.0, "entrance": 0.0, "inside": 0.0, "deep": 0.0},
            predator_response_events=0,
            mean_predator_response_latency=0.0,
            mean_sleep_debt=0.0,
            food_distance_delta=0.0,
            shelter_distance_delta=0.0,
            final_hunger=0.0,
            final_fatigue=0.0,
            final_sleep_debt=0.0,
            final_health=1.0,
            alive=True,
            reward_component_totals={},
            predator_state_ticks={"PATROL": 10},
            predator_mode_transitions=0,
            dominant_predator_state="PATROL",
            reflex_usage_rate=0.0,
            final_reflex_override_rate=0.0,
            mean_reflex_dominance=0.0,
            module_reflex_usage_rates={name: 0.0 for name in REFLEX_MODULE_NAMES},
            module_reflex_override_rates={name: 0.0 for name in REFLEX_MODULE_NAMES},
            module_reflex_dominance={name: 0.0 for name in REFLEX_MODULE_NAMES},
            module_contribution_share={name: 0.0 for name in PROPOSAL_SOURCE_NAMES},
            mean_module_credit_weights=credit,
            module_gradient_norm_means=gradients,
            dominant_module="",
            dominant_module_share=0.0,
            effective_module_count=0.0,
            module_agreement_rate=0.0,
            module_disagreement_rate=0.0,
        )

    def test_aggregate_mean_module_credit_weights_single_episode(self) -> None:
        """aggregate_episode_stats passes through single-episode credit weights unchanged."""
        stats = self._make_minimal_episode_stats(
            mean_module_credit_weights={"alert_center": 0.75}
        )
        result = aggregate_episode_stats([stats])
        self.assertAlmostEqual(result["mean_module_credit_weights"]["alert_center"], 0.75)

    def test_aggregate_module_gradient_norm_means_single_episode(self) -> None:
        """aggregate_episode_stats passes through single-episode gradient norms unchanged."""
        stats = self._make_minimal_episode_stats(
            module_gradient_norm_means={"visual_cortex": 3.5}
        )
        result = aggregate_episode_stats([stats])
        self.assertAlmostEqual(result["module_gradient_norm_means"]["visual_cortex"], 3.5)

    def test_aggregate_mean_module_credit_weights_three_episodes(self) -> None:
        """aggregate_episode_stats computes mean credit weight across multiple episodes."""
        history = [
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 0.1}),
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 0.5}),
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 0.9}),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_module_credit_weights"]["alert_center"], 0.5)

    def test_aggregate_module_gradient_norm_means_three_episodes(self) -> None:
        """aggregate_episode_stats computes mean gradient norm across multiple episodes."""
        history = [
            self._make_minimal_episode_stats(module_gradient_norm_means={"sensory_cortex": 1.0}),
            self._make_minimal_episode_stats(module_gradient_norm_means={"sensory_cortex": 2.0}),
            self._make_minimal_episode_stats(module_gradient_norm_means={"sensory_cortex": 3.0}),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["module_gradient_norm_means"]["sensory_cortex"], 2.0)

    def test_aggregate_all_proposal_source_names_in_credit_weights(self) -> None:
        """aggregate_episode_stats includes all PROPOSAL_SOURCE_NAMES in mean_module_credit_weights."""
        stats = self._make_minimal_episode_stats()
        result = aggregate_episode_stats([stats])
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, result["mean_module_credit_weights"])

    def test_aggregate_all_proposal_source_names_in_gradient_norms(self) -> None:
        """aggregate_episode_stats includes all PROPOSAL_SOURCE_NAMES in module_gradient_norm_means."""
        stats = self._make_minimal_episode_stats()
        result = aggregate_episode_stats([stats])
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, result["module_gradient_norm_means"])

    def test_aggregate_missing_module_treated_as_zero(self) -> None:
        """A module absent from one episode's mean_module_credit_weights is treated as 0.0 in the mean."""
        history = [
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 1.0}),
            self._make_minimal_episode_stats(mean_module_credit_weights={}),
        ]
        result = aggregate_episode_stats(history)
        # Mean of (1.0, 0.0) = 0.5
        self.assertAlmostEqual(result["mean_module_credit_weights"]["alert_center"], 0.5)

class AggregateBehaviorScoresMetadataTest(unittest.TestCase):
    """Additional tests for aggregate_behavior_scores covering new optional params."""

    def _make_score(self, success: bool = True) -> BehavioralEpisodeScore:
        """
        Constructs a minimal BehavioralEpisodeScore populated with deterministic test defaults.
        
        Parameters:
            success (bool): Whether the constructed score should be marked as successful (defaults to True).
        
        Returns:
            BehavioralEpisodeScore: An episode score with episode=0, seed=1, scenario="s", objective="o",
            a single check named "check_a" whose `passed` equals `success`, empty `behavior_metrics`,
            and `failures` set to [] when successful or ["check_a"] when not.
        """
        return BehavioralEpisodeScore(
            episode=0,
            seed=1,
            scenario="s",
            objective="o",
            success=success,
            checks={"check_a": make_behavior_check_result("check_a", passed=success)},
            behavior_metrics={},
            failures=[] if success else ["check_a"],
        )

    def test_none_metadata_params_become_empty_strings(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="s",
            description="d",
            objective="o",
            diagnostic_focus=None,
            success_interpretation=None,
            failure_interpretation=None,
            budget_note=None,
            check_specs=[],
        )
        self.assertEqual(result["diagnostic_focus"], "")
        self.assertEqual(result["success_interpretation"], "")
        self.assertEqual(result["failure_interpretation"], "")
        self.assertEqual(result["budget_note"], "")

    def test_no_failure_mode_in_scores_omits_distribution_keys(self) -> None:
        scores = [self._make_score(False), self._make_score(True)]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[BehaviorCheckSpec("check_a", "desc", "true")],
        )
        self.assertNotIn("primary_failure_mode", result["diagnostics"])
        self.assertNotIn("failure_mode_distribution", result["diagnostics"])

    def test_single_failure_mode_is_primary(self) -> None:
        score = self._make_score(False)
        score.behavior_metrics["failure_mode"] = "orientation_failure"
        result = aggregate_behavior_scores(
            [score],
            scenario="s",
            description="d",
            objective="o",
            check_specs=[BehaviorCheckSpec("check_a", "desc", "true")],
        )
        self.assertEqual(result["diagnostics"]["primary_failure_mode"], "orientation_failure")
        self.assertAlmostEqual(
            result["diagnostics"]["failure_mode_distribution"]["orientation_failure"],
            1.0,
        )

    def test_metadata_keys_present_even_with_no_scores(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="scenario_x",
            description="desc_x",
            objective="obj_x",
            diagnostic_focus="focus_x",
            success_interpretation="success_x",
            failure_interpretation="failure_x",
            budget_note="note_x",
            check_specs=[],
        )
        self.assertEqual(result["diagnostic_focus"], "focus_x")
        self.assertEqual(result["success_interpretation"], "success_x")
        self.assertEqual(result["failure_interpretation"], "failure_x")
        self.assertEqual(result["budget_note"], "note_x")
        self.assertEqual(result["episodes"], 0)

class AggregateEpisodeStatsNewFieldsTest(unittest.TestCase):
    """Tests for new aggregate_episode_stats fields."""

    def _minimal_episode_stats(self, **overrides) -> EpisodeStats:
        """
        Create a minimal EpisodeStats populated with sensible defaults for testing, allowing selective overrides.
        
        The provided keyword arguments are merged into a set of test-oriented default fields before constructing and returning the EpisodeStats. Use overrides to set episode identifiers, scenario/seed, reward/step counts, predator and night metrics, module/reflex/learning fields, or any other EpisodeStats fields required by a test.
        
        Parameters:
            overrides (dict): Keyword overrides merged into the default field map. Common keys include: episode, seed, training, scenario, total_reward, steps, food_eaten, night_ticks, night_role_ticks, reward_component_totals, predator_state_ticks, predator_mode_transitions, dominant_predator_state, and other EpisodeStats fields.
        
        Returns:
            EpisodeStats: An EpisodeStats instance built from the defaults with provided overrides applied.
        """
        from spider_cortex_sim.world import REWARD_COMPONENT_NAMES
        defaults = dict(
            episode=0, seed=0, training=False, scenario=None,
            total_reward=1.0, steps=10, food_eaten=1, sleep_events=0,
            shelter_entries=1, alert_events=0, predator_contacts=0,
            predator_sightings=0, predator_escapes=0,
            night_ticks=5, night_shelter_ticks=4, night_still_ticks=3,
            night_role_ticks={"outside": 1, "entrance": 0, "inside": 1, "deep": 3},
            night_shelter_occupancy_rate=0.8, night_stillness_rate=0.6,
            night_role_distribution={"outside": 0.2, "entrance": 0.0, "inside": 0.2, "deep": 0.6},
            predator_response_events=0, mean_predator_response_latency=0.0,
            mean_sleep_debt=0.3,
            food_distance_delta=2.0,
            shelter_distance_delta=1.0,
            final_hunger=0.4, final_fatigue=0.3, final_sleep_debt=0.25,
            final_health=0.9, alive=True,
            reward_component_totals={name: 0.0 for name in REWARD_COMPONENT_NAMES},
            predator_state_ticks={s: 0 for s in PREDATOR_STATES},
            predator_mode_transitions=2,
            dominant_predator_state="PATROL",
        )
        defaults.update(overrides)
        return EpisodeStats(**defaults)

    def test_aggregate_includes_mean_food_distance_delta(self) -> None:
        stats = [self._minimal_episode_stats(food_distance_delta=4.0),
                 self._minimal_episode_stats(food_distance_delta=2.0)]
        result = aggregate_episode_stats(stats)
        self.assertIn("mean_food_distance_delta", result)
        self.assertAlmostEqual(result["mean_food_distance_delta"], 3.0)

    def test_aggregate_includes_mean_shelter_distance_delta(self) -> None:
        stats = [self._minimal_episode_stats(shelter_distance_delta=6.0),
                 self._minimal_episode_stats(shelter_distance_delta=2.0)]
        result = aggregate_episode_stats(stats)
        self.assertIn("mean_shelter_distance_delta", result)
        self.assertAlmostEqual(result["mean_shelter_distance_delta"], 4.0)

    def test_aggregate_includes_mean_predator_mode_transitions(self) -> None:
        stats = [self._minimal_episode_stats(predator_mode_transitions=4),
                 self._minimal_episode_stats(predator_mode_transitions=2)]
        result = aggregate_episode_stats(stats)
        self.assertIn("mean_predator_mode_transitions", result)
        self.assertAlmostEqual(result["mean_predator_mode_transitions"], 3.0)

    def test_aggregate_includes_dominant_predator_state(self) -> None:
        patrol_ticks = {s: 0 for s in PREDATOR_STATES}
        patrol_ticks["CHASE"] = 10
        stats = [self._minimal_episode_stats(predator_state_ticks=patrol_ticks,
                                              dominant_predator_state="CHASE")]
        result = aggregate_episode_stats(stats)
        self.assertIn("dominant_predator_state", result)
        self.assertEqual(result["dominant_predator_state"], "CHASE")

    def test_aggregate_dominant_predator_state_defaults_to_patrol_when_all_zero(self) -> None:
        stats = [
            self._minimal_episode_stats(
                predator_state_ticks={"CHASE": 0, "PATROL": 0},
            )
        ]
        result = aggregate_episode_stats(stats)
        self.assertEqual(result["dominant_predator_state"], "PATROL")

    def test_aggregate_empty_history_returns_zero_deltas(self) -> None:
        result = aggregate_episode_stats([])
        self.assertEqual(result["mean_food_distance_delta"], 0.0)
        self.assertEqual(result["mean_shelter_distance_delta"], 0.0)
        self.assertEqual(result["mean_predator_mode_transitions"], 0.0)
        self.assertEqual(result["dominant_predator_state"], "PATROL")

    def test_aggregate_negative_food_delta_preserved(self) -> None:
        stats = [self._minimal_episode_stats(food_distance_delta=-3.0)]
        result = aggregate_episode_stats(stats)
        self.assertAlmostEqual(result["mean_food_distance_delta"], -3.0)

    def test_aggregate_includes_mean_predator_contacts_by_type(self) -> None:
        """
        Verifies that aggregate_episode_stats computes per-type mean predator contact counts across episodes.
        
        Provides two minimal EpisodeStats with complementary `predator_contacts_by_type` counts and asserts
        the aggregated `mean_predator_contacts_by_type` contains the arithmetic mean for each predator type.
        """
        stats = [
            self._minimal_episode_stats(predator_contacts_by_type={"visual": 2, "olfactory": 0}),
            self._minimal_episode_stats(predator_contacts_by_type={"visual": 0, "olfactory": 2}),
        ]
        result = aggregate_episode_stats(stats)
        self.assertAlmostEqual(result["mean_predator_contacts_by_type"]["visual"], 1.0)
        self.assertAlmostEqual(result["mean_predator_contacts_by_type"]["olfactory"], 1.0)

    def test_aggregate_includes_mean_module_response_by_predator_type(self) -> None:
        stats = [
            self._minimal_episode_stats(
                module_response_by_predator_type={
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                    "olfactory": {"visual_cortex": 0.1, "sensory_cortex": 0.9},
                }
            ),
            self._minimal_episode_stats(
                module_response_by_predator_type={
                    "visual": {"visual_cortex": 0.6, "sensory_cortex": 0.4},
                    "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                }
            ),
        ]
        result = aggregate_episode_stats(stats)
        self.assertAlmostEqual(
            result["mean_module_response_by_predator_type"]["visual"]["visual_cortex"],
            0.7,
        )
        self.assertAlmostEqual(
            result["mean_module_response_by_predator_type"]["olfactory"]["sensory_cortex"],
            0.85,
        )

    def test_aggregate_includes_mean_proposer_divergence_by_module(self) -> None:
        stats = [
            self._minimal_episode_stats(
                proposer_divergence_by_module={
                    "visual_cortex": 0.8,
                    "sensory_cortex": 0.6,
                }
            ),
            self._minimal_episode_stats(
                proposer_divergence_by_module={
                    "visual_cortex": 0.4,
                    "sensory_cortex": 0.2,
                }
            ),
        ]
        result = aggregate_episode_stats(stats)
        self.assertAlmostEqual(
            result["mean_proposer_divergence_by_module"]["visual_cortex"],
            0.6,
        )
        self.assertAlmostEqual(
            result["mean_proposer_divergence_by_module"]["sensory_cortex"],
            0.4,
        )

    def test_aggregate_includes_mean_action_center_differentials(self) -> None:
        stats = [
            self._minimal_episode_stats(
                action_center_gate_differential={
                    "visual_cortex": 0.6,
                    "sensory_cortex": -0.4,
                },
                action_center_contribution_differential={
                    "visual_cortex": 0.5,
                    "sensory_cortex": -0.3,
                },
            ),
            self._minimal_episode_stats(
                action_center_gate_differential={
                    "visual_cortex": 0.2,
                    "sensory_cortex": -0.2,
                },
                action_center_contribution_differential={
                    "visual_cortex": 0.1,
                    "sensory_cortex": -0.1,
                },
            ),
        ]
        result = aggregate_episode_stats(stats)
        self.assertAlmostEqual(
            result["mean_action_center_gate_differential"]["visual_cortex"],
            0.4,
        )
        self.assertAlmostEqual(
            result["mean_action_center_gate_differential"]["sensory_cortex"],
            -0.3,
        )
        self.assertAlmostEqual(
            result["mean_action_center_contribution_differential"]["visual_cortex"],
            0.3,
        )
        self.assertAlmostEqual(
            result["mean_action_center_contribution_differential"]["sensory_cortex"],
            -0.2,
        )

    def test_aggregate_includes_mean_representation_specialization_score(self) -> None:
        stats = [
            self._minimal_episode_stats(representation_specialization_score=0.8),
            self._minimal_episode_stats(representation_specialization_score=0.4),
        ]
        result = aggregate_episode_stats(stats)
        self.assertAlmostEqual(
            result["mean_representation_specialization_score"],
            0.6,
        )

class MeanMapTest(unittest.TestCase):
    """Tests for metrics._mean_map() - new in this PR."""

    def _minimal_stats(self, **kwargs) -> "EpisodeStats":
        """
        Constructs a minimal EpisodeStats instance populated with sensible defaults for use in tests.
        
        Creates an EpisodeStats populated with defaults for episode/seed metadata, rewards and step counts, night/tick and shelter metrics, predator-contact and predator-state counters, final state scalars (health, hunger, fatigue, etc.), and dictionary defaults for reward component totals and predator state ticks. Any values passed via `**kwargs` override the corresponding defaults before constructing the EpisodeStats.
        
        Parameters:
            **kwargs: Optional fields to override the built-in defaults passed through to EpisodeStats.
        
        Returns:
            An EpisodeStats instance with the default values merged with any provided overrides.
        """
        from spider_cortex_sim.world import REWARD_COMPONENT_NAMES
        from spider_cortex_sim.predator import PREDATOR_STATES
        defaults = dict(
            episode=0, seed=0, training=False, scenario=None,
            total_reward=0.0, steps=10, food_eaten=0, sleep_events=0,
            shelter_entries=0, alert_events=0, predator_contacts=0,
            predator_sightings=0, predator_escapes=0,
            night_ticks=0, night_shelter_ticks=0, night_still_ticks=0,
            night_role_ticks={"outside": 0, "entrance": 0, "inside": 0, "deep": 0},
            night_shelter_occupancy_rate=0.0, night_stillness_rate=0.0,
            night_role_distribution={"outside": 0.0, "entrance": 0.0, "inside": 0.0, "deep": 0.0},
            predator_response_events=0, mean_predator_response_latency=0.0,
            mean_sleep_debt=0.0, food_distance_delta=0.0, shelter_distance_delta=0.0,
            final_hunger=0.0, final_fatigue=0.0, final_sleep_debt=0.0,
            final_health=1.0, alive=True,
            reward_component_totals={n: 0.0 for n in REWARD_COMPONENT_NAMES},
            predator_state_ticks={s: 0 for s in PREDATOR_STATES},
            predator_mode_transitions=0,
            dominant_predator_state="PATROL",
        )
        defaults.update(kwargs)
        return EpisodeStats(**defaults)

    def test_empty_history_returns_all_zeros(self) -> None:
        result = _mean_map([], ["a", "b"], lambda s, n: 1.0)
        self.assertEqual(result, {"a": 0.0, "b": 0.0})

    def test_single_episode_returns_getter_value(self) -> None:
        stats = self._minimal_stats(food_eaten=5)
        result = _mean_map([stats], ["food_eaten"], lambda s, n: float(s.food_eaten))
        self.assertAlmostEqual(result["food_eaten"], 5.0)

    def test_multiple_episodes_returns_mean(self) -> None:
        stats1 = self._minimal_stats(food_eaten=4)
        stats2 = self._minimal_stats(food_eaten=6)
        result = _mean_map([stats1, stats2], ["food_eaten"], lambda s, n: float(s.food_eaten))
        self.assertAlmostEqual(result["food_eaten"], 5.0)

    def test_returns_dict_with_correct_keys(self) -> None:
        stats = self._minimal_stats()
        result = _mean_map([stats], ["x", "y", "z"], lambda s, n: 0.0)
        self.assertEqual(set(result.keys()), {"x", "y", "z"})
