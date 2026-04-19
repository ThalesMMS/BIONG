from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig, canonical_ablation_configs, default_brain_config
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.arbitration import (
    ARBITRATION_EVIDENCE_FIELDS,
    ARBITRATION_GATE_MODULE_ORDER,
    PRIORITY_GATING_WEIGHTS,
    VALENCE_EVIDENCE_WEIGHTS,
    VALENCE_ORDER,
    ArbitrationDecision,
    ValenceScore,
    apply_priority_gating,
    arbitration_evidence_input_dim,
    arbitration_evidence_signal_names,
    arbitration_evidence_vector,
    arbitration_gate_weight_for,
    clamp_unit,
    compute_arbitration,
    deterministic_valence_winner,
    fixed_formula_valence_scores_from_evidence,
    priority_gate_weight_for,
    proposal_contribution_share,
    warm_start_arbitration_network,
)
from spider_cortex_sim.modules import ModuleResult
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.nn import ArbitrationNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from tests.fixtures.arbitration import (
    _blank_obs,
    _module_interface,
    _vector_for,
    _valence_vector,
    _small_arbitration_evidence_vector,
    _assert_relative_scaled_logits,
)

class ArbitrationEvidenceInputDimTest(unittest.TestCase):
    """Tests for arbitration_evidence_input_dim classmethod."""

    def test_returns_24(self) -> None:
        self.assertEqual(arbitration_evidence_input_dim(), 24)

    def test_is_classmethod_accessible(self) -> None:
        # Should be callable on the class, not just an instance
        result = arbitration_evidence_input_dim()
        self.assertIsInstance(result, int)

    def test_matches_arbitration_network_input_dim(self) -> None:
        self.assertEqual(
            arbitration_evidence_input_dim(),
            ArbitrationNetwork.INPUT_DIM,
        )

    def test_equals_sum_of_evidence_fields(self) -> None:
        total = sum(len(fields) for fields in ARBITRATION_EVIDENCE_FIELDS.values())
        self.assertEqual(arbitration_evidence_input_dim(), total)

class ArbitrationEvidenceSignalNamesTest(unittest.TestCase):
    """Tests for arbitration_evidence_signal_names classmethod."""

    def test_returns_tuple(self) -> None:
        self.assertIsInstance(arbitration_evidence_signal_names(), tuple)

    def test_length_is_24(self) -> None:
        self.assertEqual(len(arbitration_evidence_signal_names()), 24)

    def test_first_signal_is_threat_predator_visible(self) -> None:
        names = arbitration_evidence_signal_names()
        self.assertEqual(names[0], "threat.predator_visible")

    def test_last_signal_is_exploration_food_smell_directionality(self) -> None:
        names = arbitration_evidence_signal_names()
        self.assertEqual(names[-1], "exploration.food_smell_directionality")

    def test_all_names_have_valence_prefix(self) -> None:
        names = arbitration_evidence_signal_names()
        valences = set(VALENCE_ORDER)
        for name in names:
            prefix = name.split(".")[0]
            self.assertIn(prefix, valences, f"Signal {name!r} has unknown valence prefix")

    def test_matches_arbitration_network_evidence_signal_names(self) -> None:
        self.assertEqual(
            arbitration_evidence_signal_names(),
            ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES,
        )

    def test_threat_signals_come_before_hunger(self) -> None:
        names = arbitration_evidence_signal_names()
        threat_indices = [i for i, n in enumerate(names) if n.startswith("threat.")]
        hunger_indices = [i for i, n in enumerate(names) if n.startswith("hunger.")]
        self.assertLess(max(threat_indices), min(hunger_indices))

    def test_hunger_signals_come_before_sleep(self) -> None:
        names = arbitration_evidence_signal_names()
        hunger_indices = [i for i, n in enumerate(names) if n.startswith("hunger.")]
        sleep_indices = [i for i, n in enumerate(names) if n.startswith("sleep.")]
        self.assertLess(max(hunger_indices), min(sleep_indices))

    def test_sleep_signals_come_before_exploration(self) -> None:
        names = arbitration_evidence_signal_names()
        sleep_indices = [i for i, n in enumerate(names) if n.startswith("sleep.")]
        exploration_indices = [i for i, n in enumerate(names) if n.startswith("exploration.")]
        self.assertLess(max(sleep_indices), min(exploration_indices))

    def test_each_valence_has_six_signals(self) -> None:
        names = arbitration_evidence_signal_names()
        for valence in VALENCE_ORDER:
            count = sum(1 for n in names if n.startswith(f"{valence}."))
            self.assertEqual(count, 6, f"Expected 6 signals for {valence}, got {count}")

class ArbitrationEvidenceVectorTest(unittest.TestCase):
    """Tests for SpiderBrain._arbitration_evidence_vector."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def _blank_evidence(self) -> dict:
        return {
            valence: {field: 0.0 for field in fields}
            for valence, fields in ARBITRATION_EVIDENCE_FIELDS.items()
        }

    def test_returns_array_of_length_24(self) -> None:
        evidence = self._blank_evidence()
        vec = arbitration_evidence_vector(evidence)
        self.assertEqual(vec.shape, (24,))

    def test_returns_float_array(self) -> None:
        evidence = self._blank_evidence()
        vec = arbitration_evidence_vector(evidence)
        self.assertEqual(vec.dtype, float)

    def test_all_zeros_for_blank_evidence(self) -> None:
        evidence = self._blank_evidence()
        vec = arbitration_evidence_vector(evidence)
        np.testing.assert_array_equal(vec, np.zeros(24))

    def test_first_element_is_threat_predator_visible(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 0.75
        vec = arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[0]), 0.75)

    def test_seventh_element_is_hunger_hunger(self) -> None:
        # threat has 6 fields, so hunger starts at index 6
        evidence = self._blank_evidence()
        evidence["hunger"]["hunger"] = 0.55
        vec = arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[6]), 0.55)

    def test_last_element_is_exploration_food_smell_directionality(self) -> None:
        evidence = self._blank_evidence()
        evidence["exploration"]["food_smell_directionality"] = 0.33
        vec = arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[-1]), 0.33)

    def test_output_matches_signal_names_ordering(self) -> None:
        evidence = self._blank_evidence()
        # Set each signal to a unique value based on its index
        signal_names = arbitration_evidence_signal_names()
        for idx, name in enumerate(signal_names):
            valence, field = name.split(".", 1)
            evidence[valence][field] = float(idx + 1) * 0.01
        vec = arbitration_evidence_vector(evidence)
        for idx in range(24):
            self.assertAlmostEqual(float(vec[idx]), float(idx + 1) * 0.01, places=8)

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
        threat_idx = VALENCE_ORDER.index("threat")
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

class ArbitrationNetworkInArbitrationNetworkTest(unittest.TestCase):
    """Additional regression tests for ArbitrationNetwork integration in SpiderBrain."""

    def _build_brain(self, seed: int = 5) -> SpiderBrain:
        return SpiderBrain(seed=seed, module_dropout=0.0)

    def _blank_observation(self) -> dict[str, np.ndarray]:
        return _blank_obs()

    def test_arbitration_network_name_constant(self) -> None:
        self.assertEqual(SpiderBrain.ARBITRATION_NETWORK_NAME, "arbitration_network")

    def test_arbitration_gate_module_order_has_six_entries(self) -> None:
        self.assertEqual(len(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER), 6)

    def test_arbitration_gate_module_order_contains_expected_modules(self) -> None:
        expected = {
            "alert_center", "hunger_center", "sleep_center",
            "visual_cortex", "sensory_cortex", SpiderBrain.MONOLITHIC_POLICY_NAME,
        }
        self.assertEqual(set(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER), expected)

    def test_arbitration_evidence_fields_has_four_valences(self) -> None:
        self.assertEqual(
            set(ARBITRATION_EVIDENCE_FIELDS.keys()),
            set(VALENCE_ORDER),
        )

    def test_each_arbitration_evidence_valence_has_six_fields(self) -> None:
        for valence, fields in ARBITRATION_EVIDENCE_FIELDS.items():
            self.assertEqual(len(fields), 6, f"Expected 6 fields for {valence}")

    def test_arbitration_network_rng_is_independent(self) -> None:
        """Arbitration RNG should be distinct from main brain RNG (seeded offset)."""
        brain = self._build_brain(seed=42)
        self.assertIsNotNone(brain.arbitration_rng)
        self.assertIsNot(brain.arbitration_rng, brain.rng)

class ProposalContributionShareTest(unittest.TestCase):
    """Unit tests for proposal_contribution_share extracted from agent.py in this PR."""

    def _make_result(self, name: str, *, active: bool = True) -> ModuleResult:
        """Create a minimal ModuleResult with the given name and active state."""
        interface = MODULE_INTERFACES[0]  # any interface works for testing
        action_dim = len(LOCOMOTION_ACTIONS)
        return ModuleResult(
            interface=interface,
            name=name,
            observation_key=interface.observation_key,
            observation=np.zeros(interface.input_dim, dtype=float),
            logits=np.zeros(action_dim, dtype=float),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=active,
        )

    def test_two_active_modules_proportional_to_l1_magnitude(self) -> None:
        """Shares are proportional to L1 norm of gated logits for active modules."""
        r1 = self._make_result("module_a")
        r2 = self._make_result("module_b")
        # r1 has total L1=1.0, r2 has total L1=3.0 → shares 0.25 and 0.75
        g1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertAlmostEqual(shares["module_a"], 0.25, places=8)
        self.assertAlmostEqual(shares["module_b"], 0.75, places=8)

    def test_shares_sum_to_one_for_active_modules(self) -> None:
        """All active module shares must sum to 1.0."""
        results = [self._make_result(f"m{i}") for i in range(4)]
        gated = [np.array([float(i + 1)] + [0.0] * 8) for i in range(4)]
        shares = proposal_contribution_share(results, gated)
        total = sum(shares.values())
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_no_active_modules_returns_all_zeros(self) -> None:
        """When no modules are active, all contribution shares are 0.0."""
        r1 = self._make_result("mod_a", active=False)
        r2 = self._make_result("mod_b", active=False)
        g1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertEqual(shares["mod_a"], 0.0)
        self.assertEqual(shares["mod_b"], 0.0)

    def test_inactive_modules_still_validate_gated_logit_count(self) -> None:
        """Length mismatches should be reported before the inactive early return."""
        r1 = self._make_result("mod_a", active=False)
        r2 = self._make_result("mod_b", active=False)
        g1 = np.zeros(9, dtype=float)
        with self.assertRaises(ValueError):
            proposal_contribution_share([r1, r2], [g1])

    def test_all_zero_gated_logits_returns_uniform_shares(self) -> None:
        """When all active modules have zero logits, shares are uniform."""
        r1 = self._make_result("mod_x")
        r2 = self._make_result("mod_y")
        g1 = np.zeros(9, dtype=float)
        g2 = np.zeros(9, dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertAlmostEqual(shares["mod_x"], 0.5, places=8)
        self.assertAlmostEqual(shares["mod_y"], 0.5, places=8)

    def test_single_active_module_gets_full_share(self) -> None:
        """A single active module should receive contribution share of 1.0."""
        r1 = self._make_result("solo")
        g1 = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1], [g1])
        self.assertAlmostEqual(shares["solo"], 1.0, places=8)

    def test_inactive_module_gets_zero_share_even_with_nonzero_logits(self) -> None:
        """Inactive modules receive 0.0 regardless of their gated logit magnitude."""
        active = self._make_result("active_mod")
        inactive = self._make_result("inactive_mod", active=False)
        g_active = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g_inactive = np.array([9.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([active, inactive], [g_active, g_inactive])
        self.assertAlmostEqual(shares["active_mod"], 1.0, places=8)
        self.assertEqual(shares["inactive_mod"], 0.0)

    def test_mixed_active_inactive_shares_sum_to_one(self) -> None:
        """With mixed active/inactive modules, shares across all entries still sum to 1.0."""
        active1 = self._make_result("act1")
        active2 = self._make_result("act2")
        inactive = self._make_result("inact", active=False)
        g1 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g_inact = np.zeros(9, dtype=float)
        shares = proposal_contribution_share([active1, active2, inactive], [g1, g2, g_inact])
        total = sum(shares.values())
        self.assertAlmostEqual(total, 1.0, places=8)
        self.assertEqual(shares["inact"], 0.0)

    def test_result_keys_match_module_names(self) -> None:
        """Return dict keys match the names of all module results, active or not."""
        results = [self._make_result(f"m{i}", active=(i % 2 == 0)) for i in range(4)]
        gated = [np.ones(9, dtype=float) * float(i) for i in range(4)]
        shares = proposal_contribution_share(results, gated)
        self.assertEqual(set(shares.keys()), {f"m{i}" for i in range(4)})

    def test_single_active_module_with_zero_logits_gets_uniform_share(self) -> None:
        """Uniform fallback: a single active module with zero logits gets share 1.0."""
        r1 = self._make_result("only_active")
        g1 = np.zeros(9, dtype=float)
        shares = proposal_contribution_share([r1], [g1])
        self.assertAlmostEqual(shares["only_active"], 1.0, places=8)

    def test_negative_logits_use_absolute_value_for_magnitude(self) -> None:
        """L1 magnitude uses absolute values, so negative logits contribute positively."""
        r1 = self._make_result("neg")
        r2 = self._make_result("pos")
        # r1 has logits [-2.0, ...], r2 has logits [2.0, ...]; same |magnitude|
        g1 = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertAlmostEqual(shares["neg"], 0.5, places=8)
        self.assertAlmostEqual(shares["pos"], 0.5, places=8)

    def test_empty_module_list_returns_empty_dict(self) -> None:
        """An empty module list should return an empty dict without error."""
        shares = proposal_contribution_share([], [])
        self.assertEqual(shares, {})

    def test_boundary_total_just_above_threshold_uses_proportional_shares(self) -> None:
        """When total L1 is just above 1e-8, proportional shares are used (not uniform)."""
        r1 = self._make_result("small_a")
        r2 = self._make_result("small_b")
        # Give r1 a tiny nonzero logit that keeps total above threshold
        g1 = np.zeros(9, dtype=float)
        g2 = np.zeros(9, dtype=float)
        # Put all weight at index 0
        g1[0] = 1e-7  # just above 1e-8 per dimension; total L1 = 1e-7 > 1e-8
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        # r1 has all the magnitude, r2 has none
        self.assertAlmostEqual(shares["small_a"], 1.0, places=6)
        self.assertAlmostEqual(shares["small_b"], 0.0, places=6)
