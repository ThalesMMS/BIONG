import unittest
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig, canonical_ablation_configs
from spider_cortex_sim.agent import BrainStep, SpiderBrain
from spider_cortex_sim.curriculum import (
    CurriculumPhaseDefinition,
    PromotionCheckCriteria,
)
from spider_cortex_sim.export import save_behavior_csv, save_summary
from spider_cortex_sim.interfaces import ACTION_TO_INDEX, LOCOMOTION_ACTIONS
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.scenarios import SCENARIO_NAMES
from spider_cortex_sim.simulation import SpiderSimulation
from spider_cortex_sim.world import SpiderWorld

from tests.fixtures.training import SpiderTrainingTestBase

class SpiderTrainingBehaviorSuiteTest(SpiderTrainingTestBase):
    def test_scenario_runner_reports_named_results(self) -> None:
        """
        Verify that run_scenarios returns a mapping keyed by the provided scenario names and that each scenario's results include expected evaluation metrics; also confirm a non-empty trace is produced whose final entry contains required debug fields.
        
        Runs the scenarios "night_rest", "recover_after_failed_chase", and "two_shelter_tradeoff" and asserts:
        - The result mapping keys exactly match the provided scenario names.
        - "night_rest" includes "mean_reward".
        - "recover_after_failed_chase" includes "mean_reward_components" and "mean_predator_mode_transitions".
        - "two_shelter_tradeoff" includes "dominant_predator_state".
        - The returned trace is non-empty and the final trace entry contains "debug", "distance_deltas", "predator_escape", and "event_log".
        """
        sim = SpiderSimulation(seed=17, max_steps=40)
        results, trace = sim.run_scenarios(
            ["night_rest", "recover_after_failed_chase", "two_shelter_tradeoff"],
            capture_trace=True,
            debug_trace=True,
        )

        self.assertEqual(set(results.keys()), {"night_rest", "recover_after_failed_chase", "two_shelter_tradeoff"})
        self.assertIn("mean_reward", results["night_rest"])
        self.assertIn("mean_reward_components", results["recover_after_failed_chase"])
        self.assertIn("dominant_predator_state", results["two_shelter_tradeoff"])
        self.assertIn("mean_predator_mode_transitions", results["recover_after_failed_chase"])
        self.assertTrue(trace)
        self.assertIn("debug", trace[-1])
        self.assertIn("distance_deltas", trace[-1])
        self.assertIn("predator_escape", trace[-1])
        self.assertIn("event_log", trace[-1])
        self.assertIsInstance(trace[-1]["event_log"], list)
        self.assertTrue(trace[-1]["event_log"])
        self.assertIsInstance(trace[-1]["event_log"][0], dict)
        self.assertIn("stage", trace[-1]["event_log"][0])
        self.assertIn("name", trace[-1]["event_log"][0])
        self.assertIn("payload", trace[-1]["event_log"][0])

    def test_behavior_suite_reports_scorecards_and_legacy_metrics(self) -> None:
        """
        Verify evaluate_behavior_suite returns complete scorecards, legacy scenario metadata, annotated rows, and a populated trace.
        
        Asserts the returned payload contains "suite", "summary", and "legacy_scenarios"; that the suite keys match the requested scenarios; that the "night_rest" scorecard includes success rate, checks (including "deep_night_shelter"), behavior metrics, failures, and legacy metrics; that returned rows are non-empty and the first row contains expected metadata (reward_profile "classic", scenario_map and evaluation_map "central_burrow", correct architecture version and non-empty architecture_fingerprint, operational_profile "default_v1" version 1, noise_profile "none", and a noise_profile_config); and that the trace is non-empty with the final trace entry containing a "debug" field.
        """
        sim = SpiderSimulation(seed=23, max_steps=20)
        payload, trace, rows = sim.evaluate_behavior_suite(
            ["night_rest", "food_deprivation"],
            capture_trace=True,
            debug_trace=True,
        )

        self.assertIn("suite", payload)
        self.assertIn("summary", payload)
        self.assertIn("legacy_scenarios", payload)
        self.assertEqual(set(payload["suite"].keys()), {"night_rest", "food_deprivation"})
        night_rest = payload["suite"]["night_rest"]
        self.assertIn("success_rate", night_rest)
        self.assertIn("checks", night_rest)
        self.assertIn("behavior_metrics", night_rest)
        self.assertIn("failures", night_rest)
        self.assertIn("legacy_metrics", night_rest)
        self.assertIn("deep_night_shelter", night_rest["checks"])
        self.assertTrue(rows)
        self.assertEqual(rows[0]["reward_profile"], "classic")
        self.assertEqual(rows[0]["scenario_map"], "central_burrow")
        self.assertEqual(rows[0]["evaluation_map"], "central_burrow")
        self.assertEqual(rows[0]["architecture_version"], SpiderBrain.ARCHITECTURE_VERSION)
        self.assertTrue(rows[0]["architecture_fingerprint"])
        self.assertEqual(rows[0]["operational_profile"], "default_v1")
        self.assertEqual(rows[0]["operational_profile_version"], 1)
        self.assertEqual(rows[0]["noise_profile"], "none")
        self.assertIn("noise_profile_config", rows[0])
        self.assertTrue(trace)
        self.assertIn("debug", trace[-1])

    def test_behavior_suite_summary_only_skips_rows(self) -> None:
        sim = SpiderSimulation(seed=23, max_steps=20)
        payload, trace, rows = sim.evaluate_behavior_suite(
            ["night_rest"],
            summary_only=True,
        )

        self.assertIn("summary", payload)
        self.assertIn("night_rest", payload["suite"])
        self.assertEqual(trace, [])
        self.assertEqual(rows, [])

    def test_behavior_suite_is_reproducible_for_same_seed(self) -> None:
        sim_a = SpiderSimulation(seed=29, max_steps=16)
        sim_b = SpiderSimulation(seed=29, max_steps=16)

        payload_a, _, rows_a = sim_a.evaluate_behavior_suite(["night_rest"])
        payload_b, _, rows_b = sim_b.evaluate_behavior_suite(["night_rest"])

        self.assertEqual(
            payload_a["suite"]["night_rest"]["success_rate"],
            payload_b["suite"]["night_rest"]["success_rate"],
        )
        self.assertEqual(
            payload_a["suite"]["night_rest"]["checks"],
            payload_b["suite"]["night_rest"]["checks"],
        )
        self.assertEqual(rows_a, rows_b)

    def test_behavior_suite_is_reproducible_for_same_seed_with_noise_profile(self) -> None:
        sim_a = SpiderSimulation(seed=29, max_steps=16, noise_profile="low")
        sim_b = SpiderSimulation(seed=29, max_steps=16, noise_profile="low")

        payload_a, _, rows_a = sim_a.evaluate_behavior_suite(["night_rest"])
        payload_b, _, rows_b = sim_b.evaluate_behavior_suite(["night_rest"])

        self.assertEqual(payload_a["suite"]["night_rest"]["checks"], payload_b["suite"]["night_rest"]["checks"])
        self.assertEqual(rows_a, rows_b)

    def test_behavior_suite_rows_include_independence_metrics(self) -> None:
        sim = SpiderSimulation(seed=61, max_steps=8)
        payload, _, rows = sim.evaluate_behavior_suite(
            names=["night_rest"],
            episodes_per_scenario=1,
            capture_trace=False,
            debug_trace=False,
        )

        self.assertIn("suite", payload)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("metric_module_contribution_alert_center", row)
        self.assertIn("metric_dominant_module", row)
        self.assertIn("metric_dominant_module_share", row)
        self.assertIn("metric_effective_module_count", row)
        self.assertIn("metric_module_agreement_rate", row)
        self.assertIn("metric_module_disagreement_rate", row)
        self.assertIn("metric_motor_slip_rate", row)
        self.assertIn("metric_mean_orientation_alignment", row)
        self.assertIn("metric_mean_terrain_difficulty", row)
