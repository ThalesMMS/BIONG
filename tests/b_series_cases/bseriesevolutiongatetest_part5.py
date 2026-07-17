from __future__ import annotations

from .shared import *
from .bseriesevolutiongatetest_helpers import BSeriesEvolutionGateTestHelpers


class BSeriesEvolutionGateTestPart5(BSeriesEvolutionGateTestHelpers, unittest.TestCase):
    def _b74_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "thalamic_rebound_hold",
        rebound_potential: float | None = 0.30,
        inhibition_aftereffect: float | None = 0.24,
        release_window: float | None = 0.22,
        rebound_lock: int | None = 4,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b73_corridor_result(episode, **kwargs)
        for item in result["trace"]:
            if decision is not None:
                item["b74_decision"] = decision
            if rebound_potential is not None:
                item["b74_rebound_potential"] = rebound_potential
            if inhibition_aftereffect is not None:
                item["b74_inhibition_aftereffect"] = inhibition_aftereffect
            if release_window is not None:
                item["b74_release_window"] = release_window
            if rebound_lock is not None:
                item["b74_rebound_lock"] = rebound_lock
        return result

    def _b75_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "timed_rebound_stride",
        release_timing_drive: float | None = 0.32,
        go_disinhibition: float | None = 0.28,
        nogo_brake: float | None = 0.24,
        burst_window: float | None = 0.30,
        release_lock: int | None = 4,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b74_corridor_result(episode, **kwargs)
        for item in result["trace"]:
            if decision is not None:
                item["b75_decision"] = decision
            if release_timing_drive is not None:
                item["b75_release_timing_drive"] = release_timing_drive
            if go_disinhibition is not None:
                item["b75_go_disinhibition"] = go_disinhibition
            if nogo_brake is not None:
                item["b75_nogo_brake"] = nogo_brake
            if burst_window is not None:
                item["b75_burst_window"] = burst_window
            if release_lock is not None:
                item["b75_release_lock"] = release_lock
        return result

    def _b76_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "timed_stride_release",
        stride_timing_signal: float | None = 0.32,
        error_correction: float | None = 0.24,
        burst_smoothing: float | None = 0.28,
        stride_gate: float | None = 0.30,
        stride_lock: int | None = 4,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b75_corridor_result(episode, **kwargs)
        for item in result["trace"]:
            if decision is not None:
                item["b76_decision"] = decision
            if stride_timing_signal is not None:
                item["b76_stride_timing_signal"] = stride_timing_signal
            if error_correction is not None:
                item["b76_error_correction"] = error_correction
            if burst_smoothing is not None:
                item["b76_burst_smoothing"] = burst_smoothing
            if stride_gate is not None:
                item["b76_stride_gate"] = stride_gate
            if stride_lock is not None:
                item["b76_stride_lock"] = stride_lock
        return result

    def _b77_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "corrective_stride_release",
        prediction_error: float | None = 0.28,
        climbing_fiber_drive: float | None = 0.26,
        corrective_timing: float | None = 0.30,
        stability_confidence: float | None = 0.24,
        error_lock: int | None = 4,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b76_corridor_result(episode, **kwargs)
        for item in result["trace"]:
            if decision is not None:
                item["b77_decision"] = decision
            if prediction_error is not None:
                item["b77_prediction_error"] = prediction_error
            if climbing_fiber_drive is not None:
                item["b77_climbing_fiber_drive"] = climbing_fiber_drive
            if corrective_timing is not None:
                item["b77_corrective_timing"] = corrective_timing
            if stability_confidence is not None:
                item["b77_stability_confidence"] = stability_confidence
            if error_lock is not None:
                item["b77_error_lock"] = error_lock
        return result

    def _b78_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "stabilized_corridor_release",
        balance_error: float | None = 0.26,
        head_stabilization: float | None = 0.28,
        locomotor_confidence: float | None = 0.30,
        slip_risk: float | None = 0.24,
        balance_lock: int | None = 4,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b77_corridor_result(episode, **kwargs)
        for item in result["trace"]:
            if decision is not None:
                item["b78_decision"] = decision
            if balance_error is not None:
                item["b78_balance_error"] = balance_error
            if head_stabilization is not None:
                item["b78_head_stabilization"] = head_stabilization
            if locomotor_confidence is not None:
                item["b78_locomotor_confidence"] = locomotor_confidence
            if slip_risk is not None:
                item["b78_slip_risk"] = slip_risk
            if balance_lock is not None:
                item["b78_balance_lock"] = balance_lock
        return result

    def test_b78_corridor_gate_accepts_vestibular_balance_progress(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b78_vestibular_balance_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b78_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["balance_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["head_stabilization_episodes"], 3)
        self.assertEqual(gate["aggregate"]["locomotor_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["slip_risk_episodes"], 3)
        self.assertEqual(gate["aggregate"]["balance_lock_or_release_episodes"], 3)

    def test_b78_corridor_gate_rejects_b77_clone_without_balance_signal(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b78_vestibular_balance_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b78_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                balance_error=None,
                head_stabilization=None,
                locomotor_confidence=None,
                slip_risk=None,
                balance_lock=None,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b78_aggregate:explicit_b78_decision_episodes",
            gate["failures"],
        )

    def test_b78_corridor_gate_rejects_stats_predator_contacts(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b78_vestibular_balance_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b78_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                contacts=1,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 0)
        self.assertIn(
            "corridor_b78_aggregate:corridor_safety_episodes",
            gate["failures"],
        )

    def test_b78_corridor_gate_keeps_b77_base_as_diagnostic(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b78_vestibular_balance_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b78_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b77_corridor_diagnostic", gate["aggregate"])

    def test_b77_corridor_gate_accepts_olivary_error_progress(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b77_olivary_error_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b77_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["climbing_fiber_drive_episodes"], 3)
        self.assertEqual(gate["aggregate"]["corrective_timing_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stability_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["error_lock_or_release_episodes"], 3)

    def test_b77_corridor_gate_rejects_b76_clone_without_error_signal(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b77_olivary_error_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b77_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                prediction_error=None,
                climbing_fiber_drive=None,
                corrective_timing=None,
                stability_confidence=None,
                error_lock=None,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b77_aggregate:explicit_b77_decision_episodes",
            gate["failures"],
        )

    def test_b77_corridor_gate_keeps_b76_base_as_diagnostic(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b77_olivary_error_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b77_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b76_corridor_diagnostic", gate["aggregate"])

    def test_b76_corridor_gate_accepts_cerebellar_stride_progress(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b76_cerebellar_stride_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b76_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stride_timing_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["error_correction_episodes"], 3)
        self.assertEqual(gate["aggregate"]["burst_smoothing_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stride_gate_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stride_lock_or_release_episodes"], 3)

    def test_b76_corridor_gate_rejects_b75_clone_without_stride_timing(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b76_cerebellar_stride_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b76_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                stride_timing_signal=None,
                error_correction=None,
                burst_smoothing=None,
                stride_gate=None,
                stride_lock=None,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b76_aggregate:explicit_b76_decision_episodes",
            gate["failures"],
        )

    def test_b76_corridor_gate_keeps_b75_base_as_diagnostic(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b76_cerebellar_stride_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b76_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b75_corridor_diagnostic", gate["aggregate"])

    def test_b75_corridor_gate_accepts_basal_thalamic_release_progress(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b75_basal_thalamic_release_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b75_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["release_timing_drive_episodes"], 3)
        self.assertEqual(gate["aggregate"]["go_disinhibition_episodes"], 3)
        self.assertEqual(gate["aggregate"]["nogo_brake_episodes"], 3)
        self.assertEqual(gate["aggregate"]["burst_window_episodes"], 3)
        self.assertEqual(gate["aggregate"]["release_lock_or_stride_episodes"], 3)

    def test_b75_corridor_gate_rejects_b74_clone_without_release_timing(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b75_basal_thalamic_release_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b75_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                release_timing_drive=None,
                go_disinhibition=None,
                nogo_brake=None,
                burst_window=None,
                release_lock=None,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b75_aggregate:explicit_b75_decision_episodes",
            gate["failures"],
        )

    def test_b75_corridor_gate_keeps_b74_base_as_diagnostic(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b75_basal_thalamic_release_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b75_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b74_corridor_diagnostic", gate["aggregate"])

    def test_b74_corridor_gate_accepts_thalamic_rebound_progress(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b74_thalamic_rebound_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b74_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["rebound_potential_episodes"], 3)
        self.assertEqual(gate["aggregate"]["inhibition_aftereffect_episodes"], 3)
        self.assertEqual(gate["aggregate"]["release_window_episodes"], 3)
        self.assertEqual(gate["aggregate"]["rebound_lock_or_release_episodes"], 3)

    def test_b74_corridor_gate_rejects_b73_clone_without_rebound(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b74_thalamic_rebound_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b74_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                rebound_potential=None,
                inhibition_aftereffect=None,
                release_window=None,
                rebound_lock=None,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b74_aggregate:explicit_b74_decision_episodes",
            gate["failures"],
        )

    def test_b74_corridor_gate_keeps_b73_base_as_diagnostic(self) -> None:
        gate_fn = getattr(
            b_series_evolution_module,
            "b74_thalamic_rebound_corridor_gate_result",
            None,
        )
        self.assertIsNotNone(gate_fn)
        results = [
            self._b74_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = gate_fn(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b73_corridor_diagnostic", gate["aggregate"])
