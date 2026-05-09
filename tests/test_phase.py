import unittest

from spider_cortex_sim.phase import derive_phase_target


class DerivePhaseTargetTest(unittest.TestCase):
    def test_awake_recovered_shelter_maps_to_post_rest_reactivate(self) -> None:
        phase = derive_phase_target(
            state={
                "sleep_phase": "AWAKE",
                "sleep_events": 3,
                "fatigue": 0.0,
                "sleep_debt": 0.0,
                "rest_streak": 4,
            },
            observation_meta={
                "day": True,
                "night": False,
                "on_shelter": True,
                "sleep_phase": "AWAKE",
            },
        )
        self.assertEqual(phase, "POST_REST_REACTIVATE")

    def test_resting_state_stays_resting_when_not_awake(self) -> None:
        phase = derive_phase_target(
            state={
                "sleep_phase": "RESTING",
                "sleep_events": 3,
                "fatigue": 0.0,
                "sleep_debt": 0.0,
                "rest_streak": 4,
            },
            observation_meta={
                "day": True,
                "night": False,
                "on_shelter": True,
                "sleep_phase": "RESTING",
            },
        )
        self.assertEqual(phase, "RECOVERED_IN_SHELTER")
