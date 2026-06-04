from __future__ import annotations

import hashlib
import json

from .shared import *

from spider_cortex_sim.ablation._catalog_diagnostic_b0_b6 import (
    B0_B6_DIAGNOSTIC_VARIANT_SPECS,
    diagnostic_b0_b6_configs,
)
from spider_cortex_sim.ablation._catalog_diagnostic_b19_b30 import (
    diagnostic_b19_b30_configs,
)
from spider_cortex_sim.ablation._catalog_diagnostic_b31_b38 import (
    diagnostic_b31_b38_configs,
)
from spider_cortex_sim.ablation._catalog_diagnostic_b39_b51 import (
    diagnostic_b39_b51_configs,
)
from spider_cortex_sim.ablation._catalog_diagnostic_b52_b62 import (
    diagnostic_b52_b62_configs,
)
from spider_cortex_sim.ablation._catalog_diagnostic_b7_b18 import (
    diagnostic_b7_b18_configs,
)
from spider_cortex_sim.ablation._catalog_shared import (
    B5_ACCEPTED_HOMEOSTASIS_PARAMS,
    build_b_series_diagnostic_configs,
)


class BSeriesCatalogSpecsTest(unittest.TestCase):
    def test_full_b_series_catalog_surface_digest_is_stable(self) -> None:
        configs = self._all_b_series_configs()
        digest = hashlib.sha256(
            json.dumps(
                self._common_surface(configs),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        self.assertEqual(len(configs), 311)
        self.assertEqual(
            tuple(configs)[:3],
            (
                "b0_legacy_semantic_policy",
                "b0_current_bridge_policy",
                "b1_capacity_h48_bridge_policy",
            ),
        )
        self.assertEqual(
            tuple(configs)[-3:],
            (
                "b62_shelter_defense_gate_h48_bridge_policy",
                "b62_defensive_mode_selector_h56_bridge_policy",
                "b62_genetic_defensive_mode_h48_bridge_policy",
            ),
        )
        self.assertEqual(
            digest,
            "8cc281b22a2fc66d1b1c5ed717ac61cba2ca1c68f00855c3d1c562e55377a7fc",
        )

    def test_b0_b6_specs_generate_existing_catalog_surface(self) -> None:
        profile_fields: dict[str, object] = {}
        from_catalog = diagnostic_b0_b6_configs(profile_fields)
        from_specs = build_b_series_diagnostic_configs(
            B0_B6_DIAGNOSTIC_VARIANT_SPECS,
            profile_fields,
        )

        self.assertEqual(tuple(from_catalog), tuple(from_specs))
        self.assertEqual(self._variant_surface(from_specs), self._EXPECTED_VARIANTS)
        self.assertEqual(self._common_surface(from_catalog), self._common_surface(from_specs))

        for name, config in from_specs.items():
            self.assertEqual(config.name, name)
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.module_dropout, 0.0)
            self.assertFalse(config.enable_reflexes)
            self.assertFalse(config.enable_auxiliary_targets)
            self.assertFalse(config.use_learned_arbitration)
            self.assertEqual(config.warm_start_scale, 0.0)
            self.assertFalse(config.enable_food_direction_bias)
            self.assertEqual(config.credit_strategy, "broadcast")
            self.assertEqual(config.disabled_modules, ())
            self.assertEqual(config.reflex_scale, 0.0)
            self.assertEqual(config.module_reflex_scales, {})

    def _all_b_series_configs(self) -> Dict[str, BrainAblationConfig]:
        configs: Dict[str, BrainAblationConfig] = {}
        for build_configs in (
            diagnostic_b0_b6_configs,
            diagnostic_b7_b18_configs,
            diagnostic_b19_b30_configs,
            diagnostic_b31_b38_configs,
            diagnostic_b39_b51_configs,
            diagnostic_b52_b62_configs,
        ):
            configs.update(build_configs({}))
        return configs

    def _variant_surface(
        self,
        configs: Dict[str, BrainAblationConfig],
    ) -> tuple[
        tuple[str, int, str, int, int | None, str | None, str | None, dict[str, object]],
        ...,
    ]:
        return tuple(
            (
                name,
                config.b_level,
                config.b_mode,
                config.b_hidden_dim,
                config.b_parent_level,
                config.b_transfer_source_checkpoint,
                config.b_controller_profile,
                self._controller_overrides(config),
            )
            for name, config in configs.items()
        )

    def _common_surface(
        self,
        configs: Dict[str, BrainAblationConfig],
    ) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
        return tuple(
            (
                name,
                tuple((field_name, getattr(config, field_name)) for field_name in self._COMMON_FIELDS),
            )
            for name, config in configs.items()
        )

    def _controller_overrides(self, config: BrainAblationConfig) -> dict[str, object]:
        if not config.b_controller_params:
            return {}
        return {
            key: value
            for key, value in config.b_controller_params.items()
            if B5_ACCEPTED_HOMEOSTASIS_PARAMS.get(key) != value
        }

    _COMMON_FIELDS = (
        "name",
        "architecture",
        "module_dropout",
        "enable_reflexes",
        "enable_auxiliary_targets",
        "use_learned_arbitration",
        "warm_start_scale",
        "enable_food_direction_bias",
        "credit_strategy",
        "disabled_modules",
        "reflex_scale",
        "module_reflex_scales",
        "b_level",
        "b_mode",
        "b_hidden_dim",
        "b_parent_level",
        "b_transfer_source_checkpoint",
        "b_transfer_min_coverage",
        "b_transfer_allow_low_coverage",
        "b_controller_profile",
        "b_controller_params",
    )

    _EXPECTED_VARIANTS = (
        ("b0_legacy_semantic_policy", 0, "legacy_semantic", 32, None, None, None, {}),
        ("b0_current_bridge_policy", 0, "current_bridge", 32, None, None, None, {}),
        (
            "b1_capacity_h48_bridge_policy",
            1,
            "current_bridge",
            48,
            0,
            "artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b1_capacity_h64_bridge_policy",
            1,
            "current_bridge",
            64,
            0,
            "artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b1_threat_guard_bridge_policy",
            1,
            "current_bridge",
            48,
            0,
            "artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b2_temporal_threat_h48_bridge_policy",
            2,
            "current_bridge",
            48,
            1,
            "artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b2_temporal_threat_h56_bridge_policy",
            2,
            "current_bridge",
            56,
            1,
            "artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b2_temporal_threat_h64_bridge_policy",
            2,
            "current_bridge",
            64,
            1,
            "artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b3_contact_memory_h48_bridge_policy",
            3,
            "current_bridge",
            48,
            2,
            "artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b3_contact_memory_strict_h48_bridge_policy",
            3,
            "current_bridge",
            48,
            2,
            "artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b3_contact_memory_h56_bridge_policy",
            3,
            "current_bridge",
            56,
            2,
            "artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b3_recurrent_guard_h48_bridge_policy",
            3,
            "current_bridge",
            48,
            2,
            "artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best",
            None,
            {},
        ),
        (
            "b4_recovery_balance_h48_bridge_policy",
            4,
            "current_bridge",
            48,
            3,
            "artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best",
            "recovery_balance",
            {},
        ),
        (
            "b4_predator_exit_memory_h48_bridge_policy",
            4,
            "current_bridge",
            48,
            3,
            "artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best",
            "predator_exit_memory",
            {},
        ),
        (
            "b4_recovery_balance_h56_bridge_policy",
            4,
            "current_bridge",
            56,
            3,
            "artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best",
            "recovery_balance_h56",
            {},
        ),
        (
            "b4_genetic_recovery_h48_bridge_policy",
            4,
            "current_bridge",
            48,
            3,
            "artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best",
            "genetic_recovery",
            {},
        ),
        (
            "b5_homeostatic_arbiter_h48_bridge_policy",
            5,
            "current_bridge",
            48,
            4,
            "artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best",
            "homeostatic_arbiter",
            {},
        ),
        (
            "b5_circadian_recovery_h48_bridge_policy",
            5,
            "current_bridge",
            48,
            4,
            "artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best",
            "circadian_recovery",
            {},
        ),
        (
            "b5_homeostatic_arbiter_h56_bridge_policy",
            5,
            "current_bridge",
            56,
            4,
            "artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best",
            "homeostatic_arbiter_h56",
            {},
        ),
        (
            "b5_genetic_homeostasis_h48_bridge_policy",
            5,
            "current_bridge",
            48,
            4,
            "artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best",
            "genetic_homeostasis",
            {},
        ),
        (
            "b6_risk_forage_arbiter_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "risk_forage_arbiter",
            {"b6_family": 1.0, "b6_risk_threshold": 0.35, "b6_corridor_hunger": 0.86},
        ),
        (
            "b6_corridor_survival_guard_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "corridor_survival_guard",
            {"b6_family": 1.0, "b6_corridor_hunger": 0.82, "b6_corridor_lock_ticks": 14.0},
        ),
        (
            "b6_threat_priority_memory_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "threat_priority_memory",
            {"b6_family": 1.0, "b6_risk_threshold": 0.22, "b6_threat_memory_ticks": 10.0},
        ),
        (
            "b6_risk_corridor_h56_bridge_policy",
            6,
            "current_bridge",
            56,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "risk_corridor_h56",
            {"b6_family": 1.0, "b6_risk_threshold": 0.32, "b6_corridor_hunger": 0.84},
        ),
        (
            "b6_genetic_risk_corridor_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "genetic_risk_corridor",
            {"b6_family": 1.0},
        ),
        (
            "b6_recurrent_context_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "recurrent_context",
            {"b6_family": 2.0, "b6_recurrent_decay": 0.70, "b6_return_lock_ticks": 8.0},
        ),
        (
            "b6_recurrent_threat_homeostasis_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "recurrent_threat_homeostasis",
            {"b6_family": 2.0, "b6_risk_threshold": 0.28, "b6_threat_memory_ticks": 12.0},
        ),
        (
            "b6_recurrent_corridor_guard_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "recurrent_corridor_guard",
            {"b6_family": 2.0, "b6_corridor_hunger": 0.82, "b6_corridor_lock_ticks": 16.0},
        ),
        (
            "b6_recurrent_context_h56_bridge_policy",
            6,
            "current_bridge",
            56,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "recurrent_context_h56",
            {"b6_family": 2.0, "b6_recurrent_decay": 0.75, "b6_corridor_hunger": 0.84},
        ),
        (
            "b6_genetic_recurrent_memory_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "genetic_recurrent_memory",
            {"b6_family": 2.0},
        ),
        (
            "b6_fused_risk_recurrent_h48_bridge_policy",
            6,
            "current_bridge",
            48,
            5,
            "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best",
            "fused_risk_recurrent",
            {"b6_family": 3.0, "b6_corridor_hunger": 0.82, "b6_threat_memory_ticks": 12.0},
        ),
    )
