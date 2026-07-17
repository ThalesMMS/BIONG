from __future__ import annotations

import hashlib
import json

from .shared import *

from spider_cortex_sim.ablation._catalog_diagnostic_direct_core import (
    DIRECT_CORE_DIAGNOSTIC_VARIANT_SPECS,
    diagnostic_direct_core_configs,
)
from spider_cortex_sim.ablation._catalog_diagnostic_direct_local_branches import (
    diagnostic_direct_local_branches_configs,
)
from spider_cortex_sim.ablation._catalog_diagnostic_direct_teacher_replay import (
    diagnostic_direct_teacher_replay_configs,
)
from spider_cortex_sim.ablation._catalog_shared import (
    build_direct_policy_diagnostic_configs,
)


class DirectPolicyCatalogBuildersTest(unittest.TestCase):
    def test_direct_core_specs_generate_existing_catalog_surface(self) -> None:
        from_catalog = diagnostic_direct_core_configs({})
        from_specs = build_direct_policy_diagnostic_configs(
            DIRECT_CORE_DIAGNOSTIC_VARIANT_SPECS,
            {},
        )

        self.assertEqual(tuple(from_catalog), tuple(from_specs))
        self.assertEqual(
            self._direct_policy_surface(from_catalog),
            self._direct_policy_surface(from_specs),
        )
        self.assertEqual(
            from_specs["direct_mlp_policy"].direct_policy_hidden_dims,
            (128, 64),
        )
        self.assertTrue(from_specs["direct_mlp_policy"].enable_food_direction_bias)
        self.assertFalse(
            from_specs["behavior_tree_oracle_policy"].enable_food_direction_bias
        )
        self.assertTrue(
            from_specs["true_monolithic_owned_option_controller_policy"]
            .direct_policy_owned_option_controller
        )

    def test_full_direct_policy_catalog_surface_digest_is_stable(self) -> None:
        configs = self._all_direct_policy_configs()
        digest = hashlib.sha256(
            json.dumps(
                self._direct_policy_surface(configs),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        self.assertEqual(len(configs), 83)
        self.assertEqual(
            tuple(configs)[:5],
            (
                "direct_mlp_policy",
                "behavior_tree_oracle_policy",
                "true_monolithic_recurrent_policy",
                "true_monolithic_recurrent_phase_policy",
                "true_monolithic_owned_option_controller_policy",
            ),
        )
        self.assertEqual(
            tuple(configs)[-5:],
            (
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_phase_option_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_option_sequence_head_post_rest_probe_replayable_teacher_distill_option_replay_policy",
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_trace_distill_option_replay_policy",
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_action_token_post_rest_probe_replayable_teacher_distill_option_replay_policy",
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_replayable_teacher_distill_option_margin_replay_policy",
            ),
        )
        self.assertEqual(
            digest,
            "3797752238383a1df1da9d622751daf64b8ef9acaebfb17f422527f3e78b5fee",
        )

    def test_phase_option_dynamics_variants_keep_phase_head_enabled(self) -> None:
        configs = self._all_direct_policy_configs()
        phase_option_dynamics = {
            name: config
            for name, config in configs.items()
            if "phase_option_dynamics" in name
        }

        self.assertTrue(phase_option_dynamics)
        self.assertEqual(
            [
                name
                for name, config in phase_option_dynamics.items()
                if not config.direct_policy_phase_head
            ],
            [],
        )

    def _all_direct_policy_configs(self) -> Dict[str, BrainAblationConfig]:
        configs: Dict[str, BrainAblationConfig] = {}
        for build_configs in (
            diagnostic_direct_core_configs,
            diagnostic_direct_teacher_replay_configs,
            diagnostic_direct_local_branches_configs,
        ):
            configs.update(build_configs({}))
        return configs

    def _direct_policy_surface(
        self,
        configs: Dict[str, BrainAblationConfig],
    ) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
        default = BrainAblationConfig()
        return tuple(
            (
                name,
                tuple(
                    (field_name, getattr(config, field_name))
                    for field_name in self._COMMON_FIELDS
                )
                + (
                    (
                        "direct_policy",
                        {
                            field_name: getattr(config, field_name)
                            for field_name in self._DIRECT_POLICY_FIELDS
                            if getattr(config, field_name) != getattr(default, field_name)
                        },
                    ),
                ),
            )
            for name, config in configs.items()
        )

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
    )

    _DIRECT_POLICY_FIELDS = tuple(
        field_name
        for field_name in BrainAblationConfig.__dataclass_fields__
        if field_name.startswith("direct_policy_")
    )
