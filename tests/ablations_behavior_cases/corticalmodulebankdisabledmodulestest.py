from __future__ import annotations

from .shared import *


class CorticalModuleBankDisabledModulesTest(unittest.TestCase):
    """Tests for CorticalModuleBank.disabled_modules feature (new in this PR)."""

    def _make_bank(self, disabled: tuple[str, ...] = ()) -> CorticalModuleBank:
        """
        Constructs a CorticalModuleBank with a fixed RNG, zero module dropout, and the specified disabled modules.
        
        Parameters:
            disabled (tuple[str, ...]): Names of modules to disable in the returned bank.
        
        Returns:
            CorticalModuleBank: Bank with action_dim equal to the number of locomotion actions, RNG seeded with 42, module_dropout set to 0.0, and disabled_modules set to `disabled`.
        """
        rng = np.random.default_rng(42)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
            disabled_modules=disabled,
        )

    def _blank_observation(self) -> dict:
        """
        Constructs a blank observation mapping for the brain's module interfaces.
        
        Returns:
            dict: Mapping from each interface's `observation_key` to a NumPy float array of zeros with length equal to that interface's `input_dim`.
        """
        obs = {}
        for spec in self._make_bank().specs:
            obs[spec.observation_key] = np.zeros(spec.input_dim, dtype=float)
        return obs

    def test_disabled_modules_stored_as_set(self) -> None:
        bank = self._make_bank(("alert_center",))
        self.assertIsInstance(bank.disabled_modules, set)

    def test_unknown_disabled_module_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                disabled_modules=("nonexistent_module",),
            )

    def test_unknown_hidden_dim_override_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                hidden_dims={"nonexistent_module": 16},
            )

    def test_non_positive_hidden_dim_override_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "int > 0"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                hidden_dims={"visual_cortex": 0},
            )

    def test_all_disabled_modules_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(
            ValueError,
            "all modules were disabled; at least one module must be enabled",
        ):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                disabled_modules=MODULE_NAMES,
            )

    def test_disabled_module_preserves_canonical_specs_and_prunes_enabled_specs(self) -> None:
        bank = self._make_bank(("hunger_center",))
        self.assertIn("hunger_center", [spec.name for spec in bank.specs])
        self.assertNotIn("hunger_center", [spec.name for spec in bank.enabled_specs])

    def test_disabled_module_is_absent_from_networks_and_results(self) -> None:
        """
        Verifies that a disabled module is removed from the bank rather than kept as an inactive zero-logit slot.
        
        Asserts the disabled module is absent from both the instantiated network map and the
        returned ModuleResult list.
        """
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        self.assertNotIn("hunger_center", bank.modules)
        self.assertNotIn("hunger_center", [result.name for result in results])

    def test_disabled_module_not_returned(self) -> None:
        bank = self._make_bank(("sleep_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        self.assertNotIn("sleep_center", [result.name for result in results])

    def test_load_state_dict_ignores_modules_absent_from_bank(self) -> None:
        source_bank = self._make_bank(())
        target_bank = self._make_bank(("hunger_center",))
        loaded = target_bank.load_state_dict(
            source_bank.state_dict(),
            modules=list(source_bank.state_dict().keys()),
        )
        expected_loaded = set(source_bank.state_dict().keys()) & set(target_bank.state_dict().keys())
        self.assertSetEqual(set(loaded), expected_loaded)

    def test_enabled_modules_remain_active(self) -> None:
        bank = self._make_bank(("alert_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        active_names = [r.name for r in results if r.active]
        self.assertNotIn("alert_center", active_names)
        self.assertIn("visual_cortex", active_names)
        self.assertIn("hunger_center", active_names)

    def test_no_disabled_modules_all_active(self) -> None:
        bank = self._make_bank(())
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        self.assertTrue(all(r.active for r in results))

    def test_dropout_keeps_one_non_disabled_module_active(self) -> None:
        rng = np.random.default_rng(42)
        bank = CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=1.0,
            disabled_modules=(
                "sensory_cortex",
                "hunger_center",
                "sleep_center",
                "alert_center",
            ),
        )
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=True)

        active_names = [result.name for result in results if result.active]
        self.assertEqual(len(active_names), 1)
        self.assertNotIn("sensory_cortex", active_names)
        self.assertNotIn("hunger_center", active_names)
        self.assertNotIn("sleep_center", active_names)
        self.assertNotIn("alert_center", active_names)
