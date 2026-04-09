import unittest

from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.reward import (
    REWARD_COMPONENT_NAMES,
    REWARD_COMPONENT_AUDIT,
    REWARD_PROFILES,
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_component_audit,
    reward_profile_audit,
    reward_total,
)
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.world import SpiderWorld
from spider_cortex_sim.world_types import TickContext, TickSnapshot


class RewardModuleTest(unittest.TestCase):
    def _profile_with_reward_updates(self, **updates: float) -> OperationalProfile:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "reward_test_profile"
        summary["version"] = 7
        allowed_keys = set(summary["reward"].keys())
        invalid_keys = sorted(set(updates) - allowed_keys)
        if invalid_keys:
            raise ValueError(
                f"Unknown reward profile field(s): {', '.join(invalid_keys)}. "
                f"Allowed keys: {', '.join(sorted(allowed_keys))}."
            )
        summary["reward"].update({name: float(value) for name, value in updates.items()})
        return OperationalProfile.from_summary(summary)

    def _tick_context(
        self,
        world: SpiderWorld,
        *,
        action_name: str,
        moved: bool,
        night: bool,
        terrain_now: str,
        was_on_shelter: bool,
        prev_food_dist: int,
        prev_shelter_dist: int,
        prev_predator_dist: int,
        prev_predator_visible: bool,
        prev_spider_pos: tuple[int, int],
        prev_lizard_pos: tuple[int, int],
        prev_rest_streak: int | None = None,
    ) -> TickContext:
        return TickContext(
            action_idx=0,
            intended_action=action_name,
            executed_action=action_name,
            motor_noise_applied=False,
            snapshot=TickSnapshot(
                tick=int(world.tick),
                spider_pos=prev_spider_pos,
                lizard_pos=prev_lizard_pos,
                was_on_shelter=bool(was_on_shelter),
                prev_shelter_role=world.shelter_role_at(prev_spider_pos),
                prev_food_dist=int(prev_food_dist),
                prev_shelter_dist=int(prev_shelter_dist),
                prev_predator_dist=int(prev_predator_dist),
                prev_predator_visible=bool(prev_predator_visible),
                night=bool(night),
                rest_streak=int(world.state.rest_streak if prev_rest_streak is None else prev_rest_streak),
            ),
            reward_components=empty_reward_components(),
            info={"distance_deltas": {}},
            moved=bool(moved),
            terrain_now=terrain_now,
        )

    def test_empty_reward_components_matches_declared_names(self) -> None:
        reward_components = empty_reward_components()
        self.assertEqual(set(reward_components.keys()), set(REWARD_COMPONENT_NAMES))
        self.assertTrue(all(value == 0.0 for value in reward_components.values()))

    def test_reward_total_and_copy_round_trip(self) -> None:
        reward_components = empty_reward_components()
        reward_components["feeding"] = 1.2
        reward_components["action_cost"] = -0.2
        copied = copy_reward_components(reward_components)
        self.assertEqual(copied, reward_components)
        self.assertAlmostEqual(reward_total(reward_components), 1.0)

    def test_apply_action_and_terrain_effects_updates_state_and_costs(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999, map_template="entrance_funnel")
        world.reset(seed=1)
        world.state.x, world.state.y = 5, world.height // 2
        reward_components = empty_reward_components()
        hunger_before = world.state.hunger
        fatigue_before = world.state.fatigue
        terrain_now = apply_action_and_terrain_effects(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            reward_components=reward_components,
        )
        self.assertEqual(terrain_now, world.terrain_at(world.spider_pos()))
        self.assertLess(reward_components["action_cost"], 0.0)
        self.assertGreater(world.state.hunger, hunger_before)
        self.assertGreater(world.state.fatigue, fatigue_before)

    def test_compute_predator_threat_detects_visible_predator(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x, world.state.y = 2, 2
        world.lizard.x, world.lizard.y = 4, 2
        self.assertTrue(
            compute_predator_threat(
                world,
                prev_predator_visible=True,
                prev_predator_dist=2,
            )
        )

    def test_apply_pressure_penalties_uses_state_levels(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.hunger = 0.8
        world.state.fatigue = 0.6
        world.state.sleep_debt = 0.4
        reward_components = empty_reward_components()
        apply_pressure_penalties(world, reward_components)
        self.assertLess(reward_components["hunger_pressure"], 0.0)
        self.assertLess(reward_components["fatigue_pressure"], 0.0)
        self.assertLess(reward_components["sleep_debt_pressure"], 0.0)

    def test_reward_profiles_contain_required_keys(self) -> None:
        """
        Verify each reward profile in REWARD_PROFILES includes the full set of required reward configuration keys.
        
        This test checks that every profile contains the following keys:
        "action_cost_stay", "action_cost_move", "hunger_pressure", "fatigue_pressure",
        "sleep_debt_pressure", "food_progress", "shelter_progress", "predator_escape",
        "predator_escape_bonus", "sleep_debt_night_awake", "sleep_debt_day_awake",
        "sleep_debt_overdue_threshold", "sleep_debt_health_penalty", "recent_contact_decay",
        "recent_pain_decay", and "death_penalty".
        
        If any key is missing from a profile, the test fails with a message identifying
        the profile name and the missing key.
        """
        required_keys = {
            "action_cost_stay", "action_cost_move",
            "hunger_pressure", "fatigue_pressure", "sleep_debt_pressure",
            "food_progress", "shelter_progress",
            "predator_escape", "predator_escape_bonus",
            "sleep_debt_night_awake", "sleep_debt_day_awake",
            "sleep_debt_overdue_threshold", "sleep_debt_health_penalty",
            "recent_contact_decay", "recent_pain_decay", "death_penalty",
        }
        for profile_name, profile in REWARD_PROFILES.items():
            for key in required_keys:
                self.assertIn(key, profile, f"Profile '{profile_name}' missing key: {key}")

    def test_reward_profiles_has_classic_and_ecological(self) -> None:
        self.assertIn("classic", REWARD_PROFILES)
        self.assertIn("ecological", REWARD_PROFILES)
        self.assertIn("austere", REWARD_PROFILES)

    def test_reward_profiles_ecological_stricter_penalties(self) -> None:
        classic = REWARD_PROFILES["classic"]
        ecological = REWARD_PROFILES["ecological"]
        self.assertGreater(ecological["hunger_pressure"], classic["hunger_pressure"])
        self.assertGreater(ecological["fatigue_pressure"], classic["fatigue_pressure"])

    def test_austere_profile_zeroes_progress_guidance_terms(self) -> None:
        austere = REWARD_PROFILES["austere"]
        self.assertEqual(austere["food_progress"], 0.0)
        self.assertEqual(austere["shelter_progress"], 0.0)
        self.assertEqual(austere["threat_shelter_progress"], 0.0)
        self.assertEqual(austere["predator_escape"], 0.0)
        self.assertEqual(austere["predator_escape_bonus"], 0.0)
        self.assertEqual(austere["day_exploration_hungry"], 0.0)
        self.assertEqual(austere["day_exploration_calm"], 0.0)

    def test_reward_component_audit_covers_declared_components(self) -> None:
        audit = reward_component_audit()
        self.assertEqual(set(audit.keys()), set(REWARD_COMPONENT_NAMES))
        self.assertEqual(set(REWARD_COMPONENT_AUDIT.keys()), set(REWARD_COMPONENT_NAMES))

    def test_reward_component_audit_classifies_progress_components(self) -> None:
        audit = reward_component_audit()
        self.assertEqual(audit["food_progress"]["category"], "progress")
        self.assertEqual(audit["shelter_progress"]["category"], "progress")
        self.assertEqual(audit["predator_escape"]["category"], "progress")

    def test_reward_profile_audit_reduces_progress_weight_in_austere_profile(self) -> None:
        classic_audit = reward_profile_audit("classic")
        austere_audit = reward_profile_audit("austere")
        self.assertEqual(austere_audit["profile"], "austere")
        self.assertEqual(
            austere_audit["component_weight_proxy"]["food_progress"],
            0.0,
        )
        self.assertEqual(
            austere_audit["component_weight_proxy"]["shelter_progress"],
            0.0,
        )
        self.assertLess(
            austere_audit["category_weight_proxy"]["progress"],
            classic_audit["category_weight_proxy"]["progress"],
        )

    def test_reward_profile_audit_exposes_configurable_and_hardcoded_mass(self) -> None:
        audit = reward_profile_audit("austere")
        self.assertIn("configurable_category_weight_proxy", audit)
        self.assertIn("hardcoded_mass", audit)
        self.assertGreater(audit["hardcoded_mass"], 0.0)
        self.assertLessEqual(
            audit["configurable_category_weight_proxy"]["progress"],
            audit["category_weight_proxy"]["progress"],
        )

    def test_reward_component_audit_returns_deep_copy_not_reference(self) -> None:
        audit1 = reward_component_audit()
        audit2 = reward_component_audit()
        audit1["food_progress"]["category"] = "mutated"
        self.assertEqual(audit2["food_progress"]["category"], "progress")

    def test_reward_component_audit_event_category_components_present(self) -> None:
        audit = reward_component_audit()
        expected_event_components = {
            "action_cost", "terrain_cost", "night_exposure",
            "predator_contact", "feeding", "resting", "shelter_entry",
            "night_shelter_bonus", "death_penalty",
        }
        actual_event_components = {
            name for name, data in audit.items()
            if data["category"] == "event"
        }
        self.assertEqual(actual_event_components, expected_event_components)

    def test_reward_component_audit_internal_pressure_components_present(self) -> None:
        audit = reward_component_audit()
        expected_pressure_components = {
            "hunger_pressure", "fatigue_pressure",
            "sleep_debt_pressure", "homeostasis_penalty",
        }
        actual_pressure_components = {
            name for name, data in audit.items()
            if data["category"] == "internal_pressure"
        }
        self.assertEqual(actual_pressure_components, expected_pressure_components)

    def test_reward_component_audit_all_entries_have_required_fields(self) -> None:
        audit = reward_component_audit()
        required_fields = {"category", "source", "gates", "config_keys", "configured_weight_keys", "shaping_risk", "notes"}
        for component_name, metadata in audit.items():
            for field in required_fields:
                self.assertIn(
                    field,
                    metadata,
                    f"Reward component {component_name!r} missing field {field!r}",
                )

    def test_reward_component_audit_shaping_risk_values_are_valid(self) -> None:
        audit = reward_component_audit()
        valid_risk_levels = {"low", "medium", "high"}
        for name, data in audit.items():
            self.assertIn(
                data["shaping_risk"],
                valid_risk_levels,
                f"Component {name!r} has unexpected shaping_risk {data['shaping_risk']!r}",
            )

    def test_reward_profile_audit_raises_key_error_for_unknown_profile(self) -> None:
        with self.assertRaises(KeyError):
            reward_profile_audit("nonexistent_profile")

    def test_reward_profile_audit_classic_has_all_expected_keys(self) -> None:
        audit = reward_profile_audit("classic")
        expected_keys = {
            "profile", "component_weight_proxy", "category_weight_proxy",
            "configurable_category_weight_proxy", "dominant_category",
            "hardcoded_mass", "non_configurable_components", "notes",
        }
        self.assertEqual(set(audit.keys()), expected_keys)

    def test_reward_profile_audit_dominant_category_is_valid(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            self.assertIn(
                audit["dominant_category"],
                {"event", "progress", "internal_pressure"},
                f"Profile {profile_name!r} has unexpected dominant_category",
            )

    def test_reward_profile_audit_non_configurable_components_includes_expected(self) -> None:
        audit = reward_profile_audit("classic")
        non_configurable = audit["non_configurable_components"]
        for expected in ("predator_contact", "feeding"):
            self.assertIn(
                expected,
                non_configurable,
                f"Expected {expected!r} in non_configurable_components",
            )
        self.assertNotIn("death_penalty", non_configurable)

    def test_reward_profile_audit_non_configurable_components_is_sorted(self) -> None:
        audit = reward_profile_audit("classic")
        components = audit["non_configurable_components"]
        self.assertEqual(components, sorted(components))

    def test_reward_profile_audit_notes_is_non_empty_list(self) -> None:
        audit = reward_profile_audit("classic")
        self.assertIsInstance(audit["notes"], list)
        self.assertGreater(len(audit["notes"]), 0)

    def test_reward_profile_audit_category_weight_proxy_has_three_categories(self) -> None:
        audit = reward_profile_audit("classic")
        self.assertEqual(
            set(audit["category_weight_proxy"].keys()),
            {"event", "progress", "internal_pressure"},
        )

    def test_reward_profile_audit_classic_total_weight_is_positive(self) -> None:
        audit = reward_profile_audit("classic")
        total = sum(audit["category_weight_proxy"].values())
        self.assertGreater(total, 0.0)

    def test_reward_profile_audit_component_weights_are_non_negative(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            for comp_name, weight in audit["component_weight_proxy"].items():
                self.assertGreaterEqual(
                    weight,
                    0.0,
                    f"Profile {profile_name!r} component {comp_name!r} has negative weight proxy",
                )

    def test_austere_profile_has_non_zero_action_costs_and_pressures(self) -> None:
        austere = REWARD_PROFILES["austere"]
        self.assertGreater(austere["action_cost_stay"], 0.0)
        self.assertGreater(austere["action_cost_move"], 0.0)
        self.assertGreater(austere["hunger_pressure"], 0.0)
        self.assertGreater(austere["fatigue_pressure"], 0.0)
        self.assertGreater(austere["sleep_debt_pressure"], 0.0)

    def test_austere_profile_zeroes_shelter_entry_and_night_bonus(self) -> None:
        austere = REWARD_PROFILES["austere"]
        self.assertEqual(austere["shelter_entry"], 0.0)
        self.assertEqual(austere["night_shelter_bonus"], 0.0)

    def test_reward_profile_audit_ecological_has_progress_category(self) -> None:
        audit = reward_profile_audit("ecological")
        self.assertIn("progress", audit["category_weight_proxy"])
        self.assertGreater(audit["category_weight_proxy"]["progress"], 0.0)

    def test_reward_component_audit_resting_has_hardcoded_weight(self) -> None:
        audit = reward_component_audit()
        self.assertIn("hardcoded_weight", audit["resting"])
        self.assertGreater(float(audit["resting"]["hardcoded_weight"]), 0.0)

    def test_reward_component_audit_death_penalty_is_configurable(self) -> None:
        audit = reward_component_audit()
        self.assertEqual(audit["death_penalty"]["config_keys"], ["death_penalty"])
        self.assertEqual(audit["death_penalty"]["configured_weight_keys"], ["death_penalty"])
        self.assertNotIn("hardcoded_weight", audit["death_penalty"])

    def test_profile_with_reward_updates_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._profile_with_reward_updates(not_a_real_reward_key=1.0)
        self.assertIn("not_a_real_reward_key", str(ctx.exception))

    def test_copy_reward_components_is_independent(self) -> None:
        original = empty_reward_components()
        original["feeding"] = 2.5
        copied = copy_reward_components(original)
        self.assertAlmostEqual(copied["feeding"], 2.5)
        copied["feeding"] = 9.9
        self.assertAlmostEqual(original["feeding"], 2.5)

    def test_reward_total_handles_negative_values(self) -> None:
        components = empty_reward_components()
        components["action_cost"] = -0.5
        components["predator_contact"] = -2.0
        components["feeding"] = 3.0
        total = reward_total(components)
        self.assertAlmostEqual(total, 0.5)

    def test_reward_total_all_zeros(self) -> None:
        components = empty_reward_components()
        self.assertAlmostEqual(reward_total(components), 0.0)

    def test_apply_action_and_terrain_effects_stay_cost(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        reward_components = empty_reward_components()
        apply_action_and_terrain_effects(
            world,
            action_name="STAY",
            moved=False,
            reward_components=reward_components,
        )
        expected_cost = -world.reward_config["action_cost_stay"]
        self.assertAlmostEqual(reward_components["action_cost"], expected_cost)

    def test_apply_action_and_terrain_effects_move_costs_more_than_stay(self) -> None:
        world_stay = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_stay.reset(seed=1)
        rc_stay = empty_reward_components()
        apply_action_and_terrain_effects(world_stay, action_name="STAY", moved=False, reward_components=rc_stay)

        world_move = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_move.reset(seed=1)
        rc_move = empty_reward_components()
        apply_action_and_terrain_effects(world_move, action_name="MOVE_RIGHT", moved=True, reward_components=rc_move)

        self.assertLess(rc_move["action_cost"], rc_stay["action_cost"])

    def test_compute_predator_threat_false_when_safe(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 1, 1
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 0.0
        world.state.recent_pain = 0.0
        result = compute_predator_threat(
            world,
            prev_predator_visible=False,
            prev_predator_dist=10,
        )
        self.assertFalse(result)

    def test_compute_predator_threat_uses_operational_profile_thresholds(self) -> None:
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(predator_threat_smell_threshold=0.0),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 1, 1
        world.lizard.x, world.lizard.y = 2, 1
        world.state.recent_contact = 0.0
        world.state.recent_pain = 0.0
        result = compute_predator_threat(
            world,
            prev_predator_visible=False,
            prev_predator_dist=10,
        )
        self.assertTrue(result)

    def test_compute_predator_threat_uses_deterministic_smell_signal(self) -> None:
        noise_profile = NoiseConfig(
            name="olfactory_drop_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 1.0, "direction_jitter": 1.0},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            noise_profile=noise_profile,
            operational_profile=self._profile_with_reward_updates(predator_threat_smell_threshold=0.0),
        )
        for episode_seed in (1, 7, 29, 101):
            world.reset(seed=episode_seed)
            world.state.x, world.state.y = 1, 1
            world.lizard.x, world.lizard.y = 2, 1
            world.state.recent_contact = 0.0
            world.state.recent_pain = 0.0

            result = compute_predator_threat(
                world,
                prev_predator_visible=False,
                prev_predator_dist=10,
            )

            self.assertTrue(result, f"Threat detection changed under episode_seed={episode_seed}")

    def test_apply_progress_and_event_rewards_food_approach(self) -> None:
        """
        Tests that moving the spider closer to a food target increases the `food_progress` reward and records distance changes.
        
        Sets up a world with a food item, moves the spider one step nearer, invokes apply_progress_and_event_rewards, and asserts that `info` contains a "distance_deltas" entry and that `reward_components["food_progress"]` is greater than 0.0.
        """
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.food_positions = [(5, 3)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.hunger = 0.7

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        prev_food_dist = world.manhattan(world.spider_pos(), world.food_positions[0])
        world.state.x = 4
        context = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=prev_food_dist,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertIn("distance_deltas", context.info)
        self.assertGreater(context.reward_components["food_progress"], 0.0)

    def test_apply_progress_food_gate_uses_operational_profile_threshold(self) -> None:
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(food_progress_hunger_threshold=0.95),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.food_positions = [(5, 3)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.hunger = 0.9

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        prev_food_dist = world.manhattan(world.spider_pos(), world.food_positions[0])
        world.state.x = 4
        context = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=prev_food_dist,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertEqual(context.reward_components["food_progress"], 0.0)

    def test_apply_progress_and_event_rewards_shelter_entry_bonus(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        world.state.x, world.state.y = sorted(entrance_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.shelter_entries = 0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=3,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertGreater(context.reward_components["shelter_entry"], 0.0)
        self.assertEqual(world.state.shelter_entries, 1)

    def test_apply_progress_and_event_rewards_day_exploration_reward(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.state.hunger = 0.5
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.tick = 0

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertGreaterEqual(context.reward_components["day_exploration"], 0.0)

    def test_apply_progress_and_event_rewards_escape_bonus(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 0.0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=3,
            prev_predator_visible=True,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertGreater(context.reward_components["predator_escape"], 0.0)
        self.assertTrue(context.predator_escape)

    def test_apply_progress_and_event_rewards_updates_predator_visible_now(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 2, 2
        world.lizard.x, world.lizard.y = 3, 2
        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        prev_predator_dist = world.manhattan(prev_spider_pos, prev_lizard_pos)

        visible_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=world.terrain_at(world.spider_pos()),
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=prev_predator_dist,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=visible_context)
        self.assertTrue(visible_context.predator_visible_now)

        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        far_lizard_pos = world.lizard_pos()
        far_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=world.terrain_at(world.spider_pos()),
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=world.manhattan(prev_spider_pos, far_lizard_pos),
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=far_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=far_context)
        self.assertFalse(far_context.predator_visible_now)

    def test_shelter_progress_gate_uses_operational_profile_fatigue_threshold(self) -> None:
        # High fatigue threshold in profile → shelter progress not awarded for moderate fatigue
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(shelter_progress_fatigue_threshold=0.99),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.fatigue = 0.5
        world.state.sleep_debt = 0.0

        shelter_cells = list(world.shelter_deep_cells or world.shelter_cells)
        if not shelter_cells:
            self.skipTest("No shelter cells")
        original_pos = world.spider_pos()
        _, prev_shelter_dist = world.nearest(shelter_cells)
        candidate_moves = {
            "MOVE_LEFT": (-1, 0),
            "MOVE_RIGHT": (1, 0),
            "MOVE_UP": (0, -1),
            "MOVE_DOWN": (0, 1),
        }
        chosen_action = None
        new_shelter_dist = prev_shelter_dist
        for action_name, (dx, dy) in candidate_moves.items():
            new_x = original_pos[0] + dx
            new_y = original_pos[1] + dy
            if not (0 <= new_x < world.width and 0 <= new_y < world.height):
                continue
            world.state.x, world.state.y = new_x, new_y
            _, candidate_shelter_dist = world.nearest(shelter_cells)
            if candidate_shelter_dist < prev_shelter_dist:
                chosen_action = action_name
                new_shelter_dist = candidate_shelter_dist
                break
        if chosen_action is None:
            self.fail("Expected at least one one-step move that reduces shelter distance.")
        self.assertGreater(prev_shelter_dist, new_shelter_dist)

        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name=chosen_action,
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=prev_shelter_dist,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=original_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertEqual(context.reward_components["shelter_progress"], 0.0)

    def test_day_exploration_hunger_threshold_uses_operational_profile(self) -> None:
        # Raise hunger threshold so that moderate hunger is treated as "calm"
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(day_exploration_hunger_threshold=0.90),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.state.hunger = 0.6  # above default 0.35, below custom 0.90
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.tick = 0

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context_hungry = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context_hungry)
        # At hunger=0.6 with threshold=0.90, should get calm exploration (not hungry bonus)
        default_world = SpiderWorld(seed=1, lizard_move_interval=999999)
        default_world.reset(seed=1)
        default_world.state.x, default_world.state.y = 3, 3
        default_world.state.hunger = 0.6
        default_world.lizard.x, default_world.lizard.y = default_world.width - 1, default_world.height - 1
        default_world.tick = 0

        default_prev_spider_pos = default_world.spider_pos()
        default_prev_lizard_pos = default_world.lizard_pos()
        context_default = self._tick_context(
            default_world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=default_prev_spider_pos,
            prev_lizard_pos=default_prev_lizard_pos,
        )
        apply_progress_and_event_rewards(default_world, tick_context=context_default)
        # Custom world has raised threshold so gets calm bonus; default world gets hungry bonus
        self.assertGreater(
            context_default.reward_components["day_exploration"],
            context_hungry.reward_components["day_exploration"],
        )

    def test_predator_threat_distance_threshold_uses_operational_profile(self) -> None:
        # Raise distance threshold so threat is detected from further away
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(predator_threat_distance_threshold=10.0),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 1, 1
        world.lizard.x, world.lizard.y = world.width - 2, world.height - 2
        world.state.recent_contact = 0.0
        world.state.recent_pain = 0.0

        result = compute_predator_threat(
            world,
            prev_predator_visible=False,
            prev_predator_dist=10,
        )
        self.assertTrue(result)

    def test_world_stores_operational_profile(self) -> None:
        profile = self._profile_with_reward_updates(predator_threat_smell_threshold=0.99)
        world = SpiderWorld(seed=1, lizard_move_interval=999999, operational_profile=profile)
        self.assertIs(world.operational_profile, profile)

    def test_apply_progress_and_event_rewards_records_distance_deltas_event(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.food_positions = [(5, 3)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.hunger = 0.7

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        prev_food_dist = world.manhattan(world.spider_pos(), world.food_positions[0])
        world.state.x = 4
        context = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=prev_food_dist,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        event_names = [e.name for e in context.event_log]
        self.assertIn("distance_deltas", event_names)
        event = next(e for e in context.event_log if e.name == "distance_deltas")
        self.assertIn("food", event.payload)
        self.assertIn("shelter", event.payload)
        self.assertIn("predator", event.payload)
        self.assertIsInstance(event.payload["food"], int)
        self.assertIsInstance(event.payload["shelter"], int)
        self.assertIsInstance(event.payload["predator"], int)

    def test_apply_progress_and_event_rewards_records_distance_deltas_in_stage_reward(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        # distance_deltas is the primary event recorded by this function directly
        event_stages = [e.stage for e in context.event_log]
        self.assertTrue(all(s == "reward" for s in event_stages), "All events from apply_progress_and_event_rewards should be in 'reward' stage")
        event_names = [e.name for e in context.event_log]
        self.assertIn("distance_deltas", event_names)

    def test_apply_progress_and_event_rewards_sets_predator_visible_now_in_context(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        self.assertFalse(context.predator_visible_now)
        apply_progress_and_event_rewards(world, tick_context=context)
        # predator_visible_now should be a bool
        self.assertIsInstance(context.predator_visible_now, bool)

    def test_apply_progress_and_event_rewards_shelter_entry_records_event(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        world.state.x, world.state.y = sorted(entrance_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.shelter_entries = 0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=False,  # Coming from outside, entering shelter
            prev_food_dist=5,
            prev_shelter_dist=3,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        if context.reward_components["shelter_entry"] > 0.0:
            event_names = [e.name for e in context.event_log]
            self.assertIn("shelter_entry", event_names)
            entry_event = next(e for e in context.event_log if e.name == "shelter_entry")
            self.assertIn("shelter_role", entry_event.payload)

    def test_apply_progress_and_event_rewards_predator_escape_sets_context_flag(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 0.0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=3,
            prev_predator_visible=True,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        # context.predator_escape should be set by the function
        self.assertIsInstance(context.predator_escape, bool)
        if context.predator_escape:
            event_names = [e.name for e in context.event_log]
            self.assertIn("predator_escape", event_names)

    def test_apply_progress_and_event_rewards_predator_escape_bonus_is_one_shot_per_threat_episode(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 1.0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        first_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=True,
            prev_food_dist=5,
            prev_shelter_dist=0,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=first_context)

        second_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=True,
            prev_food_dist=5,
            prev_shelter_dist=0,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=second_context)

        self.assertTrue(first_context.predator_escape)
        self.assertFalse(second_context.predator_escape)
        self.assertEqual(world.state.predator_escapes, 1)
        self.assertIn(
            "predator_escape",
            [event.name for event in first_context.event_log],
        )
        self.assertNotIn(
            "predator_escape",
            [event.name for event in second_context.event_log],
        )

    def test_apply_progress_and_event_rewards_info_distance_deltas_populated(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertIn("distance_deltas", context.info)
        deltas = context.info["distance_deltas"]
        self.assertIn("food", deltas)
        self.assertIn("shelter", deltas)
        self.assertIn("predator", deltas)


class AustreProfileStructureTest(unittest.TestCase):
    """Regression tests for the austere reward profile added in this PR."""

    def test_austere_profile_has_same_keys_as_classic(self) -> None:
        self.assertEqual(
            set(REWARD_PROFILES["austere"].keys()),
            set(REWARD_PROFILES["classic"].keys()),
        )

    def test_austere_profile_has_same_keys_as_ecological(self) -> None:
        self.assertEqual(
            set(REWARD_PROFILES["austere"].keys()),
            set(REWARD_PROFILES["ecological"].keys()),
        )

    def test_all_three_profiles_have_identical_key_sets(self) -> None:
        classic_keys = set(REWARD_PROFILES["classic"].keys())
        ecological_keys = set(REWARD_PROFILES["ecological"].keys())
        austere_keys = set(REWARD_PROFILES["austere"].keys())
        self.assertEqual(classic_keys, ecological_keys)
        self.assertEqual(classic_keys, austere_keys)

    def test_austere_predator_escape_stay_penalty_is_zero(self) -> None:
        self.assertEqual(REWARD_PROFILES["austere"]["predator_escape_stay_penalty"], 0.0)

    def test_austere_sleep_debt_terms_are_positive(self) -> None:
        austere = REWARD_PROFILES["austere"]
        self.assertGreater(austere["sleep_debt_night_awake"], 0.0)
        self.assertGreater(austere["sleep_debt_resting_night"], 0.0)
        self.assertGreater(austere["sleep_debt_deep_night"], 0.0)

    def test_austere_pressure_terms_are_stronger_than_ecological(self) -> None:
        # austere should have higher or equal homeostatic pressure vs. ecological
        austere = REWARD_PROFILES["austere"]
        ecological = REWARD_PROFILES["ecological"]
        self.assertGreaterEqual(austere["hunger_pressure"], ecological["hunger_pressure"])
        self.assertGreaterEqual(austere["fatigue_pressure"], ecological["fatigue_pressure"])
        self.assertGreaterEqual(austere["sleep_debt_pressure"], ecological["sleep_debt_pressure"])

    def test_all_reward_profile_names_are_registered(self) -> None:
        expected = {"classic", "ecological", "austere"}
        self.assertEqual(set(REWARD_PROFILES.keys()), expected)

    def test_austere_profile_all_values_are_finite(self) -> None:
        import math
        for key, value in REWARD_PROFILES["austere"].items():
            self.assertTrue(
                math.isfinite(value),
                f"austere[{key!r}] = {value!r} is not finite",
            )


class RewardProfileFatigueCostValuesTest(unittest.TestCase):
    """Regression tests for the current fatigue cost values across all profiles."""

    def _check_profile_fatigue_costs(self, profile_name: str) -> None:
        profile = REWARD_PROFILES[profile_name]
        self.assertAlmostEqual(profile["base_fatigue_cost"], 0.0035, places=6)
        self.assertAlmostEqual(profile["move_fatigue_cost"], 0.003, places=6)
        self.assertAlmostEqual(profile["idle_fatigue_cost"], 0.001, places=6)

    def test_classic_fatigue_costs(self) -> None:
        self._check_profile_fatigue_costs("classic")

    def test_ecological_fatigue_costs(self) -> None:
        self._check_profile_fatigue_costs("ecological")

    def test_austere_fatigue_costs(self) -> None:
        self._check_profile_fatigue_costs("austere")

    def test_move_fatigue_cost_exceeds_idle_fatigue_cost_in_all_profiles(self) -> None:
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreater(
                    profile["move_fatigue_cost"],
                    profile["idle_fatigue_cost"],
                    f"{name}: move_fatigue_cost should exceed idle_fatigue_cost",
                )


class RewardProfileNightExposureValuesTest(unittest.TestCase):
    """Regression tests for the current night-exposure values across all profiles."""

    def test_classic_night_exposure_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["night_exposure_fatigue"], 0.006, places=6)

    def test_classic_night_exposure_debt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["night_exposure_debt"], 0.009, places=6)

    def test_ecological_night_exposure_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["night_exposure_fatigue"], 0.009, places=6)

    def test_ecological_night_exposure_debt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["night_exposure_debt"], 0.014, places=6)

    def test_austere_night_exposure_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["night_exposure_fatigue"], 0.010, places=6)

    def test_austere_night_exposure_debt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["night_exposure_debt"], 0.015, places=6)

    def test_austere_exposure_fatigue_exceeds_classic(self) -> None:
        self.assertGreater(
            REWARD_PROFILES["austere"]["night_exposure_fatigue"],
            REWARD_PROFILES["classic"]["night_exposure_fatigue"],
        )

    def test_austere_exposure_debt_exceeds_classic(self) -> None:
        self.assertGreater(
            REWARD_PROFILES["austere"]["night_exposure_debt"],
            REWARD_PROFILES["classic"]["night_exposure_debt"],
        )


class RewardProfileSleepDebtPenaltyValuesTest(unittest.TestCase):
    """Regression tests for the current sleep-debt penalty values across all profiles."""

    def test_classic_sleep_debt_interrupt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["sleep_debt_interrupt"], 0.035, places=6)

    def test_ecological_sleep_debt_interrupt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["sleep_debt_interrupt"], 0.050, places=6)

    def test_austere_sleep_debt_interrupt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["sleep_debt_interrupt"], 0.060, places=6)

    def test_classic_sleep_debt_night_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["sleep_debt_night_awake"], 0.007, places=6)

    def test_ecological_sleep_debt_night_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["sleep_debt_night_awake"], 0.011, places=6)

    def test_austere_sleep_debt_night_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["sleep_debt_night_awake"], 0.012, places=6)

    def test_classic_sleep_debt_day_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["sleep_debt_day_awake"], 0.0015, places=6)

    def test_ecological_sleep_debt_day_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["sleep_debt_day_awake"], 0.002, places=6)

    def test_austere_sleep_debt_day_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["sleep_debt_day_awake"], 0.0025, places=6)

    def test_sleep_debt_interrupt_is_greater_than_night_awake_in_all_profiles(self) -> None:
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreater(
                    profile["sleep_debt_interrupt"],
                    profile["sleep_debt_night_awake"],
                    f"{name}: sleep_debt_interrupt should be greater than sleep_debt_night_awake",
                )

    def test_sleep_debt_night_awake_exceeds_day_awake_in_all_profiles(self) -> None:
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreater(
                    profile["sleep_debt_night_awake"],
                    profile["sleep_debt_day_awake"],
                    f"{name}: sleep_debt_night_awake should exceed sleep_debt_day_awake",
                )


class RewardProfileClutterNarrowFatigueValuesTest(unittest.TestCase):
    """Regression tests for the current clutter/narrow fatigue values."""

    def test_classic_clutter_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["clutter_fatigue"], 0.002, places=6)

    def test_ecological_clutter_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["clutter_fatigue"], 0.003, places=6)

    def test_austere_clutter_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["clutter_fatigue"], 0.003, places=6)

    def test_classic_narrow_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["narrow_fatigue"], 0.0015, places=6)

    def test_ecological_narrow_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["narrow_fatigue"], 0.0025, places=6)

    def test_austere_narrow_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["narrow_fatigue"], 0.0025, places=6)

    def test_clutter_fatigue_exceeds_narrow_fatigue_in_all_profiles(self) -> None:
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreaterEqual(
                    profile["clutter_fatigue"],
                    profile["narrow_fatigue"],
                    f"{name}: clutter_fatigue should be >= narrow_fatigue",
                )


if __name__ == "__main__":
    unittest.main()
