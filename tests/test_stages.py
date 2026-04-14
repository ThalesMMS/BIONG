import unittest
from unittest.mock import patch

from spider_cortex_sim import stages
from spider_cortex_sim.interfaces import ACTION_DELTAS
from spider_cortex_sim.predator import OLFACTORY_HUNTER_PROFILE, VISUAL_HUNTER_PROFILE
from spider_cortex_sim.world import ACTION_TO_INDEX, SpiderWorld
from spider_cortex_sim.world_types import StageDescriptor


class StageDescriptorTest(unittest.TestCase):
    def test_stage_descriptor_stores_fields(self) -> None:
        def stage(world: object, context: object) -> None:
            """
            No-op stage used as a placeholder in tests.
            
            This stage intentionally performs no actions and exists to satisfy a StageDescriptor's `run` contract during testing.
            
            Parameters:
                world (object): Simulation world instance (unused).
                context (object): Tick context object (unused).
            """
            return None

        descriptor = StageDescriptor(
            name="placeholder",
            run=stage,
            mutates=("context.event_log", "world.state"),
        )

        self.assertEqual(descriptor.name, "placeholder")
        self.assertIs(descriptor.run, stage)
        self.assertEqual(descriptor.mutates, ("context.event_log", "world.state"))


class TickStagesTest(unittest.TestCase):
    def test_get_stage_order_returns_registered_names(self) -> None:
        self.assertEqual(
            stages.get_stage_order(),
            [
                "action",
                "terrain_and_wakefulness",
                "predator_contact",
                "autonomic",
                "predator_update",
                "reward",
                "postprocess",
                "memory",
            ],
        )

    def test_stage_descriptors_document_mutations(self) -> None:
        self.assertTrue(stages.TICK_STAGES)
        for descriptor in stages.TICK_STAGES:
            self.assertTrue(descriptor.mutates)

    def test_validate_pipeline_accepts_registered_stage_order(self) -> None:
        stages.validate_pipeline()

    def test_validate_pipeline_rejects_mismatched_stage_order(self) -> None:
        def noop(world: object, context: object) -> None:
            """
            A no-op stage function used as a placeholder in stage registries.
            
            Accepts a world and a context but performs no modifications.
            """
            return None

        with patch.object(
            stages,
            "TICK_STAGES",
            (StageDescriptor(name="wrong", run=noop, mutates=("context.event_log",)),),
        ):
            with self.assertRaises(ValueError):
                stages.validate_pipeline()


class StageFunctionTest(unittest.TestCase):
    def _make_world(self, seed: int = 11, *, lizard_move_interval: int = 999999) -> SpiderWorld:
        """
        Create and return a freshly initialized SpiderWorld configured for tests.
        
        Parameters:
        	seed (int): RNG seed used to initialize and reset the world.
        	lizard_move_interval (int): Interval controlling lizard movement frequency; large values effectively prevent lizard moves during the test.
        
        Returns:
        	world (SpiderWorld): A SpiderWorld instance that has been constructed and reset with the given seed.
        """
        world = SpiderWorld(
            seed=seed,
            lizard_move_interval=lizard_move_interval,
            noise_profile="none",
        )
        world.reset(seed=seed)
        return world

    def _build_context(self, world: SpiderWorld, action_name: str = "STAY"):
        """
        Builds a tick context for the given world and intended action.
        
        Parameters:
            world (SpiderWorld): The simulation world to build the context for.
            action_name (str): The human-readable action name (e.g., "STAY"); defaults to "STAY".
        
        Returns:
            A tick context object populated for the given world and the action's index, suitable for processing by stage functions.
        """
        return stages.build_tick_context(world, ACTION_TO_INDEX[action_name])

    def _walkable_action(self, world: SpiderWorld) -> str:
        """
        Finds a non-"STAY" action that moves the spider to a walkable adjacent cell.
        
        Parameters:
            world (SpiderWorld): World whose current spider position and walkability are used to evaluate actions.
        
        Returns:
            str: The name of a walkable action (not "STAY").
        
        Raises:
            AssertionError: Fails the test if no suitable walkable action exists.
        """
        for action_name, (dx, dy) in ACTION_DELTAS.items():
            if action_name == "STAY":
                continue
            target = (
                max(0, min(world.width - 1, world.state.x + dx)),
                max(0, min(world.height - 1, world.state.y + dy)),
            )
            if target != world.spider_pos() and world.is_walkable(target):
                return action_name
        self.fail("Expected at least one walkable action from the current spider position.")

    def _outside_cell(self, world: SpiderWorld) -> tuple[int, int]:
        """
        Finds a walkable cell that is not a shelter, not a food position, and not the lizard's position.
        
        Parameters:
            world (SpiderWorld): The world to search.
        
        Returns:
            tuple[int, int]: The (x, y) coordinates of a qualifying cell.
        
        Raises:
            AssertionError: If no qualifying cell exists.
        """
        for y in range(world.height):
            for x in range(world.width):
                pos = (x, y)
                if (
                    world.is_walkable(pos)
                    and pos not in world.shelter_cells
                    and pos not in world.food_positions
                    and pos != world.lizard_pos()
                ):
                    return pos
        self.fail("Expected at least one walkable outside cell that is not food or the lizard position.")

    def test_build_tick_context_records_bootstrap_events(self) -> None:
        world = self._make_world()

        context = self._build_context(world, "STAY")

        self.assertEqual(context.intended_action, "STAY")
        self.assertEqual(context.executed_action, "STAY")
        self.assertFalse(context.motor_noise_applied)
        self.assertEqual([event.stage for event in context.event_log], ["pre_tick"])
        self.assertEqual([event.name for event in context.event_log], ["snapshot"])
        self.assertEqual(context.info["action"], "STAY")
        self.assertEqual(context.info["distance_deltas"], {})

    def test_run_action_stage_updates_context_and_records_movement_event(self) -> None:
        world = self._make_world(seed=13)
        action_name = self._walkable_action(world)
        context = self._build_context(world, action_name)

        stages.run_action_stage(world, context)

        self.assertTrue(context.moved)
        action_event = next(event for event in context.event_log if event.name == "action_resolved")
        self.assertEqual(action_event.stage, "action")
        self.assertIn("execution_difficulty", action_event.payload)
        self.assertIn("slip_probability", action_event.payload)
        movement_event = context.event_log[-1]
        self.assertEqual(movement_event.stage, "action")
        self.assertEqual(movement_event.name, "movement_applied")
        self.assertEqual(
            set(movement_event.payload.keys()),
            {"moved", "spider_pos", "last_move_dx", "last_move_dy"},
        )

    def test_run_terrain_stage_updates_context_and_records_effects(self) -> None:
        world = self._make_world(seed=17)
        context = self._build_context(world, "STAY")

        stages.run_action_stage(world, context)
        stages.run_terrain_stage(world, context)

        self.assertTrue(context.terrain_now)
        self.assertIsInstance(context.predator_threat, bool)
        self.assertLess(context.reward_components["action_cost"], 0.0)
        terrain_event = context.event_log[-1]
        self.assertEqual(terrain_event.stage, "terrain_and_wakefulness")
        self.assertEqual(terrain_event.name, "effects_applied")
        self.assertEqual(
            set(terrain_event.payload.keys()),
            {"terrain", "predator_threat", "interrupted_rest", "exposed_at_night", "sleep_debt", "fatigue"},
        )

    def test_run_predator_contact_stage_applies_contact_and_records_event(self) -> None:
        world = self._make_world(seed=19)
        context = self._build_context(world, "STAY")
        world.lizard.x, world.lizard.y = world.spider_pos()

        stages.run_predator_contact_stage(world, context)

        self.assertTrue(context.predator_contact_applied)
        self.assertTrue(context.info["predator_contact"])
        contact_event = next(
            event
            for event in context.event_log
            if event.stage == "predator_contact" and event.name == "predator_contact"
        )
        self.assertEqual(
            set(contact_event.payload.keys()),
            {"damage", "health", "recent_pain", "recent_contact"},
        )

    def test_run_predator_contact_stage_checks_all_predator_positions(self) -> None:
        world = self._make_world(seed=21)
        world.reset(
            seed=21,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )
        context = self._build_context(world, "STAY")
        spider_x, spider_y = world.spider_pos()
        first_x = spider_x - 1 if spider_x > 0 else spider_x + 1
        world.lizard.x, world.lizard.y = first_x, spider_y
        second_predator = world.get_predator(1)
        second_predator.x, second_predator.y = spider_x, spider_y

        stages.run_predator_contact_stage(world, context)

        self.assertTrue(context.predator_contact_applied)
        self.assertTrue(context.info["predator_contact"])

    def test_run_autonomic_stage_feeding_updates_context_and_records_events(self) -> None:
        """
        Verify running the autonomic stage applies feeding when the spider is on a food cell and records the corresponding events.
        
        Sets the spider's position as a food location, runs the autonomic stage, and expects:
        - context.fed_this_tick to be True
        - context.info["ate"] to be truthy
        - an event named "feeding" and an "autonomic_summary" event to appear in the autonomic-stage entries of context.event_log
        """
        world = self._make_world(seed=23)
        world.food_positions = [world.spider_pos()]
        context = self._build_context(world, "STAY")

        stages.run_autonomic_stage(world, context)

        self.assertTrue(context.fed_this_tick)
        self.assertTrue(context.info["ate"])
        event_names = [event.name for event in context.event_log if event.stage == "autonomic"]
        self.assertIn("feeding", event_names)
        self.assertIn("autonomic_summary", event_names)

    def test_run_autonomic_stage_applies_idle_open_penalty_outside_shelter(self) -> None:
        """
        Verifies the autonomic stage applies sleep-reset and an idle-open penalty when the spider is located outside shelter.
        
        Asserts that the spider is not fed this tick, that the autonomic event log includes "sleep_reset_off_shelter", "idle_open_penalty", and "autonomic_summary", and that the recorded action cost is less than 0.0.
        """
        world = self._make_world(seed=29)
        world.state.x, world.state.y = self._outside_cell(world)
        context = self._build_context(world, "STAY")

        stages.run_autonomic_stage(world, context)

        self.assertFalse(context.fed_this_tick)
        event_names = [event.name for event in context.event_log if event.stage == "autonomic"]
        self.assertIn("sleep_reset_off_shelter", event_names)
        self.assertIn("idle_open_penalty", event_names)
        self.assertIn("autonomic_summary", event_names)
        self.assertLess(context.reward_components["action_cost"], 0.0)

    def test_run_predator_update_stage_records_update_and_second_contact_check(self) -> None:
        """
        Check that running the predator-update stage when the lizard is colocated with the spider records the update and triggers a second predator-contact check.
        
        Sets the lizard to the spider's position, runs the predator update stage, and asserts that the context indicates no predator movement, that an event named "predator_update" exists with payload keys "moved", "mode_before", "mode_after", "transition", and "lizard_pos", and that a subsequent "predator_contact" event was applied whose payload includes "damage".
        """
        world = self._make_world(seed=31, lizard_move_interval=999999)
        context = self._build_context(world, "STAY")
        world.tick = 1
        world.lizard.x, world.lizard.y = world.spider_pos()

        stages.run_predator_update_stage(world, context)

        self.assertFalse(context.predator_moved)
        self.assertFalse(context.info["predator_moved"])
        self.assertTrue(context.predator_contact_applied)
        update_event = next(event for event in context.event_log if event.stage == "predator_update")
        self.assertEqual(update_event.name, "predator_update")
        self.assertEqual(
            set(update_event.payload.keys()),
            {
                "moved",
                "mode_before",
                "mode_after",
                "transition",
                "moved_each",
                "transitions",
                "lizard_pos",
            },
        )
        self.assertEqual(update_event.payload["moved_each"], [False])
        self.assertEqual(update_event.payload["transitions"], [])
        predator_contact_event = next(
            event
            for event in context.event_log
            if event.stage == "predator_contact" and event.name == "predator_contact"
        )
        self.assertIn("damage", predator_contact_event.payload)

    def test_run_predator_update_stage_iterates_all_predator_controllers(self) -> None:
        world = self._make_world(seed=33)
        world.reset(
            seed=33,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )
        context = self._build_context(world, "STAY")

        class StubController:
            def __init__(self, predator_index: int, moved: bool, next_mode: str | None = None) -> None:
                """
                Initialize a stub predator controller for testing predator update behavior.
                
                Parameters:
                    predator_index (int): Index of the predator this controller represents.
                    moved (bool): Whether the controller's update should report the predator as having moved.
                    next_mode (str | None): Optional predator mode to set during update; if None, mode is unchanged.
                """
                self.predator_index = predator_index
                self.moved = moved
                self.next_mode = next_mode
                self.calls = 0

            def update(self, stage_world: SpiderWorld) -> bool:
                """
                Record a controller update call, optionally set the predator's mode in the provided world, and report whether the predator moved.
                
                Parameters:
                    stage_world (SpiderWorld): The world passed to the controller; if `next_mode` is set, the predator at `predator_index` in this world will have its `mode` overwritten.
                
                Returns:
                    bool: `True` if this controller reports the predator moved, `False` otherwise.
                """
                self.calls += 1
                if self.next_mode is not None:
                    stage_world.get_predator(self.predator_index).mode = self.next_mode
                return self.moved

        first = StubController(0, moved=False)
        second = StubController(1, moved=True, next_mode="ORIENT")
        world.predator_controllers = [first, second]

        stages.run_predator_update_stage(world, context)

        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertTrue(context.predator_moved)
        self.assertTrue(context.info["predator_moved"])
        self.assertEqual(context.info["predator_moved_each"], [False, True])
        self.assertEqual(
            context.info["predator_transitions"],
            [{"index": 1, "from": "PATROL", "to": "ORIENT"}],
        )
        update_event = next(event for event in context.event_log if event.stage == "predator_update")
        self.assertEqual(update_event.payload["moved_each"], [False, True])
        self.assertEqual(
            update_event.payload["transitions"],
            [{"index": 1, "from": "PATROL", "to": "ORIENT"}],
        )

    def test_run_reward_stage_populates_distance_deltas_and_records_event(self) -> None:
        """
        Verifies that run_reward_stage populates distance delta information for food, shelter, and predator and records a corresponding event.
        
        Asserts that context.info["distance_deltas"] contains keys "food", "shelter", and "predator", that context.predator_visible_now is a boolean, and that an event named "distance_deltas" is present in the event log with payload keys "food", "shelter", and "predator".
        """
        world = self._make_world(seed=37)
        context = self._build_context(world, "STAY")

        stages.run_action_stage(world, context)
        stages.run_terrain_stage(world, context)
        stages.run_reward_stage(world, context)

        self.assertEqual(set(context.info["distance_deltas"].keys()), {"food", "shelter", "predator"})
        self.assertIsInstance(context.predator_visible_now, bool)
        delta_event = next(event for event in context.event_log if event.name == "distance_deltas")
        self.assertEqual(delta_event.stage, "reward")
        self.assertEqual(set(delta_event.payload.keys()), {"food", "shelter", "predator"})

    def test_run_postprocess_stage_sets_done_and_records_events(self) -> None:
        world = self._make_world(seed=41)
        context = self._build_context(world, "STAY")
        start_tick = world.tick
        world.state.health = 0.0

        stages.run_postprocess_stage(world, context)

        self.assertTrue(context.done)
        self.assertEqual(world.tick, start_tick + 1)
        self.assertEqual(world.state.last_reward, context.reward)
        event_names = [event.name for event in context.event_log]
        self.assertIn("reward_summary", event_names)
        self.assertIn("postprocess_complete", event_names)

    def test_run_memory_stage_records_targets(self) -> None:
        world = self._make_world(seed=43)
        context = self._build_context(world, "STAY")
        context.predator_escape = True
        world.tick = 7

        stages.run_memory_stage(world, context)

        memory_event = context.event_log[-1]
        self.assertEqual(memory_event.stage, "memory")
        self.assertEqual(memory_event.name, "memory_refreshed")
        self.assertTrue(memory_event.payload["predator_escape"])
        self.assertEqual(memory_event.payload["tick"], 7)
        self.assertEqual(
            set(memory_event.payload.keys()),
            {
                "predator_escape",
                "food_memory_target",
                "predator_memory_target",
                "shelter_memory_target",
                "escape_memory_target",
                "tick",
            },
        )

    def test_finalize_step_returns_info_with_serialized_event_log(self) -> None:
        """
        Ensure finalize_step produces next observation containing "visual", returns the provided reward and done flag, and includes an info dict with "reward_components", "state", "predator_escape", and a serialized "event_log" whose last entry is a finalize/tick_complete event.
        """
        world = self._make_world(seed=47)
        context = self._build_context(world, "STAY")
        context.reward_components["feeding"] = 1.25
        context.reward = 1.25
        context.done = False

        next_obs, reward, done, info = stages.finalize_step(world, context)

        self.assertIn("visual", next_obs)
        self.assertEqual(reward, 1.25)
        self.assertFalse(done)
        self.assertIn("reward_components", info)
        self.assertIn("state", info)
        self.assertIn("predator_escape", info)
        self.assertIn("event_log", info)
        self.assertEqual(info["event_log"][-1]["stage"], "finalize")
        self.assertEqual(info["event_log"][-1]["name"], "tick_complete")
