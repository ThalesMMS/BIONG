from __future__ import annotations

from collections import deque

from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.scenarios import get_scenario
from spider_cortex_sim.world import SpiderWorld


class ScenarioWorldHelpers:
    def _setup_world(
        self,
        name: str,
        *,
        perceptual_delay_ticks: float | None = None,
    ) -> SpiderWorld:
        """
        Create and return a SpiderWorld configured for the given scenario with deterministic initialization.

        Parameters:
            name (str): Identifier of the scenario to load.

        Returns:
            SpiderWorld: A world initialized with seed 101, the scenario's map template, and with the scenario's setup applied.
        """
        scenario = get_scenario(name)
        kwargs: dict[str, object] = {}
        if perceptual_delay_ticks is not None:
            summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
            summary["perception"]["perceptual_delay_ticks"] = perceptual_delay_ticks
            kwargs["operational_profile"] = OperationalProfile.from_summary(summary)
        world = SpiderWorld(
            seed=101,
            lizard_move_interval=1,
            map_template=scenario.map_template,
            **kwargs,
        )
        world.reset(seed=101)
        scenario.setup(world)
        return world

    def _move_towards(self, world: SpiderWorld, target: tuple[int, int]) -> str:
        """
        Determine the first move action that guides the spider along a shortest walkable path to the given target cell.

        Parameters:
            world (SpiderWorld): The simulation world containing the spider, map dimensions, and walkability info.
            target (tuple[int, int]): Destination cell as (x, y) coordinates.

        Returns:
            str: One of "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT" indicating the first step toward the target, or "STAY" if the target is unreachable or is the spider's current position.
        """
        start = world.spider_pos()
        queue = deque([start])
        parent = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == target:
                break
            for action_name, (dx, dy) in (
                ("MOVE_UP", (0, -1)),
                ("MOVE_DOWN", (0, 1)),
                ("MOVE_LEFT", (-1, 0)),
                ("MOVE_RIGHT", (1, 0)),
            ):
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nxt[0] < world.width and 0 <= nxt[1] < world.height):
                    continue
                if nxt in parent or not world.is_walkable(nxt):
                    continue
                parent[nxt] = (cell, action_name)
                queue.append(nxt)
        if target not in parent:
            return "STAY"
        step = target
        action_name = "STAY"
        while parent[step] is not None:
            prev, action_name = parent[step]
            if prev == start:
                return action_name
            step = prev
        return "STAY"
