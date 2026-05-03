import json
import unittest
from pathlib import Path


PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / ".auto-claude"
    / "specs"
    / "005-make-the-pygame-window-resizable-with-fit-to-scree"
    / "implementation_plan.json"
)


def _load_plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


class AutoClaudePlanMetadataTest(unittest.TestCase):
    def test_plan_timestamp_aliases_stay_synchronized(self) -> None:
        plan = _load_plan()

        self.assertIn("updated_at", plan)
        self.assertIn("last_updated", plan)
        self.assertEqual(plan["updated_at"], plan["last_updated"])

    def test_completed_subtasks_use_concrete_files_to_modify(self) -> None:
        plan = _load_plan()

        for phase in plan["phases"]:
            for subtask in phase["subtasks"]:
                if subtask["status"] != "completed":
                    continue

                for path in subtask["files_to_modify"]:
                    self.assertNotIn("to be filled after discovery", path)


if __name__ == "__main__":
    unittest.main()
