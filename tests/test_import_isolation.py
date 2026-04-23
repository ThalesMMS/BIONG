import json
import subprocess
import sys
import unittest
from pathlib import Path


def _extract_json_payload(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line[0] not in "{[":
            continue
        return json.loads(line)
    raise ValueError("No JSON payload found in stdout.")


class LeafImportIsolationTest(unittest.TestCase):
    def test_world_types_import_stays_decoupled_from_heavy_modules(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = """
import json
import sys

from spider_cortex_sim.world_types import TickContext

print(json.dumps({
    "imported": TickContext.__name__,
    "agent_loaded": "spider_cortex_sim.agent" in sys.modules,
    "modules_loaded": "spider_cortex_sim.modules" in sys.modules,
    "nn_loaded": "spider_cortex_sim.nn" in sys.modules,
}))
"""
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        payload = _extract_json_payload(proc.stdout)
        self.assertEqual(payload["imported"], "TickContext")
        self.assertFalse(payload["agent_loaded"])
        self.assertFalse(payload["modules_loaded"])
        self.assertFalse(payload["nn_loaded"])


class ExtractJsonPayloadTest(unittest.TestCase):
    def test_extracts_last_json_line_after_noise(self) -> None:
        stdout = "\n".join(
            [
                "debug: importing leaf module",
                '{"stale": true}',
                '{"imported": "TickContext", "agent_loaded": false}',
            ]
        )

        payload = _extract_json_payload(stdout)

        self.assertEqual(
            payload,
            {"imported": "TickContext", "agent_loaded": False},
        )

    def test_raises_when_stdout_has_no_json(self) -> None:
        with self.assertRaises(ValueError):
            _extract_json_payload("debug: no payload emitted")
