from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.scenarios.gate_reports import build_gate_report, write_gate_report


class GateReportWriterTest(unittest.TestCase):
    def test_build_and_write_gate_report_round_trips_minimal_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            behavior_csv_path = tmp_path / "behavior.csv"
            summary_path.write_text(
                json.dumps(
                    {
                        "behavior_evaluation": {
                            "suite": {
                                "night_rest": {
                                    "success_rate": 0.5,
                                    "episodes": 2,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            behavior_csv_path.write_text(
                "\n".join(
                    [
                        "simulation_seed,episode_seed,scenario,episode,success,metric_failure_mode,metric_failure_attribution,metric_food_distance_delta,metric_shelter_distance_delta,metric_night_shelter_occupancy_rate",
                        "7,1001,night_rest,0,True,,,,,",
                        "7,1002,night_rest,1,False,night_shelter_missed,arbitration,1.5,,nan",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_gate_report(
                gate_name="night_rest",
                summary_path=summary_path,
                behavior_csv_path=behavior_csv_path,
            )
            paths = write_gate_report(output_dir=tmp_path, report=report)
            json_exists = paths.report_json_path.exists()
            md_exists = paths.report_md_path.exists()
            persisted = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
            markdown = paths.report_md_path.read_text(encoding="utf-8")

        self.assertTrue(json_exists)
        self.assertTrue(md_exists)
        self.assertEqual(
            persisted["generated_files"],
            {"json": str(paths.report_json_path), "md": str(paths.report_md_path)},
        )
        self.assertEqual(persisted["per_seed"], [{"seed": 7, "episodes": 2, "passed": 1, "pass_rate": 0.5}])
        self.assertEqual(persisted["failure_modes"], [{"label": "night_shelter_missed", "count": 1}])
        self.assertEqual(persisted["representative_failures"][0]["failure_attribution"], "arbitration")
        self.assertEqual(paths.report, persisted)
        self.assertIn("## Per-seed pass rates", markdown)
        self.assertIn("## Failure modes", markdown)
        self.assertIn("## Representative failing episodes", markdown)
        self.assertIn("| 7 | 1 | arbitration |  | 1.5 |  |", markdown)

    def test_build_gate_report_tolerates_malformed_behavior_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            behavior_csv_path = tmp_path / "behavior.csv"
            summary_path.write_text(
                json.dumps({"behavior_evaluation": "malformed"}),
                encoding="utf-8",
            )
            behavior_csv_path.write_text(
                "\n".join(
                    [
                        "simulation_seed,episode_seed,scenario,episode,success",
                        "7,1001,night_rest,0,True",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_gate_report(
                gate_name="night_rest",
                summary_path=summary_path,
                behavior_csv_path=behavior_csv_path,
            )

        self.assertIsNone(report["scenario_entry"])
        self.assertEqual(report["overall"], {"episodes": 1, "passed": 1, "pass_rate": 1.0})

    def test_build_gate_report_skips_infinite_seed_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            behavior_csv_path = tmp_path / "behavior.csv"
            summary_path.write_text(
                json.dumps({"behavior_evaluation": {"suite": {}}}),
                encoding="utf-8",
            )
            behavior_csv_path.write_text(
                "\n".join(
                    [
                        "simulation_seed,episode_seed,scenario,episode,success",
                        "inf,1001,night_rest,0,True",
                        "3.7,1002,night_rest,1,True",
                        "7,1003,night_rest,2,True",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_gate_report(
                gate_name="night_rest",
                summary_path=summary_path,
                behavior_csv_path=behavior_csv_path,
            )

        self.assertEqual(report["per_seed"], [{"seed": 7, "episodes": 1, "passed": 1, "pass_rate": 1.0}])
        self.assertEqual(report["overall"], {"episodes": 1, "passed": 1, "pass_rate": 1.0})

    def test_build_gate_report_skips_unknown_pass_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            behavior_csv_path = tmp_path / "behavior.csv"
            summary_path.write_text(
                json.dumps({"behavior_evaluation": {"suite": {}}}),
                encoding="utf-8",
            )
            behavior_csv_path.write_text(
                "\n".join(
                    [
                        "simulation_seed,episode_seed,scenario,episode,success,metric_failure_mode",
                        "7,1001,night_rest,0,,unknown_failure",
                        "7,1002,night_rest,1,maybe,unknown_failure",
                        "7,1003,night_rest,2,False,real_failure",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_gate_report(
                gate_name="night_rest",
                summary_path=summary_path,
                behavior_csv_path=behavior_csv_path,
            )

        self.assertEqual(report["overall"], {"episodes": 1, "passed": 0, "pass_rate": 0.0})
        self.assertEqual(report["failure_modes"], [{"label": "real_failure", "count": 1}])

    def test_write_gate_report_persists_generated_files_and_markdown_spacing(self) -> None:
        report = {
            "gate": "night_rest",
            "overall": {"episodes": 1, "passed": 1, "pass_rate": 1.0},
            "per_seed": [{"seed": 1, "episodes": 1, "passed": 1, "pass_rate": 1.0}],
            "failure_modes": [],
            "representative_failures": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_gate_report(output_dir=tmpdir, report=report)
            persisted = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
            markdown = paths.report_md_path.read_text(encoding="utf-8")

        self.assertNotIn("generated_files", report)
        self.assertEqual(
            persisted["generated_files"],
            {"json": str(paths.report_json_path), "md": str(paths.report_md_path)},
        )
        self.assertEqual(paths.report, persisted)
        self.assertIn("## Per-seed pass rates\n\n| seed", markdown)
        self.assertIn("## Failure modes\n\n| label", markdown)
        self.assertIn("## Representative failing episodes\n\n| seed", markdown)

    def test_write_gate_report_handles_partial_representative_failure_rows(self) -> None:
        report = {
            "gate": "two_shelter_tradeoff",
            "overall": {"episodes": 1, "passed": 0, "pass_rate": 0.0},
            "per_seed": [],
            "failure_modes": [],
            "representative_failures": [{"seed": 7}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_gate_report(output_dir=tmpdir, report=report)
            markdown = paths.report_md_path.read_text(encoding="utf-8")

        self.assertIn("| 7 |  |  |  |  |  |", markdown)


if __name__ == "__main__":
    unittest.main()
