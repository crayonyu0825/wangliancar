"""Offline tests for the scene diagnosis tool.

The tool is only useful if its gates match the real decision gates. These
tests pin each gate to the behaviour the decision member actually has, using
the real `protocol.targets_usable` as the cross-check where possible.
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.scene_requirements import requires_targets  # noqa: E402
from members.decision import protocol  # noqa: E402
from scripts import diagnose_scene  # noqa: E402


def snapshot(scene=1, source="sensor:perfectPerception1", targets_valid=True,
             usable=True, sensor_types=("7",), lane_valid=True,
             traffic=None, case_id="case-1", task_id="task-1"):
    """A captain-shaped snapshot with knobs for each gate."""
    return {
        "scene_id": scene, "case_name": "case", "case_id": case_id,
        "task_id": task_id, "frame_id": 10, "timestamp": 1000,
        "valid_until": 0.0, "valid": True,
        "ego": {"frame_id": 10, "timestamp": 1000, "x": 0.0, "y": 0.0,
                "heading": 0.0, "vx": 8.3, "vy": 0.0, "speed": 8.3,
                "gear": 1, "age_ms": 0, "valid": True},
        "lane": {"lane_id": "road1", "valid": lane_valid,
                 "center_line": [[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]]},
        "traffic": traffic or {"signal_state": "UNKNOWN", "observed": False,
                               "ambiguous": False, "valid": False,
                               "stop_line_distance": -1.0},
        "targets": ([{"id": 1, "x": 60.0, "y": 0.0, "vx": 0.0, "vy": 0.0,
                      "length": 4.0, "width": 1.8, "valid": True}]
                    if targets_valid else []),
        "targets_valid": targets_valid,
        "target_source": source,
        "source_status": {"targets": {"usable": usable,
                                      "quality": "ok" if usable else "unavailable",
                                      "sensor_read_ok": source.startswith("sensor:")}},
        "sensor_configurations": [{"id": "perfectPerception1", "type": kind}
                                  for kind in sensor_types],
        "sensor_configurations_valid": True,
    }


def config(behavior_model_file=""):
    return SimpleNamespace(sensor_id="perfectPerception1",
                           behavior_model_file=behavior_model_file)


class DiagnoseSceneTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="diagnose_scene_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, payload, name="frame.json"):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
        return path

    def run_gates(self, payload, cfg=None):
        """Diagnose one snapshot, ignoring the always-expired snapshot deadline.

        A saved snapshot is always stale, so the CLI downgrades the ego gate to
        UNKNOWN and drops it from the blocker list. Mirror that here so these
        tests exercise the structural gates rather than freshness.
        """
        perception = diagnose_scene.load_perception(self.write(payload))
        cfg = cfg if cfg is not None else config()
        report = diagnose_scene.diagnose(perception, cfg)
        report["blockers"] = [name for name in report["blockers"] if name != "ego"]
        return report

    # -- the tool must agree with the real decision gate -------------------

    def test_target_gate_agrees_with_the_decision_member(self):
        cases = [
            (snapshot(source="sensor:cam", usable=True), True),
            (snapshot(source="ground_truth", usable=False), False),
            (snapshot(source="none", targets_valid=False, usable=False), False),
            (snapshot(source="sensor:cam", usable=False), False),
            (snapshot(scene=6, source="none", targets_valid=False, usable=False), True),
        ]
        for payload, expected in cases:
            perception = diagnose_scene.load_perception(self.write(payload))
            gate = diagnose_scene._target_gate(perception)
            # Cross-check against the production predicate the decision uses.
            live = protocol.targets_usable(perception) or not \
                requires_targets(perception.scene_id)
            self.assertEqual(expected, live, payload["target_source"])
            self.assertEqual(expected, gate["status"] == diagnose_scene.PASS,
                             gate["detail"])

    # -- each failure mode is distinguishable -----------------------------

    def test_healthy_target_scene_is_scoreable(self):
        report = self.run_gates(snapshot(scene=1))
        self.assertEqual([], report["blockers"])

    def test_ground_truth_fallback_blocks_a_target_scene(self):
        report = self.run_gates(snapshot(scene=1, source="ground_truth", usable=False))
        self.assertIn("targets", report["blockers"])
        self.assertIn("Ground Truth", report["gates"]["targets"]["detail"])

    def test_missing_sensor_configuration_is_reported_separately(self):
        report = self.run_gates(snapshot(scene=1, sensor_types=()))
        self.assertIn("sensor_configuration", report["blockers"])
        self.assertIn("no camera/lidar/fusion/perfect", 
                      report["gates"]["sensor_configuration"]["detail"])

    def test_lane_only_scene_needs_no_targets(self):
        report = self.run_gates(snapshot(scene=6, source="none",
                                         targets_valid=False, usable=False,
                                         sensor_types=()))
        self.assertEqual([], report["blockers"])

    def test_parking_scene_blocks_without_a_behaviour_model(self):
        report = self.run_gates(snapshot(scene=7, lane_valid=False))
        self.assertIn("behaviour_model", report["blockers"])
        self.assertIn("empty", report["gates"]["behaviour_model"]["detail"])

    def test_lane_change_and_passage_scenes_need_the_model(self):
        for scene in (16, 18, 41):
            report = self.run_gates(snapshot(scene=scene))
            self.assertIn("behaviour_model", report["blockers"], scene)

    def test_lane_failure_blocks_driving_scenes(self):
        report = self.run_gates(snapshot(scene=6, lane_valid=False))
        self.assertIn("lane", report["blockers"])

    def test_parking_scene_does_not_require_lane_context(self):
        gate = self.run_gates(snapshot(scene=8, lane_valid=False))["gates"]["lane"]
        self.assertEqual(diagnose_scene.PASS, gate["status"])

    # -- behaviour model gate details -------------------------------------

    def test_model_file_must_exist_be_verified_and_match_the_task(self):
        missing = self.run_gates(snapshot(scene=7), config("nope.json"))
        self.assertIn("missing file", missing["gates"]["behaviour_model"]["detail"])

        unverified = self.write({"version": 1, "verified": False}, "model1.json")
        report = self.run_gates(snapshot(scene=7), config(unverified))
        self.assertIn("not verified", report["gates"]["behaviour_model"]["detail"])

        wrong_scope = self.write({"version": 1, "verified": True,
                                  "scope": {"case_id": "other", "task_id": "other"}},
                                 "model2.json")
        report = self.run_gates(snapshot(scene=7), config(wrong_scope))
        self.assertIn("scoped to case/task", report["gates"]["behaviour_model"]["detail"])

        matching = self.write({"version": 1, "verified": True,
                               "scope": {"case_id": "case-1", "task_id": "task-1"}},
                              "model3.json")
        report = self.run_gates(snapshot(scene=7), config(matching))
        self.assertNotIn("behaviour_model", report["blockers"])

    # -- signal gate -------------------------------------------------------

    def test_signal_scene_reports_unknown_rather_than_pass(self):
        report = self.run_gates(snapshot(scene=10))
        gate = report["gates"]["signal"]
        self.assertEqual(diagnose_scene.UNKNOWN, gate["status"])
        self.assertEqual([], report["blockers"])

    def test_ambiguous_signal_blocks(self):
        traffic = {"signal_state": "UNKNOWN", "observed": True, "ambiguous": True,
                   "valid": False, "stop_line_distance": 30.0}
        report = self.run_gates(snapshot(scene=10, traffic=traffic))
        self.assertIn("signal", report["blockers"])

    # -- expired snapshots must not be reported as pass -------------------

    def test_expired_snapshot_is_unknown_not_pass(self):
        gate = self.run_gates(snapshot(scene=6))["gates"]["ego"]
        self.assertEqual(diagnose_scene.BLOCK, gate["status"])
        self.assertIn("process-local", gate["detail"])

    # -- CLI behaviour -----------------------------------------------------

    def test_cli_reports_blockers_and_writes_json(self):
        self.write(snapshot(scene=7), "parking.json")
        out = os.path.join(self.dir, "report.json")
        argv = sys.argv
        sys.argv = ["diagnose_scene.py", "--directory", self.dir, "--json-out", out,
                    "--quiet"]
        try:
            code = diagnose_scene.main()
        finally:
            sys.argv = argv
        self.assertEqual(0, code)
        with open(out, encoding="utf-8") as stream:
            payload = json.load(stream)
        self.assertTrue(payload["snapshot_expired_by_design"])
        self.assertEqual(1, len(payload["scenes"]))
        self.assertIn("behaviour_model", payload["scenes"][0]["blockers"])

    def test_cli_rejects_a_non_snapshot(self):
        path = self.write({"hello": "world"}, "bad.json")
        with self.assertRaises(ValueError):
            diagnose_scene.load_perception(path)

    def test_cli_rejects_a_missing_path(self):
        with self.assertRaises(ValueError):
            diagnose_scene.snapshot_files(os.path.join(self.dir, "missing"))

    def test_scene_override_changes_the_verdict(self):
        self.write(snapshot(scene=6), "frame.json")
        out = os.path.join(self.dir, "report.json")
        argv = sys.argv
        sys.argv = ["diagnose_scene.py", "--snapshot",
                    os.path.join(self.dir, "frame.json"), "--scene", "7",
                    "--json-out", out, "--quiet"]
        try:
            self.assertEqual(0, diagnose_scene.main())
        finally:
            sys.argv = argv
        with open(out, encoding="utf-8") as stream:
            payload = json.load(stream)
        self.assertIn("behaviour_model", payload["scenes"][0]["blockers"])


if __name__ == "__main__":
    unittest.main()
