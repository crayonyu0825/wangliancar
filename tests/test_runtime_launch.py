import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.config import load_config
from runtime import CaptainRuntime


class RuntimeLaunchTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt" and sys.version_info[:2]==(3,6),
                         "requires Windows target Python 3.6")
    def test_platform_batch_applies_30_kmh_default_and_preserves_overrides(self):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixture = (
            "import json, os, sys\n"
            "print(json.dumps({'speed': os.environ.get('NEVC_DECISION_CRUISE_SPEED'), "
            "'front': os.environ.get('NEVC_VEHICLE_FRONT_OFFSET_M'), "
            "'half_width': os.environ.get('NEVC_VEHICLE_HALF_WIDTH_M'), "
            "'args': sys.argv[1:]}))\n"
            "sys.exit(7 if '--fixture-fail' in sys.argv else 0)\n"
        )
        # Exercise the actual batch in an isolated project, without SDK calls.
        cases = ((None, False, 0, {}), ("4.5", True, 7, {
            "NEVC_VEHICLE_FRONT_OFFSET_M": "4.1", "NEVC_VEHICLE_HALF_WIDTH_M": "1.0"}))
        for supplied_speed, local_config, expected_exit, geometry in cases:
            with self.subTest(speed=supplied_speed, local=local_config), \
                    tempfile.TemporaryDirectory(prefix="nevc launch ") as directory:
                scripts = os.path.join(directory, "scripts")
                os.mkdir(scripts)
                batch = os.path.join(scripts, "StartCaptain.bat")
                shutil.copyfile(os.path.join(project, "scripts", "StartCaptain.bat"), batch)
                with open(os.path.join(directory, "main.py"), "w", encoding="utf-8") as stream:
                    stream.write(fixture)
                if local_config:
                    os.mkdir(os.path.join(directory, "config"))
                    with open(os.path.join(directory, "config", "local.ini"), "w") as stream:
                        stream.write("[app]\n")
                environment = os.environ.copy()
                environment['NEVC_PYTHON_EXE']=sys.executable
                environment.pop("NEVC_DECISION_CRUISE_SPEED", None)
                environment.pop("NEVC_VEHICLE_FRONT_OFFSET_M", None)
                environment.pop("NEVC_VEHICLE_HALF_WIDTH_M", None)
                environment.update(geometry)
                if supplied_speed is not None:
                    environment["NEVC_DECISION_CRUISE_SPEED"] = supplied_speed
                argument = "--fixture-fail" if expected_exit else "--fixture"
                result = subprocess.run(
                    [environment.get("COMSPEC", "cmd.exe"), "/d", "/c", batch, argument],
                    cwd=directory, env=environment, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=15)
                self.assertEqual(expected_exit, result.returncode, result.stdout)
                runtime_dir = os.path.join(directory, "runtime_data")
                console_logs = [name for name in os.listdir(runtime_dir)
                                if name.startswith("captain-console-")]
                self.assertEqual(1, len(console_logs))
                with open(os.path.join(runtime_dir, console_logs[0]), encoding="utf-8") as stream:
                    launched = json.loads(stream.read())
                self.assertEqual(supplied_speed or "8.333333333333334", launched["speed"])
                self.assertEqual(geometry.get("NEVC_VEHICLE_FRONT_OFFSET_M", "3.9187"),
                                 launched["front"])
                self.assertEqual(geometry.get("NEVC_VEHICLE_HALF_WIDTH_M", "0.9"),
                                 launched["half_width"])
                self.assertEqual(argument, launched["args"][-1])
                if local_config:
                    self.assertEqual("--config", launched["args"][0])
                    self.assertEqual(os.path.join(directory, "config", "local.ini"),
                                     os.path.abspath(launched["args"][1]))
                else:
                    self.assertEqual([argument], launched["args"])
                with open(os.path.join(runtime_dir, "launcher.log")) as stream:
                    launch_log = stream.read()
                self.assertIn("Decision cruise speed: " + (supplied_speed or "8.333333333333334"), launch_log)
                self.assertIn("Vehicle geometry: front=" + launched["front"] +
                              " m half-width=" + launched["half_width"] + " m", launch_log)
                self.assertIn("exited with code " + str(expected_exit), launch_log)

    def test_snapshot_io_failure_keeps_previous_data_and_recovers(self):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(project)
            config.runtime_dir = directory
            warnings, recovered = [], []
            logger = SimpleNamespace(info=lambda *args: recovered.append(args),
                                     warning=lambda *args: warnings.append(args))
            runtime = CaptainRuntime(config, logger)
            path = runtime.snapshot_path
            self.assertTrue(runtime._publish_json(path, {"frame": 1}))
            for target in ("runtime.os.replace", "builtins.open"):
                with self.subTest(target=target):
                    with patch(target, side_effect=PermissionError(5, "reader holds file")):
                        self.assertFalse(runtime._publish_json(path, {"frame": 2}))
                        self.assertFalse(runtime._publish_json(path, {"frame": 3}))
                    with open(path, encoding="utf-8") as stream:
                        self.assertEqual({"frame": 1}, json.load(stream))
                    self.assertTrue(runtime._publish_json(path, {"frame": 1}))
            self.assertEqual(2, len(warnings))
            self.assertEqual(2, len(recovered))
            self.assertTrue(runtime._publish_json(path, {"frame": 4}))
            with open(path, encoding="utf-8") as stream:
                self.assertEqual({"frame": 4}, json.load(stream))

    def test_duplicate_start_does_not_overwrite_primary_pid_or_connect_sdk(self):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(project)
            config.runtime_dir = directory
            messages = []
            logger = SimpleNamespace(info=lambda *args: None,
                                     warning=lambda *args: messages.append(args))
            first = CaptainRuntime(config, logger)
            second = CaptainRuntime(config, logger)
            self.assertTrue(first._acquire_instance_lock())
            try:
                first._write_pid()
                with patch.object(second, "_install_signals"), \
                     patch.object(second.adapter, "bootstrap") as bootstrap:
                    second.run()
                bootstrap.assert_not_called()
                with open(first.pid_path, encoding="ascii") as stream:
                    self.assertEqual(str(os.getpid()), stream.read())
                self.assertTrue(messages)
            finally:
                first._remove_pid()
                first._release_instance_lock()
            self.assertTrue(second._acquire_instance_lock())
            second._release_instance_lock()


if __name__ == "__main__":
    unittest.main()
