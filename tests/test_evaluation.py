"""Evaluation lifecycle tests; native calls are stand-ins, never a live case."""

import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.config import AppConfig, load_config
from runtime import CaptainRuntime
from simone_platform.evaluation import LocalEvaluation
from simone_platform.simone_adapter import SimOneAdapter


class NativeCall(object):
    def __init__(self, result=True, callback=None):
        self.result, self.callback, self.calls = result, callback, []

    def __call__(self, *args):
        self.calls.append(args)
        if self.callback:
            return self.callback(*args)
        return self.result


def native_library():
    return SimpleNamespace(InitEvaluationServiceWithLocalData=NativeCall(),
                           SaveEvaluationRecord=NativeCall())


def adapter(library):
    config = SimpleNamespace(vehicle_id="0", connect_timeout_sec=1,
                             server_ip="127.0.0.1", evaluation_enabled=True)
    a = SimOneAdapter(config, logging.getLogger("evaluation-test"))
    a.service_api = SimpleNamespace(SoInitSimOneAPI=NativeCall(),
                                   SoTerminateSimOneAPI=NativeCall())
    a.pnc_api = SimpleNamespace(SoSetDriverName=NativeCall())
    a._evaluation = LocalEvaluation(library, "3.0.0001", a.logger)
    return a


class EvaluationTests(unittest.TestCase):
    def test_binding_uses_verified_two_argument_c_abi_and_sdk_data(self):
        library = native_library()
        evaluator = LocalEvaluation(library, b"3.0.0001", logging.getLogger())
        evaluator.initialize("0")
        self.assertEqual([(b"0", True)], library.InitEvaluationServiceWithLocalData.calls)
        self.assertEqual([ctypes.c_char_p, ctypes.c_bool],
                         library.InitEvaluationServiceWithLocalData.argtypes)
        self.assertIs(ctypes.c_bool, library.SaveEvaluationRecord.restype)
        self.assertEqual([], library.SaveEvaluationRecord.argtypes)
        evaluator.initialize("0")
        self.assertEqual(1, len(library.InitEvaluationServiceWithLocalData.calls))

    def test_unknown_version_and_missing_export_fail_without_native_calls(self):
        library = native_library()
        with self.assertRaisesRegex(RuntimeError, "Unverified"):
            LocalEvaluation(library, "3.9.0", logging.getLogger())
        self.assertEqual([], library.InitEvaluationServiceWithLocalData.calls)
        with self.assertRaises(RuntimeError):
            LocalEvaluation(SimpleNamespace(), "3.0.0001", logging.getLogger())

    def test_initialization_failure_prevents_driver_setup_and_still_terminates(self):
        library = native_library()
        library.InitEvaluationServiceWithLocalData.result = False
        a = adapter(library)
        with self.assertRaises(RuntimeError):
            a.initialize()
        self.assertFalse(a._evaluation.active)
        self.assertEqual([], a.pnc_api.SoSetDriverName.calls)
        a.shutdown()
        self.assertEqual([], library.SaveEvaluationRecord.calls)
        self.assertEqual(1, len(a.service_api.SoTerminateSimOneAPI.calls))
        self.assertFalse(a.connected)

    def test_periodic_save_rate_limit_force_save_and_failure_recovery(self):
        library = native_library()
        evaluator = LocalEvaluation(library, "3.0.0001", logging.getLogger(), 1.0)
        with patch("simone_platform.evaluation.time.monotonic", return_value=10.0):
            evaluator.initialize("0")
            evaluator.flush()
        self.assertEqual([], library.SaveEvaluationRecord.calls)
        library.SaveEvaluationRecord.result = False
        with patch("simone_platform.evaluation.time.monotonic", return_value=11.0):
            self.assertFalse(evaluator.flush())
            self.assertTrue(evaluator.flush())  # throttled, status still says failure
        self.assertFalse(evaluator.status()["last_save_ok"])
        library.SaveEvaluationRecord.result = True
        with patch("simone_platform.evaluation.time.monotonic", return_value=12.0):
            self.assertTrue(evaluator.flush())
            self.assertTrue(evaluator.flush(force=True))
        self.assertTrue(evaluator.status()["last_save_ok"])
        self.assertEqual(2, evaluator.save_count)
        self.assertEqual(3, len(library.SaveEvaluationRecord.calls))

    def test_shutdown_saves_before_terminate_and_does_not_save_twice_after_termination(self):
        events = []
        library = native_library()
        library.SaveEvaluationRecord.callback = lambda: events.append("save") or True
        a = adapter(library)
        a.service_api.SoTerminateSimOneAPI.callback = lambda: events.append("terminate")
        a.initialize()
        a.shutdown()
        a.shutdown()
        self.assertEqual(["save", "terminate"], events)
        self.assertFalse(a.evaluation_status()["initialized"])

    def test_final_save_exception_cannot_skip_sdk_cleanup(self):
        library = native_library()

        def fail():
            raise OSError("fixture disk full")

        library.SaveEvaluationRecord.callback = fail
        a = adapter(library)
        a.initialize()
        with self.assertRaises(RuntimeError):
            a.shutdown()
        self.assertEqual(1, len(a.service_api.SoTerminateSimOneAPI.calls))
        self.assertFalse(a.connected)
        self.assertEqual("save_exception:OSError", a.evaluation_status()["error"])

    def test_config_enabled_by_default_and_interval_rejects_invalid_values(self):
        c = AppConfig(".", {})
        self.assertTrue(c.evaluation_enabled)
        self.assertEqual(1.0, c.evaluation_flush_interval_sec)
        self.assertFalse(AppConfig(".", {"evaluation_enabled": "false"}).evaluation_enabled)
        for value in ("nan", "inf", "0", "-1", "0.01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AppConfig(".", {"evaluation_flush_interval_sec": value})

    def test_pid_specific_stop_request_ignores_stale_process_and_stops_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            r = CaptainRuntime(SimpleNamespace(runtime_dir=directory), logging.getLogger())
            with open(r.stop_path, "w") as stream:
                stream.write(str(os.getpid() + 1))
            self.assertFalse(r._consume_stop_request())
            with open(r.stop_path, "w") as stream:
                stream.write(str(os.getpid()))
            r._wait_until_running()
            self.assertTrue(r.stop_requested)

    def test_shutdown_error_removes_pid_and_releases_instance_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config = load_config(project)
            config.runtime_dir = directory
            r = CaptainRuntime(config, logging.getLogger())
            with patch.object(r, "_install_signals"), \
                 patch.object(r.adapter, "bootstrap", side_effect=ValueError("fixture")), \
                 patch.object(r.adapter, "shutdown", side_effect=RuntimeError("save failed")):
                with self.assertRaises(RuntimeError):
                    r.run()
            self.assertFalse(os.path.exists(r.pid_path))
            self.assertIsNone(r._lock_stream)
            self.assertTrue(r._acquire_instance_lock())
            r._release_instance_lock()

    @unittest.skipUnless(os.name == "nt" and sys.version_info[:2]==(3,6),
                         "requires Windows target Python 3.6")
    def test_real_end_batch_requests_save_and_exit_without_force_kill(self):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixture = '''import os,sys,json,logging
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, PROJECT)
from core.config import load_config
from runtime import CaptainRuntime
from simone_platform.simone_adapter import SimOneAdapter
from simone_platform.evaluation import LocalEvaluation
class Native:
    def __call__(self,*args): return True
root=os.path.dirname(os.path.abspath(__file__))
config=load_config(PROJECT)
config.runtime_dir=os.path.join(root,"runtime_data")
r=CaptainRuntime(config,logging.getLogger())
a=r.adapter
a.bootstrap=lambda: None
a.load_hdmap=lambda: True
a._register_target_callback=lambda: None
a.get_case_status=lambda: a.CASE_PAUSE
class Save(Native):
    def __call__(self):
        with open(os.path.join(root,"saved.json"),"w") as f:
            json.dump({"records":[{"category":"mainvehicle","fixture":True}]},f)
        return True
class Terminate(Native):
    def __call__(self):
        assert os.path.exists(os.path.join(root,"saved.json"))
        with open(os.path.join(root,"terminated.txt"),"w") as f:f.write("saved first")
a.service_api=SimpleNamespace(SoInitSimOneAPI=Native(),SoTerminateSimOneAPI=Terminate())
a.pnc_api=SimpleNamespace(SoSetDriverName=Native())
a._evaluation=LocalEvaluation(SimpleNamespace(InitEvaluationServiceWithLocalData=Native(),SaveEvaluationRecord=Save()),"3.0.0001",r.logger)
builder=SimpleNamespace(update_case_info=lambda:{"case_name":"fixture"},scene_id=0)
with patch("runtime.RouteManager"),patch("runtime.PerceptionBuilder",return_value=builder):
    r.run()
'''.replace("PROJECT", repr(project))
        with tempfile.TemporaryDirectory(prefix="evaluation stop ") as directory:
            os.mkdir(os.path.join(directory, "scripts"))
            for name in ("KillCaptain.bat", "stop_captain.ps1"):
                shutil.copyfile(os.path.join(project, "scripts", name),
                                os.path.join(directory, "scripts", name))
            main = os.path.join(directory, "main.py")
            with open(main, "w", encoding="utf-8") as stream:
                stream.write(fixture)
            process = subprocess.Popen([sys.executable, main],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                pid = os.path.join(directory, "runtime_data", "captain.pid")
                deadline = time.monotonic() + 5
                while not os.path.isfile(pid) and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(os.path.isfile(pid), "fixture did not start")
                result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c",
                                         os.path.join(directory, "scripts", "KillCaptain.bat")],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                output = process.communicate(timeout=5)[0]
                self.assertEqual(0, result.returncode, result.stdout)
                self.assertNotIn(b"forcing stop", result.stdout)
                self.assertEqual(0, process.returncode, output)
                self.assertFalse(os.path.exists(pid))
                with open(os.path.join(directory, "saved.json")) as stream:
                    self.assertEqual(1, len(json.load(stream)["records"]))
                self.assertTrue(os.path.isfile(os.path.join(directory, "terminated.txt")))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
