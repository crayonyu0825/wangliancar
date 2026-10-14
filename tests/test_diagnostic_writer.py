"""Real wall-clock I/O backpressure must not stall vehicle computation."""
import threading
import time
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from core.diagnostic_writer import DiagnosticWriter


class DiagnosticWriterTests(unittest.TestCase):
    def test_real_clock_full_pipeline_continues_while_disk_waits_over_control_limit(self):
        from core.config import load_config
        from runtime import CaptainRuntime
        from perception.perception_builder import PerceptionBuilder
        from perception.route_manager import RouteManager
        from tests.test_pipeline_integration import SimulatedSDK
        import os
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = load_config(project)
        config.send_control = config.safety_brake_enabled = True
        config.control_calibrated = True
        for name,value in dict(wheelbase_m=2.9187,front_steer_max_rad=.5,steering_sign=1,
            throttle_per_mps=.2,brake_per_mps=.3,hold_brake=.1,emergency_brake=1.,
            max_throttle=.8,max_brake=.8).items():
            setattr(config,'control_'+name,value)
        with tempfile.TemporaryDirectory() as directory:
            config.runtime_dir = directory
            runtime = CaptainRuntime(config,SimpleNamespace(info=lambda *a:None,warning=lambda *a:None))
            self.addCleanup(runtime._diagnostics.close)
            sdk = SimulatedSDK(config,scene=6,frames=14,initial_offset=0.)
            sdk.install(runtime.adapter)
            builder = PerceptionBuilder(runtime.adapter,RouteManager(runtime.adapter,runtime.logger),6)
            builder.case_id,builder.case_name = 'real-clock-fixture','06'
            entered = threading.Event()
            original = runtime._publish_json
            def slow_disk(path,data):
                if not entered.is_set():
                    entered.set()
                    time.sleep(.8)
                return original(path,data)
            # No clock/member/sender/monitor output is replaced. Only disk I/O is delayed.
            with patch.object(runtime,'_publish_json',side_effect=slow_disk):
                runtime._loop(builder,False)
            self.assertTrue(entered.is_set())
            self.assertEqual(sdk.frames,len(sdk.sent))
            self.assertTrue(any(c['throttle']>0 for c in sdk.sent))
            self.assertTrue(all(c['brake']==0. and c['gear']==1 for c in sdk.sent))
            self.assertGreater(sdk.x,10.)
            self.assertLess(runtime._timing_summary['max_ms']['total'],500.)
            self.assertEqual(sdk.frames,runtime._timing_summary['cycles'])

    def test_blocked_disk_never_blocks_submit_and_latest_queue_is_bounded(self):
        entered, release = threading.Event(), threading.Event()
        writes = []
        def write(path, data):
            entered.set()
            release.wait(2.)
            writes.append((path,data))
            return True
        writer = DiagnosticWriter(write, capacity=3)
        self.addCleanup(lambda:(release.set(),writer.close()))
        writer.submit('active', {'frame':0})
        self.assertTrue(entered.wait(.5))
        start = time.perf_counter()
        for frame in range(100):
            writer.submit('latest', {'frame':frame})
        for frame in range(10):
            writer.submit('event-'+str(frame), {'frame':frame})
        self.assertLess(time.perf_counter()-start,.05)
        self.assertLessEqual(writer.status()['pending'],3)
        self.assertGreater(writer.status()['dropped'],0)
        self.assertFalse(writer.flush(.01))
        release.set()
        self.assertTrue(writer.close())
        self.assertEqual({'frame':9},writes[-1][1])

    def test_one_failed_write_does_not_stop_subsequent_snapshots(self):
        writes = []
        def write(path,data):
            if path == 'bad':raise PermissionError('busy')
            writes.append(data)
            return True
        writer = DiagnosticWriter(write)
        self.addCleanup(writer.close)
        writer.submit('bad',{})
        writer.submit('good',{'frame':2})
        self.assertTrue(writer.flush())
        self.assertEqual([{'frame':2}],writes)
        self.assertEqual(1,writer.status()['failed'])
