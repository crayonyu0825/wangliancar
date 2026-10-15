"""Capture a one-frame brake even when the next pipeline snapshot is clear."""
import glob
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.interfaces import DecisionMode, Target
from core.safety_supervisor import SafetyAssessment
from runtime import CaptainRuntime
from tests.test_safety_runtime import FakeAdapter, perception, decision, trajectory, controller


class BrakingEventTests(unittest.TestCase):
    def runtime(self, directory):
        config = SimpleNamespace(runtime_dir=directory, loop_hz=20, send_control=True,
                                 safety_brake_enabled=True, publish_json=True,
                                 pipeline_timeout_ms=200, sensor_timeout_ms=500)
        return CaptainRuntime(config, SimpleNamespace(info=lambda *args: None,
                                                       warning=lambda *args: None))

    def test_transient_plan_failure_keeps_full_target_frame_after_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            frames = [perception(i) for i in range(10, 16)]
            for p in frames:
                p.lane.center_line = [(0, 0), (100, 0)]
                target = Target()
                target.id, target.x, target.valid = 11646222, 78, True
                target.lane_id, target.lane_width_m = 'successor', 3.97
                p.targets = [target]
            def follow(p):
                d = decision(p)
                d.mode, d.target_speed, d.reason = DecisionMode.FOLLOW, 8.33, 'FOLLOW_TARGET'
                return d
            def plan(p, d):
                t = trajectory(p, d)
                if p.frame_id == 11:
                    t.valid = False
                    t.reason = 'crossing or oncoming obstacle motion unsupported'
                return t
            runtime.adapter = FakeAdapter(len(frames))
            iterator = iter(frames)
            with patch('runtime.decide', side_effect=follow), \
                 patch('runtime.plan', side_effect=plan), \
                 patch('runtime.compute_control', side_effect=controller), \
                 patch.object(runtime, '_sleep_remaining', return_value=None):
                runtime._loop(SimpleNamespace(build=lambda: next(iterator)), False)
            files = glob.glob(os.path.join(directory, 'braking_events', '*.json'))
            events = []
            for path in files:
                with open(path, encoding='utf-8') as stream:
                    events.append(json.load(stream))
            onset = next(e for e in events if e['event'] == 'brake_start')
            self.assertEqual(11, onset['perception']['frame_id'])
            self.assertEqual(DecisionMode.FOLLOW, onset['decision']['mode'])
            self.assertEqual(3.97, onset['perception']['targets'][0]['lane_width_m'])
            self.assertEqual([[0, 0], [100, 0]], onset['perception']['lane']['center_line'])
            self.assertEqual(10, onset['preceding_frames'][-1]['frame_id'])
            self.assertEqual('safety_sent', onset['send']['reason'])
            self.assertTrue(any(e['event'] == 'brake_release' for e in events))
            with open(runtime.pipeline_path, encoding='utf-8') as stream:
                self.assertTrue(json.load(stream)['trajectory']['valid'])
            self.assertEqual(0.2, runtime.adapter.sent[-1][1])

    def test_controller_brake_and_release_are_saved_without_safety_override(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            for frame, brake in ((1, 0), (2, 0.2), (3, 0)):
                p = perception(frame)
                d, t = decision(p), trajectory(p, decision(p))
                c = controller(p, t)
                c.brake, c.throttle, c.source = brake, 0 if brake else 0.2, 'control:TRACK'
                c.diagnostics['reference_speed_mps'] = 0 if brake else 2.0
                runtime._trace_braking(p, d, t, c, {'reason': 'sent'}, SafetyAssessment())
            runtime.flush_diagnostics()
            files = sorted(glob.glob(os.path.join(directory, 'braking_events', '*.json')))
            self.assertEqual(2, len(files))
            with open(files[0], encoding='utf-8') as stream:
                event = json.load(stream)
            self.assertEqual('normal', event['safety']['mode'])
            self.assertEqual(0.2, event['control']['brake'])
            self.assertEqual(0, event['control']['diagnostics']['reference_speed_mps'])

    def test_diagnostic_write_failure_does_not_change_command_and_capture_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory)
            p = perception()
            d, t = decision(p), trajectory(p, decision(p))
            c = controller(p, t)
            c.brake, c.throttle = 0.2, 0
            receipt = {'reason': 'sent', 'ok': True}
            with patch.object(runtime, '_publish_json', side_effect=PermissionError('busy')):
                runtime._trace_braking(p, d, t, c, receipt, SafetyAssessment())
                runtime.flush_diagnostics()
            self.assertEqual((0.2, 0, True), (c.brake, c.throttle, receipt['ok']))
            with patch.object(runtime, '_publish_json', return_value=True) as publish:
                for i in range(160):
                    c.brake = 0.2 if i % 2 else 0
                    runtime._trace_braking(p, d, t, c, receipt, SafetyAssessment())
                runtime.flush_diagnostics()
                self.assertLessEqual(publish.call_count, 79)
            self.assertEqual(8, len(runtime._braking_history))
