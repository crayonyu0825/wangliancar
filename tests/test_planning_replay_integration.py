"""Compatibility of the PR 17 offline tool with the PR 16 fixed entry.

SDK stand-ins exercise recovery, never the live simulator or vehicle.
"""
import copy
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.config import load_config
from core.serialization import perception_to_dict, to_dict
from core.traffic_quality import signal_stop_bound
from members import planning_stub
from members.planning.lane_planner import PlannerSettings
from members.planning.replay_bundle import replay_records
from perception.perception_builder import PerceptionBuilder
from perception.route_manager import RouteManager
from runtime import CaptainRuntime
from tests.test_pipeline_integration import SimulatedSDK
from tests.test_planning_replay import manifest, record
import tests.test_traffic_constraint_chain as traffic_tests


class PlanningReplayIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_geometry = planning_stub._CONTROL_GEOMETRY
        planning_stub._CONTROL_GEOMETRY = {}
        self.addCleanup(setattr, planning_stub, '_CONTROL_GEOMETRY', self.original_geometry)

    def test_shipped_example_uses_the_current_complete_settings_schema(self):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        directory = os.path.join(project, 'members', 'planning', 'examples')
        with open(os.path.join(directory, 'replay_manifest.synthetic.json'), encoding='utf-8') as stream:
            header = json.load(stream)
        with open(os.path.join(directory, 'replay_frames.synthetic.jsonl'), encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        with open(os.path.join(directory, 'replay_events.synthetic.json'), encoding='utf-8') as stream:
            events = json.load(stream)
        report = replay_records(header, rows, events=events)
        self.assertEqual(set(vars(PlannerSettings())), set(header['planner_settings']))
        self.assertEqual(2, report['summary']['replay_valid'])
        self.assertEqual(1, report['summary']['expired_inputs'])
        self.assertEqual('incomplete', report['delivery']['status'])

    def test_recorded_geometry_is_independent_of_previous_runtime_configuration(self):
        header, item = manifest(), record()
        header['planner_settings'].update(wheelbase_m=2.7, front_steer_max_rad=.5)
        expected = replay_records(header, [item], events=[])['records'][0]['replay']
        cached = {'wheelbase_m': 3.2, 'front_steer_max_rad': .4}
        planning_stub._CONTROL_GEOMETRY = cached
        observed = replay_records(header, [item], events=[])['records'][0]['replay']
        self.assertTrue(observed['valid'], observed['reason'])
        self.assertEqual(expected, observed)
        self.assertIs(cached, planning_stub._CONTROL_GEOMETRY)
        self.assertEqual({'wheelbase_m': 3.2, 'front_steer_max_rad': .4}, cached)

    def test_absent_recorded_capability_is_not_borrowed_from_runtime_cache(self):
        header, item = manifest(), record()
        item['perception']['ego'].update(x=0., y=0., speed=0., vx=0.)
        item['perception']['lane']['center_line'] = [
            [4*math.sin(i*.02), 4*(1-math.cos(i*.02))] for i in range(101)]
        expected = replay_records(header, [item], events=[])['records'][0]['replay']
        self.assertTrue(expected['valid'])
        planning_stub._CONTROL_GEOMETRY = {'wheelbase_m': 3.2, 'front_steer_max_rad': .4}
        observed = replay_records(header, [item], events=[])['records'][0]['replay']
        self.assertEqual(expected, observed)
        self.assertIsNone(header['planner_settings']['wheelbase_m'])
        self.assertIsNone(header['planner_settings']['front_steer_max_rad'])

    def test_new_signal_groups_reach_recomputed_planning(self):
        helper = traffic_tests.TrafficConstraintChainTests()
        helper.setUp()
        perception = helper.frame()
        helper.build(perception, {'10_0_-2': [helper.record(30, 2, 52)],
                                  '20_0_-2': [helper.record(80, 1, 60)]})
        self.assertTrue(perception.traffic.signal_groups_valid)
        self.assertEqual(2, len(perception.traffic.signal_groups))
        item, header = record(), manifest()
        item['perception']['ego'].update(x=10., y=0.)
        item['perception']['lane'] = to_dict(perception.lane)
        item['perception']['traffic'] = to_dict(perception.traffic)
        item['decision']['target_lane_id'] = perception.lane.lane_id
        before = copy.deepcopy(item)
        output = replay_records(header, [item], events=[])['records'][0]['replay']
        self.assertTrue(output['valid'], output['reason'])
        self.assertFalse(output['emergency_stop'])
        expected = signal_stop_bound(perception, 3.9187, .3)
        self.assertAlmostEqual(expected, output['stop_distance'])
        self.assertEqual(before, item)

    def test_recorded_stop_intent_remains_stop_on_green(self):
        helper = traffic_tests.TrafficConstraintChainTests()
        helper.setUp()
        perception = helper.frame()
        helper.build(perception, {'10_0_-2': [helper.record(30, 2, 52)]})
        self.assertEqual('GREEN', perception.traffic.signal_state)
        item = record()
        item['perception']['traffic'] = to_dict(perception.traffic)
        item['perception']['source_status']['traffic'] = copy.deepcopy(perception.source_status['traffic'])
        item['perception']['ego'].update(speed=0., vx=0.)
        item['decision'].update(mode='STOP', target_speed=0., stop_distance=0.)
        output = replay_records(manifest(), [item], events=[])['records'][0]['replay']
        self.assertTrue(output['valid'])
        self.assertEqual(0., output['initial_speed'])
        self.assertEqual(0., output['stop_distance'])
        # The tool executes recorded decisions, not a decision recovery engine.
        self.assertEqual('STOP', item['decision']['mode'])

    def _captured_pipeline(self, signal=False):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = load_config(project)
        # This offline recorder must inspect every cycle, independently of the
        # production latest-snapshot interval. Keep all original replay checks.
        config.diagnostic_interval_s = 0.0
        config.scene_id_override = 10 if signal else 1
        config.send_control = config.safety_brake_enabled = True
        config.control_calibrated = True
        config.control_wheelbase_m, config.control_front_steer_max_rad = 2.9187, .5
        rows = []
        environment = {'NEVC_DECISION_CRUISE_SPEED': '3.0',
                       'NEVC_VEHICLE_FRONT_OFFSET_M': '3.9187',
                       'NEVC_VEHICLE_HALF_WIDTH_M': '.9',
                       'NEVC_VEHICLE_REAR_OFFSET_M': '.88'}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment):
            config.runtime_dir = directory
            sdk = SimulatedSDK(config, scene=config.scene_id_override,
                               frames=360 if signal else 90, sensor=True,
                               red_light=signal, initial_offset=0.,
                               target_failures=() if signal else tuple(range(15, 24)))
            logger = SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
            with patch('time.monotonic', side_effect=lambda: sdk.now):
                runtime = CaptainRuntime(config, logger)
                sdk.install(runtime.adapter)
                if signal:
                    def light(vehicle, identity, native):
                        native.status, native.countDown = (1 if sdk.reads < 220 else 2), 10
                        return True
                    runtime.adapter.sensor_api.SoGetTrafficLights = light
                builder = PerceptionBuilder(runtime.adapter, RouteManager(runtime.adapter, logger),
                                            config.scene_id_override)
                builder.case_id, builder.case_name = 'synthetic-capture', 'synthetic-capture'
                builder.task_id = 'synthetic-capture-task'
                start = sdk.now
                def capture(p, d, t, c, receipt, safety):
                    rows.append(dict(session_id='synthetic-component-compatibility',
                        record_index=len(rows), relative_time_s=sdk.now-start,
                        capture_monotonic_s=sdk.now, recording_errors=[],
                        perception=perception_to_dict(p), decision=to_dict(d),
                        trajectory=to_dict(t), control=to_dict(c), send=copy.deepcopy(receipt),
                        safety=dict(mode=safety.mode, reason=safety.reason, candidate=to_dict(safety.control))))
                with patch.object(runtime, '_sleep_remaining', return_value=None), \
                        patch.object(runtime, '_publish_pipeline', side_effect=capture):
                    runtime._loop(builder, False)
                header = dict(session_id='synthetic-component-compatibility',
                    data_kind='synthetic_sdk_standin_not_platform_capture',
                    capture_clock='process_monotonic_seconds',
                    planner_settings=copy.deepcopy(runtime.runtime_info['planning_settings']),
                    source_quality_config=dict(sensor_timeout_ms=config.sensor_timeout_ms,
                                               max_sensor_frame_gap=config.max_sensor_frame_gap))
            events = [dict(event_id='synthetic-recovery', start_relative_time_s=rows[0]['relative_time_s'],
                           end_relative_time_s=rows[-1]['relative_time_s'])]
            report = replay_records(header, rows, events=events)
        return sdk, rows, report

    def test_full_loop_target_source_fault_and_recovery_remain_visible(self):
        sdk, rows, report = self._captured_pipeline()
        self.assertEqual(90, len(rows))
        self.assertEqual(90, report['summary']['records'])
        failed = [(item, replay) for item, replay in zip(rows, report['records'])
                  if not item['perception']['targets_valid']]
        self.assertTrue(failed)
        for item, replay in failed:
            self.assertIsNone(replay['replay'])
            self.assertFalse(replay['formal_sensor_usable'])
            self.assertIn('required_target_source_unusable', replay['issues'])
            self.assertEqual(item['safety'], replay['recorded']['safety'])
        self.assertEqual('normal', rows[-1]['safety']['mode'])
        self.assertGreater(rows[-1]['control']['throttle'], 0.)
        self.assertTrue(report['records'][-1]['replay']['valid'])
        self.assertEqual('not_executed', report['claims']['whole_pipeline_replay'])

    def test_full_loop_red_hold_green_recovery_matches_component_replay(self):
        sdk, rows, report = self._captured_pipeline(signal=True)
        red = [item for item in rows if item['perception']['traffic']['signal_state'] == 'RED']
        green = [item for item in rows if item['perception']['traffic']['signal_state'] == 'GREEN']
        self.assertTrue(red and green)
        self.assertTrue(any(item['perception']['ego']['speed'] < .05 and
                            item['control']['diagnostics'].get('state') == 'HOLD' for item in red[-30:]))
        self.assertLess(max(item['perception']['ego']['x'] + 3.9187 for item in red), 25.)
        self.assertGreater(sdk.x, 25.)
        self.assertGreater(sdk.speed, 1.)
        self.assertEqual(360, report['summary']['replayed'])
        self.assertEqual(360, report['summary']['replay_valid'])
        for item, replay in zip(rows, report['records']):
            self.assertAlmostEqual(item['trajectory']['stop_distance'], replay['replay']['stop_distance'])
        self.assertEqual('structural_only', report['delivery']['status'])


if __name__ == '__main__':
    unittest.main()
