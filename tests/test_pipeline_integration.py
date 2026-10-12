"""Run the real four-module loop with simulated SDK I/O and vehicle feedback.

No member function, perception builder, sender or safety supervisor is mocked.
The small plant and SDK stand-ins verify integration, not SimOne dynamics/ABI.
"""

import ctypes
import json
import math
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.config import load_config
from core.geometry import project_polyline
from perception.perception_builder import PerceptionBuilder
from perception.route_manager import RouteManager
from runtime import CaptainRuntime
from simone_platform.simone_adapter import _native_dict


class Enum(ctypes.c_int):
    pass


class NativeControl(ctypes.Structure):
    _fields_ = [("frame", ctypes.c_int), ("timestamp", ctypes.c_longlong),
                ("EThrottleMode", Enum), ("throttle", ctypes.c_float),
                ("EBrakeMode", Enum), ("brake", ctypes.c_float),
                ("ESteeringMode", Enum), ("steering", ctypes.c_float),
                ("handbrake", ctypes.c_bool), ("isManualGear", ctypes.c_bool),
                ("gear", Enum)]


class NativeLights(ctypes.Structure):
    _fields_ = [("frame", ctypes.c_int), ("timestamp", ctypes.c_longlong),
                ("signalLights", ctypes.c_uint)]


class Vector(list):
    def Size(self):
        return len(self)

    def GetElement(self, index):
        return self[index]


class SimulatedSDK(object):
    def __init__(self, config, scene=4, frames=240, sensor=False,
                 target_ok=True, red_light=False, repeat=False, observe=False,
                 config_ok=True, gear_feedback=1, initial_offset=0.3,
                 connected_curve=False, broken_curve=False, curve_exit_length=0,
                 target_failures=(), target_velocity=None, target_sensor_id="perfectPerception1"):
        self.config, self.scene, self.frames = config, scene, frames
        self.sensor, self.target_ok = sensor, target_ok
        self.red_light, self.repeat, self.observe = red_light, repeat, observe
        self.config_ok, self.gear_feedback = config_ok, gear_feedback
        self.target_failures, self.target_velocity = set(target_failures), target_velocity
        self.target_sensor_id = target_sensor_id
        self.connected_curve, self.broken_curve = connected_curve, broken_curve
        self.curve_exit_length = curve_exit_length
        self.reads, self.now = 0, time.monotonic()
        # Keep the initial pose inside the latest planner's 0.5 m envelope.
        self.x, self.y, self.heading, self.speed = 10.0, initial_offset, 0.03, 0.0
        self.yaw_rate = 0.0
        self.command = None
        self.sent, self.lights, self.modes = [], [], []
        self.positions, self.pipelines = [], []

    def gps(self, unused_vehicle, data):
        self.reads += 1
        self.now += 0.05
        if self.command is not None and not self.repeat:
            command = self.command
            acceleration = command["throttle"] * 4.0 - command["brake"] * 8.0
            next_speed = max(0.0, self.speed + acceleration * 0.05)
            speed = 0.5 * (self.speed + next_speed)
            angle = (command["steering"] * self.config.control_front_steer_max_rad
                     / self.config.control_steering_sign)
            self.yaw_rate = speed / self.config.control_wheelbase_m * math.tan(angle)
            middle = self.heading + self.yaw_rate * 0.025
            self.x += speed * math.cos(middle) * 0.05
            self.y += speed * math.sin(middle) * 0.05
            self.heading += self.yaw_rate * 0.05
            self.speed = next_speed
        data.frame = 1 if self.repeat else self.reads
        data.timestamp = data.frame * 50
        data.posX, data.posY, data.posZ = self.x, self.y, 0.0
        data.oriX, data.oriY, data.oriZ = 0.0, 0.0, self.heading
        data.velX = self.speed * math.cos(self.heading)
        data.velY, data.velZ = self.speed * math.sin(self.heading), 0.0
        data.accelX = data.accelY = data.accelZ = 0.0
        data.angVelZ = self.yaw_rate
        data.wheelSpeedFL = data.wheelSpeedFR = self.speed
        data.wheelSpeedRL = data.wheelSpeedRR = self.speed
        data.odometer, data.throttle, data.brake, data.steering = self.x, 0.0, 0.0, 0.0
        data.gear, data.isGPSLost = self.gear_feedback, False
        self.positions.append((self.x, self.y, self.speed))
        return True

    def targets(self, unused_vehicle, unused_sensor, data):
        data.frame, data.timestamp = self.reads, self.reads * 50
        data.objectSize, data.objects = 0, []
        if unused_sensor != self.target_sensor_id or self.reads in self.target_failures:
            return False
        if self.target_velocity is not None:
            data.objectSize = 1
            data.objects = [SimpleNamespace(id=31, type=6, posX=40.0 + self.target_velocity * self.reads * 0.05,
                posY=0.0, posZ=0.0, velX=self.target_velocity, velY=0.0, velZ=0.0,
                length=4.0, width=1.8, height=1.5, probability=1.0)]
        return self.target_ok

    def configurations(self, unused_vehicle, data):
        data.data = [SimpleNamespace(sensorId=self.target_sensor_id.encode("utf-8"),
                    sensorType=b"7", hz=20, x=0.0, y=0.0, z=1.0,
                    yaw=0.0)] if self.sensor else []
        data.dataSize = len(data.data)
        return self.config_ok

    def drive(self, unused_vehicle, data):
        self.command = _native_dict(data)
        self.sent.append(self.command)
        return True

    def signals(self, unused_vehicle, data):
        self.lights.append(_native_dict(data))
        return True

    def mode(self, unused_vehicle, value):
        self.modes.append(value)
        return True

    def install(self, adapter):
        adapter.structs = SimpleNamespace(
            SimOne_Data_Gps=SimpleNamespace,
            SimOne_Data_SensorConfigurations=SimpleNamespace,
            SimOne_Data_SensorDetections=SimpleNamespace,
            SimOne_Data_Obstacle=SimpleNamespace,
            SimOne_Data_Control=NativeControl,
            SimOne_Data_Signal_Lights=NativeLights,
            SimOne_Data_CaseInfo=SimpleNamespace,
            SimOne_Data_TrafficLight=SimpleNamespace)
        adapter.sensor_api = SimpleNamespace(
            SoGetGps=self.gps, SoGetSensorConfigurations=self.configurations,
            SoGetSensorDetections=self.targets, SoGetGroundTruth=lambda *args: False)
        if self.red_light:
            def light(unused_vehicle, unused_identifier, data):
                data.status, data.countDown = 1, 10
                return True
            adapter.sensor_api.SoGetTrafficLights = light
        adapter.pnc_api = SimpleNamespace(
            SoSetDriveMode=self.mode, SoSetDrive=self.drive,
            SoSetSignalLights=self.signals)
        adapter.service_api = SimpleNamespace(
            SoGetCaseRunStatus=lambda: 2 if self.reads < self.frames else 1)
        adapter.map_loaded = True
        def line(y):
            return Vector([SimpleNamespace(x=float(x), y=y, z=0.0)
                           for x in range(161)])
        signal = SimpleNamespace(id=42, pt=SimpleNamespace(x=25.0, y=0.0),
            validities=Vector([SimpleNamespace(roadId=1, sectionIndex=0,
                                               fromLaneId=-1, toLaneId=-1)]))
        stopline = SimpleNamespace(pt=SimpleNamespace(x=25.0, y=0.0))
        adapter.hdmap = SimpleNamespace(
            pySimPoint3D=lambda *args: args, pySimString=lambda value: value,
            getNearMostLane=lambda *args: SimpleNamespace(exists=True, laneId="1_0_-1"),
            getLaneSample=lambda *args: SimpleNamespace(exists=True,
                laneInfo=SimpleNamespace(centerLine=line(0.0),
                    leftBoundary=line(1.75), rightBoundary=line(-1.75))),
            getLaneWidth=lambda *args: SimpleNamespace(exists=True, width=3.5),
            getTrafficLightList=lambda: Vector([signal]) if self.red_light else Vector(),
            getStoplineList=lambda *args: Vector([stopline]))
        if self.connected_curve or self.broken_curve:
            # A short current lane plus a real curved successor forces the
            # complete runtime to cross the boundary and refresh the lane ID.
            gap = 0.5 if self.broken_curve else 0.0
            lines = {"1_0_-1": [(0, 0), (20, 0)],
                     "2_0_-1": [(20 + 30 * math.sin(i * 0.02),
                                  gap + 30 * (1 - math.cos(i * 0.02)))
                                 for i in range(101)]}
            if self.curve_exit_length:
                end_x, end_y = lines["2_0_-1"][-1]
                lines["2_0_-1"].extend((end_x + distance * math.cos(2.0),
                                         end_y + distance * math.sin(2.0))
                                        for distance in range(1, self.curve_exit_length + 1))
            def nearest(position):
                lane_id = min(lines, key=lambda key:
                              project_polyline(lines[key], position[0], position[1])["distance"])
                return SimpleNamespace(exists=True, laneId=lane_id)
            def sample(lane_id):
                points = Vector([SimpleNamespace(x=x, y=y, z=0.0) for x, y in lines[lane_id]])
                return SimpleNamespace(exists=True, laneInfo=SimpleNamespace(
                    centerLine=points, leftBoundary=Vector(), rightBoundary=Vector()))
            def links(lane_id):
                return SimpleNamespace(exists=True, laneLink=SimpleNamespace(
                    leftNeighborLaneId="", rightNeighborLaneId="",
                    successorLaneIds=Vector(["2_0_-1"] if lane_id == "1_0_-1" else []),
                    predecessorLaneIds=Vector(["1_0_-1"] if lane_id == "2_0_-1" else [])))
            adapter.hdmap.getNearMostLane = nearest
            adapter.hdmap.getLaneSample = sample
            adapter.hdmap.getLaneLink = links


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Preserve the earlier low-speed regression fixtures independently
        # of the current launch/default cruise; the 30 km/h test uses defaults.
        environment = patch.dict(os.environ, {"NEVC_DECISION_CRUISE_SPEED": "3.0"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_raw_gearbox_positions_cannot_become_reverse_or_park_commands(self):
        for feedback in (-1, 0, 1, 2, 3, 6):
            with self.subTest(feedback=feedback):
                sdk = self.run_pipeline(scene=6, frames=24, config_ok=False,
                                        target_ok=False, gear_feedback=feedback)
                self.assertEqual(feedback, sdk.perception["ego"]["gear"])
                self.assertEqual(sdk.frames, len(sdk.sent))
                self.assertTrue(all(c["gear"] == 1 for c in sdk.sent))
                self.assertTrue(all(p["trajectory"]["valid"] and p["control"]["valid"]
                                    and p["send"]["reason"] == "sent" for p in sdk.pipelines))
                self.assertGreater(sdk.speed, 0.0)

    def test_snapshot_locks_cannot_interrupt_normal_control_or_safety_braking(self):
        cases = ({"scene": 6, "config_ok": False, "target_ok": False,
                  "gear_feedback": 0},
                 {"scene": 1, "sensor": True, "target_ok": False})
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for options in cases:
            with self.subTest(scene=options["scene"]), tempfile.TemporaryDirectory() as directory:
                config = load_config(project)
                config.runtime_dir = directory
                config.scene_id_override = options["scene"]
                config.send_control = config.safety_brake_enabled = True
                warnings = []
                logger = SimpleNamespace(info=lambda *args: None,
                                         warning=lambda *args: warnings.append(args))
                sdk = SimulatedSDK(config, frames=20, **options)
                with patch("time.monotonic", side_effect=lambda: sdk.now):
                    runtime = CaptainRuntime(config, logger)
                    sdk.install(runtime.adapter)
                    builder = PerceptionBuilder(runtime.adapter, RouteManager(runtime.adapter, logger),
                                                config.scene_id_override)
                    with patch.object(runtime, "_sleep_remaining", return_value=None), \
                         patch("runtime.os.replace", side_effect=PermissionError(5, "reader holds file")):
                        runtime._loop(builder, False)
                self.assertEqual(sdk.frames, len(sdk.sent))
                snapshot_warnings = [w for w in warnings if "诊断快照写入失败" in w[0]]
                # Each latest snapshot still warns only once. Additional
                # immutable brake-event files may report their own failure.
                warned_paths = [w[1] for w in snapshot_warnings]
                self.assertEqual(1, warned_paths.count(runtime.snapshot_path))
                self.assertEqual(1, warned_paths.count(runtime.pipeline_path))
                if options["scene"] == 6:
                    self.assertGreater(sdk.sent[-1]["throttle"], 0.0)
                    self.assertGreater(sdk.speed, 0.0)
                else:
                    self.assertTrue(all(c["throttle"] == 0 and c["brake"] > 0
                                        for c in sdk.sent))

    def run_pipeline(self, **options):
        project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = load_config(project)
        config.scene_id_override = options.get("scene", 4)
        config.send_control = not options.get("observe", False)
        config.safety_brake_enabled = True
        with tempfile.TemporaryDirectory() as directory:
            config.runtime_dir = directory
            logger = SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
            sdk = SimulatedSDK(config, **options)
            # Engines retain the clock callable at construction. Construct all
            # modules under the same simulation clock as the SDK and validator.
            with patch("time.monotonic", side_effect=lambda: sdk.now):
                runtime = CaptainRuntime(config, logger)
                sdk.install(runtime.adapter)
                builder = PerceptionBuilder(runtime.adapter, RouteManager(runtime.adapter, logger),
                                            config.scene_id_override)
                builder.case_id, builder.case_name = "integration", "integration"
                publish = runtime._publish_pipeline
                def capture(*args):
                    publish(*args)
                    with open(runtime.pipeline_path, encoding="utf-8") as stream:
                        sdk.pipelines.append(json.load(stream))
                with patch.object(runtime, "_sleep_remaining", return_value=None), \
                     patch.object(runtime, "_publish_pipeline", side_effect=capture):
                    runtime._loop(builder, False)
            with open(runtime.snapshot_path, encoding="utf-8") as stream:
                sdk.perception = json.load(stream)
        return sdk

    def test_starts_and_tracks_with_real_modules_and_optional_sensors_absent(self):
        for scene in (4, 6):
            with self.subTest(scene=scene):
                sdk = self.run_pipeline(scene=scene)
                self.assertEqual(sdk.frames, len(sdk.sent))
                self.assertGreater(sdk.sent[0]["throttle"], 0.0)
                self.assertGreater(sdk.x, 25.0)
                self.assertAlmostEqual(3.0, sdk.speed, delta=0.15)
                self.assertLess(abs(sdk.y), 0.12)
                self.assertEqual("not_configured", sdk.perception["source_status"]["targets"]["sensor_presence"])
                self.assertEqual(sdk.frames, len(sdk.lights))
                for pipeline, native in zip(sdk.pipelines, sdk.sent):
                    self.assertEqual("DecisionEngine", pipeline["runtime"]["engine"])
                    self.assertEqual("decision-behaviors-runtime-stop-v1", pipeline["runtime"]["version"])
                    self.assertEqual(3.0, pipeline["runtime"]["cruise_speed"])
                    frame = pipeline["perception_frame_id"]
                    for key in ("decision", "trajectory", "control"):
                        self.assertTrue(pipeline[key]["valid"], pipeline[key])
                        self.assertEqual(frame, pipeline[key]["frame_id"])
                    self.assertEqual("sent", pipeline["send"]["reason"])
                    self.assertTrue(pipeline["send"]["ok"])
                    self.assertEqual(frame, native["frame"])
                    self.assertEqual(frame * 50, native["timestamp"])
                    self.assertEqual(1, native["gear"])
                    self.assertFalse(native["handbrake"] or native["isManualGear"])
                    self.assertEqual((0, 0, 0), (native["EThrottleMode"],
                                     native["EBrakeMode"], native["ESteeringMode"]))

    def test_required_empty_target_stream_allows_launch(self):
        sdk = self.run_pipeline(scene=1, sensor=True, frames=30)
        self.assertTrue(sdk.perception["targets_valid"])
        self.assertEqual([], sdk.perception["targets"])
        self.assertGreater(sdk.sent[0]["throttle"], 0.0)
        self.assertEqual("normal", sdk.pipelines[-1]["safety"]["mode"])

    def test_required_empty_targets_work_across_tasks_with_discovered_nondefault_id(self):
        for scene in (1, 14, 15, 25):
            with self.subTest(scene=scene):
                sdk = self.run_pipeline(scene=scene, sensor=True, frames=12,
                                        target_sensor_id="actualTarget2")
                self.assertGreater(sdk.speed, 0.0)
                self.assertEqual("sensor:actualTarget2", sdk.perception["target_source"])
                self.assertTrue(all(p["target_input"]["state"] == "valid_empty"
                                    and p["send"]["reason"] == "sent" for p in sdk.pipelines))

    def test_target_losses_stop_then_recover_through_real_four_module_chain(self):
        failures = (8, 9) + tuple(range(45, 76))
        for scene in (14, 15):
            with self.subTest(scene=scene):
                sdk = self.run_pipeline(scene=scene, sensor=True, frames=130,
                                        target_sensor_id="actualTarget2", target_failures=failures)
                by_frame = {p["perception_frame_id"]: p for p in sdk.pipelines}
                for number in failures:
                    p = by_frame[number]
                    self.assertEqual("unavailable", p["target_input"]["state"])
                    self.assertEqual("required_targets_unavailable", p["safety"]["reason"])
                    self.assertEqual("safety_sent", p["send"]["reason"])
                    self.assertEqual(0.0, p["safety"]["candidate"]["throttle"])
                    self.assertGreater(p["safety"]["candidate"]["brake"], 0.0)
                self.assertTrue(any(p["safety"]["reason"] == "recovery_waiting_for_fresh_frames"
                                    for p in sdk.pipelines))
                self.assertEqual("normal", sdk.pipelines[-1]["safety"]["mode"])
                self.assertEqual("valid_empty", sdk.pipelines[-1]["target_input"]["state"])
                self.assertEqual("sent", sdk.pipelines[-1]["send"]["reason"])
                self.assertGreater(sdk.sent[-1]["throttle"], 0.0)

    def test_static_and_moving_leads_enter_real_chain_without_source_protection(self):
        with patch.dict(os.environ, {"NEVC_VEHICLE_FRONT_OFFSET_M": "3.9187",
                                    "NEVC_VEHICLE_HALF_WIDTH_M": "0.9"}):
            for scene, velocity in ((1, 0.0), (14, 2.0), (15, 2.0)):
                with self.subTest(scene=scene, velocity=velocity):
                    sdk = self.run_pipeline(scene=scene, sensor=True, frames=60,
                                            target_sensor_id="actualTarget2", target_velocity=velocity)
                    self.assertGreater(sdk.speed, 0.0)
                    self.assertTrue(all(p["target_input"]["state"] == "valid_objects"
                                        for p in sdk.pipelines))
                    self.assertTrue(all(p["safety"]["reason"] != "required_targets_unavailable"
                                        for p in sdk.pipelines))
                    self.assertTrue(all(p["send"]["ok"] for p in sdk.pipelines))
                    if velocity == 0.0:
                        # A distant stationary lead permits approach while
                        # KEEP_LANE carries the explicit bounded stop distance.
                        self.assertTrue(all(p["decision"]["reason"].startswith("STOP_TARGET:")
                                            and p["decision"]["stop_distance"] > 0
                                            for p in sdk.pipelines))
                    else:
                        self.assertTrue(any(p["decision"]["mode"] == "FOLLOW"
                                            for p in sdk.pipelines))

    def test_default_30_kmh_reaches_straight_cruise_and_keeps_curve_limits(self):
        with patch.dict(os.environ):
            os.environ.pop("NEVC_DECISION_CRUISE_SPEED", None)
            straight = self.run_pipeline(scene=4, frames=360)
            # A straight exit separates curve-following from terminal braking;
            # the existing end-of-map test separately checks genuine stops.
            curved = self.run_pipeline(scene=6, connected_curve=True,
                                       curve_exit_length=160, frames=350)
        cruise = 30.0 / 3.6
        self.assertAlmostEqual(cruise, straight.speed, delta=0.2)
        self.assertGreater(straight.x, 90.0)
        self.assertAlmostEqual(cruise, straight.pipelines[-1]["runtime"]["cruise_speed"])
        self.assertAlmostEqual(cruise,
                               straight.pipelines[-1]["control"]["diagnostics"]["reference_speed_mps"])
        self.assertTrue(all(p["send"]["reason"] == "sent" for p in straight.pipelines))
        self.assertGreater(curved.y, 30.0)
        moving_bend = []
        for position, pipeline in zip(curved.positions, curved.pipelines):
            self.assertAlmostEqual(cruise, pipeline["decision"]["target_speed"])
            self.assertTrue(pipeline["trajectory"]["valid"], pipeline["trajectory"]["reason"])
            self.assertTrue(pipeline["control"]["valid"], pipeline["control"]["errors"])
            if 1.0 < position[1] < 40.0:
                moving_bend.append(pipeline)
                self.assertEqual("normal", pipeline["safety"]["mode"], pipeline["trajectory"]["reason"])
                self.assertEqual("TRACK", pipeline["control"]["diagnostics"]["state"])
        self.assertTrue(moving_bend)
        self.assertTrue(all(p["control"]["diagnostics"]["reference_speed_mps"] < cruise
                            for p in moving_bend))

    def test_crosses_lane_boundary_tracks_curve_then_stops_at_true_map_end(self):
        sdk = self.run_pipeline(scene=6, connected_curve=True, frames=650)
        self.assertEqual(sdk.frames, len(sdk.sent))
        self.assertEqual("2_0_-1", sdk.perception["lane"]["lane_id"])
        self.assertGreater(sdk.y, 40.0)
        self.assertAlmostEqual(2.0, sdk.heading, delta=0.12)
        terminal = (20 + 30 * math.sin(2.0), 30 * (1 - math.cos(2.0)))
        for position, pipeline in zip(sdk.positions, sdk.pipelines):
            self.assertTrue(pipeline["decision"]["valid"])
            self.assertTrue(pipeline["trajectory"]["valid"], pipeline["trajectory"]["reason"])
            self.assertTrue(pipeline["control"]["valid"], pipeline["control"]["errors"])
            if pipeline["send"]["reason"] != "sent":
                # The independent monitor may override a late, infeasible
                # braking profile at the true terminal. It must never do so
                # at the current-lane/curve boundary or during the bend.
                self.assertEqual("safety_sent", pipeline["send"]["reason"])
                self.assertLess(math.hypot(position[0] - terminal[0], position[1] - terminal[1]), 2.0)
        self.assertEqual("successor-continuation-v4", pipeline["runtime"]["route_reference_version"])
        crossing = [(position, pipeline) for position, pipeline in zip(sdk.positions, sdk.pipelines)
                    if 18.5 <= position[0] <= 23.0 and position[1] < 1.0]
        self.assertTrue(crossing)
        self.assertTrue(all(position[2] > 2.0 for position, _ in crossing))
        self.assertTrue(all(p["control"]["diagnostics"]["state"] == "TRACK" for _, p in crossing))
        self.assertIn(["1_0_-1", "2_0_-1"],
                      [p["route_reference"]["lane_ids"] for p in sdk.pipelines])
        self.assertLess(sdk.speed, 0.1)
        self.assertEqual("HOLD", sdk.pipelines[-1]["control"]["diagnostics"]["state"])
        self.assertTrue(sdk.pipelines[-1]["trajectory"]["stop_required"])
        self.assertEqual("map_end", sdk.pipelines[-1]["route_reference"]["status"])

    def test_disconnected_successor_does_not_remove_boundary_stop(self):
        sdk = self.run_pipeline(scene=6, broken_curve=True, frames=200)
        self.assertEqual("1_0_-1", sdk.perception["lane"]["lane_id"])
        self.assertEqual("disconnected_successor", sdk.pipelines[-1]["route_reference"]["status"])
        self.assertEqual("HOLD", sdk.pipelines[-1]["control"]["diagnostics"]["state"])
        self.assertLess(sdk.x, 20.0)
        self.assertLess(sdk.speed, 0.1)
        self.assertTrue(all(p["trajectory"]["valid"] for p in sdk.pipelines))

    def test_live_observed_unknown_optional_sensor_and_neutral_feedback_can_start(self):
        sdk = self.run_pipeline(scene=6, frames=40, config_ok=False,
                                target_ok=False, gear_feedback=0)
        self.assertEqual("unknown", sdk.perception["source_status"]["targets"]["sensor_presence"])
        self.assertFalse(sdk.perception["targets_valid"])
        self.assertEqual(0, sdk.perception["ego"]["gear"])
        self.assertTrue(all(p["trajectory"]["valid"] and p["control"]["valid"]
                            for p in sdk.pipelines))
        self.assertTrue(all(c["gear"] == 1 for c in sdk.sent))
        self.assertGreater(sdk.sent[0]["throttle"], 0.0)
        self.assertGreater(sdk.speed, 0.0)

    def test_real_planner_failure_reason_survives_runtime_validation(self):
        sdk = self.run_pipeline(scene=6, frames=4, initial_offset=0.6)
        trajectory = sdk.pipelines[-1]["trajectory"]
        self.assertFalse(trajectory["valid"])
        self.assertIn("lateral recovery requires explicit vehicle/steering capability", trajectory["reason"])
        self.assertEqual(trajectory["reason"], trajectory["errors"][0])
        self.assertEqual(sdk.pipelines[-1]["perception_frame_id"], trajectory["frame_id"])
        self.assertTrue(all(c["throttle"] == 0 and c["brake"] > 0 for c in sdk.sent))

    def test_required_target_failure_cannot_become_clear_road(self):
        sdk = self.run_pipeline(scene=1, sensor=True, target_ok=False, frames=8)
        self.assertEqual("controlled_stop", sdk.pipelines[-1]["safety"]["mode"])
        self.assertTrue(all(c["throttle"] == 0.0 and c["brake"] > 0.0
                            for c in sdk.sent))

    def test_red_light_from_map_and_sdk_reaches_stop_and_hold(self):
        sdk = self.run_pipeline(red_light=True, frames=20)
        self.assertEqual("RED", sdk.perception["traffic"]["signal_state"])
        self.assertTrue(all(p["decision"]["mode"] == "STOP" for p in sdk.pipelines))
        self.assertTrue(all(p["control"]["valid"] for p in sdk.pipelines))
        self.assertTrue(all(c["throttle"] == 0.0 and c["brake"] > 0.0
                            for c in sdk.sent))

    def test_observe_mode_runs_every_module_without_sdk_writes(self):
        sdk = self.run_pipeline(observe=True, frames=8)
        self.assertEqual([], sdk.sent)
        self.assertEqual([], sdk.modes)
        self.assertTrue(sdk.pipelines[-1]["control"]["valid"])
        self.assertGreater(sdk.pipelines[-1]["control"]["throttle"], 0.0)
        self.assertEqual("observe_mode", sdk.pipelines[-1]["send"]["reason"])

    def test_duplicate_frames_are_not_resent_and_stall_reaches_brake(self):
        sdk = self.run_pipeline(repeat=True, frames=12)
        self.assertEqual("gps_frame_repeated", sdk.pipelines[1]["send"]["reason"])
        self.assertEqual("fault_stop", sdk.pipelines[-1]["safety"]["mode"])
        self.assertGreater(sdk.sent[0]["throttle"], 0.0)
        self.assertTrue(all(c["throttle"] == 0.0 and c["brake"] > 0.0
                            for c in sdk.sent[1:]))


if __name__ == "__main__":
    unittest.main()
