"""Exercise the SDK boundary with Python 3.6-compatible stand-ins."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simone_platform.simone_adapter import SimOneAdapter


class Vector(list):
    def Size(self):
        return len(self)

    def GetElement(self, index):
        return self[index]


class TrafficStatus(ctypes.c_int):
    pass


class TrafficNative(ctypes.Structure):
    _fields_ = [("status", TrafficStatus), ("countDown", ctypes.c_int)]


class AdapterTests(unittest.TestCase):
    @staticmethod
    def _adapter():
        config = SimpleNamespace(vehicle_id="0", sensor_id="perfectPerception1")
        return SimOneAdapter(config, SimpleNamespace())

    def test_snapshot_keeps_reference_data_and_failed_target_status(self):
        adapter = self._adapter()
        gps = SimpleNamespace(frame=10, timestamp=1000, posX=1.0, posY=2.0,
                              posZ=0.0, oriX=0.0, oriY=0.0, oriZ=0.0,
                              velX=3.0, velY=0.0, velZ=0.0,
                              accelX=-2.0, accelY=0.0, accelZ=0.0,
                              angVelZ=0.2, wheelSpeedFL=3.0, wheelSpeedFR=3.0,
                              wheelSpeedRL=3.0, wheelSpeedRR=3.0, odometer=20.0,
                              throttle=0.0, brake=0.5, steering=0.0, gear=1,
                              isGPSLost=False)
        route = SimpleNamespace(wayPointsSize=1,
                                wayPoints=[SimpleNamespace(
                                    index=2, posX=20.0, posY=0.0,
                                    heading_x=0.0, heading_y=0.0,
                                    heading_z=0.0, heading_w=1.0)])
        sensor_config = SimpleNamespace(sensorId=b"cam1", sensorType=b"Camera",
                                        hz=20, x=0.0, y=0.0, z=1.0, yaw=0.0)
        configs = SimpleNamespace(dataSize=1, data=[sensor_config])
        environment = SimpleNamespace(timeOfDay=12.0, cloudDensity=0.0,
                                      fogDensity=0.0, rainDensity=0.2,
                                      snowDensity=0.0, groundHumidityLevel=0.0,
                                      groundDirtyLevel=0.0)
        adapter.structs = SimpleNamespace(
            SimOne_Data_Gps=lambda: gps,
            SimOne_Data_SensorDetections=lambda: SimpleNamespace(),
            SimOne_Data_Obstacle=lambda: SimpleNamespace(),
            SimOne_Data_WayPoints=lambda: route,
            SimOne_Data_SensorConfigurations=lambda: configs,
            SimOne_Data_Environment=lambda: environment)
        adapter.sensor_api = SimpleNamespace(
            SoGetGps=lambda vehicle, data: True,
            SoGetSensorDetections=lambda vehicle, sensor, data: False,
            SoGetGroundTruth=lambda vehicle, data: False,
            SoGetSensorConfigurations=lambda vehicle, data: True,
            SoGetEnvironment=lambda data: True)
        adapter.pnc_api = SimpleNamespace(SoGetWayPoints=lambda vehicle, data: True)
        result = adapter.read_raw_snapshot()
        self.assertFalse(result["targets_valid"])
        self.assertEqual("none", result["target_source"])
        self.assertEqual("cam1", result["source_status"]["targets"]["sensor_id"])
        self.assertEqual("configured", result["source_status"]["targets"]["sensor_presence"])
        self.assertEqual([(20.0, 0.0)], result["route_points"])
        self.assertEqual([0.0, 0.0, 0.0, 1.0],
                         result["route_waypoints"][0]["heading_quaternion"])
        self.assertTrue(result["sensor_configurations_valid"])
        self.assertEqual("cam1", result["sensor_configurations"][0]["id"])
        self.assertTrue(result["environment_valid"])
        self.assertEqual(-2.0, result["gps"]["ax"])
        obstacle = SimpleNamespace(id=4, type=TrafficStatus(6), posX=10.0,
                                   posY=0.0, posZ=0.0, velX=0.0, velY=0.0,
                                   velZ=0.0, length=4.0, width=2.0,
                                   height=1.5)
        adapter.structs.SimOne_Data_Obstacle = lambda: SimpleNamespace(
            obstacleSize=1, obstacle=[obstacle], frame=10, timestamp=1000)
        adapter.sensor_api.SoGetGroundTruth = lambda vehicle, data: True
        recovered = adapter.read_raw_snapshot()
        self.assertTrue(recovered["targets_valid"])
        self.assertEqual("ground_truth", recovered["target_source"])
        self.assertEqual(6, recovered["targets"][0]["type"])
        self.assertIsNone(recovered["targets"][0]["heading"])
        sensor_config.sensorType = b"Radar"
        adapter._reference_update = 0.0
        calls = []
        adapter.sensor_api.SoGetSensorDetections = lambda *args: calls.append(args) or False
        absent = adapter.read_raw_snapshot()
        self.assertEqual([], calls)
        # Radar is a real sensor but not a detection source, which is distinct
        # from having no sensors and from an unrecognised detection type.
        self.assertEqual("no_detection_sensor",
                         absent["source_status"]["targets"]["sensor_presence"])
        self.assertEqual("non_detection_sensors_only",
                         absent["source_status"]["targets"]["sensor_catalog"]["status"])
        self.assertEqual("", absent["source_status"]["targets"]["sensor_id"])

    def test_traffic_uses_lane_stopline_and_caches_map_lookup(self):
        adapter = self._adapter()
        stop = SimpleNamespace(pt=SimpleNamespace(x=23.0, y=1.0),
                               boundaryKnots=Vector([SimpleNamespace(x=22.0, y=-1.0),
                                                     SimpleNamespace(x=22.0, y=1.0)]))
        light = SimpleNamespace(id=42, pt=SimpleNamespace(x=25.0, y=0.0))
        calls = []

        def stoplines(signal, lane_id):
            calls.append(lane_id)
            return Vector([stop])

        adapter.map_loaded = True
        adapter.hdmap = SimpleNamespace(getTrafficLightList=lambda: Vector([light]),
                                        getStoplineList=stoplines,
                                        pySimString=lambda value: value)
        adapter.structs = SimpleNamespace(SimOne_Data_TrafficLight=TrafficNative)

        def read_light(vehicle, identifier, native):
            native.status = 1
            native.countDown = 9
            return True

        adapter.sensor_api = SimpleNamespace(SoGetTrafficLights=read_light)
        for unused in range(2):
            observed = adapter.read_traffic("1_0_-1")
            self.assertEqual(1, observed[0]["status"])
            self.assertEqual(22.0, observed[0]["stop_line_x"])
            self.assertEqual(9, observed[0]["count_down"])
        self.assertEqual(["1_0_-1"], calls)
        # The association survives failure; the last successful RED is not reused.
        adapter.sensor_api.SoGetTrafficLights = lambda *args: False
        missing = adapter.read_traffic("1_0_-1")
        self.assertEqual(42, missing[0]["opendrive_id"])
        self.assertEqual(22.0, missing[0]["stop_line_x"])
        self.assertEqual(0, missing[0]["status"])
        self.assertFalse(missing[0]["read_ok"])
        self.assertTrue(adapter.last_traffic_query["association_valid"])
        self.assertFalse(adapter.last_traffic_query["read_ok"])
        adapter.sensor_api.SoGetTrafficLights = read_light
        self.assertTrue(adapter.read_traffic("1_0_-1")[0]["read_ok"])

    def test_empty_map_signals_are_distinct_from_missing_map_api(self):
        adapter = self._adapter()
        adapter.map_loaded = True
        adapter.hdmap = SimpleNamespace(getTrafficLightList=lambda: Vector([]),
                                       getStoplineList=lambda *args: Vector([]))
        self.assertEqual([], adapter.read_traffic("lane"))
        self.assertTrue(adapter.last_traffic_query["association_valid"])
        adapter.map_loaded = False
        self.assertEqual([], adapter.read_traffic("lane"))
        self.assertFalse(adapter.last_traffic_query["association_valid"])

    def test_auxiliary_sensor_sources_stay_separate(self):
        adapter = self._adapter()
        imu = SimpleNamespace(accelX=-1.0, accelY=0.0, accelZ=0.0,
                              angVelZ=0.3, rotX=0.0, rotY=0.0, rotZ=0.1)
        radar_hit = SimpleNamespace(id=7, type=TrafficStatus(6), posX=10.0,
                                    posY=1.0, velX=2.0, velY=0.0, range=11.0,
                                    rangeRate=-1.0, probability=0.8)
        radar = SimpleNamespace(detectNum=1, detections=[radar_hit],
                                frame=12, timestamp=1200)
        line = SimpleNamespace(lineID=1, lineType=TrafficStatus(1),
                               lineColor=TrafficStatus(4), linewidth=0.1,
                               linePoints=[], linecurveParameter=SimpleNamespace(
                                   C0=0.0, C1=0.0, C2=0.0, C3=0.0,
                                   firstPoints=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                   endPoints=SimpleNamespace(x=1.0, y=0.0, z=0.0), length=1.0))
        lane = SimpleNamespace(frame=12, timestamp=1200, id=3,
                               laneType=TrafficStatus(1), laneLeftID=2, laneRightID=4,
                               lanePredecessorID=[0], laneSuccessorID=[0],
                               ll_Line=line, l_Line=line, c_Line=line,
                               r_Line=line, rr_Line=line)
        ultrasonic = SimpleNamespace(
            sensorId=b"rear1", obstacleNum=1,
            obstacleDetections=[SimpleNamespace(obstacleRanges=1.2, x=-1.2,
                                                y=0.0, z=0.0)],
            frame=12, timestamp=1200)
        ultrasonics = SimpleNamespace(ultrasonicRadarNum=1,
                                     ultrasonicRadars=[ultrasonic])
        adapter.structs = SimpleNamespace(
            SimOne_Data_IMU=lambda: imu,
            SimOne_Data_RadarDetection=lambda: radar,
            SimOne_Data_LaneInfo=lambda: lane,
            SimOne_Data_UltrasonicRadars=lambda: ultrasonics)
        adapter.sensor_api = SimpleNamespace(
            SoGetImu=lambda vehicle, data: True,
            SoGetRadarDetections=lambda vehicle, sensor, data: True,
            SoGetSensorLaneInfo=lambda vehicle, sensor, data: True,
            SoGetUltrasonicRadars=lambda vehicle, data: True)
        result = adapter._read_auxiliary_data([
            {"id": "radar1", "type": "3"},
            {"id": "camera1", "type": "Camera"}])
        self.assertTrue(result["imu_valid"])
        self.assertEqual(6, result["radar_detections"][0]["type"])
        self.assertTrue(result["radar_status"]["radar1"])
        self.assertEqual(1, result["sensor_lane_observations"][0]["left_line_type"])
        self.assertTrue(result["ultrasonic_valid"])
        self.assertEqual("rear1", result["ultrasonic_detections"][0]["sensor_id"])

    def test_frame_age_uses_local_monotonic_time(self):
        adapter = self._adapter()
        with patch("simone_platform.simone_adapter.time.monotonic",
                   side_effect=[10.0, 10.25, 10.5]):
            self.assertEqual(0, adapter._frame_age_ms("gps", 12))
            self.assertEqual(250, adapter._frame_age_ms("gps", 12))
            self.assertEqual(0, adapter._frame_age_ms("gps", 13))

    def test_regressed_frame_stays_invalid_until_high_watermark(self):
        adapter = self._adapter()
        with patch("simone_platform.simone_adapter.time.monotonic",
                   side_effect=[10.0, 10.1, 10.2, 10.3]):
            self.assertEqual(0, adapter._frame_age_ms("gps", 12))
            self.assertEqual(-2, adapter._frame_age_ms("gps", 11))
            self.assertAlmostEqual(200, adapter._frame_age_ms("gps", 12), delta=1)
            self.assertEqual(0, adapter._frame_age_ms("gps", 13))

    def test_empty_radar_success_preserves_frame_metadata(self):
        adapter = self._adapter()
        adapter.structs = SimpleNamespace(SimOne_Data_RadarDetection=lambda:
            SimpleNamespace(detectNum=0, detections=[], frame=12, timestamp=1200))
        adapter.sensor_api = SimpleNamespace(SoGetRadarDetections=lambda *args: True)
        data = adapter._read_auxiliary_data([{"id": "radar1", "type": "Radar"}])
        self.assertEqual([], data["radar_detections"])
        self.assertTrue(data["radar_status"]["radar1"])
        self.assertEqual(12, adapter._source_status["radar:radar1"]["frame_id"])

    def test_ultrasonic_sensor_header_is_preserved(self):
        adapter = self._adapter()
        sensor = SimpleNamespace(sensorId=b"rear", frame=8, timestamp=800,
                                 obstacleNum=0, obstacleDetections=[])
        adapter.structs = SimpleNamespace(SimOne_Data_UltrasonicRadars=lambda:
            SimpleNamespace(ultrasonicRadarNum=1, ultrasonicRadars=[sensor]))
        adapter.sensor_api = SimpleNamespace(SoGetUltrasonicRadars=lambda *args: True)
        data = adapter._read_auxiliary_data([])
        self.assertTrue(data["ultrasonic_valid"])
        self.assertEqual([], data["ultrasonic_detections"])
        self.assertEqual(8, adapter._source_status["ultrasonic:rear"]["frame_id"])

    def test_map_traffic_signs_are_preserved_without_guessing_speed(self):
        adapter = self._adapter()
        sign = SimpleNamespace(
            id=9, type="274", subType="", value="40", unit="km/h",
            isDynamic=False, heading=0.0,
            pt=SimpleNamespace(x=5.0, y=0.0, z=0.0),
            validities=Vector([SimpleNamespace(roadId=1, sectionIndex=0,
                fromLaneId=-1, toLaneId=-1)]))
        adapter.map_loaded = True
        adapter.hdmap = SimpleNamespace(getTrafficSignList=lambda: Vector([sign]))
        adapter.pnc_api = SimpleNamespace()
        adapter.sensor_api = SimpleNamespace()
        data = adapter._read_reference_data()
        self.assertTrue(data["traffic_signs_valid"])
        self.assertEqual("40", data["traffic_signs"][0]["value"])
        self.assertEqual(-1, data["traffic_signs"][0]["validities"][0]["from_lane_id"])


if __name__ == "__main__":
    unittest.main()
