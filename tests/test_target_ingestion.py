"""Regression cases for official sensor discovery and source validity."""

import ctypes
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from perception.perception_builder import PerceptionBuilder
from core.interfaces import LaneContext
from core.serialization import perception_to_dict
from simone_platform.sensor_catalog import sensor_kind, target_sensor_ids
from simone_platform.simone_adapter import SimOneAdapter
from simone_platform.sdk_compat import polling_structs
from test_sdk_compat import legacy_sdk


class DetectionEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("id", ctypes.c_int), ("type", ctypes.c_int)] + [
        (name, ctypes.c_float) for name in (
            "posX", "posY", "posZ", "velX", "velY", "velZ", "length",
            "width", "height", "probability")]


class DetectionPacket(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("timestamp", ctypes.c_int64), ("frame", ctypes.c_int),
               ("objectSize", ctypes.c_int), ("objects", DetectionEntry * 2)]


def packet(frame=10, objects=1):
    value = DetectionPacket()
    value.frame, value.timestamp, value.objectSize = frame, frame * 50, objects
    if objects:
        item = value.objects[0]
        item.id, item.type, item.posX = 31, 6, 20.0
        item.length, item.width, item.height, item.probability = 4.0, 2.0, 1.5, 1.0
    return value


def catalog(*sources, **kwargs):
    return {"sensor_configurations_valid": kwargs.get("valid", True),
            "sensor_configurations_reason": "ok" if sources else "read_failed",
            "sensor_configurations": [{"id": identifier, "type": kind}
                                      for identifier, kind in sources]}


class TargetIngestionTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(vehicle_id="0", sensor_id="preferred",
                                      sensor_timeout_ms=500, max_sensor_frame_gap=2,
                                      pipeline_timeout_ms=200)
        self.logs = []
        self.adapter = SimOneAdapter(self.config, SimpleNamespace(
            info=lambda *args: self.logs.append(args)))
        self.adapter.structs = SimpleNamespace(SimOne_Data_SensorDetections=DetectionPacket)
        self.calls = []
        self.responses = {}

        def read(vehicle, identifier, destination):
            self.calls.append(identifier)
            value = self.responses.get(identifier)
            if isinstance(value, Exception):
                raise value
            if value is None:
                return False
            ctypes.memmove(ctypes.addressof(destination), ctypes.addressof(value),
                           ctypes.sizeof(value))
            return True
        self.adapter.sensor_api = SimpleNamespace(SoGetSensorDetections=read)

    def test_numeric_types_and_platform_names_use_the_right_capabilities(self):
        for value, expected in ((b"1", "camera"), (2, "lidar"), ("7", "perfect"),
                                ("9", "fusion"), ("ObjectBasedCamera", "camera"),
                                ("MMWRadar", "radar"), ("UltrasonicRadar", "ultrasonic")):
            self.assertEqual(expected, sensor_kind(value))
        value = catalog(("radar", "3"), ("cam", "1"), ("lidar", "2"),
                        ("perfect", "7"), ("fusion", "9"))
        self.assertEqual(["perfect", "cam", "lidar", "fusion"],
                         target_sensor_ids(value["sensor_configurations"], "perfect"))
        self.assertEqual(["cam", "lidar", "perfect", "fusion"],
                         target_sensor_ids(value["sensor_configurations"], "radar"))

    def test_failed_first_source_falls_back_to_another_configured_source(self):
        self.responses["lidar"] = packet()
        result = self.adapter._read_target_sources(catalog(("cam", "1"), ("lidar", "2")), 10)
        self.assertEqual(["cam", "lidar"], self.calls)
        self.assertEqual("sensor:lidar", result[1])
        self.assertEqual(31, result[0][0]["id"])
        self.assertTrue(result[4]["id_verified"])
        self.assertEqual("configured", result[4]["sensor_presence"])

    def test_preferred_sensor_cannot_override_an_incompatible_type(self):
        self.responses["cam"] = packet()
        result = self.adapter._read_target_sources(catalog(("preferred", "3"), ("cam", "1")), 10)
        self.assertEqual(["cam"], self.calls)
        self.assertEqual("sensor:cam", result[1])

    def test_successful_empty_sensor_packet_is_valid_evidence(self):
        self.responses["preferred"] = packet(objects=0)
        result = self.adapter._read_target_sources(catalog(("preferred", "7")), 10)
        self.assertEqual([], result[0])
        self.assertEqual("sensor:preferred", result[1])
        self.assertEqual("empty", result[4]["reason"])

    def test_confirmed_absent_target_sensor_does_not_probe_default_id(self):
        result = self.adapter._read_target_sources(catalog(("radar", "3")), 10)
        self.assertEqual([], self.calls)
        self.assertIsNone(result[0])
        # A radar-only vehicle, a vehicle with no sensors at all, and an
        # unrecognised detection type are three different situations with
        # three different fixes, so the report must not merge them.
        self.assertEqual("no_detection_sensor", result[4]["sensor_presence"])
        self.assertEqual("non_detection_sensors_only",
                         result[4]["sensor_catalog"]["status"])

    def test_unrecognised_detection_type_is_not_reported_as_absent(self):
        """A naming difference must not look like a missing sensor.

        The reading rule is unchanged: an unknown type is still not a target
        source. The point is that the report distinguishes "the platform
        published no detection sensor" from "it published one under a type this
        binding does not know".
        """
        result = self.adapter._read_target_sources(
            catalog(("obj", "ObjectBasedSensor")), 10)
        self.assertEqual([], self.calls)
        self.assertIsNone(result[0])
        self.assertEqual("unclassified_types", result[4]["sensor_presence"])
        report = result[4]["sensor_catalog"]
        self.assertEqual("unclassified_sensor_types_only", report["status"])
        self.assertEqual([], report["detection_ids"])
        self.assertEqual("ObjectBasedSensor", report["unclassified"][0]["raw_type"])

    def test_missing_catalog_is_unknown_and_retries_are_bounded(self):
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.0):
            first = self.adapter._read_target_sources(catalog(valid=False), 10)
            second = self.adapter._read_target_sources(catalog(valid=False), 10)
        self.assertEqual(["preferred"], self.calls)
        self.assertEqual("unknown", first[4]["sensor_presence"])
        self.assertEqual("retry_backoff", second[4]["attempts"][0]["reason"])
        self.responses["preferred"] = packet()
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=11.1):
            recovered = self.adapter._read_target_sources(catalog(valid=False), 10)
        self.assertEqual("sensor:preferred", recovered[1])
        self.assertFalse(recovered[4]["id_verified"])

    def test_stalled_primary_stream_falls_back_without_relabeling_it_fresh(self):
        self.responses["preferred"], self.responses["other"] = packet(), packet(frame=11)
        sources = catalog(("preferred", "7"), ("other", "1"))
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.0):
            self.adapter._read_target_sources(sources, 10)
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.6):
            result = self.adapter._read_target_sources(sources, 11)
        self.assertEqual("sensor:other", result[1])
        self.assertEqual("stale", result[4]["attempts"][0]["reason"])

    def test_future_or_regressed_packet_is_not_clear_road_evidence(self):
        self.responses["preferred"] = packet(frame=20, objects=0)
        result = self.adapter._read_target_sources(catalog(("preferred", "7")), 10)
        self.assertIsNone(result[0])
        self.assertEqual("unsynchronized", result[4]["attempts"][0]["reason"])
        self.adapter._target_probe_after.clear()
        self.responses["preferred"] = packet(frame=19, objects=0)
        result = self.adapter._read_target_sources(catalog(("preferred", "7")), 19)
        self.assertIsNone(result[0])
        self.assertEqual("regressed", result[4]["attempts"][0]["reason"])

    def test_bad_object_count_and_nonfinite_data_allow_other_source(self):
        for bad in (packet(objects=3), packet()):
            if bad.objectSize == 1:
                bad.objects[0].posX = float("nan")
            self.responses["preferred"], self.responses["other"] = bad, packet()
            self.adapter._target_probe_after.clear()
            result = self.adapter._read_target_sources(catalog(("preferred", "7"), ("other", "1")), 10)
            self.assertEqual("sensor:other", result[1])
            self.assertTrue(any("SENSOR_TARGETS_INVALID:preferred" in error for error in result[5]))

    def test_callback_discovers_real_id_without_catalog_or_guessed_probes(self):
        value = packet()
        self.adapter._receive_target_detection(b"0", b"actualCamera", ctypes.pointer(value))
        value.objects[0].posX = 999.0  # native storage is immediately reusable
        result = self.adapter._read_target_sources(catalog(valid=False), 10)
        self.assertEqual([], self.calls)
        self.assertEqual("sensor:actualCamera", result[1])
        self.assertEqual(20.0, result[0][0]["x"])
        self.assertEqual("callback", result[4]["transport"])
        self.assertTrue(result[4]["id_verified"])
        self.assertEqual("unknown", result[4]["sensor_presence"])

    def test_callback_for_another_vehicle_is_ignored(self):
        self.adapter._receive_target_detection(b"1", b"cam", ctypes.pointer(packet()))
        self.assertEqual({}, self.adapter._target_streams)

    def test_invalid_callback_does_not_publish_empty_success(self):
        self.adapter._receive_target_detection(b"0", b"cam", ctypes.pointer(packet(objects=3)))
        result = self.adapter._read_target_sources(catalog(valid=False), 10)
        self.assertIsNone(result[0])
        self.assertEqual(["cam"], self.calls)

    def test_expired_callback_can_recover_by_polling_the_discovered_id(self):
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.0):
            self.adapter._receive_target_detection(b"0", b"cam", ctypes.pointer(packet()))
        self.responses["cam"] = packet(frame=11)
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.6):
            result = self.adapter._read_target_sources(catalog(valid=False), 11)
        self.assertEqual(["cam"], self.calls)
        self.assertEqual("poll", result[4]["transport"])
        self.assertEqual(11, result[2])

    def test_new_callback_recovers_during_poll_failure_backoff(self):
        sources = catalog(("preferred", "7"))
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.0):
            self.adapter._read_target_sources(sources, 10)
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.1):
            self.adapter._receive_target_detection(b"0", b"preferred", ctypes.pointer(packet(frame=11)))
            result = self.adapter._read_target_sources(sources, 11)
        self.assertEqual(["preferred"], self.calls)
        self.assertEqual("sensor:preferred", result[1])

    def test_native_callback_uses_repaired_stride_and_stays_alive(self):
        self.adapter.structs, unused_repairs = polling_structs(legacy_sdk(), "3.0.0001")
        self.adapter.sdk_version = "3.0.0001"
        retained = []
        def register(callback):
            retained.append(callback)
            return True
        self.adapter.sensor_api.SimoneAPI = SimpleNamespace(SetSensorDetectionsUpdateCB=register)
        self.adapter._register_target_callback()
        self.assertEqual("registered", self.adapter._target_callback_status)
        kind = self.adapter.structs.SimOne_Data_SensorDetections
        value = kind()
        value.frame, value.timestamp, value.objectSize = 10, 500, 0
        retained[0](b"0", b"real", ctypes.pointer(value))
        self.assertIs(self.adapter._target_callback, retained[0])
        self.assertEqual(228, ctypes.sizeof(kind._fields_[1][1]._type_))
        self.assertEqual([], self.adapter._target_streams["real"]["targets"])

    def test_unknown_callback_abi_is_rejected_without_registration(self):
        self.adapter.sdk_version = "3.0.0001"
        registered = []
        self.adapter.sensor_api.SimoneAPI = SimpleNamespace(
            SetSensorDetectionsUpdateCB=lambda *args: registered.append(args))
        self.adapter._register_target_callback()
        self.assertEqual([], registered)
        self.assertTrue(self.adapter._target_callback_status.startswith("unavailable"))

    def test_shutdown_clears_streams_and_late_callbacks_cannot_repopulate_them(self):
        value = packet()
        self.adapter._receive_target_detection(b"0", b"cam", ctypes.pointer(value))
        self.adapter._last_frame_times["targets:sensor:cam"] = (10, 1.0)
        self.adapter.connected = True
        self.adapter.service_api = SimpleNamespace(SoTerminateSimOneAPI=lambda: None)
        self.adapter.shutdown()
        self.adapter._receive_target_detection(b"0", b"cam", ctypes.pointer(value))
        self.assertEqual({}, self.adapter._target_streams)
        self.assertEqual({}, self.adapter._last_frame_times)

    def test_returned_callback_packet_does_not_mutate_the_cached_native_snapshot(self):
        self.adapter._receive_target_detection(b"0", b"cam", ctypes.pointer(packet()))
        result = self.adapter._read_target_sources(catalog(valid=False), 10)
        result[0][0]["x"] = -999.0
        self.assertEqual(20.0, self.adapter._target_streams["cam"]["targets"][0]["x"])

    def test_failed_catalog_is_retried_before_five_second_reference_cache(self):
        self.adapter.pnc_api = SimpleNamespace()
        self.adapter.structs.SimOne_Data_SensorConfigurations = SimpleNamespace
        calls = []
        def configurations(unused_vehicle, data):
            calls.append(True)
            data.dataSize, data.data = 0, []
            return len(calls) > 1
        self.adapter.sensor_api.SoGetSensorConfigurations = configurations
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.0):
            first = self.adapter._read_reference_data()
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.2):
            self.adapter._read_reference_data()
        with patch("simone_platform.simone_adapter.time.monotonic", return_value=10.6):
            recovered = self.adapter._read_reference_data()
        self.assertEqual(2, len(calls))
        self.assertEqual("read_failed", first["sensor_configurations_reason"])
        self.assertEqual("empty", recovered["sensor_configurations_reason"])
        self.assertTrue(recovered["sensor_configurations_valid"])

    def test_callback_sensor_packet_reaches_public_perception_and_json_metadata(self):
        self.adapter._receive_target_detection(b"0", b"cam", ctypes.pointer(packet()))
        targets, source, frame, timestamp, meta, unused_errors = (
            self.adapter._read_target_sources(catalog(valid=False), 10))
        meta.update(read_ok=True, sensor_read_ok=True, frame_id=frame,
                    timestamp=timestamp, age_ms=0, clock="sdk_frame")
        raw = {"gps": {"frame": 10, "timestamp": 500, "x": 0.0, "y": 0.0,
                       "heading": 0.0, "vx": 1.0, "vy": 0.0, "gear": 1, "age_ms": 0},
               "targets": targets, "target_source": source, "targets_valid": True,
               "targets_frame": frame, "targets_timestamp": timestamp, "targets_age_ms": 0,
               "source_status": {"targets": meta}}
        route = SimpleNamespace(update=lambda ego: LaneContext())
        self.adapter.read_traffic = lambda lane: []
        perception = PerceptionBuilder(self.adapter, route).build_from_raw(raw)
        self.assertTrue(perception.targets_valid)
        self.assertTrue(perception.source_status["targets"]["usable"])
        self.assertEqual("sensor:cam", perception.target_source)
        self.assertEqual(31, perception.targets[0].id)
        self.assertEqual("callback", perception.source_status["targets"]["transport"])
        published = json.loads(json.dumps(perception_to_dict(perception), allow_nan=False))
        self.assertTrue(published["source_status"]["targets"]["id_verified"])
        self.assertEqual(31, published["targets"][0]["id"])


if __name__ == "__main__":
    unittest.main()
