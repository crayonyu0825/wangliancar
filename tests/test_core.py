import math
import unittest

from core.geometry import calculate_ttc, normalize_angle, world_to_ego
from core.interfaces import ControlOut
from core.scene_requirements import requires_targets
from simone_platform.case_resolver import resolve_scene_id


class CoreTests(unittest.TestCase):
    def test_target_stream_is_required_only_by_relevant_scenes(self):
        for scene_id in (4, 5, 6, 7, 8, 9, 10, 19, 20, 29):
            self.assertFalse(requires_targets(scene_id))
        for scene_id in (1, 3, 11, 18, 21, 28, 30, 41):
            self.assertTrue(requires_targets(scene_id))
        self.assertTrue(requires_targets(0))

    def test_scene_id_from_chinese_case_name(self):
        self.assertEqual(4, resolve_scene_id("04.直道车道偏离抑制-测试"))
        self.assertEqual(6, resolve_scene_id("06_车道居中控制"))
        self.assertEqual(0, resolve_scene_id("未编号场景"))
        self.assertEqual(9, resolve_scene_id("04.场景", 9))

    def test_world_to_ego(self):
        longitudinal, lateral = world_to_ego(0.0, 0.0, math.pi / 2.0, 0.0, 10.0)
        self.assertAlmostEqual(10.0, longitudinal, places=5)
        self.assertAlmostEqual(0.0, lateral, places=5)

    def test_ttc(self):
        ttc, closing = calculate_ttc(20.0, 10.0, 5.0, 0.0, 0.0)
        self.assertAlmostEqual(4.0, ttc)
        self.assertAlmostEqual(5.0, closing)

    def test_control_clamp(self):
        output = ControlOut()
        output.throttle = 2.0
        output.brake = -1.0
        output.steering = -3.0
        output.clamp()
        self.assertEqual(1.0, output.throttle)
        self.assertEqual(0.0, output.brake)
        self.assertEqual(-1.0, output.steering)

    def test_sensor_type_map_overrides_the_builtin_table(self):
        """A platform naming its detection sensor differently must be fixable
        by configuration, not by editing this module."""
        from simone_platform.sensor_catalog import (parse_type_map, sensor_kind,
                                                    sensor_catalog_report,
                                                    target_sensor_ids)
        configurations = [{"id": "obj1", "type": "ObjectBasedSensor"}]
        self.assertEqual("unknown", sensor_kind("ObjectBasedSensor"))
        self.assertEqual([], target_sensor_ids(configurations, ""))
        self.assertEqual("unclassified_sensor_types_only",
                         sensor_catalog_report(configurations)["status"])

        mapping = parse_type_map("ObjectBasedSensor=perfect")
        self.assertEqual("perfect", sensor_kind("ObjectBasedSensor", mapping))
        self.assertEqual(["obj1"], target_sensor_ids(configurations, "", mapping))
        self.assertEqual("detection_source_configured",
                         sensor_catalog_report(configurations, mapping)["status"])

    def test_sensor_type_map_rejects_invalid_entries(self):
        from simone_platform.sensor_catalog import parse_type_map
        for bad in ("Foo", "Foo=banana", "=camera", "Foo="):
            with self.assertRaises(ValueError):
                parse_type_map(bad)
        self.assertEqual({}, parse_type_map(""))
        self.assertEqual({}, parse_type_map(None))
        self.assertEqual({"cam": "camera"},
                         parse_type_map("Cam=Camera, "))

    def test_sensor_type_map_is_loaded_and_validated_by_config(self):
        from core.config import AppConfig
        ok = AppConfig(".", {"sensor_type_map": "Obj=camera"})
        self.assertEqual({"obj": "camera"}, ok.sensor_type_map)
        self.assertEqual({}, AppConfig(".", {}).sensor_type_map)
        with self.assertRaises(ValueError):
            AppConfig(".", {"sensor_type_map": "Obj=notakind"})


if __name__ == "__main__":
    unittest.main()
