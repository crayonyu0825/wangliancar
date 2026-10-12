"""Selected SDK crosswalk -> real map/Sensor builder -> shared strict reader.

Native adapters and visibility below are authored test inputs, not SimOne
runtime calibration. We never author a verified crossing environment by hand.
"""
import copy
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS

from core.interfaces import Perception
from core.passage_facts import read_crosswalk_facts, read_passage_facts, _exit_boundary
from core.region_geometry import rectangle, outline_contains_point, simple_outline
from core.serialization import perception_to_dict
from perception.maneuver_environment import ManeuverEnvironmentBuilder
from scripts.replay_decision import _assign
from simone_platform.map_observations import MapObservationReader
from tests import test_local_road_regions as local_fixture
from tests.test_map_observations import map_api
from tests.test_maneuver_perception import profile, target


class CrosswalkPassageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, 'visibility.json')
        self.now = 100.
        self.clock = lambda: self.now
        self.adapter, self.metadata, self.route, self.p = local_fixture.LocalRoadRegionTests().fixture()
        self.builder = ManeuverEnvironmentBuilder(self.route, NS(vehicle_id='0', sensor_timeout_ms=500,
                                                                sensor_visibility_file=self.path), self.clock)
        self.vehicle = NS(front_offset_m=3.9, rear_offset_m=1., half_width_m=.9)
        native = MapObservationReader(map_api()).lane_objects([self.p.lane.lane_id])
        for key, value in native.items():
            setattr(self.p, key, copy.deepcopy(value))
        self.write_profile(profile(self.p))
        self.build()

    def write_profile(self, model):
        with open(self.path, 'w', encoding='utf-8') as stream:
            json.dump(model, stream)

    def build(self):
        self.p.ego.frame_id = self.p.frame_id
        self.p.valid_until = self.now + .2
        self.p.maneuver_environment = self.builder.build(self.p)
        return self.p.maneuver_environment

    def read(self):
        return read_passage_facts(self.p, 'crosswalk:5', self.vehicle, self.clock)

    def crossing(self):
        return self.p.maneuver_environment.crossing_regions[0]

    def test_actual_sdk_and_current_map_sensor_chain_is_accepted(self):
        self.assertTrue(self.p.map_crosswalks_valid)
        self.assertEqual('clear', self.crossing()['occupancy'])
        scene = self.read()
        self.assertEqual('CROSSWALK', scene['kind'])
        self.assertEqual([[(44., -2.), (46., -2.), (46., 2.), (44., 2.)]], scene['conflict_polygons'])
        self.assertEqual(self.p.frame_id, scene['frame_id'])
        self.assertEqual([], scene['objects'])
        self.assertFalse(scene['already_inside'])
        self.assertFalse(scene['exit_progress_reached'])
        self.assertAlmostEqual(30., scene['junction']['entry_distance_m'])
        self.assertAlmostEqual(36., scene['junction']['exit_distance_m'])
        self.assertEqual('sdk_stop_line', scene['junction']['entry_kind'])
        self.assertEqual(4, scene['junction']['entry_stop_line']['id'])
        self.assertEqual('UNKNOWN', scene['junction']['right_of_way'])
        self.assertFalse(scene['junction']['rule_verified'])
        self.assertEqual(self.p.maneuver_environment.dynamic_valid_until, scene['source_valid_until_s'])

    def test_associated_catalogue_is_not_mistaken_for_exhaustive_absence(self):
        self.assertFalse(self.p.map_observation_status['map_crosswalks']['coverage_complete'])
        self.p.map_observation_status['map_crosswalks']['signal_catalog_complete'] = False
        self.assertEqual('CROSSWALK', self.read()['kind'])

    def test_original_data_is_not_changed_and_returned_objects_are_detached(self):
        self.p.targets = [target(25., 0.)]
        self.build()
        before = perception_to_dict(self.p)
        scene = self.read()
        self.assertEqual(before, perception_to_dict(self.p))
        scene['junction']['entry_stop_line']['boundary_knots'][0][0] = -100.
        scene['objects'][0]['x'] = -100.
        scene['corridor'].polygons[0][0] = (-100., -100.)
        self.assertEqual(before, perception_to_dict(self.p))

    def test_full_body_front_can_enter_before_rear_axle(self):
        self.p.ego.x = 40.2
        self.build()
        scene = self.read()
        self.assertLess(self.p.ego.x, 44.)
        self.assertTrue(scene['already_inside'])
        self.assertFalse(scene['exit_progress_reached'])

    def test_axle_past_far_edge_does_not_mean_tail_exited(self):
        self.p.ego.x = 46.5
        self.build()
        scene = self.read()
        self.assertGreater(self.p.ego.x, 46.)
        self.assertTrue(scene['already_inside'])
        self.assertFalse(scene['exit_progress_reached'])
        self.assertLess(scene['junction']['exit_distance_m'], 0.)
        body = rectangle(46.5, 0., 0., 3.9, 1., .9)
        self.assertLessEqual(scene['exit_region'].body_clearance(body, lambda: None, 100000), 0.)

    def test_whole_body_exit_and_exit_region_use_measured_current_pose(self):
        self.p.ego.x = 48.
        self.build()
        scene = self.read()
        self.assertFalse(scene['already_inside'])
        self.assertTrue(scene['exit_progress_reached'])
        body = rectangle(48., 0., 0., 3.9, 1., .9)
        self.assertGreater(scene['exit_region'].body_clearance(body, lambda: None, 100000), 0.)
        self.assertAlmostEqual(-8., scene['junction']['entry_distance_m'])
        self.assertEqual(4, scene['junction']['entry_stop_line']['id'])

    def test_exit_geometry_does_not_include_observed_road_before_crosswalk(self):
        polygon = self.read()['exit_region'].polygons[0]
        self.assertFalse(outline_contains_point((20., 0.), polygon))
        self.assertFalse(outline_contains_point((45., 0.), polygon))
        self.assertTrue(outline_contains_point((50., 0.), polygon))
        self.assertTrue(all(v[0] >= 46. for v in polygon))
        self.assertEqual(75., max(v[0] for v in polygon))

    def test_stop_line_distance_is_recomputed_not_copied_from_published_scalar(self):
        self.crossing()['entry_distance_m'] = 1000.
        self.assertAlmostEqual(30., self.read()['junction']['entry_distance_m'])
        self.p.ego.x = 30.
        self.build()
        self.assertAlmostEqual(10., self.read()['junction']['entry_distance_m'])

    def test_unrelated_stop_line_association_does_not_create_a_stop_gate(self):
        self.p.map_stop_lines[0]['association_signal_ids'] = [88]
        self.build()
        self.assertIsNone(self.crossing()['entry_stop_line'])
        scene = self.read()
        self.assertEqual('crosswalk_boundary', scene['junction']['entry_kind'])
        self.assertAlmostEqual(34., scene['junction']['entry_distance_m'])

    def test_empty_valid_stopline_query_is_distinct_from_unreadable_query(self):
        self.p.map_stop_lines = []
        self.build()
        self.assertEqual('crosswalk_boundary', self.read()['junction']['entry_kind'])
        self.p.map_stop_lines_valid = False
        with self.assertRaisesRegex(ValueError, 'STOP_LINE_SOURCE_UNKNOWN'):
            self.read()

    def test_occupied_crosswalk_keeps_all_target_types_and_unknown_heading(self):
        pedestrian, vehicle = target(45., 0., heading=None, identifier=9), target(25., 0., identifier=10)
        pedestrian.type, vehicle.type = 5, 1
        self.p.targets = [pedestrian, vehicle]
        self.build()
        scene = self.read()
        self.assertEqual('occupied', scene['junction']['occupancy'])
        self.assertEqual([9], scene['junction']['object_ids'])
        self.assertEqual([9, 10], [v['id'] for v in scene['objects']])
        self.assertIsNone(scene['objects'][0]['heading'])
        self.assertEqual([5, 1], [v['type'] for v in scene['objects']])

    def test_unknown_object_motion_cannot_be_treated_as_static_or_dropped(self):
        self.p.targets = [target(45., 0.)]
        self.p.targets[0].vx = None
        self.build()
        with self.assertRaises(ValueError): self.read()

    def test_ground_truth_and_failed_stale_sensor_are_not_formal_clearance(self):
        original = copy.deepcopy(self.p)
        for mutation in ('ground_truth', 'failed', 'stale', 'wrong_frame'):
            self.p = copy.deepcopy(original)
            if mutation == 'ground_truth': self.p.target_source = 'ground_truth'
            elif mutation == 'failed': self.p.source_status['targets']['usable'] = False
            elif mutation == 'stale': self.p.source_status['targets']['quality'] = 'stale'
            else: self.p.source_status['targets']['frame_id'] = self.p.targets_frame_id + 1
            with self.assertRaises(ValueError, msg=mutation): self.read()

    def test_empty_targets_without_verified_full_visibility_are_not_clear(self):
        model = profile(self.p)
        model['profiles'][0]['complete_detections'] = False
        self.write_profile(model)
        self.build()
        with self.assertRaises(ValueError): self.read()
        model = profile(self.p)
        model['profiles'][0]['body_region_m'] = [[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]]
        self.write_profile(model)
        self.build()
        with self.assertRaises(ValueError): self.read()

    def test_partial_view_of_crosswalk_cannot_authorize_unseen_exit(self):
        model = profile(self.p)
        model['profiles'][0]['body_region_m'] = [[-20., -20.], [38., -20.], [38., 20.], [-20., 20.]]
        self.write_profile(model)
        self.build()
        self.assertTrue(self.crossing()['coverage_verified'])
        self.assertFalse(next(v for v in self.p.maneuver_environment.road_regions
                              if v['role'] == 'current')['coverage_verified'])
        with self.assertRaises(ValueError): self.read()

    def test_original_sensor_deadline_is_preserved_and_expiration_rejected(self):
        self.p.source_status['targets']['age_ms'] = 450
        self.build()
        self.assertAlmostEqual(100.05, self.read()['source_valid_until_s'])
        self.now = 100.051
        with self.assertRaises(ValueError): self.read()

    def test_expiry_during_reader_work_cannot_publish_scene(self):
        class Clock(object):
            def __init__(self): self.calls = 0
            def __call__(self):
                self.calls += 1
                return 100. if self.calls < 8 else 100.21
        with self.assertRaises(ValueError):
            read_crosswalk_facts(self.p, 'crosswalk:5', self.vehicle, Clock())

    def test_source_query_status_failures_are_not_geometry_authority(self):
        original = copy.deepcopy(self.p)
        for field, value in (('read_ok', False), ('source', 'ground_truth'), ('clock', 'process_monotonic'),
                             ('api', 'other'), ('version', 'other'), ('invalid_records', 1),
                             ('invalid_records', False), ('errors', ['query:failed'])):
            self.p = copy.deepcopy(original)
            self.p.map_observation_status['map_crosswalks'][field] = value
            with self.assertRaises(ValueError, msg=field): self.read()

    def test_duplicate_or_missing_selected_native_and_published_ids_reject(self):
        original = copy.deepcopy(self.p)
        for mutation in ('native_duplicate', 'native_missing', 'published_duplicate', 'published_missing'):
            self.p = copy.deepcopy(original)
            if mutation == 'native_duplicate': self.p.map_crosswalks.append(copy.deepcopy(self.p.map_crosswalks[0]))
            elif mutation == 'native_missing': self.p.map_crosswalks = []
            elif mutation == 'published_duplicate': self.p.maneuver_environment.crossing_regions.append(copy.deepcopy(self.crossing()))
            else: self.p.maneuver_environment.crossing_regions = []
            with self.assertRaises(ValueError, msg=mutation): self.read()

    def test_native_boundary_plane_or_association_cannot_be_replaced(self):
        original = copy.deepcopy(self.p)
        for mutation in ('boundary', 'plane', 'lane', 'source'):
            self.p = copy.deepcopy(original)
            if mutation == 'boundary': self.p.map_crosswalks[0]['boundary_knots'][0][0] = 43.
            elif mutation == 'plane': self.p.map_crosswalks[0]['boundary_knots'][0][2] = .2
            elif mutation == 'lane': self.p.map_crosswalks[0]['lane_ids'] = ['9_0_-1']
            else: self.p.map_crosswalks[0]['source'] = 'unverified'
            with self.assertRaises(ValueError, msg=mutation): self.read()

    def test_matched_odr_map_and_current_road_must_be_verified(self):
        self.metadata['verified'] = False
        self.build()
        with self.assertRaises(ValueError): self.read()

    def test_sensor_packet_cannot_drop_target_or_falsify_crosswalk_occupancy(self):
        self.p.targets = [target(45., 0.)]
        self.build()
        original = copy.deepcopy(self.p)
        for mutation in ('drop', 'occupied_ids', 'clear'):
            self.p = copy.deepcopy(original)
            if mutation == 'drop': self.p.maneuver_environment.objects = []
            elif mutation == 'occupied_ids': self.crossing()['object_ids'] = []
            else: self.crossing()['occupancy'] = 'clear'
            with self.assertRaises(ValueError, msg=mutation): self.read()

    def test_published_stop_line_must_equal_original_sdk_line(self):
        self.crossing()['entry_stop_line']['boundary_knots'][0][0] += 1.
        with self.assertRaisesRegex(ValueError, 'STOP_LINE_SOURCE_MISMATCH'): self.read()

    def test_ambiguous_same_station_stop_lines_are_rejected(self):
        another = copy.deepcopy(self.p.map_stop_lines[0])
        another['id'] = 6
        another['boundary_knots'] = [[40., -3., 0.], [40., 3., 0.]]
        self.p.map_stop_lines.append(another)
        self.build()
        with self.assertRaisesRegex(ValueError, 'STOP_LINE_AMBIGUOUS'): self.read()

    def test_public_json_roundtrip_keeps_real_source_facts(self):
        serial = json.loads(json.dumps(perception_to_dict(self.p), allow_nan=False))
        p = Perception()
        _assign(p, serial, 'perception')
        scene = read_crosswalk_facts(p, 'crosswalk:5', self.vehicle, self.clock)
        self.assertEqual('CROSSWALK', scene['kind'])
        self.assertAlmostEqual(30., scene['junction']['entry_distance_m'])

    def test_invalid_identifier_or_missing_vehicle_dimension_reject(self):
        for identifier in ('crosswalk:-1', 'crosswalk:05', 'crosswalk:true', 'crosswalk:', 'crosswalk:6'):
            with self.assertRaises(ValueError, msg=identifier):
                read_crosswalk_facts(self.p, identifier, self.vehicle, self.clock)
        self.vehicle.rear_offset_m = None
        with self.assertRaisesRegex(ValueError, 'VEHICLE_GEOMETRY_UNKNOWN'): self.read()

    def test_downstream_cut_preserves_concavity_instead_of_filling_hull(self):
        left = [(0., 2., 0.), (50., 2., 0.), (55., .4, 0.), (60., 2., 0.), (75., 2., 0.)]
        right = [(0., -2., 0.), (75., -2., 0.)]
        a = _exit_boundary(left, (46., 0., 0.), 0., lambda: None)
        b = _exit_boundary(right, (46., 0., 0.), 0., lambda: None)
        polygon = simple_outline(a + list(reversed(b)))
        self.assertIn((55., .4), polygon)
        self.assertFalse(outline_contains_point((55., 1.), polygon))
        self.assertTrue(outline_contains_point((55., 0.), polygon))

    def test_downstream_cut_is_rotation_translation_invariant(self):
        angle, offset = .7, (100., -25.)
        def transform(p):
            return (offset[0] + math.cos(angle)*p[0] - math.sin(angle)*p[1],
                    offset[1] + math.sin(angle)*p[0] + math.cos(angle)*p[1], 0.)
        boundary = [transform(p) for p in ((0., 2.), (50., 2.), (75., 2.))]
        result = _exit_boundary(boundary, transform((46., 0.)), angle, lambda: None)
        expected = [transform(p)[:2] for p in ((46., 2.), (50., 2.), (75., 2.))]
        for actual, point in zip(result, expected):
            self.assertAlmostEqual(point[0], actual[0])
            self.assertAlmostEqual(point[1], actual[1])

    def test_returning_boundary_or_no_downstream_road_is_not_guessed(self):
        for boundary in ([[(0., 2., 0.), (50., 2., 0.), (40., 2., 0.), (75., 2., 0.)],
                         [(0., 2., 0.), (45., 2., 0.)]]):
            with self.assertRaises(ValueError):
                _exit_boundary(boundary, (46., 0., 0.), 0., lambda: None)


if __name__ == '__main__':
    unittest.main()
