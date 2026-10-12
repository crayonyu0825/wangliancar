"""Formal producer-to-consumer source gates, local scope and source hysteresis."""
import copy
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.maneuver_facts import ManeuverFacts
from core.behavior_contract import TaskContext, BehaviorFrame
from core.serialization import perception_to_dict
from core.interfaces import Perception
from perception.maneuver_environment import ManeuverEnvironmentBuilder
from members.decision.behaviors.environment import lane_candidates
from members.decision.behaviors.lane_change import LaneChangePolicy
from members.decision.behaviors.observations import MotionObject
from members.decision.behaviors.stop_policies import BlockedRoadPolicy, BlockingObservation
from members.decision.behaviors.coordinator import BehaviorCoordinator
from members.decision.settings import DecisionSettings
from members.planning.maneuver_scene import read_maneuver_scene
from members.planning.candidate_validation import validate_candidate
from tests import test_local_road_regions as local_fixture
from tests.test_maneuver_perception import profile, target
from tests.test_decision_maneuver_policies import lane, caps
from tests.test_decision_behavior_contract import frame
from tests.test_planning_candidate_validation import point, vehicle, limits, budget
from scripts.replay_decision import _assign


class FactConsumerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, 'visibility.json')
        self.now = 100.
        self.clock = lambda: self.now
        self.adapter, self.metadata, self.route, self.p = local_fixture.LocalRoadRegionTests().fixture()
        self.builder = ManeuverEnvironmentBuilder(self.route, NS(vehicle_id='0', sensor_timeout_ms=500,
                                                                sensor_visibility_file=self.path), self.clock)
        self.write_profile(profile(self.p))
        self.build()

    def write_profile(self, model):
        with open(self.path, 'w', encoding='utf-8') as stream:
            json.dump(model, stream)

    def build(self):
        p = self.p
        p.ego.frame_id = p.frame_id
        p.valid_until = self.now + .2
        p.maneuver_environment = self.builder.build(p)
        return p.maneuver_environment

    def bound(self):
        for unused in range(50):
            self.p.frame_id += 1
            self.p.targets_frame_id = self.p.frame_id
            self.now += .001
            self.build()
            if self.p.maneuver_environment.neighbor_lanes[0]['crossing_range_verified']:
                return
        self.fail('Synthetic native boundary binding did not finish')

    def behavior_frame(self):
        p = self.p
        return BehaviorFrame(TaskContext(p.case_id, p.task_id, p.scene_id, 'synthetic-reader'), p.frame_id,
                             self.now, p.valid_until, p.ego.x, p.ego.y, p.ego.heading, p.ego.speed, usable=True)

    def test_actual_local_producer_is_read_without_whole_thousand_metre_lane(self):
        facts = ManeuverFacts(self.p, self.clock)
        road = facts.road('1_0_-2')
        self.assertEqual(75., road['center_line'][-1][0])
        self.assertEqual(1000., self.p.lane.center_line[-1][0])
        self.assertTrue(road['coverage'])
        self.assertEqual([], facts.objects())
        road['polygon'][0] = (500., 500.)
        self.assertNotEqual(road['polygon'], facts.road('1_0_-2')['polygon'])

    def test_version_task_frame_timestamp_and_validity_gate(self):
        original = copy.deepcopy(self.p)
        for field, value in (('contract_version', 'unknown'), ('task_id', 'other'), ('scene_id', 8),
                             ('frame_id', 2), ('timestamp', 999), ('valid', False)):
            self.p = copy.deepcopy(original)
            setattr(self.p.maneuver_environment, field, value)
            with self.assertRaises(ValueError, msg=field):
                ManeuverFacts(self.p, self.clock)
        self.p = copy.deepcopy(original)
        self.p.ego.frame_id = -1
        with self.assertRaises(ValueError): ManeuverFacts(self.p, self.clock)

    def test_reader_cannot_borrow_another_current_environment_or_expired_frame(self):
        facts = ManeuverFacts(self.p, self.clock)
        self.p.maneuver_environment = copy.deepcopy(self.p.maneuver_environment)
        with self.assertRaises(ValueError): facts.objects()
        self.now += .21
        with self.assertRaises(ValueError): ManeuverFacts(self.p, self.clock)

    def test_anonymous_or_noninteger_context_cannot_authorize_new_behavior(self):
        original = copy.deepcopy(self.p)
        for field, value in (('case_id', ''), ('task_id', None), ('scene_id', True), ('timestamp', True)):
            self.p = copy.deepcopy(original)
            setattr(self.p, field, value)
            setattr(self.p.maneuver_environment, field, value)
            with self.assertRaises(ValueError, msg=field): ManeuverFacts(self.p, self.clock)

    def test_map_digest_semantics_and_lane_binding_must_all_match(self):
        original = copy.deepcopy(self.p)
        for field, value in (('verified', False), ('semantic_verified', False), ('geometry_bound', False),
                             ('digest', 'unverified'), ('current_lane_id', '2_0_-1')):
            self.p = copy.deepcopy(original)
            self.p.maneuver_environment.map_semantics[field] = value
            with self.assertRaises(ValueError, msg=field): ManeuverFacts(self.p, self.clock).road('1_0_-2')
        self.p = copy.deepcopy(original)
        self.p.maneuver_environment.road_regions[0]['map_digest'] = 'a' * 32
        with self.assertRaises(ValueError): ManeuverFacts(self.p, self.clock).road(self.p.lane.lane_id)

    def test_future_renewed_or_wrong_clock_evidence_never_borrows_static_geometry(self):
        original = copy.deepcopy(self.p)
        for field, value in (('frame_id', 2), ('frame_id', True), ('observed_at_s', 100.1),
                             ('valid_until_s', 100.3), ('clock_id', 'another_process'), ('source_kind', 'map')):
            self.p = copy.deepcopy(original)
            self.p.maneuver_environment.road_regions[0]['evidence'][field] = value
            with self.assertRaises(ValueError, msg=field): ManeuverFacts(self.p, self.clock).road(self.p.lane.lane_id)

    def test_dynamic_deadline_rechecked_after_map_work_and_not_refreshed(self):
        self.p.source_status['targets']['age_ms'] = 450
        self.build()
        facts = ManeuverFacts(self.p, self.clock)
        self.assertAlmostEqual(100.05, facts.road(self.p.lane.lane_id)['evidence']['valid_until_s'])
        self.now = 100.051
        self.assertTrue(facts.road(self.p.lane.lane_id, False))
        with self.assertRaises(ValueError): facts.road(self.p.lane.lane_id)

    def test_diagnostic_ground_truth_and_bad_source_metadata_cannot_supply_objects(self):
        original = copy.deepcopy(self.p)
        for update in ('source', 'usable', 'quality', 'frame'):
            self.p = copy.deepcopy(original)
            if update == 'source': self.p.target_source = 'ground_truth'
            elif update == 'usable': self.p.source_status['targets']['usable'] = False
            elif update == 'quality': self.p.source_status['targets']['quality'] = 'regressed'
            else: self.p.source_status['targets']['frame_id'] = 2
            with self.assertRaises(ValueError, msg=update): ManeuverFacts(self.p, self.clock).objects()

    def test_object_geometry_identity_and_original_packet_are_matched(self):
        self.p.targets = [target(25., 3.5, heading=None)]
        self.build()
        self.assertIsNone(ManeuverFacts(self.p, self.clock).objects()[0]['heading'])
        original = copy.deepcopy(self.p)
        for field, value in (('x', 26.), ('source_frame_id', 0), ('source', 'ground_truth'),
                             ('heading', 0.), ('length', float('nan'))):
            self.p = copy.deepcopy(original)
            self.p.maneuver_environment.objects[0][field] = value
            with self.assertRaises(ValueError, msg=field): ManeuverFacts(self.p, self.clock).objects()
        self.p = copy.deepcopy(original)
        self.p.maneuver_environment.objects.append(copy.deepcopy(self.p.maneuver_environment.objects[0]))
        with self.assertRaises(ValueError): ManeuverFacts(self.p, self.clock).objects()

    def test_empty_packet_without_model_is_not_coverage(self):
        self.builder.visibility.filename = ''
        self.build()
        self.assertEqual([], ManeuverFacts(self.p, self.clock).objects())
        with self.assertRaises(ValueError): ManeuverFacts(self.p, self.clock).road(self.p.lane.lane_id)

    def test_visibility_frame_completeness_and_original_plane_are_checked(self):
        original = copy.deepcopy(self.p)
        for field, value in (('complete_detections', False), ('source_frame_id', 0), ('pose_frame_id', 0),
                             ('pose_source', 'guessed'), ('verification_reference', ''), ('reference_z_m', .2)):
            self.p = copy.deepcopy(original)
            self.p.maneuver_environment.coverage_regions[0][field] = value
            with self.assertRaises(ValueError, msg=field): ManeuverFacts(self.p, self.clock).road(self.p.lane.lane_id)

    def test_duplicate_roads_and_replaced_outline_are_rejected(self):
        self.p.maneuver_environment.road_regions.append(copy.deepcopy(self.p.maneuver_environment.road_regions[0]))
        with self.assertRaises(ValueError): ManeuverFacts(self.p, self.clock).road(self.p.lane.lane_id)
        self.p.maneuver_environment.road_regions.pop()
        self.p.maneuver_environment.road_regions[0]['polygon'][0] = (500., 500.)
        with self.assertRaises(ValueError): ManeuverFacts(self.p, self.clock).road(self.p.lane.lane_id)

    def test_scope_checks_continue_after_source_expires_during_read(self):
        calls = [0]
        def clock():
            calls[0] += 1
            return 100. if calls[0] < 6 else 100.21
        with self.assertRaises(ValueError): ManeuverFacts(self.p, clock).road(self.p.lane.lane_id)

    def test_bound_crossing_is_an_exact_subsequence_not_a_nearby_line(self):
        self.bound()
        facts = ManeuverFacts(self.p, self.clock)
        crossing = facts.crossing('1_0_-2')
        self.assertTrue(crossing['permitted_fragments'])
        original = copy.deepcopy(self.p)
        for field in ('geometry', 'road_s', 'permission', 'map'):
            self.p = copy.deepcopy(original)
            value = self.p.maneuver_environment.neighbor_lanes[0]
            if field == 'geometry': value['crossing_ranges_world'][0]['shared_boundary_world'][0] = (1., 1.76, 0.)
            elif field == 'road_s': value['crossing_ranges_world'][0]['road_s_samples_m'][0] += .1
            elif field == 'permission': value['crossing_ranges_world'][0]['marking']['crossing_permitted'] = False
            else: value['marking_map_digest'] = 'b' * 32
            with self.assertRaises(ValueError, msg=field): ManeuverFacts(self.p, self.clock).crossing('1_0_-2')

    def test_local_d03_candidate_uses_real_boundaries_and_preserves_permission_fragments(self):
        self.bound()
        candidates, failures = lane_candidates(self.p, self.behavior_frame(), self.clock)
        self.assertEqual([], failures)
        self.assertEqual(1, len(candidates))
        candidate = candidates[0]
        self.assertEqual(75., candidate.center_line[-1][0])
        self.assertTrue(LaneChangePolicy(3.5, .9, .9).opportunity(self.behavior_frame(), candidate)[0])
        self.assertTrue(candidate.crossing_ranges)
        self.assertFalse(self.p.maneuver_environment.neighbor_lanes[0]['dynamic_coverage_verified'])

    def test_candidate_partial_or_solid_crossing_cannot_be_extended_by_flag(self):
        f = frame(speed=5.)
        candidate = lane(f)
        candidate.crossing_boundary = [(-150., 1.75), (200., 1.75)]
        candidate.crossing_ranges = [[(-150., 1.75), (5., 1.75)], [(20., 1.75), (200., 1.75)]]
        result = LaneChangePolicy(3.5, .9, .9).opportunity(f, candidate)
        self.assertEqual((False, 'CROSSING_MANEUVER_WINDOW_UNVERIFIED'), result)

    def test_no_binding_or_unknown_coverage_does_not_make_lane_candidate(self):
        candidates, failures = lane_candidates(self.p, self.behavior_frame(), self.clock)
        self.assertEqual([], candidates)
        self.assertTrue(failures)
        self.bound()
        self.p.maneuver_environment.coverage_regions = []
        self.assertEqual([], lane_candidates(self.p, self.behavior_frame(), self.clock)[0])

    def test_rotated_and_unknown_heading_objects_cannot_hide_outside_width_summary(self):
        f = frame(speed=5.)
        policy = LaneChangePolicy(3.5, .9, .9)
        for heading in (math.pi / 2, None):
            candidate = lane(f, [MotionObject(1, 15., 8.5, 0., 0., 10., .2, heading)])
            self.assertEqual('TARGET_LANE_FRONT_GAP', policy.opportunity(f, candidate)[1])
        unknown = MotionObject(1, 0., 0., 0., 0., 4., 2., None)
        radius = math.sqrt(5.)
        self.assertEqual([radius, radius, -radius, -radius], [p[0] for p in unknown.footprint()])
        self.assertIsNone(unknown.heading)

    def test_current_clearance_requires_old_location_in_view_and_same_map_plane(self):
        facts = ManeuverFacts(self.p, self.clock)
        self.assertTrue(facts.road_clear_at(self.p.lane.lane_id, (30., 0., 0.)))
        self.assertFalse(facts.road_clear_at(self.p.lane.lane_id, (90., 0., 0.)))
        self.assertFalse(facts.road_clear_at(self.p.lane.lane_id, (30., 0., .2)))
        self.p.targets = [target(30., 0.)]
        self.build()
        self.assertFalse(ManeuverFacts(self.p, self.clock).road_clear_at(self.p.lane.lane_id, (30., 0., 0.)))

    def test_historical_sensor_pose_and_deadline_survive_the_consumers(self):
        p = self.p
        for kind in ('gps', 'targets'):
            p.source_status[kind].update(frame_id=1, timestamp=50, clock='sdk_frame')
        p.targets_timestamp = 50
        p.ego.timestamp = 50
        self.build()
        self.now += .05
        p.frame_id, p.timestamp = 2, 100
        p.ego.timestamp = 100
        p.source_status['gps'].update(frame_id=2, timestamp=100)
        p.ego.x = 60.
        p.lane = self.route.update(p.ego, True)
        self.build()
        facts = ManeuverFacts(p, self.clock)
        self.assertEqual(1, facts.coverage([(10., 0., 0.)])[0]['pose_frame_id'])
        self.assertEqual(100., facts.coverage([(10., 0., 0.)])[0]['observed_at_s'])
        self.assertEqual(100.25, facts.coverage([(10., 0., 0.)])[0]['valid_until_s'])

    def test_planning_scene_keeps_partial_visibility_as_a_separate_constraint(self):
        model = profile(self.p)
        model['profiles'][0]['body_region_m'] = [[-20., -20.], [20., -20.], [20., 20.], [-20., 20.]]
        self.write_profile(model)
        self.build()
        self.assertFalse(self.p.maneuver_environment.road_regions[0]['coverage_verified'])
        scene = read_maneuver_scene(self.p, [self.p.lane.lane_id], self.clock)
        self.assertEqual('geometry_and_current_visibility_only', scene['authority'])
        self.assertEqual(75., scene['roads'][0]['center_line'][-1][0])
        with patch('members.planning.candidate_validation.time.monotonic', self.clock):
            samples = [point(10.), point(12., at=2.)]
            result = validate_candidate(samples, 1, vehicle(), limits(), scene['coverage'], [], budget())
            self.assertEqual('safe', result['status'], result)
            samples = [point(28.), point(32., at=4.)]
            result = validate_candidate(samples, 1, vehicle(), limits(), scene['coverage'], [], budget())
            self.assertEqual('CORRIDOR_COLLISION', result['reason_code'], result)
        self.assertEqual(100.2, scene['source_valid_until_s'])

    def test_planning_neighbor_requires_current_crossing_identity(self):
        with self.assertRaises(ValueError): read_maneuver_scene(self.p, ['1_0_-1', '1_0_-2'], self.clock)
        self.bound()
        scene = read_maneuver_scene(self.p, ['1_0_-1', '1_0_-2'], self.clock)
        self.assertTrue(scene['crossings'])
        self.assertEqual(2, len(scene['roads']))
        with self.assertRaises(ValueError): read_maneuver_scene(self.p, ['1_0_-1', 'unknown'], self.clock)
        with self.assertRaises(ValueError): read_maneuver_scene(self.p, ['1_0_-1', '1_0_-1'], self.clock)

    def test_public_replay_reader_preserves_original_deadlines_and_local_shapes(self):
        saved = json.loads(json.dumps(perception_to_dict(self.p), allow_nan=False))
        rebuilt = _assign(Perception(), saved, 'perception')
        road = ManeuverFacts(rebuilt, self.clock).road(self.p.lane.lane_id)
        self.assertEqual(100.2, road['evidence']['valid_until_s'])
        self.assertEqual(75., road['center_line'][-1][0])

    def test_production_coordinator_supplies_source_qualified_blockage_release(self):
        from core.interfaces import DecisionTarget
        settings = DecisionSettings(front_offset_m=3.5, half_width_m=.9)
        coordinator = BehaviorCoordinator(settings, self.clock)
        self.p.scene_id = 11
        self.p.targets = [target(30., 0.)]
        self.build()
        output = DecisionTarget().bind(self.p)
        output.valid, output.target_speed = True, 1.
        coordinator.observe_legacy(self.p, output, NS(blockage_candidates=[(9, 15.)]), self.now)
        self.assertIsNotNone(coordinator.blocked.session.intent_id)
        self.p.targets = []
        for unused in range(2):
            self.now += .01
            self.p.frame_id += 1
            self.p.targets_frame_id = self.p.frame_id
            self.build()
            coordinator.observe_legacy(self.p, output, NS(blockage_candidates=[]), self.now)
        self.assertEqual('COMPLETED', coordinator.blocked.session.status)
        self.assertEqual('BLOCKAGE_CLEARED', coordinator.blocked.session.reason_code)
        self.assertIsNone(output.behavior_request)

    def test_coordinator_map_change_cannot_clear_an_anchor_from_another_map(self):
        from core.interfaces import DecisionTarget
        coordinator = BehaviorCoordinator(DecisionSettings(front_offset_m=3.5, half_width_m=.9), self.clock)
        self.p.scene_id = 11
        self.p.targets = [target(30., 0.)]
        self.build()
        output = DecisionTarget().bind(self.p)
        coordinator.observe_legacy(self.p, output, NS(blockage_candidates=[(9, 15.)]), self.now)
        self.p.targets = []
        for unused in range(3):
            self.now += .01
            self.p.frame_id += 1
            self.p.targets_frame_id = self.p.frame_id
            self.build()
            self.p.maneuver_environment.map_semantics['digest'] = 'b' * 32
            for road in self.p.maneuver_environment.road_regions: road['map_digest'] = 'b' * 32
            coordinator.observe_legacy(self.p, output, NS(blockage_candidates=[]), self.now)
        self.assertNotEqual('COMPLETED', coordinator.blocked.session.status)
        self.assertEqual('BLOCKAGE_EVIDENCE_UNKNOWN', coordinator.snapshot()['policies'][1]['reason_code'])


class BlockageSourceTests(unittest.TestCase):
    def start(self):
        self.policy = BlockedRoadPolicy()
        f = frame()
        self.policy.evaluate(f, BlockingObservation('car', 15., True), caps(f))

    def clear(self, index, when, source_index, observed, source='sensor:formal', **kwargs):
        f = frame(index, when)
        return self.policy.evaluate(f, BlockingObservation('prior', None, False, verified_clear=True,
            source=source, source_frame_id=source_index, source_observed_at_s=observed), caps(f), **kwargs)

    def test_repeated_sensor_packet_cannot_count_as_two_clear_observations(self):
        self.start()
        self.clear(2, .1, 2, .1)
        self.assertNotEqual('COMPLETE', self.clear(3, .2, 2, .1).phase)
        self.assertEqual('COMPLETE', self.clear(4, .3, 4, .3).phase)

    def test_unknown_source_between_clear_frames_breaks_confirmation(self):
        self.start()
        self.clear(2, .1, 2, .1)
        f = frame(3, .2)
        self.policy.evaluate(f, BlockingObservation('prior', None, False, usable=False), caps(f))
        self.assertNotEqual('COMPLETE', self.clear(4, .3, 4, .3).phase)
        self.assertEqual('COMPLETE', self.clear(5, .4, 5, .4).phase)

    def test_source_switch_regression_and_future_do_not_stack_confirmation(self):
        for arguments in ((3, .2, 3, .2, 'sensor:other'), (3, .2, 1, .05, 'sensor:formal'),
                          (3, .2, 4, .3, 'sensor:formal')):
            self.start()
            self.clear(2, .1, 2, .1)
            self.assertNotEqual('COMPLETE', self.clear(*arguments).phase)

    def test_safety_override_and_paused_frame_break_clear_confirmation(self):
        self.start()
        self.clear(2, .1, 2, .1)
        self.clear(3, .2, 3, .2, safety_override=True)
        self.assertNotEqual('COMPLETE', self.clear(4, .3, 4, .3).phase)
        self.assertEqual('COMPLETE', self.clear(5, .4, 5, .4).phase)

    def test_paused_and_unusable_gps_break_clear_confirmation(self):
        for flags in ({'paused': True}, {'usable': False}):
            self.start()
            self.clear(2, .1, 2, .1)
            f = frame(3, .2, **flags)
            self.policy.evaluate(f, BlockingObservation('prior', None, False, verified_clear=True,
                source='sensor:formal', source_frame_id=3, source_observed_at_s=.2), caps(f))
            self.assertNotEqual('COMPLETE', self.clear(4, .3, 4, .3).phase)
            self.assertEqual('COMPLETE', self.clear(5, .4, 5, .4).phase)


if __name__ == '__main__': unittest.main()
