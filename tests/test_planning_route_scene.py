"""Real selected SDK/ODR route producer, independent physical source checks.

The ordinary seam model supplies a 50 ms route-work budget as a positive-test
prerequisite. It does not establish the production 5 ms default or 20 Hz rate.
"""
import copy
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.interfaces import Perception
from core.region_geometry import rectangle, outline_contains_point
from core.serialization import perception_to_dict
from members.planning.candidate_validation import validate_candidate
from members.planning.maneuver_scene import ManeuverGeometryCache
from members.planning.route_scene import read_route_scene
from perception.maneuver_environment import ManeuverEnvironmentBuilder, _evidence
from perception.route_environment import publish_route
from perception.route_manager import RouteManager
from scripts.replay_decision import _assign
from tests import test_local_road_regions as local_fixture
from tests import test_driving_integration as driving_fixture
from tests.test_maneuver_map import map_adapter, junction_document
from tests.test_maneuver_perception import profile, perception, target
from tests.test_planning_candidate_validation import point, vehicle, limits, budget


class PlanningRouteSceneTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.clock = lambda: self.now
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = os.path.join(temporary.name, 'visibility.json')
        self.adapter,self.metadata = map_adapter(junction_document())
        self.api = self.adapter.hdmap
        self.api.segments = {'1_0_-1':[(0.,0.),(20.,0.)],
                             '2_0_-1':[(20.,0.),(40.,0.)], '3_0_-1':[(40.,0.),(60.,0.)]}
        self.api.successors = {'1_0_-1':['2_0_-1'],'2_0_-1':['3_0_-1']}
        self.api.predecessors = {'2_0_-1':['1_0_-1'],'3_0_-1':['2_0_-1']}
        self.api.current,self.api.neighbor = '1_0_-1',''
        self.api.getRoadST = lambda identity,p:NS(exists=True,s=p[0]-{'1':0.,'2':20.,'3':40.}[identity.GetString()])
        self.route = RouteManager(self.adapter,NS(warning=lambda *args:None))
        self.p = perception()
        with open(path,'w',encoding='utf-8') as stream: json.dump(profile(self.p),stream)
        self.builder = ManeuverEnvironmentBuilder(self.route,NS(vehicle_id='0',sensor_timeout_ms=500,
                                                               sensor_visibility_file=path),self.clock)
        self.build()

    def build(self):
        self.p.ego.frame_id = self.p.frame_id
        self.p.valid_until = self.now+.2
        self.p.lane = self.route.update(self.p.ego,True)
        self.p.maneuver_environment = self.builder.build(self.p)
        return self.p.maneuver_environment

    def published(self):
        return self.p.maneuver_environment.map_semantics['route_geometry']

    def scene(self, cache=None):
        return read_route_scene(self.p,self.clock,cache)

    def test_real_selected_route_boundaries_extend_beyond_first_lane(self):
        self.assertTrue(self.published()['geometry_valid'],self.published())
        self.assertEqual(2,len(self.published()['verified_joins']))
        self.assertEqual(self.p.maneuver_environment.map_semantics,
                         perception_to_dict(self.p)['maneuver_environment']['map_semantics'])
        for join in self.published()['verified_joins']:
            self.assertIsInstance(join['source_point'],list)
            self.assertIsInstance(join['target_point'],list)
        scene = self.scene()
        self.assertEqual(['1_0_-1','2_0_-1','3_0_-1'],scene['route_lane_ids'])
        self.assertEqual(['current','forward','forward'],[v['role'] for v in scene['roads']])
        self.assertEqual((60.,0.,0.),scene['reference'][-1])
        self.assertTrue(any(outline_contains_point((30.,0.),v) for v in scene['corridor'].polygons))
        self.assertTrue(any(outline_contains_point((50.,0.),v) for v in scene['corridor'].polygons))
        self.assertEqual(self.p.maneuver_environment.dynamic_valid_until,scene['source_valid_until_s'])

    def test_nominal_full_body_can_cross_two_actual_sources_at_join(self):
        scene = self.scene()
        body = rectangle(20.1,0.,0.,3.9,.9,.9)
        self.assertGreater(scene['corridor'].body_clearance(body,lambda:None,100000),0.)
        path = [point(17.,speed=1.,heading=0.,at=0.),point(21.,speed=1.,heading=0.,at=4.)]
        report = validate_candidate(path,1,vehicle(3.9,.9,.9,2.8),limits(),scene['corridor'],[],budget())
        self.assertEqual('safe',report['status'],report)

    def test_sdk_lane_switch_retains_only_previous_selected_tail(self):
        original_deadline = self.scene()['source_valid_until_s']
        self.p.ego.x,self.api.current = 20.1,'2_0_-1'
        self.p.frame_id += 1
        self.p.targets_frame_id = self.p.frame_id
        self.now += .05
        self.build()
        self.assertTrue(self.published()['geometry_valid'],self.published())
        scene = self.scene()
        self.assertEqual(['1_0_-1','2_0_-1','3_0_-1'],scene['route_lane_ids'])
        self.assertEqual('predecessor',scene['roads'][0]['role'])
        self.assertGreater(scene['source_valid_until_s'],original_deadline)
        # This is current Sensor evidence, not a reused old source deadline.
        self.assertEqual(self.p.frame_id,scene['roads'][0]['geometry_evidence']['frame_id'])
        body = rectangle(20.1,0.,0.,3.9,.9,.9)
        self.assertGreater(scene['corridor'].body_clearance(body,lambda:None,100000),0.)

    def test_first_frame_on_new_road_does_not_invent_an_incoming_lane(self):
        self.route._route_geometry_state.reset(None)
        self.p.ego.x,self.api.current = 20.1,'2_0_-1'
        self.build()
        scene = self.scene()
        self.assertEqual(['2_0_-1','3_0_-1'],scene['route_lane_ids'])
        body = rectangle(20.1,0.,0.,3.9,.9,.9)
        self.assertLessEqual(scene['corridor'].body_clearance(body,lambda:None,100000),0.)

    def test_predecessor_is_freshly_requeried_not_only_cached(self):
        original = self.api.getLaneSample
        queried = []
        def observed(identity):
            queried.append(identity.GetString())
            return original(identity)
        self.api.getLaneSample = observed
        self.p.ego.x,self.api.current = 20.1,'2_0_-1'
        self.build()
        self.assertIn('1_0_-1',queried)
        self.api.segments['1_0_-1'] = [(0.,0.),(19.,0.)]
        self.build()
        self.assertFalse(self.published()['geometry_valid'])
        self.assertIn('REFERENCE_MISMATCH',self.published()['reason'])

    def test_scene_change_invalidates_incoming_history(self):
        self.p.ego.x,self.api.current = 20.1,'2_0_-1'
        self.p.scene_id += 1
        self.build()
        self.assertNotIn('1_0_-1',self.scene()['route_lane_ids'])

    def test_changed_route_hint_invalidates_incoming_history(self):
        self.route._route_hint = ((20.,0.),(40.,0.))
        self.p.ego.x,self.api.current = 20.1,'2_0_-1'
        self.build()
        self.assertNotIn('1_0_-1',self.scene()['route_lane_ids'])

    def test_current_body_outside_first_source_is_not_repaired(self):
        self.p.ego.x = .1
        self.build()
        body = rectangle(.1,0.,0.,3.9,.9,.9)
        self.assertLessEqual(self.scene()['corridor'].body_clearance(body,lambda:None,100000),0.)

    def test_parallel_neighbor_is_never_published_as_a_forward_road(self):
        self.api.neighbor = '1_0_-2'
        self.api.segments['1_0_-2'] = [(0.,3.5),(20.,3.5)]
        self.build()
        self.assertEqual(['1_0_-1','2_0_-1','3_0_-1'],self.scene()['route_lane_ids'])
        self.assertFalse(any(outline_contains_point((10.,3.5),v) for v in self.scene()['corridor'].polygons))

    def test_wrong_declared_travel_native_type_or_height_stays_protected(self):
        original_type,original_st = self.api.getLaneType,self.api.getRoadST
        for mutation in ('type','direction','height'):
            self.api.getLaneType,self.api.getRoadST = original_type,original_st
            self.api.height = 0.
            if mutation == 'type': self.api.getLaneType=lambda *args:NS(exists=True,laneType='shoulder')
            elif mutation == 'direction': self.api.getRoadST=lambda identity,p:NS(exists=True,s=20.-original_st(identity,p).s)
            else:
                original_sample = self.api.getLaneSample
                def sample(identity):
                    value=original_sample(identity)
                    if identity.GetString()=='2_0_-1':
                        for name in ('centerLine','leftBoundary','rightBoundary'):
                            for point in getattr(value.laneInfo,name): point.z=.5
                    return value
                self.api.getLaneSample=sample
            self.build()
            if mutation in ('type','direction'):
                self.assertFalse(self.published()['geometry_valid'],mutation)
                with self.assertRaises(ValueError,msg=mutation): self.scene()
            else:
                # RouteManager rejects the height mismatch before publishing a
                # successor; the genuine finite current prefix remains usable.
                scene = self.scene()
                self.assertEqual(['1_0_-1'],scene['route_lane_ids'])
                self.assertEqual((20.,0.,0.),scene['reference'][-1])
                self.assertLessEqual(scene['corridor'].body_clearance(
                    rectangle(20.1,0.,0.,3.9,.9,.9),lambda:None,100000),0.)

    def test_current_map_identity_failure_cannot_publish_route(self):
        self.metadata['verified'] = False
        self.build()
        self.assertFalse(self.published()['geometry_valid'])
        with self.assertRaises(ValueError): self.scene()

    def test_actual_source_gap_remains_a_gap_not_a_connecting_hull(self):
        original = self.api.getLaneSample
        def sample(identity):
            value = original(identity)
            if identity.GetString()=='2_0_-1':
                # Current center samples still meet. Both actual target borders
                # begin farther forward, leaving a real unsupported sliver.
                for name in ('leftBoundary','rightBoundary'):
                    getattr(value.laneInfo,name)[0].x=20.1
            return value
        self.api.getLaneSample = sample
        self.build()
        self.assertFalse(self.published()['geometry_valid'])
        self.assertEqual('LOCAL_CUT_OUTSIDE_BOUNDARY',self.published()['reason'])
        scene = self.scene()
        self.assertFalse(any(outline_contains_point((20.05,0.),v) for v in scene['corridor'].polygons))
        body = rectangle(20.,0.,0.,1.,1.,.5)
        self.assertLessEqual(scene['corridor'].body_clearance(body,lambda:None,100000),0.)

    def test_bad_spans_never_repair_a_selected_route(self):
        self.p.lane.forward_lane_spans[1]['start_index'] += 1
        env = self.p.maneuver_environment
        publish_route(self.p,env,self.route,env.coverage_regions,self.builder._route_test_query if hasattr(self.builder,'_route_test_query') else None,
                      _evidence,clock=self.clock)
        self.assertFalse(self.published()['geometry_valid'])
        self.assertEqual(['1_0_-1'],self.scene()['route_lane_ids'])

    def test_query_work_and_clock_budgets_reject_without_renewing_facts(self):
        env = self.p.maneuver_environment
        for setting in ({'max_queries':1},{'max_checks':1},{'max_seconds':1e-12}):
            publish_route(self.p,env,self.route,env.coverage_regions,None,_evidence,clock=self.clock,**setting)
            self.assertFalse(self.published()['geometry_valid'],setting)
            self.assertIn('BUDGET_EXHAUSTED',self.published()['reason'])
        self.now += .21
        publish_route(self.p,env,self.route,env.coverage_regions,None,_evidence,clock=self.clock)
        self.assertFalse(self.published()['geometry_valid'])
        self.assertEqual('ROUTE_SOURCE_EXPIRED',self.published()['reason'])

    def test_current_sensor_objects_and_original_deadline_are_preserved(self):
        self.p.targets = [target(25.,0.,heading=None)]
        self.p.source_status['targets']['age_ms'] = 450
        self.build()
        scene = self.scene()
        self.assertEqual([9],[v['id'] for v in scene['objects']])
        self.assertIsNone(scene['objects'][0]['heading'])
        self.assertAlmostEqual(100.05,scene['source_valid_until_s'])
        self.now = 100.051
        with self.assertRaises(ValueError): self.scene()

    def test_ground_truth_and_unknown_motion_cannot_supply_route_clearance(self):
        self.p.target_source = 'ground_truth'
        self.build()
        with self.assertRaises(ValueError): self.scene()
        self.p.target_source = 'sensor:perfect'
        self.p.targets = [target(25.,0.)]
        self.p.targets[0].vx = None
        self.build()
        with self.assertRaises(ValueError): self.scene()

    def test_no_cache_or_json_replay_refreshes_deadlines(self):
        cache = ManeuverGeometryCache(4,100000)
        first = self.scene(cache)
        self.assertIs(first['corridor'],self.scene(cache)['corridor'])
        restored = Perception()
        _assign(restored,json.loads(json.dumps(perception_to_dict(self.p),allow_nan=False)),'perception')
        self.assertEqual(first['reference'],read_route_scene(restored,self.clock,cache)['reference'])
        self.now += .21
        with self.assertRaises(ValueError): read_route_scene(restored,self.clock,cache)

    def test_returned_road_edits_do_not_change_formal_source(self):
        before = perception_to_dict(self.p)
        scene = self.scene()
        scene['roads'][0]['center_line'][0] = (-100.,-100.,0.)
        scene['roads'][0]['polygon'][0] = (-100.,-100.)
        scene['objects'].append({'id':100})
        self.assertEqual(before,perception_to_dict(self.p))

    def test_bad_route_reference_or_evidence_cannot_extend_actual_source(self):
        original = copy.deepcopy(self.p)
        for mutation in ('reference','deadline','frame','map','join'):
            self.p = copy.deepcopy(original)
            r = self.published()
            if mutation == 'reference': r['reference'][-1] = (1000.,0.,0.)
            elif mutation == 'deadline': r['dynamic_evidence']['valid_until_s'] += 1.
            elif mutation == 'frame': r['roads'][0]['geometry_evidence']['frame_id'] += 1
            elif mutation == 'map': r['map_digest'] = 'a'*32
            else: r['verified_joins'][0]['odr_link_verified'] = False
            with self.assertRaises(ValueError,msg=mutation): self.scene()

    def test_long_current_road_is_local_and_not_truncated_to_first_short_lane(self):
        adapter,unused,route,p = local_fixture.LocalRoadRegionTests().fixture()
        self.builder.route_manager,self.route,self.p = route,route,p
        self.build()
        scene = self.scene()
        self.assertEqual(1000.,p.lane.center_line[-1][0])
        self.assertEqual((75.,0.,0.),scene['reference'][-1])
        self.assertEqual(['1_0_-1'],scene['route_lane_ids'])
        self.assertTrue(self.published()['geometry_valid'],self.published())

    def test_ordinary_fixed_entries_physically_cross_current_successor_seam(self):
        # Same actual formal builder and real fixed decide/plan/control entries;
        # the authored bicycle plant replaces only vehicle response and SDK I/O.
        # This positive model test supplies an explicit work budget. The 5 ms
        # production default is not a Windows scheduling or realtime guarantee;
        # separate negative tests still check the original bounded API directly.
        def model_route_budget(*args,**kwargs):
            kwargs['max_seconds'] = .05
            return publish_route(*args,**kwargs)
        route_patch = patch('perception.route_environment.publish_route',model_route_budget)
        route_patch.start()
        self.addCleanup(route_patch.stop)
        h = driving_fixture.DrivingIntegrationTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        h.passage_id = None
        crossed = False
        for unused in range(240):
            decision,trajectory,control,safety = h.cycle()
            self.assertIsNone(decision.behavior_request)
            self.assertTrue(trajectory.planning_metrics['ordinary_dynamic_speed'])
            self.assertTrue(trajectory.planning_metrics['execution_robust'])
            # Independent explicit world rectangle; source center membership
            # alone would miss the rear body spanning the actual lane seam.
            for longitudinal in (-.9,3.9):
                for lateral in (-.9,.9):
                    x = h.plant.x+longitudinal*math.cos(h.plant.heading)-lateral*math.sin(h.plant.heading)
                    y = h.plant.y+longitudinal*math.sin(h.plant.heading)+lateral*math.cos(h.plant.heading)
                    self.assertTrue(0. < x < 60.)
                    self.assertTrue(-1.75 < y < 1.75)
            if h.plant.x > 22.:
                crossed = True
                self.assertEqual('2_0_-1',h.p.lane.lane_id)
                self.assertIn('1_0_-1',read_route_scene(h.p,h.builder.maneuver_builder.clock)['route_lane_ids'])
                break
            h.advance(control)
        self.assertTrue(crossed,(h.plant.x,h.plant.speed,trajectory.errors))


if __name__ == '__main__':
    unittest.main()
