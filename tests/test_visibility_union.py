"""Original-packet union geometry, formal producers and downstream gates."""
import copy
import json
import math
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.visibility_query import VisibilityQuery
from core.maneuver_facts import ManeuverFacts
from core.serialization import perception_to_dict
from core.interfaces import Perception
from perception.maneuver_environment import ManeuverEnvironmentBuilder
from scripts.replay_decision import _assign
from tests.test_maneuver_perception import perception,profile,target
from tests import test_maneuver_fact_consumers as fixture
from tests.test_planning_candidate_validation import rectangle


def record(polygon,**changes):
    value=dict(polygon=polygon,sensor_id='perfect',source='sensor:perfect',source_kind='sensor',
        complete_detections=True,coverage_verified=True,clock_id='process_monotonic',
        frame_id=1,source_frame_id=1,pose_frame_id=1,pose_source='current_frame_pose',
        observed_at_s=100.,valid_until_s=100.2,reference_z_m=0.,verification_reference='explicit synthetic test model')
    value.update(changes); return value


class VisibilityUnionTests(unittest.TestCase):
    def query(self,**changes):
        value=dict(source='sensor:perfect',source_frame_id=1,observed_at_s=100.,
                   valid_until_s=100.2,clock=lambda:100.,max_checks=100000)
        value.update(changes); return VisibilityQuery(**value)

    def test_adjacent_regions_cover_full_body_across_exact_internal_seam(self):
        q=self.query(); parts=[record(rectangle(-2.,-2.,2.,3.)),record(rectangle(2.,-2.,6.,3.))]
        self.assertTrue(q.contains(rectangle(0.,0.,4.,2.),parts),q.reason)
        self.assertEqual('COVERAGE_COMPLETE_PACKET_UNION',q.reason)

    def test_hole_and_tiny_positive_gap_are_not_filled_between_visible_corners(self):
        holes=[rectangle(-1.,-1.,5.,.5),rectangle(-1.,1.5,5.,3.),
               rectangle(-1.,.5,1.,1.5),rectangle(2.,.5,5.,1.5)]
        gaps=[rectangle(-1.,-1.,2.,3.),rectangle(2.+1e-10,-1.,5.,3.)]
        for polygons in (holes,gaps):
            q=self.query()
            self.assertFalse(q.contains(rectangle(0.,0.,4.,2.),[record(p) for p in polygons]),q.reason)

    def test_single_convex_profile_preserves_exact_closed_boundary(self):
        q=self.query(); ring=rectangle(0.,0.,4.,2.)
        self.assertTrue(q.contains(ring,[record(ring)]),q.reason)
        self.assertFalse(q.contains(rectangle(-1e-10,0.,4.,2.),[record(ring)]))

    def test_tiny_concavity_cannot_borrow_legacy_convex_tolerance(self):
        # Legacy float turn tolerance accepts this notch although its exact
        # interior excludes part of the queried rectangle's top edge.
        notch=[(0.,0.),(4.,0.),(4.,2.),(2.,2.-1e-10),(0.,2.)]
        from core.region_geometry import convex_polygon
        self.assertEqual(5,len(convex_polygon(notch)))
        q=self.query(); self.assertFalse(q.contains(rectangle(0.,0.,4.,2.),[record(notch)]))
        self.assertIn('NOT_EXACT_CONVEX',q.reason)
        parts=[record(rectangle(-2.,-2.,2.,3.)),record(rectangle(2.,-2.,6.,3.))]
        q=self.query(); self.assertFalse(q.contains(notch,parts))
        self.assertIn('NOT_EXACT_CONVEX',q.reason)

    def test_source_frame_pose_height_and_time_are_not_fused(self):
        first=record(rectangle(-2.,-2.,2.,3.)); second=record(rectangle(2.,-2.,6.,3.))
        for changes in ({'source':'sensor:other'},{'sensor_id':'other'},{'source_frame_id':2},
                        {'pose_frame_id':2},{'pose_source':'sensor_pose_history_exact_frame'},
                        {'observed_at_s':99.9},{'reference_z_m':.1},{'clock_id':'other'},
                        {'complete_detections':False},{'verification_reference':''}):
            q=self.query(); wrong=dict(second,**changes)
            self.assertFalse(q.contains(rectangle(0.,0.,4.,2.),[first,wrong]),changes)
            self.assertIn('MISMATCH',q.reason)

    def test_count_and_original_source_deadlines_bound_cached_queries(self):
        parts=[record(rectangle(-2.,-2.,2.,3.)),record(rectangle(2.,-2.,6.,3.))]
        body=rectangle(0.,0.,4.,2.); q=self.query()
        self.assertTrue(q.contains(body,parts),q.reason)
        old=q.checks; q.max_checks=old+1
        self.assertFalse(q.contains(body,parts)); self.assertLessEqual(q.checks,q.max_checks+1)
        now=[100.]; q=self.query(clock=lambda:now[0])
        self.assertTrue(q.contains(body,parts),q.reason); now[0]=100.2
        self.assertFalse(q.contains(body,parts)); self.assertIn('DEADLINE',q.reason)

    def test_shorter_region_deadline_is_not_extended_by_another_region(self):
        parts=[record(rectangle(-2.,-2.,2.,3.)),record(rectangle(2.,-2.,6.,3.),valid_until_s=100.01)]
        q=self.query(clock=lambda:100.02)
        self.assertFalse(q.contains(rectangle(0.,0.,4.,2.),parts)); self.assertIn('DEADLINE',q.reason)

    def test_static_cache_cannot_hide_changed_polygon_or_invalid_packet(self):
        parts=[record(rectangle(-2.,-2.,2.,3.)),record(rectangle(2.,-2.,6.,3.))]
        q=self.query(); body=rectangle(0.,0.,4.,2.)
        self.assertTrue(q.contains(body,parts),q.reason)
        parts[1]['polygon']=rectangle(2.1,-2.,6.,3.)
        self.assertFalse(q.contains(body,parts))
        parts[1]['polygon']=rectangle(2.,-2.,6.,3.); parts[1]['coverage_verified']=False
        self.assertFalse(q.contains(body,parts)); self.assertIn('MISMATCH',q.reason)

    def test_concave_lane_uses_original_strip_without_covering_its_missing_corner(self):
        left=[(0.,0.),(0.,4.),(4.,4.)]; right=[(2.,0.),(2.,2.),(4.,2.)]
        outline=left+list(reversed(right))
        parts=[record(rectangle(-1.,-1.,3.,5.)),record(rectangle(-1.,1.,5.,5.))]
        q=self.query(); self.assertTrue(q.contains(outline,parts,left,right),q.reason)
        parts=[record(rectangle(-1.,-1.,3.,3.)),record(rectangle(1.,1.,5.,5.))]
        q=self.query(); self.assertFalse(q.contains(outline,parts,left,right),q.reason)

    def test_changed_lane_outline_and_nonconvex_unbound_query_fail_closed(self):
        left=[(0.,0.),(0.,4.),(4.,4.)]; right=[(2.,0.),(2.,2.),(4.,2.)]
        parts=[record(rectangle(-1.,-1.,3.,5.)),record(rectangle(-1.,1.,5.,5.))]
        q=self.query(); outline=left+list(reversed(right)); wrong=list(outline); wrong[0]=(.1,0.)
        self.assertFalse(q.contains(wrong,parts,left,right)); self.assertIn('MISMATCH',q.reason)
        q=self.query(); self.assertFalse(q.contains(outline,parts))


class FormalVisibilityUnionTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.FactConsumerTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)

    def model(self,gap=0.):
        value=profile(self.f.p); first=value['profiles'][0]
        first['body_region_m']=[[-100.,-20.],[15.,-20.],[15.,20.],[-100.,20.]]
        second=copy.deepcopy(first)
        second['body_region_m']=[[15.+gap,-20.],[150.,-20.],[150.,20.],[15.+gap,20.]]
        value['profiles'].append(second); return value

    def build(self,model):
        self.f.write_profile(model); self.f.build(); return self.f.p.maneuver_environment

    def test_formal_roads_and_lane_candidate_consume_original_packet_union(self):
        from members.decision.behaviors.environment import lane_candidates
        env=self.build(self.model()); self.f.bound()
        self.assertTrue(all(r['coverage_verified'] for r in env.road_regions))
        facts=ManeuverFacts(self.f.p,self.f.clock)
        self.assertEqual(2,len(facts.road('1_0_-2')['coverage']))
        candidates,failures=lane_candidates(self.f.p,self.f.behavior_frame(),self.f.clock)
        self.assertEqual(1,len(candidates),failures)

    def test_hole_cannot_be_bypassed_by_forged_road_coverage_flag(self):
        env=self.build(self.model(gap=1e-10))
        for r in env.road_regions:
            r['coverage_verified']=True; r['evidence']['coverage_verified']=True
        with self.assertRaises(ValueError): ManeuverFacts(self.f.p,self.f.clock).road('1_0_-2')

    def test_reader_budget_and_view_withdrawal_do_not_borrow_prepared_union(self):
        self.build(self.model()); facts=ManeuverFacts(self.f.p,self.f.clock)
        self.assertTrue(facts.road('1_0_-2')['coverage'])
        self.f.p.maneuver_environment.coverage_regions.pop()
        with self.assertRaises(ValueError): facts.road('1_0_-2')
        self.build(self.model())
        with self.assertRaises(ValueError): ManeuverFacts(self.f.p,self.f.clock,visibility_max_checks=1).road('1_0_-2')

    def test_json_replay_retains_both_regions_and_original_deadlines(self):
        self.build(self.model()); replay=Perception()
        _assign(replay,json.loads(json.dumps(perception_to_dict(self.f.p))),'perception')
        facts=ManeuverFacts(replay,self.f.clock); road=facts.road('1_0_-2')
        self.assertEqual(2,len(road['coverage']))
        self.assertTrue(all(v['observed_at_s']==self.f.p.maneuver_environment.dynamic_observed_at_s
                            and v['valid_until_s']==self.f.p.maneuver_environment.dynamic_valid_until
                            for v in road['coverage']))

    def test_producer_budget_exhaustion_keeps_unproved_regions_unknown(self):
        p=perception(); model=profile(p)
        path=os.path.join(self.f.directory.name,'limited_visibility.json')
        with open(path,'w',encoding='utf8') as stream: json.dump(model,stream)
        config=NS(vehicle_id='0',sensor_timeout_ms=500,sensor_visibility_file=path)
        env=ManeuverEnvironmentBuilder(NS(),config,self.f.clock,visibility_max_checks=1).build(p)
        self.assertEqual('unknown',env.parking_spaces[0]['occupancy']); self.assertFalse(env.free_regions)
        self.assertIn('BUDGET',env.status['coverage']['query_reason'])

    def test_source_expiry_during_map_work_withdraws_all_new_clear_certificates(self):
        f=self.f; original=f.route.read_maneuver_map
        def late(*args,**kwargs):
            result=original(*args,**kwargs); f.now+=.21; return result
        with patch.object(f.route,'read_maneuver_map',side_effect=late): env=self.build(self.model())
        self.assertFalse(env.coverage_regions); self.assertFalse(env.free_regions)
        self.assertFalse(any(v['coverage_verified'] for v in env.road_regions+env.parking_spaces))
        self.assertEqual('source_expired_during_environment_build',env.status['coverage']['reason'])

    def test_clock_regression_during_map_work_cannot_publish_clear_regions(self):
        f=self.f; original=f.route.read_maneuver_map
        def regressed(*args,**kwargs):
            result=original(*args,**kwargs); f.now-=.01; return result
        with patch.object(f.route,'read_maneuver_map',side_effect=regressed): env=self.build(self.model())
        self.assertFalse(env.coverage_regions); self.assertFalse(env.free_regions)
        self.assertFalse(any(v['coverage_verified'] for v in env.road_regions+env.parking_spaces))

    def test_split_bay_is_rechecked_by_formal_parking_reader(self):
        from simone_platform.map_observations import MapObservationReader
        from tests.test_map_observations import parking,map_api
        f=self.f; catalogue=MapObservationReader(map_api(parks=[parking()])).catalog()
        f.p.parking_spaces=catalogue['parking_spaces']; f.p.map_observation_status=catalogue['map_observation_status']
        model=self.model(); model['profiles'][0]['body_region_m'][1][0]=12.5
        model['profiles'][0]['body_region_m'][2][0]=12.5
        model['profiles'][1]['body_region_m'][0][0]=12.5
        model['profiles'][1]['body_region_m'][3][0]=12.5
        env=self.build(model); self.assertEqual('empty',env.parking_spaces[0]['occupancy'])
        facts=ManeuverFacts(f.p,f.clock); self.assertEqual(2,len(facts.parking(7)['coverage']))
        env.coverage_regions.pop()
        with self.assertRaises(ValueError): facts.parking(7)
        # A body on the internal visibility seam remains a real obstacle.
        f.p.targets=[target(22.5,4.5)]
        env=self.build(model); self.assertEqual('occupied',env.parking_spaces[0]['occupancy'])
        self.assertEqual([9],env.parking_spaces[0]['blocking_object_ids'])
        self.assertFalse(any(v.get('id')=='parking:7' for v in env.free_regions))
        with self.assertRaises(ValueError): ManeuverFacts(f.p,f.clock).parking(7)

    def test_split_crosswalk_keeps_object_prediction_and_unknown_rules(self):
        from simone_platform.map_observations import MapObservationReader
        from tests.test_map_observations import map_api
        f=self.f; values=MapObservationReader(map_api()).lane_objects([f.p.lane.lane_id])
        f.p.map_crosswalks,f.p.map_crosswalks_valid=values['map_crosswalks'],True
        f.p.map_stop_lines,f.p.map_stop_lines_valid=values['map_stop_lines'],True
        model=self.model(); model['profiles'][0]['body_region_m'][1][0]=35.
        model['profiles'][0]['body_region_m'][2][0]=35.
        model['profiles'][1]['body_region_m'][0][0]=35.
        model['profiles'][1]['body_region_m'][3][0]=35.
        f.p.targets=[target(45.,0.),target(45.,-10.,identifier=10)]; f.p.targets[1].vy=10.
        item=self.build(model).crossing_regions[0]
        self.assertTrue(item['coverage_verified']); self.assertEqual('occupied',item['occupancy'])
        self.assertEqual([9],item['object_ids']); self.assertEqual([9,10],item['predicted_object_ids'])
        self.assertFalse(item['rule_verified']); self.assertFalse(item['exit_coverage_verified'])

    def test_historical_packet_union_uses_original_rotated_pose_and_deadline(self):
        from tests.test_sensor_pose_history import headers,old_packet,Clock
        clock=Clock(); p=headers(perception(),1,50,heading=math.pi/2)
        model=profile(p); first=model['profiles'][0]
        first['body_region_m']=[[-10.,-5.],[0.,-5.],[0.,5.],[-10.,5.]]
        second=copy.deepcopy(first); second['body_region_m']=[[0.,-5.],[10.,-5.],[10.,5.],[0.,5.]]
        model['profiles'].append(second)
        path=os.path.join(self.f.directory.name,'historical_union.json')
        with open(path,'w',encoding='utf8') as stream: json.dump(model,stream)
        builder=ManeuverEnvironmentBuilder(NS(),NS(vehicle_id='0',sensor_timeout_ms=500,
            sensor_visibility_file=path),clock)
        builder.build(p); clock.now=100.1; headers(p,2,100,x=60.); old_packet(p,1,50)
        p.maneuver_environment=env=builder.build(p)
        views=ManeuverFacts(p,clock).coverage([(9.,-1.,0.),(11.,1.,0.)])
        self.assertEqual(2,len(views))
        self.assertTrue(all(v['pose_frame_id']==1 and v['pose_source']=='sensor_pose_history_exact_frame'
                            and v['observed_at_s']==100. and v['valid_until_s']==100.5 for v in views))
        q=VisibilityQuery(p.target_source,1,100.,env.dynamic_valid_until,clock)
        self.assertTrue(q.contains(rectangle(9.,-1.,11.,1.),views),q.reason)
        self.assertFalse(q.contains(rectangle(59.,-1.,61.,1.),views))
