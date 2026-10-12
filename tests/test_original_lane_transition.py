"""Real producer/source refresh across an SDK current-lane switch, no SDK send."""
import copy
import json
import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.interfaces import Perception,Trajectory
from core.maneuver_facts import ManeuverFacts
from core.serialization import perception_to_dict
from scripts.replay_decision import _assign
from members.planning.lane_change_generator import plan_lane_change_candidate,PredictionEnvelope
from members.planning.maneuver_scene import read_maneuver_scene,ManeuverGeometryCache
from tests import test_maneuver_fact_consumers as producer_fixture
from tests.test_planning_lane_change_generator import search
from tests.test_planning_candidate_validation import vehicle,limits,budget
from tests.test_maneuver_map import document,road,section,mark
from tests.test_maneuver_perception import target
from tests.test_control_maneuvers import BidirectionalPlant
from tests.control_benchmark import vehicle as control_vehicle
from members.control.controller import ControlEngine
from core.region_geometry import outline_contains_path
from core.geometry import normalize_angle


SOURCE,TARGET='1_0_-1','1_0_-2'


class OriginalLaneTransitionTests(unittest.TestCase):
    def setUp(self):
        self.f=producer_fixture.FactConsumerTests(); self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.route._maneuver_map.crossing_ranges.clock=self.f.clock
        self.api=self.f.adapter.hdmap
        original_link=self.api.getLaneLink
        def links(native):
            value=original_link(native)
            if self.api.identity(native)==TARGET:
                value.laneLink.rightNeighborLaneId=self.api.pySimString(SOURCE)
            return value
        self.api.getLaneLink=links
        self.api.getRoadMark=lambda point,native:NS(exists=True,
            left=NS(type='broken',sOffset=0.),right=NS(type='broken',sOffset=0.))
        self.f.bound()

    def update(self,x=None,y=None,heading=None,current=None):
        p=self.f.p
        for key,value in (('x',x),('y',y),('heading',heading)):
            if value is not None: setattr(p.ego,key,value)
        if current is not None:
            self.api.current=current
            p.lane=self.f.route.update(p.ego,True)
        p.frame_id+=1; p.targets_frame_id=p.frame_id; self.f.now+=.001
        self.f.build()

    def switch(self):
        self.update(18.,2.2,.15,TARGET)
        self.assertEqual(TARGET,self.f.p.lane.lane_id)
        self.assertEqual(SOURCE,self.f.p.lane.right_lane_id)

    def read(self): return ManeuverFacts(self.f.p,self.f.clock).crossing(TARGET,SOURCE)

    def plan(self,**changes):
        values=dict(perception=self.f.p,target_lane_id=TARGET,initial_curvature_m_inv=0.,
            speed_cap_mps=3.,vehicle=vehicle(3.9,.9,.9,2.9),limits=limits(),search=search(),
            budget=budget(max_checks=1000000),prediction_envelopes={},clock=self.f.clock)
        values.update(changes)
        return plan_lane_change_candidate(**values)

    def restrict(self,direction,rule='RHT'):
        data=document(road(length=1000,sections=section(0,mark(change=direction)),rule=rule))
        model=self.f.route._maneuver_map
        # The existing adapter identity service returns these selected bytes.
        import hashlib
        self.f.metadata['digest']=hashlib.md5(data).hexdigest()
        self.f.adapter.read_map_document=lambda:(data,dict(self.f.metadata))
        model._read()
        self.update(current=SOURCE)
        self.f.bound()

    def test_original_direction_refresh_is_available_on_first_switched_frame(self):
        before=ManeuverFacts(self.f.p,self.f.clock).crossing(TARGET)
        self.switch(); after=self.read()
        self.assertEqual(SOURCE,after['source_lane_id'])
        self.assertEqual(TARGET,after['lane_id'])
        self.assertEqual(before['source_lane_boundaries'],after['source_lane_boundaries'])
        self.assertEqual(before['permitted_source_positions'],after['permitted_source_positions'])
        self.assertEqual('left',after['side'])
        self.assertEqual(self.f.p.frame_id,after['evidence']['frame_id'])

    def test_original_permission_and_reverse_permission_are_independent(self):
        self.restrict('decrease')
        self.switch()
        self.assertTrue(self.read()['permitted_fragments'])
        # Refresh B->A separately until its own source sampling completes.
        for unused in range(100):
            self.update()
            record=self.f.p.maneuver_environment.neighbor_lanes[0]
            if record['crossing_range_verified']: break
        reverse=ManeuverFacts(self.f.p,self.f.clock).crossing(SOURCE)
        self.assertEqual([],reverse['permitted_fragments'])
        self.assertTrue(self.read()['permitted_fragments'])

    def test_allowed_reverse_cannot_authorize_forbidden_original_direction(self):
        self.restrict('increase'); self.switch()
        self.assertEqual([],self.read()['permitted_fragments'])
        result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=(35.,3.5,0.))
        self.assertFalse(result['points'],result)

    def test_actual_perception_to_frozen_goal_replans_through_lane_switch(self):
        first=self.plan(); self.assertEqual('safe',first['status'],first)
        goal=first['target_pose']
        self.switch()
        for x,y,yaw in ((18.,2.2,.15),(24.,3.2,.04)):
            self.update(x,y,yaw)
            result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=goal)
            self.assertEqual('safe',result['status'],result)
            self.assertEqual(goal,result['target_pose'])
            self.assertEqual((x,y,yaw),(result['points'][0].x,result['points'][0].y,result['points'][0].heading))
            self.assertEqual((SOURCE,TARGET),(result['source_lane_id'],result['target_lane_id']))

    def test_switched_source_requires_explicit_original_source_and_frozen_goal(self):
        self.switch()
        self.assertEqual('TARGET_ALREADY_CURRENT_LANE',self.plan()['reason_code'])
        self.assertEqual('FROZEN_LANE_CHANGE_GOAL_REQUIRED',self.plan(source_lane_id=SOURCE)['reason_code'])

    def test_unknown_or_unrelated_original_source_cannot_borrow_pair(self):
        self.switch()
        for source in ('other',TARGET):
            result=self.plan(source_lane_id=source,fixed_goal_pose=(35.,3.5,0.))
            self.assertFalse(result['points'])
        with self.assertRaises(ValueError):
            read_maneuver_scene(self.f.p,[SOURCE,TARGET],self.f.clock,source_lane_id='other')

    def test_frozen_goal_does_not_move_with_current_projection_or_changed_reference(self):
        for goal in ((5.,3.5,0.),(35.,4.,0.),(35.,3.5,.1),(1001.,3.5,0.),None):
            result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=goal)
            if goal is None:
                self.assertEqual('safe',result['status'],result)
            else:
                self.assertFalse(result['points'],result)
                self.assertIn(result['reason_code'],('FROZEN_LANE_CHANGE_GOAL_OUTSIDE_CURRENT_REFERENCE',
                                                     'TARGET_REFERENCE_EXTRAPOLATION'))

    def test_unknown_native_original_marking_withdraws_cached_permission(self):
        self.switch(); self.assertTrue(self.read()['permitted_fragments'])
        original=self.api.getRoadMark
        self.api.getRoadMark=lambda point,native:NS(exists=False) if self.api.identity(native)==SOURCE else original(point,native)
        self.update()
        with self.assertRaises(ValueError): self.read()
        self.assertFalse(self.plan(source_lane_id=SOURCE,fixed_goal_pose=(35.,3.5,0.))['points'])

    def test_source_native_type_or_section_offset_mismatch_withdraws_binding(self):
        self.switch(); original=self.api.getRoadMark
        for value in (NS(type='solid',sOffset=0.),NS(type='broken',sOffset=2.)):
            self.api.getRoadMark=lambda point,native,v=value:(NS(exists=True,left=v,right=v)
                if self.api.identity(native)==SOURCE else original(point,native))
            self.update()
            with self.assertRaises(ValueError): self.read()

    def test_map_failure_does_not_publish_a_prior_source_record(self):
        self.switch(); self.f.metadata['verified']=False; self.update()
        self.assertNotIn('crossing_to_current',self.f.p.maneuver_environment.neighbor_lanes[0])
        with self.assertRaises(ValueError): self.read()

    def test_missing_original_binding_cannot_initialize_from_reverse_permission(self):
        self.switch(); self.f.route._maneuver_map.crossing_ranges.clear(); self.update()
        with self.assertRaises(ValueError): self.read()
        self.assertFalse(self.plan(source_lane_id=SOURCE,fixed_goal_pose=(35.,3.5,0.))['points'])

    def test_original_geometry_change_cannot_reuse_same_id_certificate(self):
        self.switch()
        self.api.segments[SOURCE]=[(0.,.05),(1000.,.05)]
        self.f.route._neighbor_samples.clear(); self.update()
        with self.assertRaises(ValueError): self.read()

    def test_shared_budget_includes_original_native_recheck(self):
        self.switch()
        model=self.f.route._maneuver_map
        items=self.f.route.read_neighbor_lanes(self.f.p.ego,self.f.p.lane)
        calls=[]; st=self.api.getRoadST; marks=self.api.getRoadMark
        self.api.getRoadST=lambda *args:(calls.append('st'),st(*args))[1]
        self.api.getRoadMark=lambda *args:(calls.append('mark'),marks(*args))[1]
        meta,unused=model.observe(self.f.p.ego,self.f.p.lane,items)
        self.assertLessEqual(meta['boundary_binding_budget']['atomic_steps'],16)
        self.assertLessEqual(len(calls),17)  # Existing current-road-s read plus shared work.
        self.assertTrue(items[0]['crossing_to_current']['crossing_range_verified'])

    def test_json_replay_keeps_original_direction_and_current_source_expiry(self):
        self.switch(); before=self.read()
        replay=Perception(); _assign(replay,json.loads(json.dumps(perception_to_dict(self.f.p))),'perception')
        after=ManeuverFacts(replay,self.f.clock).crossing(TARGET,SOURCE)
        self.assertEqual(before,after)
        self.f.now+=.21
        with self.assertRaises(ValueError): ManeuverFacts(replay,self.f.clock).crossing(TARGET,SOURCE)

    def test_old_sensor_dynamic_data_cannot_borrow_refreshed_map_permission(self):
        self.switch(); self.f.p.source_status['targets']['age_ms']=450; self.update()
        self.assertTrue(self.read()['permitted_fragments'])
        self.f.now+=.051
        result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=(35.,3.5,0.))
        self.assertFalse(result['points'])

    def test_cache_does_not_change_original_source_after_sdk_lane_switch(self):
        self._use_functional_clock()
        cache=ManeuverGeometryCache(4,1000000)
        first=self.plan(geometry_cache=cache); self.assertEqual('safe',first['status'],first)
        self.switch()
        result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=first['target_pose'],geometry_cache=cache)
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(first['target_pose'],result['target_pose'])
        scene=read_maneuver_scene(self.f.p,[SOURCE,TARGET],self.f.clock,source_lane_id=SOURCE)
        self.assertEqual(SOURCE,scene['crossings'][0]['source_lane_id'])

    def test_original_and_target_perimeters_are_rechecked_after_switch(self):
        self.switch(); before=copy.deepcopy(self.f.p)
        for change in ('source','target','identity','side'):
            self.f.p=copy.deepcopy(before)
            original=self.f.p.maneuver_environment.neighbor_lanes[0]
            prior=original['crossing_to_current']
            if change=='source': original['left_boundary'][0]=(0.,2.,0.)
            elif change=='target': prior['left_boundary'][0]=(0.,6.,0.)
            elif change=='identity': prior['source_lane_id']='other'
            else: prior['side']='right'
            with self.assertRaises(ValueError,msg=change): self.read()

    def test_cached_geometry_cannot_borrow_new_incomplete_visibility(self):
        self._use_functional_clock()
        cache=ManeuverGeometryCache(4,1000000)
        first=self.plan(geometry_cache=cache); self.assertEqual('safe',first['status'],first)
        self.switch()
        self.f.p.maneuver_environment.coverage_regions=[]
        result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=first['target_pose'],geometry_cache=cache)
        self.assertFalse(result['points'])

    def test_cached_geometry_does_not_renew_work_or_source_deadline_during_search(self):
        self._use_functional_clock()
        cache=ManeuverGeometryCache(4,1000000)
        first=self.plan(geometry_cache=cache)
        self.assertEqual('safe',first['status'],first)
        self.switch()
        from members.planning import lane_change_generator as generator
        original=generator.generate_lane_change
        for lifetime,elapsed,reason in ((.01,.02,'BUDGET_EXHAUSTED'),
                                        (5.,.21,'MANEUVER_FRAME_EXPIRED_OR_MISMATCHED')):
            work=budget(max_checks=1000000,deadline_monotonic_s=self.f.now+lifetime)
            def consume_time(*args,**kwargs):
                self.f.now+=elapsed
                return original(*args,**kwargs)
            with patch.object(generator,'generate_lane_change',side_effect=consume_time):
                result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=first['target_pose'],
                                 geometry_cache=cache,budget=work)
            self.assertFalse(result['points'],result)
            self.assertEqual(reason,result['reason_code'],result)

    def test_new_unknown_yaw_object_is_checked_after_sdk_lane_switch(self):
        self.switch(); self.f.p.targets=[target(25.,3.5,heading=None)]; self.update()
        self.assertTrue(self.read()['permitted_fragments'])
        result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=(35.,3.5,0.),
                         prediction_envelopes={9:PredictionEnvelope(100.,.1,.1)})
        self.assertFalse(result['points'],result)

    def test_source_native_recheck_cannot_bypass_clock_budget(self):
        self.switch()
        class LateClock(object):
            def __init__(self): self.now=100.
            def __call__(self): self.now+=.01; return self.now
        self.f.route._maneuver_map.crossing_ranges.clock=LateClock()
        self.update()
        with self.assertRaises(ValueError): self.read()

    def test_original_native_queries_each_withdraw_when_they_exhaust_budget(self):
        self.switch()
        binder=self.f.route._maneuver_map.crossing_ranges
        binder.clock=self.f.clock
        originals=dict(getRoadST=self.api.getRoadST,getRoadMark=self.api.getRoadMark)
        for method in ('getRoadST','getRoadMark'):
            calls=[]
            def slow_query(*args):
                # Only the projected original border consumes time; the
                # existing current-road-s read remains available.
                point=args[1] if method=='getRoadST' else args[0]
                if abs(point[1]-1.75)<1e-8:
                    calls.append(method); self.f.now+=.006
                return originals[method](*args)
            setattr(self.api,method,slow_query)
            self.update()
            self.assertTrue(calls,method)
            prior=self.f.p.maneuver_environment.neighbor_lanes[0]['crossing_to_current']
            self.assertEqual('SOURCE_NATIVE_RECHECK_BUDGET',prior['crossing_scope_reason'])
            with self.assertRaises(ValueError): self.read()
            setattr(self.api,method,originals[method])

    def test_original_native_probe_uses_its_border_and_own_road_station(self):
        self.switch(); original_mark=self.api.getRoadMark; original_st=self.api.getRoadST
        probes=[]
        def mark_read(point,native):
            if self.api.identity(native)==SOURCE:
                probes.append(point)
                return original_mark(point,native) if abs(point[1]-1.75)<1e-8 else NS(exists=False)
            return original_mark(point,native)
        self.api.getRoadMark=mark_read
        self.update()
        self.assertTrue(self.read()['permitted_fragments'])
        prior=self.f.p.maneuver_environment.neighbor_lanes[0]['crossing_to_current']
        self.assertFalse(prior['crossing_allowed_at_ego'])
        self.assertTrue(prior['crossing_allowed_at_source_probe'])
        self.assertTrue(probes); self.assertTrue(all(abs(v[1]-1.75)<1e-8 for v in probes))
        # Current target-lane road-s remains readable; only original-border
        # road-s fails. It cannot be replaced by the target observation.
        self.api.getRoadST=lambda native,point:(NS(exists=False,s=0.)
            if abs(point[1]-1.75)<1e-8 else original_st(native,point))
        self.update()
        with self.assertRaises(ValueError): self.read()

    def test_unknown_sdk_original_lane_type_withdraws_prior_source(self):
        self.switch()
        self.api.getLaneType=lambda native:NS(exists=True,
            laneType='unknown' if self.api.identity(native)==SOURCE else 'driving')
        self.f.route._neighbor_samples.clear(); self.update()
        with self.assertRaises(ValueError): self.read()

    def test_reversed_travel_keeps_logical_side_and_native_side_distinct(self):
        self.f.p.ego.x,self.f.p.ego.heading=60.,math.pi
        self.restrict('decrease','LHT')
        before=ManeuverFacts(self.f.p,self.f.clock).crossing(TARGET)
        self.assertEqual('right',before['side'])
        self.update(50.,2.2,math.pi-.15,TARGET)
        after=self.read()
        self.assertEqual('right',after['side'])
        self.assertEqual(before['source_lane_boundaries'],after['source_lane_boundaries'])
        self.assertTrue(after['permitted_fragments'])
        result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=(35.,3.5,math.pi))
        self.assertEqual('safe',result['status'],result)

    def _use_functional_clock(self):
        virtual_clock=patch('time.monotonic',self.f.clock)
        virtual_clock.start(); self.addCleanup(virtual_clock.stop)
        # Functional source/model assertions use the same virtual time.
        # Expiry during search and native-query deadlines are separate tests.

    def test_generated_replans_and_real_controller_cross_sdk_identity_boundary(self):
        self._use_functional_clock()
        calibration=control_vehicle()
        geometry=vehicle(3.9,.9,.9,calibration.wheelbase_m)
        motion=limits(max_front_steer_rad=calibration.front_steer_max_rad)
        first=self.plan(speed_cap_mps=1.5,vehicle=geometry,limits=motion)
        self.assertEqual('safe',first['status'],first)
        goal=first['target_pose']
        plant=BidirectionalPlant(x=10.,lag=.3)
        engine=ControlEngine(calibration,clock=self.f.clock)
        switched,settled=False,0
        for unused in range(700):
            p=self.f.p
            p.ego.speed,p.ego.yaw_rate=plant.speed,plant.yaw_rate
            p.ego.vx,p.ego.vy=plant.speed*math.cos(plant.heading),plant.speed*math.sin(plant.heading)
            current=TARGET if plant.y>=1.75 else SOURCE
            self.update(plant.x,plant.y,plant.heading,current)
            switched=switched or current==TARGET
            curvature=math.tan(plant.steering*calibration.front_steer_max_rad/calibration.steering_sign)/calibration.wheelbase_m
            result=self.plan(source_lane_id=SOURCE,fixed_goal_pose=goal,speed_cap_mps=1.5,
                             initial_curvature_m_inv=curvature,vehicle=geometry,limits=motion)
            if result['status']!='safe':
                prior=p.maneuver_environment.neighbor_lanes[0].get('crossing_to_current',{})
                diagnostics={key:prior.get(key) for key in ('crossing_scope_reason',
                    'marking_semantics_verified','crossing_range_verified','source_marking_recheck')}
                self.fail((plant.x,plant.y,plant.heading,result,diagnostics))
            self.assertEqual(goal,result['target_pose'])
            t=Trajectory().bind(p); t.valid=True
            t.points=result['points']; t.target_speed=1.5; t.target_lane_id=TARGET
            output=engine.compute(p,t)
            self.assertTrue(output.valid,output.errors)
            self.assertFalse(output.throttle>0 and output.brake>0)
            # Whole-body containment and yaw, not the SDK lane label alone.
            c,s=math.cos(plant.heading),math.sin(plant.heading)
            corners=[(plant.x+a*c-b*s,plant.y+a*s+b*c)
                     for a,b in ((3.9,.9),(3.9,-.9),(-.9,-.9),(-.9,.9))]
            inside=outline_contains_path(corners+[corners[0]],[(0.,1.75),(1000.,1.75),(1000.,5.25),(0.,5.25)])
            arrived=current==TARGET and inside and abs(normalize_angle(plant.heading))<.1 and plant.x>28.
            settled=settled+1 if arrived else 0
            if settled>=5: break
            plant.step(output,.05,calibration); self.f.now+=.049
        self.assertTrue(switched)
        self.assertGreaterEqual(settled,5,(plant.x,plant.y,plant.heading))


if __name__=='__main__': unittest.main()
