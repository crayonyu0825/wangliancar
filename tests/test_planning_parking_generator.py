"""Authored P04 segments and formal source consumption; no SDK/control send."""
import copy
import json
import math
import time
import unittest
from fractions import Fraction
from unittest.mock import patch

from core.boundary_lineage import encode_position
from core.maneuver_facts import ManeuverFacts
from core.interfaces import Perception
from core.geometry import project_polyline,normalize_angle
from core.serialization import perception_to_dict
from scripts.replay_decision import _assign
from simone_platform.map_observations import MapObservationReader
from members.planning.candidate_validation import CorridorRegion
from members.planning.crossing_corridor import CrossingCorridor
from members.planning.parking_generator import ParkingSearch,generate_parking_segment
from members.planning.parking_scene import read_parking_scene,plan_parking_candidate
from members.planning.lane_change_generator import PredictionEnvelope
from members.planning.maneuver_scene import ManeuverGeometryCache
from tests.test_planning_candidate_validation import vehicle,limits,budget,rectangle,obstacle
from tests import test_maneuver_fact_consumers as producer_fixture
from tests.test_maneuver_perception import target
from tests.test_map_observations import map_api,parking,pt,Vector
from tests.test_control_maneuvers import BidirectionalPlant,maneuver_input
from tests.control_benchmark import vehicle as control_vehicle
from members.control.controller import ControlEngine


def search(**changes):
    values=dict(tangent_scales=(.7,1.,1.4,2.),end_tangent_scales=(.7,1.,1.4),
                speed_scales=(1.,.5),spacing_m=.2,max_points=1000)
    values.update(changes)
    return ParkingSearch(**values)


def bay_access(knots,polygons,edge=0):
    descriptor=((tuple(knots[edge])+(0.,),tuple(knots[(edge+1)%4])+(0.,)),
                (tuple(knots[(edge+3)%4])+(0.,),tuple(knots[(edge+2)%4])+(0.,)),'left',
                ((encode_position(0,Fraction(0)),encode_position(0,Fraction(1))),))
    return CrossingCorridor(tuple(tuple(tuple(p) for p in v) for v in polygons),descriptor,1000000)


PARALLEL=((7.,1.75),(-1.5,1.75),(-1.5,-1.75),(7.,-1.75))
PERPENDICULAR=((-1.75,5.),(-1.75,-1.5),(1.75,-1.5),(1.75,5.))


class ParkingGeneratorTests(unittest.TestCase):
    def generate(self,**changes):
        road=rectangle(-15.,-15.,25.,15.)
        values=dict(start_pose=(10.,3.5,0.),goal_pose=(0.,0.,0.),initial_speed_mps=0.,
            initial_curvature_m_inv=0.,motion_direction=-1,speed_cap_mps=.8,
            vehicle=vehicle(3.9,.9,.9,2.9),limits=limits(),
            access_corridor=bay_access(PARALLEL,[road,PARALLEL]),coverage=CorridorRegion([road]),
            goal_corridor=CorridorRegion([PARALLEL]),obstacles=[],search=search(),
            budget=budget(max_checks=1000000,deadline_monotonic_s=time.monotonic()+20.))
        values.update(changes)
        return generate_parking_segment(**values)

    def ready(self,result):
        self.assertEqual('safe',result['status'],result)
        self.assertEqual('nominal_candidate_only',result['authority'])
        self.assertTrue(result['stop_required']); self.assertTrue(result['precision_stop'])
        self.assertGreater(result['target_speed'],0.)
        self.assertEqual(0.,result['points'][-1].speed)

    def test_parallel_reverse_generates_actual_stopped_goal_and_body_yaw(self):
        result=self.generate(); self.ready(result)
        self.assertEqual(-1,result['motion_direction'])
        points=result['points']
        self.assertEqual((10.,3.5,0.,0.),(points[0].x,points[0].y,points[0].heading,points[0].speed))
        self.assertEqual((0.,0.,0.),(points[-1].x,points[-1].y,points[-1].heading))
        self.assertTrue(all(b.x<a.x for a,b in zip(points,points[1:])))
        self.assertTrue(all(abs(p.heading)<math.pi/2 for p in points))
        # Travel is backwards while body heading stays approximately forwards.
        self.assertLess(points[1].x-points[0].x,0.)
        distance=sum(math.hypot(b.x-a.x,b.y-a.y) for a,b in zip(points,points[1:]))
        self.assertAlmostEqual(distance,result['stop_distance'])

    def test_reverse_speed_time_acceleration_and_stopping_are_consistent(self):
        result=self.generate(initial_speed_mps=.3); self.ready(result)
        self.assertEqual(.3,result['points'][0].speed)
        for a,b in zip(result['points'],result['points'][1:]):
            ds=math.hypot(b.x-a.x,b.y-a.y); dt=b.relative_time-a.relative_time
            self.assertGreater(dt,0.)
            self.assertAlmostEqual(ds,.5*(a.speed+b.speed)*dt,places=10)
            self.assertLessEqual((b.speed-a.speed)/dt,3.+1e-9)
            self.assertLessEqual((a.speed-b.speed)/dt,4.+1e-9)
            self.assertGreaterEqual(a.speed,0.)

    def test_perpendicular_reverse_and_forward_exit_are_generated_from_goals(self):
        road=rectangle(-15.,-15.,25.,20.)
        access=bay_access(PERPENDICULAR,[road,PERPENDICULAR],3)
        entry=self.generate(start_pose=(10.,10.,0.),goal_pose=(0.,0.,math.pi/2),
            access_corridor=access,coverage=CorridorRegion([road]),goal_corridor=CorridorRegion([PERPENDICULAR]))
        self.ready(entry)
        self.assertAlmostEqual(math.pi/2,entry['points'][-1].heading)
        exit_path=self.generate(start_pose=(0.,0.,math.pi/2),goal_pose=(10.,10.,0.),motion_direction=1,
            access_corridor=access,coverage=CorridorRegion([road]),goal_corridor=CorridorRegion([road]))
        self.ready(exit_path)
        self.assertEqual(1,exit_path['motion_direction'])
        self.assertEqual(0.,exit_path['points'][-1].heading)

    def test_parallel_forward_exit_ends_stopped_in_aisle(self):
        road=rectangle(-15.,-15.,25.,15.)
        result=self.generate(start_pose=(0.,0.,0.),goal_pose=(10.,3.5,0.),motion_direction=1,
                             goal_corridor=CorridorRegion([road]))
        self.ready(result)

    def test_slot_side_and_rear_cannot_be_crossed_even_when_union_contains_them(self):
        for start,goal in (((-10.,0.,0.),(0.,0.,0.)),((0.,-6.,math.pi/2),(0.,0.,math.pi/2))):
            result=self.generate(start_pose=start,goal_pose=goal,motion_direction=1)
            self.assertFalse(result['points'],result)
            self.assertTrue(any(v.get('access_report',{}).get('reason_code')=='CORRIDOR_COLLISION'
                                for v in result['attempts']),result)

    def test_full_target_body_must_fit_selected_bay(self):
        narrow=CorridorRegion([rectangle(-1.,-.5,4.,.5)])
        result=self.generate(goal_corridor=narrow)
        self.assertFalse(result['points'])
        self.assertTrue(any(v.get('goal_pose_report',{}).get('reason_code')=='CORRIDOR_COLLISION'
                            for v in result['attempts']),result)

    def test_rear_vehicle_intrusion_and_unknown_orientation_are_not_ignored(self):
        for yaw in (0.,None):
            result=self.generate(obstacles=[obstacle(5.,1.5,length=4.,width=2.,heading=yaw)])
            self.assertFalse(result['points'])
            self.assertTrue(any(v.get('access_report',{}).get('reason_code')=='OBSTACLE_COLLISION'
                                for v in result['attempts']),result)

    def test_view_holes_and_unknown_rear_space_withdraw_path(self):
        views=CorridorRegion([rectangle(6.,-10.,25.,15.),rectangle(-15.,-10.,4.,15.)])
        result=self.generate(coverage=views)
        self.assertFalse(result['points'])
        self.assertTrue(any(v.get('visibility_report',{}).get('reason_code')=='CORRIDOR_COLLISION'
                            for v in result['attempts']),result)

    def test_prediction_horizon_remains_finite(self):
        result=self.generate(obstacles=[obstacle(20.,10.,horizon=.1)])
        self.assertFalse(result['points'])
        self.assertTrue(any(v.get('access_report',{}).get('reason_code')=='PREDICTION_HORIZON'
                            for v in result['attempts']),result)

    def test_stationary_goal_accepts_list_tuple_without_creating_motion_or_dwell_completion(self):
        result=self.generate(start_pose=[0.,0.,0.]); self.ready(result)
        self.assertEqual(0.,result['stop_distance'])
        self.assertTrue(all(p.x==p.y==p.speed==0. for p in result['points']))
        self.assertNotIn('hold_completed',result)
        self.assertNotIn('behavior_feedback',result)
        moving=self.generate(start_pose=[0.,0.,0.],initial_speed_mps=.2)
        self.assertFalse(moving['points'])

    def test_stationary_heading_change_cannot_rotate_vehicle_in_place(self):
        result=self.generate(start_pose=(0.,0.,.2))
        self.assertFalse(result['points'])

    def test_invalid_direction_geometry_limits_search_and_access_are_rejected(self):
        for changes in (dict(motion_direction=True),dict(motion_direction=0),dict(goal_pose=None),
                        dict(initial_curvature_m_inv=None),dict(vehicle=None),dict(search=search(max_points=2)),
                        dict(access_corridor=CorridorRegion([rectangle()]))):
            result=self.generate(**changes)
            self.assertEqual('invalid',result['status'],result); self.assertFalse(result['points'])

    def test_one_work_budget_for_all_candidates_and_checks(self):
        for work in (budget(max_checks=10),budget(deadline_monotonic_s=time.monotonic()-1.)):
            result=self.generate(budget=work)
            self.assertEqual('BUDGET_EXHAUSTED',result['reason_code'])
            self.assertLessEqual(result['checks'],work.max_checks+1)
            self.assertFalse(result['points'])

    def test_mirrored_bay_and_goal_produce_mirrored_body_yaw(self):
        normal=self.generate(); self.ready(normal)
        mirrored=tuple((x,-y) for x,y in PARALLEL)
        road=rectangle(-15.,-15.,25.,15.)
        result=self.generate(start_pose=(10.,-3.5,0.),access_corridor=bay_access(mirrored,[road,mirrored]),
                             goal_corridor=CorridorRegion([mirrored]))
        self.ready(result)
        for a,b in zip(normal['points'],result['points']):
            self.assertAlmostEqual(a.x,b.x); self.assertAlmostEqual(a.y,-b.y)
            self.assertAlmostEqual(a.heading,-b.heading)
            self.assertAlmostEqual(a.relative_time,b.relative_time)

    def test_parallel_uses_supplied_long_side_not_sdk_front_by_default(self):
        result=self.generate(); self.ready(result)
        self.assertNotEqual(result['attempts'][-1]['tangent_scale'],result['attempts'][-1]['end_tangent_scale'])
        road=rectangle(-15.,-15.,25.,15.)
        wrong=self.generate(access_corridor=bay_access(PARALLEL,[road,PARALLEL],3))
        self.assertFalse(wrong['points'])

    def test_tight_slot_does_not_turn_failed_finite_search_into_a_safe_path(self):
        narrow=((5.,1.75),(-1.5,1.75),(-1.5,-1.75),(5.,-1.75))
        road=rectangle(-15.,-15.,25.,15.)
        result=self.generate(access_corridor=bay_access(narrow,[road,narrow]),goal_corridor=CorridorRegion([narrow]))
        self.assertFalse(result['points'])
        self.assertEqual('NO_CERTIFIED_PARKING_SEGMENT',result['reason_code'])

    def test_terminal_straight_preserves_reverse_goal_and_zero_speed(self):
        model=search(tangent_scales=(1.,1.4,2.),end_tangent_scales=(.7,1.),
                     speed_scales=(1.,),terminal_straight_m=(2.,))
        road=rectangle(-15.,-15.,25.,15.)
        result=self.generate(start_pose=(10.,0.,0.),search=model,
            access_corridor=bay_access(PARALLEL,[road,PARALLEL],3)); self.ready(result)
        self.assertEqual(2.,result['attempts'][-1]['terminal_straight_m'])
        points=result['points']; tail=[p for p in points if p.x<=2.+1e-9]
        self.assertGreater(len(tail),2)
        for p in tail:
            self.assertAlmostEqual(0.,p.y); self.assertAlmostEqual(0.,p.heading)
        self.assertEqual((0.,0.,0.,0.),(points[-1].x,points[-1].y,points[-1].heading,points[-1].speed))
        for a,b in zip(points,points[1:]):
            self.assertLessEqual(math.hypot(b.x-a.x,b.y-a.y),model.spacing_m+1e-9)
            self.assertAlmostEqual(math.hypot(b.x-a.x,b.y-a.y),
                .5*(a.speed+b.speed)*(b.relative_time-a.relative_time),places=10)

    def test_terminal_straight_cannot_bypass_objects_visibility_or_finite_work(self):
        model=search(tangent_scales=(1.,1.4,2.),end_tangent_scales=(.7,1.),
                     speed_scales=(1.,),terminal_straight_m=(2.,))
        road=rectangle(-15.,-15.,25.,15.)
        access=bay_access(PARALLEL,[road,PARALLEL],3)
        self.ready(self.generate(start_pose=(10.,0.,0.),search=model,access_corridor=access))
        for changes in (dict(obstacles=[obstacle(1.,0.,length=1.,width=1.,heading=None)]),
                        dict(coverage=CorridorRegion([rectangle(3.,-15.,25.,15.)])),
                        dict(budget=budget(max_checks=5))):
            result=self.generate(start_pose=(10.,0.,0.),search=model,access_corridor=access,**changes)
            self.assertFalse(result['points'],result)

    def test_terminal_candidate_family_and_point_count_remain_bounded(self):
        for model in (search(terminal_straight_m=(-1.,)),search(terminal_straight_m=(float('nan'),)),
                      search(terminal_straight_m=(0.,1.,2.)),
                      search(tangent_scales=(1.,),end_tangent_scales=(1.,),speed_scales=(1.,),
                             terminal_straight_m=(100.,),max_points=70)):
            result=self.generate(search=model)
            self.assertFalse(result['points'],result)
        self.assertEqual('CANDIDATE_POINT_LIMIT',result['attempts'][0]['reason_code'])


class ParkingFactsAndSceneTests(unittest.TestCase):
    def setUp(self):
        f=producer_fixture.FactConsumerTests(); f.setUp(); self.addCleanup(f.doCleanups)
        self.f=f
        # Source quad front is at x=20, aisle x<20; rear x=27.
        native=parking(7)
        native.pt=pt(23.5,0.)
        native.boundaryKnots=Vector([pt(20.,1.75),pt(27.,1.75),pt(27.,-1.75),pt(20.,-1.75)])
        native.heading=pt(-1.,0.)
        catalogue=MapObservationReader(map_api(parks=[native])).catalog()
        f.p.parking_spaces=catalogue['parking_spaces']
        f.p.parking_spaces_valid=catalogue['parking_spaces_valid']
        f.p.map_observation_status=catalogue['map_observation_status']
        f.p.ego.heading=math.pi
        f.build()

    def facts(self): return ManeuverFacts(self.f.p,self.f.clock)

    def plan(self,**changes):
        f=self.f
        values=dict(perception=f.p,space_id=7,stage='REVERSE_ENTRY',goal_pose=(25.,0.,math.pi),
            motion_direction=-1,initial_curvature_m_inv=0.,speed_cap_mps=.8,
            vehicle=vehicle(3.9,.9,.9,2.9),limits=limits(),search=search(),
            budget=budget(max_checks=1000000),prediction_envelopes={},road_lane_ids=[f.p.lane.lane_id],
            access_edge_index=3,clock=f.clock)
        values.update(changes)
        return plan_parking_candidate(**values)

    def test_actual_catalogue_environment_to_generated_reverse_segment(self):
        bay=self.facts().parking(7)
        self.assertEqual('empty',bay['occupancy'])
        self.assertEqual([bay['boundary_knots'][0],bay['boundary_knots'][3]],bay['entrance_edge'])
        result=self.plan()
        self.assertEqual('safe',result['status'],result)
        self.assertEqual((25.,0.,math.pi),result['goal_pose'])
        self.assertEqual(0.,result['points'][-1].speed)
        self.assertLessEqual(result['source_valid_until_s'],self.f.p.valid_until)

    def test_public_json_replay_preserves_original_bay_and_deadlines(self):
        replay=Perception(); _assign(replay,json.loads(json.dumps(perception_to_dict(self.f.p))),'perception')
        before=self.facts().parking(7)
        after=ManeuverFacts(replay,self.f.clock).parking(7)
        self.assertEqual(before,after)

    def test_catalogue_failure_duplicate_id_original_geometry_and_entry_change_rejected(self):
        original=copy.deepcopy(self.f.p)
        for change in ('catalogue','duplicate','knots','entry','bound','heading','plane'):
            self.f.p=copy.deepcopy(original)
            p=self.f.p
            if change=='catalogue': p.map_observation_status['parking_spaces']['read_ok']=False
            elif change=='duplicate': p.parking_spaces.append(copy.deepcopy(p.parking_spaces[0]))
            elif change=='knots': p.maneuver_environment.parking_spaces[0]['boundary_knots'][0]=(21.,1.75,0.)
            elif change=='entry': p.parking_spaces[0]['entrance_edge']=p.parking_spaces[0]['boundary_knots'][1:3]
            elif change=='bound': p.maneuver_environment.parking_spaces[0]['boundary'][0]=(21.,1.75)
            elif change=='heading': p.maneuver_environment.parking_spaces[0]['heading']=0.
            else: p.parking_spaces[0]['boundary_knots'][0]=(20.,1.75,.2)
            with self.assertRaises(ValueError,msg=change): self.facts().parking(7)

    def test_unknown_map_heading_is_preserved_not_used_as_vehicle_goal(self):
        p=self.f.p
        p.parking_spaces[0].update(heading_valid=False,heading=0.)
        self.f.build()
        self.assertIsNone(self.facts().parking(7)['map_heading_rad'])
        self.assertEqual('safe',self.plan()['status'])

    def test_occupied_unobserved_and_expired_space_cannot_become_free(self):
        original=copy.deepcopy(self.f.p)
        self.f.p.targets=[target(23.,0.)]; self.f.build()
        with self.assertRaises(ValueError): self.facts().parking(7)
        self.f.p=copy.deepcopy(original)
        self.f.p.maneuver_environment.coverage_regions=[]
        with self.assertRaises(ValueError): self.facts().parking(7)
        self.f.p=copy.deepcopy(original); self.f.now+=.21
        with self.assertRaises(ValueError): self.facts().parking(7)

    def test_read_returns_detached_geometry_and_does_not_grant_road_rules(self):
        bay=self.facts().parking(7); bay['boundary'][0]=(100.,100.)
        self.assertNotEqual(bay['boundary'],self.facts().parking(7)['boundary'])
        scene=read_parking_scene(self.f.p,7,[self.f.p.lane.lane_id],True,3,self.f.clock)
        self.assertEqual('bay_access_geometry_and_current_visibility_only',scene['authority'])

    def test_forward_exit_reuses_current_geometry_and_stage_contract(self):
        self.f.p.ego.x=25.; self.f.build()
        result=self.plan(stage='EXIT',goal_pose=(10.,0.,math.pi),motion_direction=1)
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(1,result['motion_direction'])
        self.assertEqual(0.,result['points'][-1].speed)

    def test_stage_direction_and_actual_opposite_motion_are_rejected(self):
        for changes in (dict(stage='PARKED_DWELL'),dict(motion_direction=1),dict(stage='EXIT')):
            result=self.plan(**changes); self.assertFalse(result['points'])
        self.f.p.ego.vx=-1.; self.f.p.ego.speed=1.
        result=self.plan()
        self.assertEqual('ACTUAL_MOTION_OPPOSES_PARKING_STAGE',result['reason_code'])

    def test_prediction_bounds_match_original_object_set_and_age(self):
        self.f.p.targets=[target(65.,0.,heading=None)]; self.f.build()
        result=self.plan(); self.assertEqual('PREDICTION_OBJECT_SET_MISMATCH',result['reason_code'])
        result=self.plan(prediction_envelopes={9:PredictionEnvelope(100.,.1,.1)})
        self.assertEqual('safe',result['status'],result)

    def test_source_expiry_or_map_change_during_search_withdraws_points(self):
        from members.planning import parking_scene
        original=parking_scene.generate_parking_segment
        def expire(*args):
            result=original(*args); self.f.now+=.21; return result
        with patch.object(parking_scene,'generate_parking_segment',side_effect=expire): result=self.plan()
        self.assertFalse(result['points']); self.assertNotEqual('safe',result['status'])

    def test_cache_and_unknown_aisle_never_extend_scope(self):
        cache=ManeuverGeometryCache(3,1000000)
        first=self.plan(geometry_cache=cache); self.assertEqual('safe',first['status'],first)
        second=self.plan(geometry_cache=cache); self.assertEqual('safe',second['status'],second)
        unknown=self.plan(road_lane_ids=['unknown'])
        self.assertEqual('PARKING_ROAD_UNAVAILABLE',unknown['reason_code'])
        self.f.now+=.21
        expired=self.plan(geometry_cache=cache); self.assertFalse(expired['points'])

    def test_access_edge_is_explicit_and_bound_to_original_quad(self):
        for index in (None,True,-1,4):
            result=self.plan(access_edge_index=index)
            self.assertEqual('PARKING_ACCESS_EDGE_UNAVAILABLE',result['reason_code'])
        wrong=self.plan(access_edge_index=1)
        self.assertFalse(wrong['points'])

    def test_map_change_during_search_cannot_reuse_old_candidate(self):
        from members.planning import parking_scene
        original=parking_scene.generate_parking_segment
        def changed(*args):
            result=original(*args)
            self.f.p.maneuver_environment.map_semantics['digest']='a'*32
            return result
        with patch.object(parking_scene,'generate_parking_segment',side_effect=changed): result=self.plan()
        self.assertFalse(result['points'])
        self.assertEqual('PARKING_SOURCE_IDENTITY_CHANGED_DURING_SEARCH',result['reason_code'])

    def test_formal_curved_entry_into_bay_outside_aisle_uses_actual_union_and_entry(self):
        native=parking(7)
        native.pt=pt(21.75,-5.25); native.heading=pt(0.,1.)
        native.boundaryKnots=Vector([pt(20.,-1.75),pt(20.,-8.75),pt(23.5,-8.75),pt(23.5,-1.75)])
        catalogue=MapObservationReader(map_api(parks=[native])).catalog()
        self.f.p.parking_spaces=catalogue['parking_spaces']
        self.f.p.map_observation_status=catalogue['map_observation_status']
        self.f.p.ego.x,self.f.p.ego.y,self.f.p.ego.heading=28.,1.,0.
        self.f.build()
        result=self.plan(goal_pose=(21.75,-6.,math.pi/2),road_lane_ids=['1_0_-1','1_0_-2'],
                         search=search(tangent_scales=(1.,),end_tangent_scales=(1.4,),speed_scales=(1.,)))
        self.assertEqual('safe',result['status'],result)
        self.assertLess(result['points'][-1].y,-1.75)
        self.assertAlmostEqual(math.pi/2,result['points'][-1].heading)


class GeneratedParkingControlTests(unittest.TestCase):
    def test_generated_parallel_and_perpendicular_reverse_dwell_and_exit(self):
        for kind in ('parallel','perpendicular'):
            with self.subTest(kind=kind): self.execute(kind)

    def execute(self,kind):
        generator=ParkingGeneratorTests()
        start=(10.,3.5,0.) if kind=='parallel' else (10.,10.,0.)
        goal=(0.,0.,0.) if kind=='parallel' else (0.,0.,math.pi/2)
        knots=PARALLEL if kind=='parallel' else PERPENDICULAR
        road=rectangle(-15.,-15.,25.,20.)
        access=bay_access(knots,[road,knots],0 if kind=='parallel' else 3)
        entry=generator.generate(start_pose=start,goal_pose=goal,access_corridor=access,
            coverage=CorridorRegion([road]),goal_corridor=CorridorRegion([knots]))
        generator.ready(entry)
        exit_path=generator.generate(start_pose=goal,goal_pose=start,motion_direction=1,
            access_corridor=access,coverage=CorridorRegion([road]),goal_corridor=CorridorRegion([road]))
        generator.ready(exit_path)
        now=[0.]; calibration=control_vehicle()
        engine=ControlEngine(calibration,clock=lambda:now[0])
        plant=BidirectionalPlant(x=start[0],y=start[1],heading=start[2],lag=.3)
        parked_at,exit_at,finished=None,None,False
        for frame in range(1,2201):
            candidate=entry if parked_at is None else exit_path
            points=candidate['points']
            reference=[(p.x,p.y,p.speed) for p in points]
            projection=project_polyline(reference,plant.x,plant.y)
            remaining=max(0.,candidate['stop_distance']-projection['s'])
            p,t=maneuver_input(plant,reference,frame,target=candidate['target_speed'],
                direction=candidate['motion_direction'],stop_distance=remaining,precision=True,
                dwell=10. if parked_at is None else 0.,parking=parked_at is None)
            t.points=copy.deepcopy(points)
            output=engine.compute(p,t)
            self.assertTrue(output.valid,(kind,frame,output.errors))
            self.assertFalse(output.throttle>0 and output.brake>0)
            if parked_at is None and output.diagnostics['state']=='DWELL':
                parked_at=now[0]
                self.assertLessEqual(plant.speed,.05)
                self.assertLess(math.hypot(plant.x-goal[0],plant.y-goal[1]),.16)
                self.assertLess(abs(normalize_angle(plant.heading-goal[2])),.15)
            elif parked_at is not None:
                if output.throttle>0 and exit_at is None:
                    exit_at=now[0]; self.assertEqual(1,output.gear)
                if now[0]-parked_at<10.:
                    self.assertEqual(0.,output.throttle); self.assertTrue(output.handbrake)
                if (exit_at is not None and output.diagnostics.get('stop_pose_arrived') is True
                        and plant.speed<=.05):
                    finished=True; break
            plant.step(output,.05,calibration); now[0]+=.05
        self.assertIsNotNone(parked_at,(kind,plant.x,plant.y,plant.heading))
        self.assertIsNotNone(exit_at,(kind,'no exit propulsion'))
        self.assertGreaterEqual(exit_at-parked_at,10.)
        self.assertTrue(finished,(kind,plant.x,plant.y,plant.heading))
        self.assertLess(math.hypot(plant.x-start[0],plant.y-start[1]),.16)
        self.assertLess(abs(normalize_angle(plant.heading-start[2])),.15)
        self.assertEqual(1,plant.direction)


if __name__=='__main__': unittest.main()
