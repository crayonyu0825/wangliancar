"""Real formal producer to D04, with explicit synthetic task/model inputs."""
import copy
import math
import unittest
from unittest.mock import patch

from core.behavior_contract import GoalPose,Capabilities
from core.corridor_region import CorridorRegion as SharedRegion
from core.maneuver_facts import ManeuverFacts
from members.decision.behaviors.parking import ParkingArea,ParkingPolicy
from members.decision.behaviors.parking_environment import ParkingMission,parking_facts
from members.decision.behaviors.observations import Evidence,MotionObject,vehicle_footprint
from members.planning.corridor_region import CorridorRegion as LegacyRegion
from simone_platform.map_observations import MapObservationReader
from tests import test_planning_parking_generator as fixture
from tests.test_map_observations import parking,pt,Vector,map_api
from tests.test_planning_candidate_validation import rectangle
from tests.test_decision_behavior_contract import feedback
from tests.test_maneuver_perception import target


class ParkingAreaTests(unittest.TestCase):
    def area(self,regions,views=None,**changes):
        values=dict(regions=regions,views=views or [rectangle(-10.,-10.,10.,10.)],
                    max_checks=1000000,valid_until_s=101.,clock=lambda:100.)
        values.update(changes)
        return ParkingArea(**values)

    def test_legacy_planning_import_is_the_same_shared_implementation(self):
        self.assertIs(SharedRegion,LegacyRegion)

    def test_body_across_real_shared_seam_is_contained(self):
        region=self.area([rectangle(-2.,-2.,2.,3.),rectangle(2.,-2.,6.,3.)])
        body=vehicle_footprint(GoalPose(2.,0.,0.),2.,1.,.5)
        self.assertTrue(region.contains(body),region.reason)

    def test_internal_hole_and_positive_gap_cannot_hide_between_body_corners(self):
        hole=[rectangle(-1.,-1.,5.,.5),rectangle(-1.,1.5,5.,3.),
              rectangle(-1.,.5,1.,1.5),rectangle(2.,.5,5.,1.5)]
        gap=[rectangle(-1.,-1.,2.,3.),rectangle(2.+1e-10,-1.,5.,3.)]
        for regions in (hole,gap):
            area=self.area(regions)
            self.assertFalse(area.contains([(0.,0.),(4.,0.),(4.,2.),(0.,2.)]))

    def test_partial_visibility_is_not_filled_by_the_road_union(self):
        area=self.area([rectangle(-4.,-4.,6.,4.)],[rectangle(-4.,-4.,1.,4.)])
        self.assertFalse(area.contains(vehicle_footprint(GoalPose(0.,0.,0.),3.,1.,.5)))

    def test_budget_deadline_and_unknown_clock_withdraw_clearance(self):
        body=vehicle_footprint(GoalPose(0.,0.,0.),2.,1.,.5)
        for change in ({'max_checks':1},{'valid_until_s':100.},{'clock':lambda:float('nan')}):
            area=self.area([rectangle(-4.,-4.,6.,4.)],**change)
            self.assertFalse(area.contains(body)); self.assertIn('BUDGET_OR_DEADLINE',area.reason)


class FormalParkingDecisionTests(unittest.TestCase):
    def setUp(self):
        self.source=fixture.ParkingFactsAndSceneTests(); self.source.setUp()
        self.addCleanup(self.source.doCleanups)
        self.f=self.source.f
        manager=patch('time.monotonic',self.f.clock); manager.start(); self.addCleanup(manager.stop)
        native=parking(7); native.pt=pt(21.75,-5.25); native.heading=pt(0.,1.)
        native.boundaryKnots=Vector([pt(20.,-1.75),pt(20.,-8.75),pt(23.5,-8.75),pt(23.5,-1.75)])
        catalogue=MapObservationReader(map_api(parks=[native])).catalog()
        self.f.p.parking_spaces=catalogue['parking_spaces']
        self.f.p.map_observation_status=catalogue['map_observation_status']
        self.f.p.ego.heading=0.; self.f.build()
        self.extents=(3.9,.9,.9); self.policy=ParkingPolicy(*self.extents)

    def frame(self): return self.f.behavior_frame()

    def mission(self,**changes):
        f=self.frame()
        values=dict(evidence=Evidence(f.context,f.frame_id,f.observed_at_s,f.valid_until_s,
            usable=True,coverage_verified=True,source='synthetic_task_snapshot',source_kind='task'),
            mission_id='synthetic-perpendicular-bay-task',map_digest=ManeuverFacts(self.f.p,self.f.clock).map_digest(),
            space_id=7,road_lane_ids=('1_0_-1','1_0_-2'),access_edge_index=3,body_heading_rad=math.pi/2,
            approach_pose=GoalPose(15.,1.,0.),position_pose=GoalPose(28.,1.,0.),exit_goal=GoalPose(15.,1.,0.))
        values.update(changes)
        return ParkingMission(**values)

    def read(self,mission=None,**changes):
        values=dict(perception=self.f.p,frame=self.frame(),mission=mission or self.mission(),
            vehicle_extents=self.extents,position_uncertainty_m=.1,velocity_uncertainty_mps=.1,
            geometry_max_checks=1000000,clock=self.f.clock)
        values.update(changes)
        return parking_facts(**values)

    def caps(self,actions=('PARK','PATH','REVERSE','DWELL','FEEDBACK')):
        f=self.frame(); return Capabilities(f.context,actions,f.observed_at_s,f.valid_until_s,usable=True)

    def advance(self,pose=None,speed=0.):
        self.f.now+=.05; self.f.p.frame_id+=1; self.f.p.targets_frame_id=self.f.p.frame_id
        if pose is not None:
            self.f.p.ego.x,self.f.p.ego.y,self.f.p.ego.heading=pose.x,pose.y,pose.body_heading_rad
        self.f.p.ego.speed=speed
        self.f.p.ego.vx=speed*math.cos(self.f.p.ego.heading)
        self.f.p.ego.vy=speed*math.sin(self.f.p.ego.heading)
        self.f.build()

    def evaluate(self,reply=None,capabilities=None,**read_changes):
        return self.policy.evaluate(self.frame(),self.read(**read_changes),capabilities or self.caps(),reply)

    def arrived(self,r,pose,**changes):
        values=dict(actual_pose=pose,actual_standstill_confirmed=True,goal_pose_arrived=True,progress=1.,
                    motion_direction=r.goal.motion_direction,candidate_id=r.goal.candidate_id)
        values.update(changes)
        return feedback(r,self.frame(),producer='runtime',status='ARRIVED',**values)

    def test_formal_bay_outside_aisle_selects_real_rear_axle_goal(self):
        facts=self.read(); result=self.policy.evaluate(self.frame(),facts,self.caps())
        self.assertTrue(result.request.dispatch_allowed,result.reason)
        self.assertIsInstance(facts.free_area,ParkingArea)
        self.assertEqual((21.75,-6.75,math.pi/2),(self.policy.park_goal.x,self.policy.park_goal.y,
                                               self.policy.park_goal.body_heading_rad))
        self.assertEqual(7,result.request.goal.parking_space_id)
        self.assertGreater(facts.free_area.checks,0)

    def test_missing_task_pose_map_or_access_cannot_be_inferred_from_bay(self):
        for changes in ({'approach_pose':None},{'access_edge_index':None},{'position_pose':None}):
            with self.assertRaises(ValueError): self.mission(**changes)
        with self.assertRaisesRegex(ValueError,'MISSION_MAP_MISMATCH'):
            self.read(self.mission(map_digest='a'*32))
        mission=self.mission(); mission.access_edge_index=True
        with self.assertRaises(ValueError): self.read(mission)

    def test_expired_or_diagnostic_task_source_is_not_an_authorized_mission(self):
        mission=self.mission(); mission.evidence.source_kind='diagnostic_ground_truth'
        with self.assertRaisesRegex(ValueError,'TASK_MISSION_UNVERIFIED'): self.read(mission)
        mission=self.mission(); mission.evidence.valid_until_s=self.f.now
        with self.assertRaises(ValueError): self.read(mission)

    def test_short_original_source_deadline_is_preserved(self):
        mission=self.mission(); mission.evidence.valid_until_s=self.f.now+.03
        facts=self.read(mission)
        self.assertEqual(self.f.now+.03,facts.evidence.valid_until_s)
        self.f.now+=.04
        self.assertFalse(facts.free_area.contains(vehicle_footprint(GoalPose(10.,0.,0.),*self.extents)))

    def test_unknown_coverage_and_unmatched_vehicle_cannot_dispatch(self):
        facts=self.read(geometry_max_checks=1)
        self.assertFalse(facts.front_coverage_verified); self.assertFalse(facts.rear_coverage_verified)
        r=self.policy.evaluate(self.frame(),facts,self.caps())
        self.assertFalse(r.request is not None and r.request.dispatch_allowed)
        r=self.policy.evaluate(self.frame(),self.read(vehicle_extents=(3.5,.9,.9)),self.caps())
        self.assertEqual('PARKING_VEHICLE_GEOMETRY_MISMATCH',r.reason)
        self.f.p.maneuver_environment.coverage_regions=[]
        with self.assertRaises(ValueError): self.read()

    def test_selected_mission_map_goal_and_access_remain_stable(self):
        first=self.evaluate(); identity=first.request.intent_id
        for changes in ({'mission_id':'other'},{'access_edge_index':0},{'exit_goal':GoalPose(14.,1.,0.)}):
            self.advance()
            r=self.evaluate(mission=self.mission(**changes))
            self.assertEqual(identity,r.request.intent_id)
            self.assertFalse(r.request.dispatch_allowed)
            self.assertEqual('PARKING_ACTIVE_MISSION_OR_MAP_CHANGED',r.reason)

    def test_new_stage_reticks_new_capabilities_and_does_not_replay_old_feedback(self):
        r=self.evaluate().request
        self.advance(self.mission().approach_pose)
        result=self.evaluate(self.arrived(r,self.mission().approach_pose))
        self.assertEqual('POSITION',result.request.stage)
        self.assertTrue(result.request.dispatch_allowed,result.reason)
        self.assertIsNone(self.policy.session.last_feedback)
        self.advance(self.mission().position_pose)
        result=self.evaluate(self.arrived(result.request,self.mission().position_pose),
                             self.caps(('PARK','PATH','FEEDBACK')))
        self.assertEqual('REVERSE_ENTRY',result.request.stage)
        self.assertFalse(result.request.dispatch_allowed)
        self.assertTrue(result.hold_required)

    def test_feedback_at_goal_does_not_override_current_gps_or_speed(self):
        r=self.evaluate().request
        self.advance()
        result=self.evaluate(self.arrived(r,self.mission().approach_pose))
        self.assertEqual('APPROACH',result.request.stage)
        self.advance(self.mission().approach_pose,speed=.2)
        result=self.evaluate(self.arrived(result.request,self.mission().approach_pose))
        self.assertEqual('APPROACH',result.request.stage)

    def test_formal_policy_runs_original_stages_and_cannot_shorten_ten_second_dwell(self):
        r=self.evaluate().request; identity=r.intent_id
        for goal,expected in ((self.mission().approach_pose,'POSITION'),
                              (self.mission().position_pose,'REVERSE_ENTRY')):
            self.advance(goal); result=self.evaluate(self.arrived(r,goal)); r=result.request
            self.assertEqual(expected,r.stage); self.assertTrue(r.dispatch_allowed,result.reason)
        goal=self.policy.park_goal
        self.advance(goal); r=self.evaluate(self.arrived(r,goal)).request
        self.assertEqual('PARKED_DWELL',r.stage); obligation=r.goal.stop_obligation_id
        self.advance(goal)
        r=self.evaluate(self.arrived(r,goal,hold_completed=True,standstill_duration_s=9.9)).request
        self.assertEqual('PARKED_DWELL',r.stage)
        self.advance(goal)
        r=self.evaluate(self.arrived(r,goal,hold_completed=True,standstill_duration_s=10.)).request
        self.assertEqual('EXIT_PREPARE',r.stage)
        self.advance(goal); result=self.evaluate(self.arrived(r,goal)); r=result.request
        self.assertEqual('EXIT',r.stage); self.assertEqual(1,r.goal.motion_direction)
        self.assertFalse(result.hold_required)
        self.advance(self.mission().exit_goal)
        r=self.evaluate(self.arrived(r,self.mission().exit_goal)).request
        self.assertEqual('COMPLETED',r.status); self.assertEqual(identity,r.intent_id)
        self.assertEqual('parking:7',obligation)

    def test_new_formal_object_intruding_on_current_body_blocks_existing_intent(self):
        first=self.evaluate()
        self.f.p.targets=[target(self.f.p.ego.x,self.f.p.ego.y,heading=None)]
        self.advance()
        result=self.evaluate()
        self.assertEqual(first.request.intent_id,result.request.intent_id)
        self.assertFalse(result.request.dispatch_allowed); self.assertTrue(result.hold_required)

    def test_formal_selected_goal_feeds_existing_reverse_generator_outside_aisle(self):
        self.evaluate(); goal=self.policy.park_goal; mission=self.mission()
        self.advance(mission.position_pose)
        result=self.source.plan(goal_pose=(goal.x,goal.y,goal.body_heading_rad),
                                road_lane_ids=mission.road_lane_ids,access_edge_index=mission.access_edge_index)
        self.assertEqual('safe',result['status'],result)
        self.assertEqual((goal.x,goal.y,goal.body_heading_rad),result['goal_pose'])
        self.assertEqual(-1,result['motion_direction']); self.assertEqual(0.,result['points'][-1].speed)

    def test_body_occupancy_advances_original_object_age_and_explicit_error(self):
        facts=self.read(); actual=GoalPose(10.,0.,0.)
        facts.objects=(MotionObject(9,20.,0.,-50.,0.,1.,1.,None),)
        facts.position_uncertainty_m=facts.velocity_uncertainty_mps=0.
        self.assertTrue(self.policy._body_clear(actual,facts))
        facts.object_age_s=.2
        self.assertFalse(self.policy._body_clear(actual,facts))

    def test_mutable_public_goal_cannot_replace_the_frozen_stage_pose(self):
        result=self.evaluate(); expected=self.mission().approach_pose.to_dict()
        result.request.goal.goal_pose.x+=1.
        self.advance(); blocked=self.evaluate()
        self.assertFalse(blocked.request.dispatch_allowed)
        self.assertEqual('PARKING_ACTIVE_STAGE_GOAL_CHANGED',blocked.reason)
        self.assertEqual(expected,blocked.request.goal.goal_pose.to_dict())

    def test_changed_error_model_cannot_silently_reuse_the_selected_mission(self):
        self.evaluate(); self.advance()
        result=self.evaluate(position_uncertainty_m=0.)
        self.assertEqual('PARKING_ACTIVE_MISSION_OR_MAP_CHANGED',result.reason)
        self.assertFalse(result.request.dispatch_allowed)

    def test_unknown_mutated_stage_withdraws_without_losing_bound_goal(self):
        initial=self.evaluate(); goal=initial.request.goal.goal_pose.to_dict()
        self.policy.session.stage='UNKNOWN_EXTENSION'
        self.advance(); result=self.evaluate()
        self.assertFalse(result.request.dispatch_allowed)
        self.assertEqual(goal,result.request.goal.goal_pose.to_dict())
        self.advance(); repeated=self.evaluate()
        self.assertFalse(repeated.request.dispatch_allowed)
        self.assertEqual(0.,repeated.request.goal.speed_cap_mps)


if __name__=='__main__': unittest.main()
