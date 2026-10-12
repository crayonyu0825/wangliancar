"""Geometry and recovery checks for the registered production lane generator."""
import math
import unittest

from core.road_region import RoadRegion
from core.behavior_protocol import GoalPose,behavior_frame
from core.behavior_environment import vehicle_footprint
import tests.test_joint_behavior_m1 as m1
from members import decision_stub,planning_stub


class JointM2Tests(unittest.TestCase):
    setUp=m1.JointM1Tests.setUp
    sample=m1.JointM1Tests.sample
    step=m1.JointM1Tests.step
    def curved_sample(self,index):
        from core.geometry import project_polyline
        p=m1.JointM1Tests.sample(self,index)
        radius=500.0
        def bend(s,offset):
            angle=s/radius
            return ((radius-offset)*math.sin(angle),radius-(radius-offset)*math.cos(angle))
        for lane in p.behavior_environment.lanes:
            offset=lane.center_line[0][1]
            lane.center_line=[bend(s,offset) for s in range(-80,301,5)]
            lane.left_boundary=[bend(s,offset+1.75) for s in range(-80,301,5)]
            lane.right_boundary=[bend(s,offset-1.75) for s in range(-80,301,5)]
        chosen=min(p.behavior_environment.lanes,key=lambda lane:project_polyline(lane.center_line,p.ego.x,p.ego.y)['distance'])
        for lane in p.behavior_environment.lanes:
            lane.side='current' if lane is chosen else 'left' if lane.lane_id=='lane-2' else 'right'
        p.lane.lane_id=chosen.lane_id;p.lane.center_line=chosen.center_line
        p.lane.left_boundary=chosen.left_boundary;p.lane.right_boundary=chosen.right_boundary
        p.behavior_environment.coverage.regions=([(-95,-20),(350,-20),(350,150),(-95,150)],)
        p.behavior_environment.road_regions=(RoadRegion(
            [bend(s,-1.75) for s in range(-70,291,5)]+
            [bend(s,5.25) for s in reversed(range(-70,291,5))]),)
        target=p.targets[0]
        target.x,target.y=bend(self.lead_x,0)
        target.vx,target.vy=1.4*math.cos(self.lead_x/radius),1.4*math.sin(self.lead_x/radius)
        target.heading=self.lead_x/radius
        target.lane_id='lane-1';target.same_lane=chosen.lane_id=='lane-1'
        target.longitudinal_distance=project_polyline(p.lane.center_line,target.x,target.y)['s']-project_polyline(p.lane.center_line,p.ego.x,p.ego.y)['s']
        return p

    def test_real_entries_complete_curved_overtake_and_return(self):
        self.sample=self.curved_sample
        moved=False
        for index in range(1,2201):
            p,d,t,c=self.step(index)
            moved=moved or decision_stub.decision_behavior_info()['runtime_behavior']['phase']=='PASS'
            if moved and decision_stub.decision_behavior_info()['runtime_behavior']['phase']=='COMPLETE':break
        self.assertTrue(moved,self.records[-20:])
        self.assertEqual('COMPLETE',decision_stub.decision_behavior_info()['runtime_behavior']['phase'],self.records[-20:])
        from core.geometry import project_polyline
        road=p.behavior_environment.lane('lane-1')
        self.assertLess(project_polyline(road.center_line,self.plant.x,self.plant.y)['distance'],0.5)

    def test_real_entries_bypass_a_static_obstruction(self):
        original=self.sample
        def static_sample(index):
            p=original(index)
            t=p.targets[0];t.x,t.vx=45.0,0.0;t.longitudinal_distance=45-self.plant.x
            return p
        self.sample=static_sample
        seen=False
        for i in range(1,1401):
            p,d,t,c=self.step(i)
            seen=seen or d.behavior_request is not None and d.behavior_request.maneuver=='AVOID'
            if seen and decision_stub.decision_behavior_info()['runtime_behavior']['phase']=='COMPLETE':break
        self.assertTrue(seen,self.records[-20:])
        self.assertGreater(self.plant.x-0.9,47.5)
        self.assertEqual('COMPLETE',decision_stub.decision_behavior_info()['runtime_behavior']['phase'])

    def test_task_required_merge_has_actual_whole_body_completion(self):
        original=self.sample
        def merge_sample(index):
            p=original(index);p.targets=[];p.behavior_environment.desired_lane_id='lane-2'
            return p
        self.sample=merge_sample
        for i in range(1,1001):
            p,d,t,c=self.step(i)
            if decision_stub.decision_behavior_info()['runtime_behavior']['phase']=='COMPLETE':break
        self.assertEqual('COMPLETE',decision_stub.decision_behavior_info()['runtime_behavior']['phase'],self.records[-20:])
        self.assertGreater(self.plant.y,2.8)
        self.assertEqual('lane-2',p.lane.lane_id)

    def test_danger_disappearance_needs_covered_clear_frames_not_history_expiry(self):
        p=self.sample(1);p.targets[0].x=6.0;p.targets[0].vx=0.0
        self.exchange.prepare(p,planning_stub.planning_capabilities(p));decision_stub.decide(p)
        for i in range(2,30):
            self.now+=0.05;p=self.sample(i);p.targets=[]
            p.behavior_environment.coverage.evidence.coverage_verified=False
            self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
            self.assertIn(d.mode,('STOP','EMERGENCY_BRAKE'))
        for i in (30,31,32):
            self.now+=0.05;p=self.sample(i);p.targets=[]
            self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
        self.assertGreater(d.target_speed,0)

    def test_changed_purpose_before_crossing_cancels_without_lateral_execution(self):
        for i in range(1,120):
            p,d,t,c=self.step(i)
            if d.behavior_request is not None and d.behavior_request.authorization=='PREPARE':break
        p=self.sample(i+1)
        p.targets[0].vx=5.0
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p)
        self.assertIsNone(d.behavior_request)
        self.assertEqual('IDLE',decision_stub.decision_behavior_info()['runtime_behavior']['phase'])
        self.assertLess(abs(self.plant.y),0.01)

    def test_missing_execution_timing_cannot_authorize_a_dynamic_candidate(self):
        for i in range(1,140):
            p=self.sample(i)
            p.behavior_environment.coverage.timing_evidence=None
            self.exchange.prepare(p,planning_stub.planning_capabilities(p))
            d=decision_stub.decide(p)
            out=planning_stub.plan(p,d)
            if d.behavior_request is not None and d.behavior_request.authorization=='EVALUATE_ONLY':
                self.assertEqual('REJECTED',out.planning_feedback[0].status)
                self.assertIn('timing',out.planning_feedback[0].reason_code)
                self.assertIsNone(out.execution_path_id)
                return
            # Same successful model sender is used only for preparation lamps.
            from members import control_stub
            c=control_stub.compute_control(p,out)
            self.exchange.finish(p,out,c,{'attempted':True,'ok':True})
            self.now+=0.05
        self.fail('expected an explicit evaluation timing rejection')

    def test_temporal_error_sweep_preserves_a_late_crossing_hazard(self):
        from members.planning.candidate_validation import ObstaclePrediction,_obstacle_gap
        body=vehicle_footprint(GoalPose(0,0,0),1,1,0.5)
        nominal=ObstaclePrediction('cross',0,3,0.5,0.5,0,0,-3,5,0,0)
        uncertain=ObstaclePrediction('cross',0,3,0.5,0.5,0,0,-3,5,0,0,0,1)
        self.assertGreater(_obstacle_gap(body,nominal,0),0)
        self.assertLessEqual(_obstacle_gap(body,uncertain,0),0)

    def test_nonconvex_road_does_not_fill_a_forbidden_notch(self):
        road=RoadRegion([(-5,-3),(5,-3),(5,3),(1,3),(1,0),(-1,0),(-1,3),(-5,3)])
        self.assertGreater(road.body_gap([(-4,-2),(-2,-2),(-2,-1),(-4,-1)]),0)
        self.assertEqual(0,road.body_gap([(-2,1),(2,1),(2,2),(-2,2)]))
        with self.assertRaises(ValueError):
            RoadRegion([(0,0),(3,3),(0,3),(3,0)])

    def test_crossing_permission_withdrawal_rejects_the_active_candidate(self):
        for i in range(1,150):
            p,d,t,c=self.step(i)
            if t.execution_path_id is not None:
                break
        self.assertIsNotNone(t.execution_path_id)
        p=self.sample(i+1)
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p)
        p.behavior_environment.lane(d.behavior_request.goal.target_lane_id).marking='SOLID'
        out=planning_stub.plan(p,d)
        self.assertIsNone(out.execution_path_id)
        self.assertEqual('REJECTED',out.planning_feedback[0].status)
        self.assertIn('crossing authority',out.planning_feedback[0].reason_code)

    def test_lost_environment_after_authorization_is_braking_not_recentering(self):
        for i in range(1,150):
            p,d,t,c=self.step(i)
            if t.execution_path_id is not None:break
        p=self.sample(i+1)
        p.behavior_environment=None
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p)
        self.assertEqual('EMERGENCY_BRAKE',d.mode)
        self.assertEqual(0,d.target_speed)
        self.assertIsNone(d.behavior_request)

    def test_curved_verified_road_and_lane_transition_are_computed_from_geometry(self):
        # An actual curved source/target corridor, transformed from the model
        # station coordinates, exercises the production quintic generator.
        p=self.sample(1)
        radius=500.0
        def bend(point):
            station,offset=point
            angle=station/radius
            return ((radius-offset)*math.sin(angle),radius-(radius-offset)*math.cos(angle))
        for lane in p.behavior_environment.lanes:
            offset=lane.center_line[0][1]
            lane.center_line=[bend((s,offset)) for s in range(-80,301,5)]
            lane.left_boundary=[bend((s,offset+1.75)) for s in range(-80,301,5)]
            lane.right_boundary=[bend((s,offset-1.75)) for s in range(-80,301,5)]
        origin=p.behavior_environment.lane('lane-1')
        p.lane.center_line=origin.center_line
        p.behavior_environment.road_regions=(RoadRegion(
            [bend((s,-1.75)) for s in range(-70,291,5)]+
            [bend((s,5.25)) for s in reversed(range(-70,291,5))]),)
        from core.behavior_protocol import RuntimeRequest,BehaviorGoal,LightIntent
        f=behavior_frame(p,self.now)
        req=RuntimeRequest(f,'curve-evaluation','OVERTAKE','EVALUATE_OUT',1,1,
            BehaviorGoal(target_lane_id='lane-2',speed_cap_mps=5,
                         light_intent=LightIntent(left_signal=True)),origin_lane_id='lane-1',target_id=1)
        p.targets[0].x,p.targets[0].y=bend((19,0))
        p.targets[0].vx,p.targets[0].vy=1.4*math.cos(19/radius),1.4*math.sin(19/radius)
        p.targets[0].heading=19/radius
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);d.behavior_request=req
        out=planning_stub.plan(p,d)
        self.assertEqual('PLANNED',out.planning_feedback[0].status,out.planning_feedback[0].reason_code)
        self.assertIsNone(out.execution_path_id)


if __name__=='__main__':unittest.main()
