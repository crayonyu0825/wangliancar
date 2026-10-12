"""Real parking decision/search/control with explicit bidirectional plant."""
import math
import unittest
import tests.test_joint_behavior_m1 as m1

from core.behavior_protocol import GoalPose
from core.parking_protocol import ParkingBay,ParkingTask
from core.behavior_environment import Evidence
from core.validation import validate_output
from core.interfaces import DecisionTarget,Trajectory,ControlOut
from members import decision_stub,planning_stub,control_stub
from tests.test_control_maneuvers import BidirectionalPlant
from tests.control_benchmark import vehicle


class JointM3Tests(unittest.TestCase):
    setUp=m1.JointM1Tests.setUp

    def sample(self,index,vertical=False):
        p=m1.JointM1Tests.sample(self,index)
        p.targets=[]
        p.scene_id=7
        p.ego.vx=self.plant.direction*self.plant.speed*math.cos(self.plant.heading)
        p.ego.vy=self.plant.direction*self.plant.speed*math.sin(self.plant.heading)
        from core.behavior_protocol import TaskContext
        context=TaskContext(p.case_id,p.task_id,p.scene_id,p.session_id)
        proof=Evidence(context,index,self.now,self.now+0.2,True,True,'explicit_model_park',source_kind='synthetic')
        for lane in p.behavior_environment.lanes:lane.evidence=proof
        p.behavior_environment.coverage.evidence=proof
        p.behavior_environment.coverage.timing_evidence=proof
        p.behavior_environment.task_evidence=proof
        boundary=[(8.5,7),(11.5,7),(11.5,13),(8.5,13)] if vertical else [(7,-1.5),(13,-1.5),(13,1.5),(7,1.5)]
        heading=math.pi/2 if vertical else 0.0
        bay=ParkingBay(1,boundary,heading,'empty',proof)
        dims=dict(front_offset_m=3.5,rear_offset_m=0.9,half_width_m=0.9)
        body=self.plant.x,self.plant.y,self.plant.heading
        from core.behavior_environment import vehicle_footprint
        if bay.region.body_gap(vehicle_footprint(GoalPose(*body),3.5,0.9,0.9))>0:
            bay.occupancy='self'
        exit_goal=GoalPose(10,23,math.pi/2) if vertical else GoalPose(25,0,0)
        p.behavior_environment.parking_task=ParkingTask('parking-model',proof,[bay],
            [[(-25,-25),(40,-25),(40,40),(-25,40)]],exit_goal,10.0,0)
        return p

    def step(self,index,vertical=False):
        p=self.sample(index,vertical)
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);self.assertTrue(d.valid,d.reason)
        validate_output(d,DecisionTarget,p)
        t=planning_stub.plan(p,d)
        self.assertTrue(t.valid,(index,t.reason,[v.reason_code for v in t.planning_feedback]))
        validate_output(t,Trajectory,d)
        c=control_stub.compute_control(p,t)
        self.assertTrue(c.valid,(index,c.errors,t.reason))
        validate_output(c,ControlOut,t)
        self.exchange.finish(p,t,c,{'attempted':True,'ok':True})
        state=decision_stub.decision_behavior_info()['parking_runtime']
        self.records.append((index,round(self.plant.x,2),round(self.plant.y,2),round(self.plant.heading,2),
                             round(self.plant.speed,2),state['phase'],state['segment_index'],
                             [v.reason_code for v in t.planning_feedback]))
        self.plant.step(c,0.05,vehicle());self.now+=0.05
        return p,d,t,c,state

    def test_green_cannot_release_unfinished_actual_signal_dwell(self):
        from tests.test_decision import add_red_light
        self.plant=BidirectionalPlant(speed=1.5)
        green=False;stationary=0.0;released=False
        for i in range(1,1201):
            p=m1.JointM1Tests.sample(self,i);p.targets=[]
            env=p.behavior_environment
            env.stop_rule_id='model-rule';env.rule_stop_duration_s=5.0;env.rule_stop_strict=True
            add_red_light(p,20-self.plant.x);p.traffic.signal_id=42
            p.source_status['traffic']={'usable':True,'association_valid':True,'read_ok':True}
            if stationary>1.0:green=True;p.traffic.signal_state='GREEN'
            self.exchange.prepare(p,planning_stub.planning_capabilities(p))
            d=decision_stub.decide(p);self.assertTrue(d.valid,d.reason)
            t=planning_stub.plan(p,d);self.assertTrue(t.valid,t.reason)
            c=control_stub.compute_control(p,t);self.assertTrue(c.valid,c.errors)
            self.exchange.finish(p,t,c,{'attempted':True,'ok':True})
            at_stop=abs(self.plant.x-16.2)<0.15 and self.plant.speed<0.05
            if at_stop:stationary+=0.05
            if green and stationary<5.0:self.assertEqual(0,c.throttle)
            self.plant.step(c,0.05,vehicle());self.now+=0.05
            if green and self.plant.x>17:
                released=True;break
        self.assertTrue(released)
        self.assertGreater(stationary,5.0)

    def test_unknown_or_other_bay_never_authorizes_reverse(self):
        self.plant=BidirectionalPlant(x=15)
        for occupancy in ('unknown','other','self'):
            p=self.sample(1)
            p.behavior_environment.parking_task.bays[0].occupancy=occupancy
            p.behavior_environment.parking_task.ego_target_id=None
            self.exchange.prepare(p,planning_stub.planning_capabilities(p))
            d=decision_stub.decide(p);t=planning_stub.plan(p,d)
            self.assertNotEqual(-1,t.motion_direction)
            self.assertIsNone(t.execution_path_id)

    def test_rear_intrusion_rejects_selected_reverse_candidate(self):
        self.plant=BidirectionalPlant(x=15)
        self.step(1)
        p=self.sample(2);self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p)
        self.assertEqual('EXECUTE_SELECTED',d.behavior_request.authorization)
        from tests.test_decision import add_target
        target=add_target(p,-3,0,lane_id='lane-1')
        target.id=77;target.x,target.y=12,0
        t=planning_stub.plan(p,d)
        self.assertIsNone(t.execution_path_id)
        self.assertEqual('REJECTED',t.planning_feedback[0].status)

    def test_unexpected_reverse_and_untrusted_self_identity_protect(self):
        self.plant=BidirectionalPlant(x=15,speed=0.3);self.plant.direction=-1
        p=self.sample(1);self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p)
        self.assertEqual('EMERGENCY_BRAKE',d.mode)
        self.assertIn('UNAUTHORIZED',d.reason)
        from tests.test_decision import add_target
        self.plant.speed=0
        p=self.sample(2)
        target=add_target(p,20,0,lane_id='lane-1');target.id=0
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);t=planning_stub.plan(p,d)
        self.assertEqual('REJECTED',t.planning_feedback[0].status)
        self.assertIn('ego target identity',t.planning_feedback[0].reason_code)

    def test_frame_rollback_requires_new_parking_evaluation(self):
        self.plant=BidirectionalPlant(x=15)
        self.step(10);self.step(11)
        p=self.sample(1);self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);self.assertTrue(d.valid,d.reason)
        self.assertTrue(d.behavior_request is None or d.behavior_request.authorization!='EXECUTE_SELECTED')
        self.now+=0.05
        p=self.sample(2);self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);self.assertTrue(d.valid,d.reason)
        self.assertEqual('EVALUATE_ONLY',d.behavior_request.authorization)
        t=planning_stub.plan(p,d);self.assertTrue(t.valid,t.reason)
        self.assertEqual('PLANNED',t.planning_feedback[0].status)
        self.assertEqual(2,t.planning_feedback[0].candidate_reference.source_frame_id)

    def test_vehicle_that_does_not_fit_is_not_selected(self):
        from core.road_region import RoadRegion
        self.plant=BidirectionalPlant(x=15)
        p=self.sample(1)
        p.behavior_environment.parking_task.bays[0].region=RoadRegion([(7,-.5),(13,-.5),(13,.5),(7,.5)])
        self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
        self.assertIsNone(d.behavior_request)
        self.assertEqual('NO_OBSERVED_FITTING_EMPTY_BAY',decision_stub.decision_behavior_info()['parking_runtime']['reason_code'])

    def test_parking_interrupted_by_emergency_is_researched_and_completed(self):
        from core.emergency_protocol import EmergencyTask
        self.plant=BidirectionalPlant(x=15);original_sample=self.sample;seen=False;resumed=False
        def sample(index,vertical=False):
            p=original_sample(index,vertical)
            if index>=30:
                p.behavior_environment.emergency_task=EmergencyTask('parking-interruption',p.behavior_environment.task_evidence,
                    index<55,[[(-25,-25),(40,-25),(40,40),(-25,40)]],(0.0,),'lane-1')
            return p
        self.sample=sample
        for i in range(1,2401):
            p,d,t,c,state=self.step(i)
            info=decision_stub.decision_behavior_info()['emergency_runtime']
            seen=seen or info['phase'] not in ('IDLE','COMPLETE')
            if i>55 and seen and d.behavior_request is not None and d.behavior_request.maneuver=='PARK':
                resumed=resumed or d.behavior_request.authorization=='EVALUATE_ONLY'
            if state['phase']=='COMPLETE':break
        self.assertTrue(seen);self.assertTrue(resumed,self.records[-15:])
        self.assertEqual('COMPLETE',state['phase'],self.records[-15:])
        self.assertAlmostEqual(25,self.plant.x,delta=.2)

    def test_real_parallel_parking_dwell_and_exit(self):
        self.plant=BidirectionalPlant(x=15)
        stopped_duration=0.0;entered=False
        for i in range(1,2201):
            p,d,t,c,state=self.step(i)
            parked=(abs(self.plant.x-8.7)<0.15 and abs(self.plant.y)<0.15 and self.plant.speed<0.05)
            entered=entered or parked
            if parked:stopped_duration+=0.05
            if state['phase']=='COMPLETE':break
        self.assertTrue(entered,self.records[-25:])
        self.assertGreaterEqual(stopped_duration,10.0)
        self.assertEqual('COMPLETE',state['phase'],self.records[-25:])
        self.assertAlmostEqual(25,self.plant.x,delta=0.2)
        self.assertLess(self.plant.speed,0.05)

    def test_real_vertical_parking_uses_computed_segments_and_exit(self):
        self.plant=BidirectionalPlant()
        directions=set();stopped_duration=0.0
        for i in range(1,3001):
            p,d,t,c,state=self.step(i,vertical=True)
            directions.add(t.motion_direction)
            parked=(math.hypot(self.plant.x-10,self.plant.y-8.7)<0.15 and self.plant.speed<0.05)
            if parked:stopped_duration+=0.05
            if state['phase']=='COMPLETE':break
        self.assertEqual({1,-1},directions)
        self.assertGreaterEqual(stopped_duration,10.0)
        self.assertEqual('COMPLETE',state['phase'],self.records[-25:])
        self.assertLess(math.hypot(self.plant.x-10,self.plant.y-23),0.2)


if __name__=='__main__':unittest.main()
