"""Real emergency choice, controller and independent supervisor in model world."""
import unittest
import math
from types import SimpleNamespace
import tests.test_joint_behavior_m1 as m1
from core.emergency_protocol import EmergencyTask
from core.safety_supervisor import SafetySupervisor
from core.validation import validate_output
from core.interfaces import Trajectory,ControlOut,DecisionTarget
from members import decision_stub,planning_stub,control_stub
from tests.control_benchmark import Plant,vehicle
from tests.test_decision import add_target


class JointM5Tests(unittest.TestCase):
    setUp=m1.JointM1Tests.setUp

    def sample(self,index,danger=True,offsets=(0,1.5,-1.5),both_blocked=False):
        p=m1.JointM1Tests.sample(self,index);p.targets=[]
        proof=p.behavior_environment.task_evidence
        p.behavior_environment.emergency_task=EmergencyTask('observed-hazard',proof,danger,
            [[(-80,-1.7),(380,-1.7),(380,5.2),(-80,5.2)]],offsets,'lane-1')
        if danger:
            target=add_target(p,11.5-self.plant.x,0,lane_id='lane-1')
            target.id=77;target.x,target.y,target.length,target.width,target.heading=11.5,0,0.5,0.4,0.0
            if both_blocked:
                target.length,target.width=4,12
        return p

    def step(self,index,**kwargs):
        p=self.sample(index,**kwargs);self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);self.assertTrue(d.valid,d.reason);validate_output(d,DecisionTarget,p)
        t=planning_stub.plan(p,d);self.assertTrue(t.valid,t.reason);validate_output(t,Trajectory,d)
        c=control_stub.compute_control(p,t);self.assertTrue(c.valid,c.errors);validate_output(c,ControlOut,t)
        assessment=self.supervisor.evaluate(p,d,t,c)
        actual=assessment.control or c
        self.exchange.finish(p,t,actual,{'attempted':True,'ok':True},assessment.active)
        state=decision_stub.decision_behavior_info()['emergency_runtime']
        self.records.append((index,round(self.plant.x,3),round(self.plant.y,3),round(self.plant.speed,3),state,
                             [v.reason_code for v in t.planning_feedback],actual.steering,actual.brake))
        self.plant.step(actual,.05,vehicle());self.now+=.05
        return p,d,t,c,assessment,state

    def supervisor_setup(self):
        settings=planning_stub._settings_with_control_geometry()
        config=SimpleNamespace(send_control=True,safety_recovery_frames=1)
        self.supervisor=SafetySupervisor(config,planning_settings=settings,decision_settings=decision_stub.decision_settings())

    def test_first_cycle_brakes_and_verified_combined_path_can_steer(self):
        self.plant=Plant(speed=5);self.supervisor_setup();turned=False
        for i in range(1,31):
            p,d,t,c,a,state=self.step(i,offsets=(0,1.5))
            self.assertGreater(a.control.brake if a.control else c.brake,0)
            self.assertEqual(0,a.control.throttle if a.control else c.throttle)
            if i==1:self.assertEqual('EVALUATE_ONLY',d.behavior_request.authorization)
            if t.emergency_maneuver_authorized:
                self.assertEqual('BRAKE_LEFT',state['selected'])
                self.assertEqual('emergency_stop',a.mode)
                turned=turned or abs(a.control.steering)>0.005
        self.assertTrue(turned,self.records)
        self.assertLess(self.plant.x+3.5,11.25)
        self.assertLess(self.plant.speed,0.05)

    def test_actual_stopped_clear_hazard_requires_new_route_recovery_authority(self):
        self.plant=Plant(speed=5);self.supervisor_setup();recovery=False
        for i in range(1,600):
            p,d,t,c,a,state=self.step(i,danger=i<35,offsets=(0,1.5))
            if state['phase']=='RECOVER_EVAL' and d.behavior_request is not None:
                self.assertEqual('EVALUATE_ONLY',d.behavior_request.authorization)
                self.assertIsNone(t.execution_path_id)
                recovery=True
            if state['phase']=='COMPLETE':break
        self.assertTrue(recovery,self.records[-12:])
        self.assertEqual('COMPLETE',state['phase'],self.records[-12:])

    def test_displaced_actual_stop_rejoins_route_before_completion(self):
        self.plant=Plant(y=0.5,speed=5);self.supervisor_setup();moved=False
        for i in range(1,1101):
            p,d,t,c,a,state=self.step(i,danger=i<35,offsets=(0,1.5))
            moved=moved or state['phase']=='RECOVER_EXEC' and self.plant.speed>0.2
            if state['phase']=='COMPLETE':break
        self.assertTrue(moved,self.records[-15:])
        self.assertEqual('COMPLETE',state['phase'],self.records[-15:])
        self.assertLess(abs(self.plant.y),0.15)

    def test_all_candidates_blocked_keep_immediate_brake(self):
        self.plant=Plant(speed=5);self.supervisor_setup()
        for i in range(1,10):
            _,_,t,c,a,_=self.step(i,both_blocked=True)
            self.assertFalse(t.emergency_maneuver_authorized)
            output=a.control or c
            self.assertEqual(0,output.throttle);self.assertEqual(0,output.steering);self.assertGreater(output.brake,0)

    def test_invalid_target_source_cannot_keep_verified_emergency_steering(self):
        self.plant=Plant(speed=5);self.supervisor_setup()
        self.step(1,offsets=(0,1.5))
        p=self.sample(2,offsets=(0,1.5));self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);t=planning_stub.plan(p,d);c=control_stub.compute_control(p,t)
        self.assertTrue(t.emergency_maneuver_authorized)
        p.targets_valid=False
        a=self.supervisor.evaluate(p,d,t,c)
        self.assertEqual(0,a.control.steering)
        self.assertGreater(a.control.brake,0)

    def test_wrong_danger_identity_and_new_path_obstacle_remove_steering_authority(self):
        self.plant=Plant(speed=5);self.supervisor_setup()
        self.step(1,offsets=(0,1.5))
        p=self.sample(2,offsets=(0,1.5));self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);self.assertEqual('EXECUTE_SELECTED',d.behavior_request.authorization)
        p.behavior_environment.emergency_task.danger_id='changed'
        t=planning_stub.plan(p,d);self.assertFalse(t.emergency_maneuver_authorized)
        self.assertEqual('REJECTED',t.planning_feedback[0].status)


if __name__=='__main__':unittest.main()
