"""Actual destination, obligation and finite progress through fixed entries."""
import unittest
import tests.test_joint_behavior_m1 as m1
from core.task_protocol import TaskRouteFacts
from core.behavior_protocol import GoalPose
from members import decision_stub,planning_stub,control_stub
from tests.control_benchmark import Plant,vehicle


class JointTaskTests(unittest.TestCase):
    setUp=m1.JointM1Tests.setUp

    def sample(self,index,reason='task_destination',obligations=()):
        p=m1.JointM1Tests.sample(self,index);p.targets=[]
        p.behavior_environment.task_progress=TaskRouteFacts(p.behavior_environment.task_evidence,'selected-model-route',
            [('segment-a','lane-1',0,25)],'segment-a',min(25,self.plant.x),reason,
            [(23,-1.5),(30,-1.5),(30,1.5),(23,1.5)],GoalPose(25,0,0),2.0,True,obligations)
        return p

    def step(self,index,**kwargs):
        p=self.sample(index,**kwargs);self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);self.assertTrue(d.valid,d.reason)
        t=planning_stub.plan(p,d);self.assertTrue(t.valid,t.reason)
        c=control_stub.compute_control(p,t);self.assertTrue(c.valid,c.errors)
        self.exchange.finish(p,t,c,{'attempted':True,'ok':True})
        state=decision_stub.decision_behavior_info()['task_runtime']
        self.plant.step(c,.05,vehicle());self.now+=.05
        return p,d,t,c,state

    def test_actual_destination_stop_completes(self):
        self.plant=Plant(speed=1)
        for i in range(1,900):
            p,d,t,c,state=self.step(i)
            if state['phase']=='COMPLETE':break
        self.assertEqual('COMPLETE',state['phase'])
        self.assertAlmostEqual(25,self.plant.x,delta=.15)
        self.assertLess(self.plant.speed,.05)

    def test_unfinished_obligation_and_window_end_cannot_complete(self):
        self.plant=Plant(x=25,speed=0)
        for i in range(1,12):
            _,_,_,_,state=self.step(i,obligations=('signal:42',))
            self.assertNotEqual('COMPLETE',state['phase'])
        p=self.sample(12,reason='window_end');self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p)
        self.assertEqual(0,d.target_speed)
        self.assertNotEqual('COMPLETE',decision_stub.decision_behavior_info()['task_runtime']['phase'])

    def test_finite_reference_stops_whole_body_and_never_finishes_task(self):
        self.plant=Plant(speed=1)
        for i in range(1,600):
            p,d,t,c,state=self.step(i,reason='map_end')
            self.assertNotEqual('COMPLETE',state['phase'])
        self.assertLessEqual(self.plant.x+3.5,25.05)
        self.assertLess(self.plant.speed,.05)

    def test_return_leg_jump_is_rejected(self):
        self.plant=Plant(speed=1)
        self.step(1)
        p=self.sample(2);p.behavior_environment.task_progress.progress_s_m=20
        self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
        self.assertEqual('EMERGENCY_BRAKE',d.mode)
        self.assertIn('UNREACHABLE',decision_stub.decision_behavior_info()['task_runtime']['reason_code'])


if __name__=='__main__':unittest.main()
