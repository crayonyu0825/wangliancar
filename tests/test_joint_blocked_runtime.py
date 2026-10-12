"""Unresolved road obstruction warning through the actual entry points."""
import unittest
import tests.test_joint_behavior_m1 as m1
from members import decision_stub,planning_stub,control_stub
from tests.control_benchmark import Plant,vehicle


class JointBlockedTests(unittest.TestCase):
    setUp=m1.JointM1Tests.setUp

    def test_real_obstruction_keeps_warning_while_stopped_and_clears_with_coverage(self):
        self.plant=Plant(speed=1);warned=False;recovered=False
        for i in range(1,1101):
            p=m1.JointM1Tests.sample(self,i);p.scene_id=11
            from core.behavior_protocol import TaskContext
            context=TaskContext(p.case_id,p.task_id,p.scene_id,p.session_id)
            env=p.behavior_environment
            for lane in env.lanes:lane.evidence.context=context;lane.marking='SOLID';lane.crossing_range_m=None
            env.coverage.evidence.context=context;env.task_evidence.context=context
            p.targets[0].vx=p.targets[0].vy=0
            if i>=600:p.targets=[]
            self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
            self.assertTrue(d.valid,d.reason);t=planning_stub.plan(p,d);self.assertTrue(t.valid,t.reason)
            c=control_stub.compute_control(p,t);self.assertTrue(c.valid,c.errors)
            self.exchange.finish(p,t,c,{'attempted':True,'ok':True})
            if 300<i<600:
                self.assertTrue(c.hazard_signal)
                self.assertLess(self.plant.speed,.05)
                warned=True
            if i>600 and self.plant.speed>1:recovered=True;break
            self.plant.step(c,.05,vehicle());self.now+=.05
        self.assertTrue(warned);self.assertTrue(recovered)


if __name__=='__main__':unittest.main()
