"""Real entry points and plant; formal junction facts are model inputs."""
import unittest
import tests.test_joint_behavior_m1 as m1
from core.intersection_protocol import IntersectionTask
from core.behavior_exchange import BehaviorExchange
from core.behavior_protocol import GoalPose,TaskContext
from core.behavior_environment import Evidence
from members import decision_stub,planning_stub,control_stub
from tests.control_benchmark import Plant,vehicle
from tests.test_decision import add_target


class JointM4Tests(unittest.TestCase):
    setUp=m1.JointM1Tests.setUp

    def sample(self,index,blocked=False,rule='EGO_PRIORITY',window=60,covered=True):
        p=m1.JointM1Tests.sample(self,index,covered=covered);p.targets=[]
        proof=p.behavior_environment.task_evidence
        p.behavior_environment.intersection_task=IntersectionTask('junction-a',proof,[(-10,0),(70,0)],30,40,
            [[(20,-5),(30,-5),(30,5),(20,5)]],[(-10,-1.75),(80,-1.75),(80,1.75),(-10,1.75)],
            [(30,-1.75),(80,-1.75),(80,1.75),(30,1.75)],[(self.now,self.now+window)],rule,True,2.0)
        if blocked:
            obj=add_target(p,40-self.plant.x,0,lane_id='lane-1');obj.id=88;obj.x,obj.y=40,0
        return p

    def step(self,index,**kwargs):
        p=self.sample(index,**kwargs);self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        d=decision_stub.decide(p);self.assertTrue(d.valid,d.reason)
        t=planning_stub.plan(p,d);self.assertTrue(t.valid,(d.reason,t.reason))
        c=control_stub.compute_control(p,t);self.assertTrue(c.valid,c.errors)
        self.exchange.finish(p,t,c,{'attempted':True,'ok':True})
        state=decision_stub.decision_behavior_info()['intersection_runtime']
        self.records.append((index,round(self.plant.x,2),round(self.plant.speed,2),state['phase'],state['reason_code'],
                             [v.reason_code for v in t.planning_feedback]))
        self.plant.step(c,0.05,vehicle());self.now+=0.05
        return p,d,t,c,state

    def test_real_wait_for_exit_space_then_whole_body_passage(self):
        self.plant=Plant(speed=1.4)
        before=[];passed=False;released_at=None
        for i in range(1,1401):
            blocked=i<500
            p,d,t,c,state=self.step(i,blocked=blocked)
            if blocked:before.append(self.plant.x+3.5)
            elif released_at is None:released_at=self.now
            if state['phase']=='EXITED':passed=True;break
        self.assertLessEqual(max(before),20.05,self.records[-20:])
        self.assertTrue(passed,self.records[-20:])
        self.assertGreater(self.plant.x-0.9,30)
        self.assertLess(state['authorized_at']-released_at,5.0)

    def test_unknown_rule_coverage_and_short_window_do_not_adopt(self):
        for kwargs in ({'rule':'UNKNOWN'},{'covered':False},{'window':0.3}):
            self.plant=Plant(speed=0);control_stub._engine.reset()
            decision_stub.reset_decision();planning_stub.configure_planning(None);self.exchange=BehaviorExchange()
            for i in range(1,15):
                p,d,t,c,state=self.step(i,**kwargs)
                self.assertIsNone(t.execution_path_id)
            self.assertLess(self.plant.x+3.5,20)

    def test_group_returning_target_and_clear_recovery(self):
        self.plant=Plant(x=12,speed=0)
        executed=False
        for i in range(1,1101):
            p=self.sample(i)
            if i<100 or 140<=i<180:
                for j,y,vy in ((1,-2,0.6),(2,2,-0.6)):
                    obj=add_target(p,25-self.plant.x,0,lane_id='crossing');obj.id=90+j
                    obj.x,obj.y,obj.vx,obj.vy,obj.heading=25,y,0,vy,1.57079632679
                    obj.length,obj.width=0.8,0.6
            self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
            self.assertTrue(d.valid,d.reason);t=planning_stub.plan(p,d);self.assertTrue(t.valid,t.reason)
            c=control_stub.compute_control(p,t);self.assertTrue(c.valid,c.errors)
            self.exchange.finish(p,t,c,{'attempted':True,'ok':True})
            state=decision_stub.decision_behavior_info()['intersection_runtime']
            if i<180:self.assertLessEqual(self.plant.x+3.5,20.05)
            executed=executed or t.execution_path_id is not None
            self.plant.step(c,0.05,vehicle());self.now+=0.05
            if state['phase']=='EXITED':break
        self.assertTrue(executed)
        self.assertEqual('EXITED',state['phase'])

    def test_inside_permission_withdrawal_never_reuses_entry_stop(self):
        self.plant=Plant(speed=1.4)
        for i in range(1,500):
            p,d,t,c,state=self.step(i)
            if self.plant.x+3.5>20.3:break
        p=self.sample(i+1);p.behavior_environment.intersection_task.signal_permits=False
        self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
        self.assertEqual('EMERGENCY_BRAKE',d.mode)
        self.assertEqual(-1,d.stop_distance)
        t=planning_stub.plan(p,d);self.assertTrue(t.emergency_stop)

    def test_exit_with_only_half_body_space_is_not_admitted(self):
        from core.road_region import RoadRegion
        p=self.sample(1)
        p.behavior_environment.intersection_task.exit_region=RoadRegion([(30,-2),(33,-2),(33,2),(30,2)])
        self.exchange.prepare(p,planning_stub.planning_capabilities(p));d=decision_stub.decide(p)
        self.assertNotEqual('EXECUTE_SELECTED',d.behavior_request.authorization)
        self.assertLessEqual(d.stop_distance,16.2)

    def test_nonconvex_conflict_not_filled_by_hull(self):
        from core.road_region import RoadRegion
        region=RoadRegion([(0,0),(6,0),(6,6),(4,6),(4,2),(2,2),(2,6),(0,6)])
        self.assertFalse(region.intersects([(2.5,3),(3.5,3),(3.5,4),(2.5,4)]))
        self.assertTrue(region.intersects([(0.5,3),(1.5,3),(1.5,4),(0.5,4)]))


if __name__=='__main__':unittest.main()
