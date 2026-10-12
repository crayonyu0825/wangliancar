"""Computed emergency geometry, authored analytic obstacles, no vehicle I/O."""
import copy
import math
import time
import unittest

from members.planning.emergency_planner import generate_emergency_candidates
from members.planning.candidate_validation import CorridorRegion,ExecutionEnvelope
from members.planning.dynamic_speed import DynamicSearch
from tests.test_planning_candidate_validation import vehicle,limits,budget,rectangle,obstacle
from tests.test_planning_lane_change_generator import search


class EmergencyCandidateTests(unittest.TestCase):
    def setUp(self):
        self.car=vehicle(.7,.5,.25,.7)
        self.motion=limits(max_speed_mps=3.,max_deceleration_mps2=3.,distance_tolerance_m=.01)
        self.region=CorridorRegion([rectangle(-3.,-3.,15.,5.)])
        self.envelope=ExecutionEnvelope(.02,.003,.02,.05)

    def solve(self,objects=(),regions=None,work=None,dynamic=None):
        regions=regions or [dict(action='BRAKE',goal=(7.,0.,0.),corridor=self.region,
            coverage=self.region,goal_region=self.region),dict(action='BRAKE_LEFT',goal=(8.,2.,0.),
            corridor=self.region,coverage=self.region,goal_region=self.region)]
        return generate_emergency_candidates((0.,0.,0.),2.,0.,1.99,self.car,self.motion,regions,
            list(objects),search(spacing_m=.2),work or budget(max_checks=1000000),self.envelope,dynamic)

    def test_actual_start_terminal_stop_and_no_acceleration_in_clear_candidates(self):
        report=self.solve()
        self.assertEqual('safe',report['status'],report)
        self.assertEqual({'BRAKE','BRAKE_LEFT'},set(v['action'] for v in report['candidates']))
        for candidate in report['candidates']:
            points=candidate['points']
            self.assertEqual((0.,0.,2.),(points[0].x,points[0].y,points[0].speed))
            self.assertEqual(0.,points[-1].speed)
            self.assertLessEqual(max(v.speed for v in points),2.)
            for a,b in zip(points,points[1:]):
                self.assertLessEqual(b.speed,a.speed+1e-9)
                self.assertAlmostEqual(math.hypot(b.x-a.x,b.y-a.y),
                    .5*(a.speed+b.speed)*(b.relative_time-a.relative_time),places=6)

    def test_straight_collision_removed_while_real_lateral_candidate_survives(self):
        report=self.solve([obstacle(4.,0.,length=.3,width=.3,horizon=30.)])
        self.assertEqual('safe',report['status'],report)
        self.assertEqual(['BRAKE_LEFT'],[v['action'] for v in report['candidates']])
        path=report['candidates'][0]['points']
        self.assertGreater(max(p.y for p in path),1.9)
        for point in path:
            if 3.3<point.x<4.8:self.assertGreater(point.y,.4)

    def test_exhaustion_never_publishes_a_partial_candidate(self):
        report=self.solve(work=budget(max_checks=1))
        self.assertEqual('inconclusive',report['status']);self.assertFalse(report['candidates'])

    def test_expired_work_remains_brake_only(self):
        report=self.solve(work=budget(deadline_monotonic_s=time.monotonic()-1.))
        self.assertEqual('inconclusive',report['status']);self.assertTrue(report['immediate_brake_required'])

    def test_node_budget_is_shared_across_actions_and_retracts_partial_candidates(self):
        report=self.solve(dynamic=DynamicSearch(max_nodes=1))
        self.assertEqual('inconclusive',report['status'],report)
        self.assertEqual('EMERGENCY_NODE_BUDGET_EXHAUSTED',report['reason_code'])
        self.assertTrue(report['budget_exhausted']);self.assertFalse(report['candidates'])
        self.assertEqual(1,report['nodes'])
        self.assertEqual(1,len(report['attempts']))
        self.assertEqual('safe',report['attempts'][0]['status'])
        self.assertEqual(report['nodes'],sum(v['nodes'] for v in report['attempts']))
        self.assertGreater(report['checks'],sum(v['checks'] for v in report['attempts']))

    def test_missing_authorized_space_does_not_invent_an_escape(self):
        empty=CorridorRegion([rectangle(-3.,-.5,15.,.5)])
        report=self.solve(regions=[dict(action='BRAKE_LEFT',goal=(8.,2.,0.),corridor=empty,
                                       coverage=empty,goal_region=empty)])
        self.assertNotEqual('safe',report['status']);self.assertFalse(report['candidates'])


if __name__=='__main__':unittest.main()
