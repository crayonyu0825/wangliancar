"""Independent whole-body/window checks on computed passage trajectories."""
import math
import time
import unittest

from core.interfaces import TrajectoryPoint
from core.corridor_region import CorridorRegion
from core.maneuver_progress import ProgressTracker
from members.planning.candidate_validation import VehicleGeometry, MotionLimits, ValidationBudget, ExecutionEnvelope
from members.planning.passage_planner import PassageConstraints, plan_passage


def ring(a, b, c, d):
    return [(a,b),(c,b),(c,d),(a,d)]


class PassageAlgorithmTests(unittest.TestCase):
    def solve(self, waiting=False, windows=((0.,30.),), exit_polygon=None, initial=2., conflict=None, **changes):
        options=dict(shape=[(float(x),0.,0.) for x in range(0,31)],initial_speed=initial,speed_cap=3.,
            vehicle=VehicleGeometry(3.9,.9,.9,2.9),limits=MotionLimits(8.33,1.,2.,1.5,.5,.8,.1,.03),
            corridor=CorridorRegion([ring(-10.,-5.,80.,5.)]),coverage=CorridorRegion([ring(-20.,-20.,100.,20.)]),
            obstacles=[],envelope=ExecutionEnvelope(.1,.02,.1,.2),
            constraints=PassageConstraints('junction',[ring(10.,-4.,15.,4.)] if conflict is None else conflict,
                CorridorRegion([ring(15.,-5.,80.,5.)] if exit_polygon is None else [exit_polygon]),
                windows,.3,.2,right_of_way=True,waiting=waiting),
            budget=ValidationBudget(200000,16,.002,time.monotonic()+4.))
        options.update(changes)
        return plan_passage(**options)

    def test_complete_front_entry_tail_exit_inside_allowed_window(self):
        result=self.solve()
        self.assertEqual('safe',result['status'],result)
        self.assertTrue(result['execution_robust'])
        # Independent straight-body geometry: front first crosses x=10 before
        # the rear axle; tail leaves x=15 only after rear axle exceeds 15.9.
        inside=[]
        for p in result['points']:
            if p.x+3.9>=10. and p.x-.9<=15.:
                inside.append(p.relative_time)
        self.assertTrue(inside)
        conservative=result['body_occupancy_windows_s'][0]
        self.assertLessEqual(conservative[0],min(inside))
        self.assertGreaterEqual(conservative[1],max(inside))
        self.assertGreater(result['points'][-1].x-.9,15.)

    def test_wait_points_stop_whole_car_before_entry_without_zero_global_cap(self):
        result=self.solve(waiting=True)
        self.assertEqual('safe',result['status'],result)
        self.assertTrue(result['stop_required'])
        self.assertEqual(2.,result['points'][0].speed)
        self.assertEqual(0.,result['points'][-1].speed)
        self.assertLess(max(p.x+3.9 for p in result['points']),10.-.3)
        self.assertGreater(result['stop_distance'],0.)

    def test_exit_half_car_and_short_window_reject_without_partial_success(self):
        short=self.solve(exit_polygon=ring(29.,-1.,31.,1.))
        self.assertNotEqual('safe',short['status'])
        self.assertFalse(short['points'])
        window=self.solve(windows=((0.,1.),))
        self.assertNotEqual('safe',window['status'])
        self.assertFalse(window['points'])

    def test_thin_crossing_is_not_missed_between_path_points(self):
        result=self.solve(waiting=True,conflict=[ring(10.401,-4.,10.402,4.)])
        self.assertEqual('safe',result['status'],result)
        self.assertLess(result['points'][-1].x+3.9,10.401-.3)

    def test_initial_speed_cannot_be_rewritten_to_make_stop_feasible(self):
        result=self.solve(waiting=True,initial=8.)
        self.assertNotEqual('safe',result['status'])
        self.assertFalse(result['points'])

    def test_body_covered_hole_in_exit_is_preserved(self):
        exit_region=CorridorRegion([ring(20.,-4.,29.9,4.),ring(30.1,-4.,40.,4.),
                                    ring(29.9,1.,30.1,4.),ring(29.9,-4.,30.1,-1.)])
        constraints=PassageConstraints('junction',[ring(10.,-4.,15.,4.)],exit_region,
                                       ((0.,30.),),.3,.2,right_of_way=True)
        result=self.solve(constraints=constraints)
        self.assertNotEqual('safe',result['status'])

    def test_expired_budget_does_not_return_previously_generated_points(self):
        result=self.solve(budget=ValidationBudget(200000,16,.002,time.monotonic()-1.))
        self.assertEqual('inconclusive',result['status'])
        self.assertFalse(result['points'])

    def test_rotation_and_translation_use_same_geometry_algorithm(self):
        angle=.6;c,s=math.cos(angle),math.sin(angle)
        def xy(p):return (100.+c*p[0]-s*p[1],-50.+s*p[0]+c*p[1])
        def polygon(p):return [xy(v) for v in p]
        options=dict(shape=[xy((float(x),0.))+(angle,) for x in range(31)],
            corridor=CorridorRegion([polygon(ring(-10.,-5.,80.,5.))]),
            coverage=CorridorRegion([polygon(ring(-20.,-20.,100.,20.))]),
            constraints=PassageConstraints('rotated',[polygon(ring(10.,-4.,15.,4.))],
                CorridorRegion([polygon(ring(15.,-5.,80.,5.))]),((0.,30.),),.3,.2,right_of_way=True))
        result=self.solve(**options)
        self.assertEqual('safe',result['status'],result)
        self.assertAlmostEqual(angle,result['points'][0].heading)
        self.assertAlmostEqual(xy((30.,0.))[0],result['points'][-1].x)


class ActualProgressTests(unittest.TestCase):
    def path(self):
        return [TrajectoryPoint(float(x),0.,1.,0.,float(x)) for x in range(11)]

    def test_actual_projection_progress_and_duplicate_have_no_virtual_motion(self):
        tracker=ProgressTracker()
        first=tracker.observe('a',self.path(),(0.,0.,0.),1.,1,10.,3.,1.)
        self.assertEqual(0.,first['progress'])
        second=tracker.observe('a',self.path(),(.2,0.,0.),1.,2,10.2,3.,1.)
        self.assertAlmostEqual(.02,second['progress'])
        duplicate=tracker.observe('a',self.path(),(.2,0.,0.),1.,2,10.4,3.,1.)
        self.assertEqual(second['progress'],duplicate['progress'])
        self.assertFalse(duplicate['distinct'])

    def test_pause_regression_teleport_and_fake_initial_anchor_rejected(self):
        for pose,frame,now in (((5.,0.,0.),2,10.1),((0.,0.,0.),0,10.1),((0.,0.,0.),2,11.)):
            tracker=ProgressTracker()
            tracker.observe('a',self.path(),(0.,0.,0.),1.,1,10.,3.,1.)
            with self.assertRaises(ValueError):tracker.observe('a',self.path(),pose,1.,frame,now,3.,1.)
        with self.assertRaises(ValueError):ProgressTracker().observe('a',self.path(),(5.,0.,0.),1.,1,10.,3.,1.)

    def test_loop_cannot_jump_to_far_returning_branch(self):
        tracker=ProgressTracker()
        path=[TrajectoryPoint(0.,0.,1.,0.,0.),TrajectoryPoint(10.,0.,1.,0.,10.),
              TrajectoryPoint(0.,.1,1.,math.pi,20.)]
        tracker.observe('loop',path,(0.,0.,0.),1.,1,10.,3.,1.)
        result=tracker.observe('loop',path,(.1,.05,0.),1.,2,10.1,3.,1.)
        self.assertLess(result['station_m'],1.)


if __name__=='__main__':unittest.main()
