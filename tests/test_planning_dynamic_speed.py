"""Authored dynamic worlds; no SDK, real calibration, or scoring claims."""
import math
import time
import unittest

from core.interfaces import TrajectoryPoint
from members.planning.candidate_validation import ExecutionEnvelope,validate_candidate
from members.planning.dynamic_speed import DynamicSearch,solve_dynamic_speed
from tests.test_planning_candidate_validation import vehicle,limits,obstacle,rectangle,budget


class ExecutionEnvelopeTests(unittest.TestCase):
    def validate(self,points,road=None,targets=None,envelope=None):
        return validate_candidate(points,1,vehicle(.4,.3,.2,.3),limits(),
            rectangle() if road is None else road,[] if targets is None else targets,
            budget(max_checks=100000),envelope)

    def test_nominal_safe_crossing_is_not_safe_when_execution_can_be_late(self):
        points=[TrajectoryPoint(0.,0.,2.,0.,0.),TrajectoryPoint(4.,0.,2.,0.,2.)]
        targets=[obstacle(2.,4.,vy=-2.,horizon=10.)]
        nominal=self.validate(points,targets=targets)
        robust=self.validate(points,targets=targets,envelope=ExecutionEnvelope(0.,0.,0.,1.))
        self.assertEqual('safe',nominal['status'],nominal)
        self.assertFalse(nominal['execution_robust'])
        # At nominal progress x=2(t=1), one-second late execution sees the
        # crossing target actually at (2,0), not the nominal (2,2).
        self.assertNotEqual('safe',robust['status'],robust)
        self.assertEqual('OBSTACLE_COLLISION',robust['reason_code'])

    def test_position_tracking_error_is_checked_against_real_road_edge(self):
        points=[TrajectoryPoint(0.,0.,1.,0.,0.),TrajectoryPoint(1.,0.,1.,0.,1.)]
        road=rectangle(-2.,-.3,3.,.3)
        self.assertEqual('safe',self.validate(points,road=road)['status'])
        result=self.validate(points,road=road,envelope=ExecutionEnvelope(.15,0.,0.,0.))
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'])

    def test_body_yaw_error_expands_front_corner_sweep(self):
        points=[TrajectoryPoint(0.,0.,1.,0.,0.),TrajectoryPoint(1.,0.,1.,0.,1.)]
        result=self.validate(points,road=rectangle(-2.,-.25,3.,.25),
                             envelope=ExecutionEnvelope(0.,.3,0.,0.))
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'])

    def test_late_execution_requires_longer_prediction_coverage(self):
        points=[TrajectoryPoint(0.,0.,1.,0.,0.),TrajectoryPoint(2.,0.,1.,0.,2.)]
        result=self.validate(points,targets=[obstacle(30.,10.,horizon=2.1)],
                             envelope=ExecutionEnvelope(0.,0.,0.,.2))
        self.assertEqual('PREDICTION_HORIZON',result['reason_code'])
        self.assertAlmostEqual(2.2,result['details']['required_horizon_s'])


class DynamicSpeedTests(unittest.TestCase):
    def solve(self,targets=None,**changes):
        arguments=dict(shape=[(float(i),0.,0.) for i in range(9)],initial_speed_mps=1.,
            speed_cap_mps=2.,vehicle=vehicle(.4,.3,.2,.3),limits=limits(),corridor=rectangle(),
            coverage=rectangle(),obstacles=[] if targets is None else targets,
            budget=budget(max_checks=1000000,deadline_monotonic_s=time.monotonic()+10.),
            execution_envelope=ExecutionEnvelope(.02,.01,.1,.1),
            search=DynamicSearch(dt_s=.5,acceleration_samples=(-.5,0.,.5,1.),
                                 beam_width=32,max_steps=80,max_nodes=5000,reserve_s=.01))
        arguments.update(changes)
        return solve_dynamic_speed(**arguments)

    def assert_physics(self,result,initial=1.,cap=2.,stop=False):
        self.assertEqual('safe',result['status'],result)
        self.assertTrue(result['execution_robust'])
        self.assertEqual(initial,result['points'][0].speed)
        points=result['points']
        self.assertEqual((0.,0.,0.),(points[0].x,points[0].y,points[0].relative_time))
        self.assertAlmostEqual(8.,points[-1].x)
        for a,b in zip(points,points[1:]):
            dt=b.relative_time-a.relative_time
            distance=math.hypot(b.x-a.x,b.y-a.y)
            self.assertGreater(dt,0.)
            self.assertAlmostEqual(distance,.5*(a.speed+b.speed)*dt,places=8)
            self.assertLessEqual((b.speed-a.speed)/dt,3.+1e-8)
            self.assertLessEqual((a.speed-b.speed)/dt,4.+1e-8)
            self.assertGreaterEqual(b.speed,0.)
            self.assertLessEqual(b.speed,max(initial,cap)+1e-8)
        if stop:
            self.assertEqual(0.,points[-1].speed)

    def test_clear_profile_preserves_initial_speed_and_exact_distance_time(self):
        result=self.solve(); self.assert_physics(result)
        self.assertGreater(result['points'][-1].speed,1.)
        self.assertLess(result['planning_horizon_s'],8.)
        self.assertGreater(result['legal_report']['clearance_lower_bound_m'],0.)

    def test_spatial_low_speed_notch_is_solved_not_discarded(self):
        result=self.solve(spatial_speed_limits=[(0.,1.),(2.,1.),(3.,.4),(4.,.4),(6.,1.5),(8.,1.5)])
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(1.,result['points'][0].speed)
        for a,b in zip(result['points'],result['points'][1:]):
            acceleration=(b.speed-a.speed)/(b.relative_time-a.relative_time)
            for x in (3.,3.5,4.):
                if a.x<=x<=b.x:
                    speed=math.sqrt(max(0.,a.speed*a.speed+2*acceleration*(x-a.x)))
                    self.assertLessEqual(speed,.4+1e-8)
        self.assertGreater(result['planning_horizon_s'],6.)

    def test_spatial_cap_drops_between_geometry_points_and_remains_binding(self):
        result=self.solve(shape=[(0.,0.,0.),(8.,0.,0.)],
                          spatial_speed_limits=[(0.,1.),(3.,1.),(3.5,.3),(4.,.3),(8.,1.)])
        self.assertEqual('safe',result['status'],result)
        self.assertTrue(any(abs(p.x-3.5)<1e-8 and p.speed<=.3+1e-8 for p in result['points']))
        self.assertTrue(any(abs(p.x-4.)<1e-8 and p.speed<=.3+1e-8 for p in result['points']))

    def test_spatial_initial_overspeed_is_disclosed_and_braked_without_rewriting(self):
        result=self.solve(initial_speed_mps=1.,spatial_speed_limits=[(0.,.5),(8.,.5)],
                          search=DynamicSearch(dt_s=.1,acceleration_samples=(-4.,-2.,0.,1.),
                                               beam_width=32,max_steps=200,max_nodes=10000))
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(1.,result['points'][0].speed)
        self.assertEqual(.5,result['initial_overspeed_mps'])
        for a,b in zip(result['points'],result['points'][1:]):
            acceleration=(b.speed-a.speed)/(b.relative_time-a.relative_time)
            for index in range(11):
                x=a.x+(b.x-a.x)*index/10.
                speed2=a.speed*a.speed+2*acceleration*(x-a.x)
                # Independent exact reachable braking floor at 4 m/s^2.
                self.assertLessEqual(speed2,max(.25,1.-8*x)+1e-8)

    def test_malformed_or_regressing_spatial_limits_cannot_be_ignored(self):
        for values in ([(2.,1.),(1.,.5)],[(0.,float('nan'))],[(0.,-1.)],[(9.,1.)]):
            result=self.solve(spatial_speed_limits=values)
            self.assertEqual('SPATIAL_SPEED_LIMITS_INVALID',result['reason_code'],result)
            self.assertFalse(result['points'])

    def test_no_acceleration_authority_binds_seed_not_only_beam_choices(self):
        caps=[(0.,2.),(2.,.6),(4.,2.),(8.,2.)]
        ordinary=self.solve(initial_speed_mps=2.,spatial_speed_limits=caps,stop_required=True)
        braking=self.solve(initial_speed_mps=2.,spatial_speed_limits=caps,stop_required=True,
                           allow_acceleration=False)
        self.assertEqual('safe',ordinary['status'],ordinary)
        self.assertTrue(any(b.speed>a.speed+1e-8 for a,b in zip(ordinary['points'],ordinary['points'][1:])))
        self.assertEqual('safe',braking['status'],braking)
        self.assertEqual(2.,braking['points'][0].speed);self.assertEqual(0.,braking['points'][-1].speed)
        self.assertFalse(braking['allow_acceleration'])
        for a,b in zip(braking['points'],braking['points'][1:]):
            acceleration=(b.speed-a.speed)/(b.relative_time-a.relative_time)
            self.assertLessEqual(acceleration,1e-8)
            self.assertGreaterEqual(acceleration,-4.-1e-8)

    def test_no_acceleration_flag_requires_real_boolean(self):
        result=self.solve(allow_acceleration=0)
        self.assertEqual('ACCELERATION_AUTHORITY_INVALID',result['reason_code'])
        self.assertFalse(result['points'])

    def test_actual_speed_above_new_cap_is_preserved_then_physically_reduced(self):
        result=self.solve(initial_speed_mps=3.,speed_cap_mps=2.,stop_required=True)
        self.assert_physics(result,initial=3.,stop=True)
        self.assertEqual(3.,result['points'][0].speed)
        self.assertLess(result['points'][1].speed,3.)
        self.assertEqual(8.,result['stop_distance'])

    def test_infeasible_stop_distance_does_not_rewrite_measurement(self):
        result=self.solve(initial_speed_mps=4.,stop_required=True,stop_distance_m=1.)
        self.assertEqual('STOP_DISTANCE_INFEASIBLE',result['reason_code'])
        self.assertEqual('unsafe',result['status'])
        self.assertFalse(result['points']); self.assertFalse(result['goal_reached'])

    def test_crossing_object_requires_changed_time_profile_not_object_removal(self):
        crossing=obstacle(3.,1.5,length=.2,width=.2,vy=-1.,horizon=20.)
        result=self.solve([crossing]); self.assert_physics(result)
        self.assertGreater(result['rejected_edges'].get('OBSTACLE_COLLISION',0),0)
        # Independent densely sampled rectangle separation. This is an
        # approximate .01s audit, not the continuous validator's proof.
        minimum=float('inf')
        for a,b in zip(result['points'],result['points'][1:]):
            dt=b.relative_time-a.relative_time
            acceleration=(b.speed-a.speed)/dt
            for sample in range(int(math.ceil(dt/.01))+1):
                local=min(dt,sample*.01)
                x=a.x+a.speed*local+.5*acceleration*local*local
                at=a.relative_time+local
                target_y=1.5-at
                dx=max(0.,3.-.1-(x+.4),(x-.3)-(3.+.1))
                dy=max(0.,target_y-.1-.2,-.2-(target_y+.1))
                minimum=min(minimum,math.hypot(dx,dy))
        self.assertGreater(minimum,.1)

    def test_second_stationary_target_remains_after_first_target_leaves(self):
        result=self.solve([obstacle(3.,1.5,vy=-1.,identity='crossing'),
                           obstacle(5.,0.,length=1.,identity='stopped-second')])
        self.assertNotEqual('safe',result['status'],result)
        self.assertFalse(result['goal_reached']); self.assertFalse(result['points'])
        self.assertGreater(result['rejected_edges'].get('OBSTACLE_COLLISION',0),0)

    def test_input_target_order_does_not_change_safe_time_profile(self):
        crossing=obstacle(3.,1.5,vy=-1.,identity='crossing')
        far=obstacle(20.,5.,identity='far')
        first=self.solve([crossing,far]); second=self.solve([far,crossing])
        self.assert_physics(first); self.assert_physics(second)
        self.assertAlmostEqual(first['planning_horizon_s'],second['planning_horizon_s'])

    def test_unknown_velocity_or_execution_bounds_cannot_create_time_safety(self):
        for changes,reason in ((dict(execution_envelope=None),'EXECUTION_ENVELOPE_UNAVAILABLE'),
                               (dict(targets=[obstacle(30.,5.,vx=None)]),'PREDICTIONS_INVALID')):
            if 'targets' in changes:
                targets=changes.pop('targets'); result=self.solve(targets,**changes)
            else:
                result=self.solve(**changes)
            self.assertEqual(reason,result['reason_code'])
            self.assertFalse(result['points'])

    def test_admission_window_includes_execution_early_and_late_bounds(self):
        result=self.solve(progress_time_windows=[(2.,4.,3.,9.)])
        self.assert_physics(result)
        entered=None; departed=None
        for a,b in zip(result['points'],result['points'][1:]):
            for value,name in ((1.98,'entered'),(4.02,'departed')):
                if a.x<=value<=b.x:
                    acceleration=(b.speed-a.speed)/(b.relative_time-a.relative_time)
                    dist=value-a.x
                    elapsed=(dist/a.speed if abs(acceleration)<1e-9 else
                             2*dist/(a.speed+math.sqrt(a.speed*a.speed+2*acceleration*dist)))
                    if name=='entered': entered=a.relative_time+elapsed
                    else: departed=a.relative_time+elapsed
        self.assertIsNotNone(entered); self.assertIsNotNone(departed)
        self.assertGreaterEqual(entered-.1,3.-1e-8)
        self.assertLessEqual(departed+.1,9.+1e-8)

    def test_too_short_window_remains_infeasible(self):
        result=self.solve(progress_time_windows=[(2.,4.,1.,1.1)])
        self.assertNotEqual('safe',result['status'],result)
        self.assertFalse(result['points'])

    def test_node_and_reserved_wall_clock_budgets_do_not_claim_safe(self):
        crossing=obstacle(3.,1.5,vy=-1.)
        for changes in (dict(search=DynamicSearch(max_nodes=1)),
                        dict(budget=budget(max_checks=1)),
                        dict(budget=budget(deadline_monotonic_s=time.monotonic()+.001))):
            result=self.solve([crossing],**changes)
            self.assertEqual('BUDGET_EXHAUSTED',result['reason_code'],result)
            self.assertFalse(result['points']); self.assertFalse(result['execution_robust'])

    def test_finite_prediction_expiry_does_not_get_extended_for_search(self):
        result=self.solve([obstacle(30.,5.,horizon=.25)])
        self.assertNotEqual('safe',result['status'],result)
        self.assertFalse(result['points'])

    def test_reverse_stop_keeps_body_yaw_and_nonnegative_speed(self):
        result=self.solve(shape=[(-float(i),0.,0.) for i in range(9)],
                          initial_speed_mps=.5,speed_cap_mps=1.,motion_direction=-1,
                          stop_required=True)
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(.5,result['points'][0].speed)
        self.assertEqual(-8.,result['points'][-1].x)
        self.assertEqual(0.,result['points'][-1].speed)
        self.assertTrue(all(p.heading==0. and p.speed>=0. for p in result['points']))

    def test_explicit_jerk_limit_is_checked_on_published_profile(self):
        result=self.solve(search=DynamicSearch(dt_s=.5,acceleration_samples=(-1.,0.,1.),
            max_jerk_mps3=.1,max_nodes=5000))
        self.assert_physics(result)
        # Initial acceleration is explicitly zero; under this sample set the
        # only legal finite-difference jerk transition is continued zero.
        for a,b in zip(result['points'],result['points'][1:]):
            self.assertAlmostEqual(0.,(b.speed-a.speed)/(b.relative_time-a.relative_time))

    def test_stop_distance_uses_actual_geometry_progress_without_extrapolation(self):
        result=self.solve(stop_required=True,stop_distance_m=5.)
        self.assertEqual('safe',result['status'],result)
        self.assertAlmostEqual(5.,result['points'][-1].x)
        self.assertAlmostEqual(5.,result['stop_distance'])
        self.assertEqual(0.,result['points'][-1].speed)

    def test_insufficient_geometry_is_not_reported_as_requested_stop_completion(self):
        result=self.solve(stop_required=True,stop_distance_m=10.)
        self.assertEqual('PATH_COVERAGE_INSUFFICIENT',result['reason_code'])
        self.assertFalse(result['goal_reached']); self.assertFalse(result['points'])

    def test_known_stop_boundary_requires_explicit_stop_contract(self):
        result=self.solve(stop_distance_m=5.,stop_required=False)
        self.assertEqual('STOP_CONTRACT_INVALID',result['reason_code'])
        self.assertFalse(result['points'])


class RobustGeneratedLaneChangeTests(unittest.TestCase):
    def test_generated_transition_uses_explicit_error_bounds_and_preserves_pose(self):
        from tests.test_planning_lane_change_generator import LaneChangeGeneratorTests
        result=LaneChangeGeneratorTests().run_search(
            execution_envelope=ExecutionEnvelope(.01,.002,.1,.1))
        self.assertEqual('safe',result['status'],result)
        self.assertTrue(result['execution_robust'])
        self.assertTrue(result['legal_report']['execution_robust'])
        self.assertEqual((10.,0.,0.,2.),(result['points'][0].x,result['points'][0].y,
                                       result['points'][0].heading,result['points'][0].speed))
        self.assertEqual((35.,3.5),(result['points'][-1].x,result['points'][-1].y))
        self.assertGreater(result['nodes'],0)
        self.assertGreater(result['checks'],0)

    def test_new_lower_cap_does_not_rewrite_measured_start_speed(self):
        from tests.test_planning_lane_change_generator import LaneChangeGeneratorTests
        result=LaneChangeGeneratorTests().run_search(initial_speed_mps=2.,speed_cap_mps=1.5,
            execution_envelope=ExecutionEnvelope(.01,.002,.1,.1))
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(2.,result['points'][0].speed)
        self.assertLessEqual(result['points'][-1].speed,1.5+1e-9)
        self.assertLess(result['points'][1].speed,2.)


if __name__=='__main__': unittest.main()
