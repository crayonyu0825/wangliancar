"""Synthetic GPS and successful/failed receipts, never field calibration."""
import math
import unittest

from core.execution_monitor import ExecutionMonitor
from core.interfaces import Perception,Trajectory,TrajectoryPoint
from members.planning.candidate_validation import ExecutionEnvelope


def actual(frame=1,now=100.,x=0.,y=0.,yaw=0.,speed=1.):
    p=Perception(); p.valid=p.ego.valid=True
    p.frame_id=p.ego.frame_id=frame
    p.timestamp=p.ego.timestamp=frame*10000  # SDK units deliberately not local seconds.
    p.valid_until=now+10.; p.case_id='authored'; p.task_id='tracking'; p.scene_id=6
    p.ego.x,p.ego.y,p.ego.heading,p.ego.speed=x,y,yaw,speed
    p.source_status={'gps':{'usable':True,'quality':'fresh','frame_id':frame,'observed_at_s':now}}
    return p


def candidate(p,points=None,direction=1):
    t=Trajectory().bind(p); t.valid=True; t.candidate_id='candidate-'+str(p.frame_id)
    t.motion_direction=direction
    t.points=(points if points is not None else [
        TrajectoryPoint(p.ego.x,p.ego.y,p.ego.speed,p.ego.heading,0.),
        TrajectoryPoint(p.ego.x+direction*2*math.cos(p.ego.heading),
                        p.ego.y+direction*2*math.sin(p.ego.heading),p.ego.speed,p.ego.heading,
                        2./p.ego.speed if p.ego.speed>0 else 2.)])
    if p.ego.speed==0 and points is None:
        t.points[-1].x,t.points[-1].y=p.ego.x,p.ego.y
    return t


class ExecutionMonitorTests(unittest.TestCase):
    def setUp(self):
        self.envelope=ExecutionEnvelope(.03,.02,.1,.2)
        self.monitor=ExecutionMonitor(max_speed_mps=3.,speed_error_mps=.05,max_gap_s=.5)
        self.sent={'attempted':True,'ok':True}

    def arm(self,p=None,t=None,envelope=None):
        p=p or actual(); t=t or candidate(p)
        report=self.monitor.record_sent(p,t,envelope or self.envelope,100.,self.sent)
        self.assertEqual('ARMED',report['status'],report)
        return p,t

    def inspect(self,p,envelope=None,t=None,now=None):
        return self.monitor.observe(p,t or candidate(p),envelope or self.envelope,
                                    p.source_status['gps']['observed_at_s'] if now is None else now)

    def assert_fault(self,result,reason):
        self.assertTrue(result['active'],result); self.assertTrue(result['brake_required'])
        self.assertEqual(reason,result['reason_code'])

    def test_observe_only_does_not_arm_or_manufacture_send_evidence(self):
        p=actual(); result=self.inspect(p)
        self.assertEqual('UNARMED',result['status']); self.assertFalse(result['active'])
        for receipt in (None,{'attempted':False,'ok':True},{'attempted':True,'ok':False}):
            self.monitor.record_sent(p,candidate(p),self.envelope,100.,receipt)
        self.assertEqual(0,len(self.monitor.history))

    def test_missing_speed_error_cannot_assume_perfect_tracking(self):
        self.monitor.speed_error_mps=None
        self.assert_fault(self.inspect(actual()),'UNVERIFIED_SPEED_ERROR')

    def test_measured_physical_speed_limit_applies_even_before_first_send(self):
        self.assert_fault(self.inspect(actual(speed=3.1)),'EXECUTION_HARD_SPEED_LIMIT')

    def test_valid_next_gps_tracks_the_previous_sent_candidate(self):
        self.arm(); result=self.inspect(actual(2,100.2,x=.2))
        self.assertFalse(result['active'],result)
        self.assertEqual('EXECUTION_WITHIN_BOUNDS',result['reason_code'])
        self.assertLessEqual(result['details']['position_error_m'],.03+1e-9)

    def test_new_candidate_anchored_to_wrong_actual_position_cannot_hide_old_error(self):
        self.arm(); p=actual(2,100.2,x=.2,y=.15)
        self.assert_fault(self.inspect(p),'EXECUTION_POSITION_OR_TIME_ERROR')
        self.assertEqual('candidate-1',self.monitor.history[-1]['candidate_id'])
        # Even an ok receipt cannot silently replace failed prior tracking.
        self.assert_fault(self.monitor.record_sent(p,candidate(p),self.envelope,100.2,self.sent),
                          'EXECUTION_POSITION_OR_TIME_ERROR')

    def test_yaw_and_speed_deviations_are_separate_real_feedback_failures(self):
        self.arm()
        self.assert_fault(self.inspect(actual(2,100.2,x=.2,yaw=.2)),'EXECUTION_HEADING_ERROR')
        self.assert_fault(self.inspect(actual(2,100.2,x=.2,speed=1.2)),'EXECUTION_SPEED_ERROR')

    def test_asymmetric_late_bound_does_not_become_early_bound(self):
        env=ExecutionEnvelope(.001,.001,.01,.3); self.arm(envelope=env)
        self.assertFalse(self.inspect(actual(2,100.4,x=.1),env)['active'])
        self.assert_fault(self.inspect(actual(3,100.45,x=.6),env),'EXECUTION_POSITION_OR_TIME_ERROR')

    def test_constant_acceleration_uses_integrated_progress_not_time_fraction(self):
        p=actual(speed=0.)
        t=candidate(p,[TrajectoryPoint(0.,0.,0.,0.,0.),TrajectoryPoint(2.,0.,2.,0.,2.)])
        self.arm(p,t,ExecutionEnvelope(.001,.001,0.,0.))
        result=self.inspect(actual(2,100.5,x=.125,speed=.5),ExecutionEnvelope(.001,.001,0.,0.))
        self.assertFalse(result['active'],result)
        self.assertAlmostEqual(.5,result['details']['nominal_time_s'])

    def test_position_and_yaw_cannot_be_checked_at_different_times(self):
        env=ExecutionEnvelope(.001,.001,2.,2.); p=actual(speed=0.)
        t=candidate(p,[TrajectoryPoint(0.,0.,0.,0.,0.),TrajectoryPoint(2.,0.,2.,1.,2.)])
        self.arm(p,t,env)
        self.assert_fault(self.inspect(actual(2,100.2,x=.02,yaw=.8,speed=.2),env),
                          'EXECUTION_HEADING_ERROR')

    def test_position_and_speed_cannot_be_checked_at_different_times(self):
        env=ExecutionEnvelope(.001,.001,2.,2.); p=actual(speed=0.)
        t=candidate(p,[TrajectoryPoint(0.,0.,0.,0.,0.),TrajectoryPoint(2.,0.,2.,0.,2.)])
        self.arm(p,t,env)
        self.assert_fault(self.inspect(actual(2,100.2,x=.02,speed=1.8),env),'EXECUTION_SPEED_ERROR')

    def test_reverse_body_heading_is_not_rotated_by_pi(self):
        p=actual(); self.arm(p,candidate(p,direction=-1))
        self.assertFalse(self.inspect(actual(2,100.2,x=-.2),t=candidate(actual(2,100.2,x=-.2),direction=-1))['active'])

    def test_yaw_wraparound_is_continuous(self):
        env=ExecutionEnvelope(.001,.01,0.,0.); p=actual(yaw=math.pi-.05)
        t=candidate(p,[TrajectoryPoint(0.,0.,1.,math.pi-.05,0.),
                       TrajectoryPoint(-1.,0.,1.,-math.pi+.05,1.)])
        self.arm(p,t,env)
        self.assertFalse(self.inspect(actual(2,100.2,x=-.2,yaw=math.pi-.03),env)['active'])

    def test_repeated_out_of_order_future_and_gap_fail_without_new_execution(self):
        self.arm(); self.assertFalse(self.inspect(actual(2,100.2,x=.2))['active'])
        self.assert_fault(self.inspect(actual(2,100.25,x=.25)),'EXECUTION_REPEATED_FRAME')
        self.assert_fault(self.inspect(actual(1,100.25,x=.25)),'EXECUTION_OUT_OF_ORDER_FRAME')
        p=actual(3,100.3,x=.3); p.source_status['gps']['observed_at_s']=101.
        self.assert_fault(self.inspect(p,now=100.3),'EXECUTION_FUTURE_OBSERVATION')
        self.assert_fault(self.inspect(actual(3,100.8,x=.8)),'EXECUTION_OBSERVATION_GAP')

    def test_timestamp_regression_cannot_be_hidden_by_new_frame(self):
        self.arm(); self.assertFalse(self.inspect(actual(2,100.1,x=.1))['active'])
        p=actual(3,100.2,x=.2); p.timestamp=p.ego.timestamp=10000
        self.assert_fault(self.inspect(p),'EXECUTION_OUT_OF_ORDER_TIMESTAMP')

    def test_raw_sdk_timestamp_is_never_compared_to_local_seconds(self):
        self.arm(); p=actual(2,100.1,x=.1); p.timestamp=p.ego.timestamp=10**15
        self.assertFalse(self.inspect(p)['active'])

    def test_capture_time_not_late_processing_time_drives_actual_pose_check(self):
        env=ExecutionEnvelope(.001,.001,0.,0.); self.arm(envelope=env)
        result=self.inspect(actual(2,100.1,x=.1),env,now=100.2)
        self.assertFalse(result['active'],result)
        self.assertAlmostEqual(.1,result['details']['nominal_time_s'])

    def test_wider_current_envelope_does_not_reinterpret_old_promise(self):
        self.arm(); wide=ExecutionEnvelope(10.,1.,10.,10.)
        self.assert_fault(self.inspect(actual(2,100.2,x=.2,y=.2),wide),'EXECUTION_POSITION_OR_TIME_ERROR')

    def test_caller_mutation_cannot_change_recorded_geometry_or_bounds(self):
        p,t=self.arm(); t.points[1].y=20.; self.envelope.position_error_m=10.
        self.assertFalse(self.inspect(actual(2,100.1,x=.1))['active'])
        self.assert_fault(self.inspect(actual(3,100.2,x=.2,y=.2)),'EXECUTION_POSITION_OR_TIME_ERROR')

    def test_duplicate_receipt_cannot_reset_timeline(self):
        p,t=self.arm()
        self.assert_fault(self.monitor.record_sent(p,t,self.envelope,100.1,self.sent),
                          'EXECUTION_DUPLICATE_SEND_CANNOT_RENEW')
        self.assertEqual(100.,self.monitor.history[-1]['sent_at_s'])

    def test_context_change_requires_trusted_reset(self):
        self.arm(); p=actual(2,100.1,x=.1); p.task_id='another'
        self.assert_fault(self.inspect(p),'EXECUTION_CONTEXT_CHANGED_RESET_REQUIRED')
        self.monitor.reset(); self.assertEqual('UNARMED',self.inspect(p)['status'])

    def test_expired_old_candidate_is_not_renewed_by_fresh_current_candidate(self):
        p=actual(); t=candidate(p); t.valid_until=100.1; self.arm(p,t)
        self.assert_fault(self.inspect(actual(2,100.2,x=.2)),'EXECUTION_SENT_CANDIDATE_EXPIRED')

    def test_horizon_exhaustion_requires_new_candidate_instead_of_silent_holding(self):
        env=ExecutionEnvelope(.001,.001,0.,0.); p=actual()
        t=candidate(p,[TrajectoryPoint(0.,0.,1.,0.,0.),TrajectoryPoint(.1,0.,1.,0.,.1)])
        self.arm(p,t,env)
        self.assert_fault(self.inspect(actual(2,100.2,x=.2),env),'EXECUTION_TIMELINE_EXHAUSTED')

    def test_history_is_bounded_and_stationary_zero_speed_is_supported(self):
        self.monitor=ExecutionMonitor(3.,.05,max_entries=2)
        for frame in range(1,5):
            now=100.+(frame-1)*.1; p=actual(frame,now,speed=0.)
            result=self.monitor.record_sent(p,candidate(p),self.envelope,now,self.sent)
            self.assertEqual('ARMED',result['status'],result)
        self.assertEqual(2,len(self.monitor.history))
        self.assertFalse(self.inspect(actual(5,100.4,speed=0.))['active'])

    def test_excessive_candidate_points_are_rejected(self):
        self.monitor=ExecutionMonitor(3.,.05,max_points=2)
        p=actual(); t=candidate(p); t.points.insert(1,TrajectoryPoint(.1,0.,1.,0.,.1))
        self.assert_fault(self.monitor.record_sent(p,t,self.envelope,100.,self.sent),
                          'EXECUTION_CANDIDATE_INVALID_OR_TOO_LARGE')

    def test_emergency_brake_command_never_claims_instant_zero_actual_speed(self):
        p=actual(speed=1.)
        t=candidate(p,[TrajectoryPoint(0.,0.,0.,0.,0.),TrajectoryPoint(0.,0.,0.,0.,.1)])
        t.emergency_stop=t.stop_required=True;t.target_speed=0.
        result=self.inspect(p,t=t)
        self.assertEqual('UNARMED',result['status'],result)
        self.assertEqual('brake_command_only_no_tracking_promise',result['details']['current_candidate_authority'])
        self.monitor.record_sent(p,t,self.envelope,100.,self.sent)
        self.assertFalse(self.monitor.history)

    def test_current_emergency_brake_does_not_erase_prior_actual_tracking_error(self):
        self.arm();p=actual(2,100.2,x=.2,y=.3)
        t=candidate(p,[TrajectoryPoint(.2,.3,0.,0.,0.),TrajectoryPoint(.2,.3,0.,0.,.1)])
        t.emergency_stop=t.stop_required=True;t.target_speed=0.
        self.assert_fault(self.inspect(p,t=t),'EXECUTION_POSITION_OR_TIME_ERROR')
        self.assert_fault(self.monitor.record_sent(p,t,self.envelope,100.2,self.sent),
                          'EXECUTION_POSITION_OR_TIME_ERROR')
        self.assertEqual('candidate-1',self.monitor.history[-1]['candidate_id'])


if __name__=='__main__':
    unittest.main()
