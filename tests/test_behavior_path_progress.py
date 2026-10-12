"""Independent actual GPS progress tests; receipts never stand for movement."""
import copy
import unittest

from core.behavior_channel import BehaviorTransport
from core.interfaces import DecisionTarget,ControlOut,TrajectoryPoint
from tests.test_execution_monitor import actual,candidate


class BehaviorPathProgressTests(unittest.TestCase):
    def setUp(self):
        self.now=100.
        self.transport=BehaviorTransport(('PASSAGE','PATH','FEEDBACK'),lambda:self.now)
        self.bounds={'max_speed_mps':3.,'max_acceleration_mps2':2.,'position_tolerance_m':.02}
        self.original=[TrajectoryPoint(0.,0.,1.,0.,0.),TrajectoryPoint(10.,0.,1.,0.,10.)]

    def observe(self,frame,x,y=0.,speed=1.,points=None,receipt=None,bounds=None,goal=None,control_observation=None):
        p=actual(frame,self.now,x=x,y=y,speed=speed)
        self.transport.prepare(p)
        d=DecisionTarget().bind(p); d.valid=True
        identity={'intent_id':'passage','stage':'ENTER','revision':1}
        d.behavior_request=dict(identity,task_context=self.transport.context.to_dict(),
            produced_frame_id=frame,valid_until_s=p.valid_until,stop_distance_m=-1.,
            maneuver='PASSAGE',candidate_id='original-candidate')
        if goal is not None: d.behavior_request['goal_pose']=goal
        t=candidate(p,points=points or copy.deepcopy(self.original)); t.candidate_id='original-candidate'
        t.behavior_identity=identity
        t.planning_metrics={'progress_bounds':copy.deepcopy(self.bounds if bounds is None else bounds)}
        c=ControlOut().bind(t); c.valid=True; c.execution_observation=control_observation
        self.transport.observe(p,d,t,c,False,receipt)
        return self.transport.feedbacks[-1]

    def test_actual_progress_for_passage_without_goal_never_claims_arrival(self):
        first=self.observe(1,0.,receipt={'attempted':True,'ok':True})
        self.assertEqual(0.,first['progress']); self.assertEqual('EXECUTING',first['status'])
        self.now+=.1
        second=self.observe(2,.1)
        self.assertAlmostEqual(.01,second['progress'])
        self.assertEqual('gps_reachable_path_progress',second['evidence_level'])
        self.assertFalse(second['goal_pose_arrived']); self.assertFalse(second['hold_completed'])
        self.assertEqual(.1,second['planning_metrics']['actual_path_progress']['station_m'])

    def test_newly_shortened_replan_cannot_turn_small_actual_motion_into_completion(self):
        self.observe(1,0.); self.now+=.1
        short=[TrajectoryPoint(.1,0.,1.,0.,0.),TrajectoryPoint(.2,0.,1.,0.,.1)]
        reply=self.observe(2,.1,points=short)
        self.assertAlmostEqual(.01,reply['progress'])
        self.assertAlmostEqual(9.9,reply['planning_metrics']['actual_path_progress']['remaining_m'])

    def test_successful_send_with_stationary_actual_gps_produces_zero_progress(self):
        self.observe(1,0.); self.now+=.1
        reply=self.observe(2,0.,receipt={'attempted':True,'ok':True})
        self.assertEqual(0.,reply['progress']); self.assertFalse(reply['hold_completed'])

    def test_real_unreachable_pose_jump_is_failed_even_if_receipt_succeeds(self):
        self.observe(1,0.); self.now+=.1
        reply=self.observe(2,9.,receipt={'attempted':True,'ok':True})
        self.assertEqual('FAILED',reply['status']); self.assertEqual(0.,reply['progress'])
        self.assertEqual('PROGRESS_ACTUAL_POSE_JUMP',reply['reason_code'])

    def test_actual_side_departure_is_failed_in_original_path_window(self):
        self.observe(1,0.); self.now+=.1
        reply=self.observe(2,.1,y=.2)
        self.assertEqual('FAILED',reply['status'])
        self.assertIn('PROGRESS_',reply['reason_code'])

    def test_larger_replan_physical_bounds_cannot_allow_jump(self):
        self.observe(1,0.); self.now+=.1
        reply=self.observe(2,2.,bounds={'max_speed_mps':100.,'max_acceleration_mps2':100.,'position_tolerance_m':10.})
        self.assertEqual('FAILED',reply['status']); self.assertEqual('PROGRESS_ACTUAL_POSE_JUMP',reply['reason_code'])

    def test_repeated_gps_cannot_increase_progress_by_mutating_pose(self):
        self.observe(1,0.); self.now+=.1
        reply=self.observe(1,.1)
        self.assertEqual('FAILED',reply['status'])
        self.assertEqual('PROGRESS_REPEATED_FRAME_CHANGED_POSE',reply['reason_code'])

    def test_actual_speed_over_model_limit_is_failed(self):
        self.observe(1,0.); self.now+=.1
        reply=self.observe(2,.1,speed=3.1)
        self.assertEqual('FAILED',reply['status'])
        self.assertEqual('PROGRESS_ACTUAL_SPEED_OUTSIDE_LIMITS',reply['reason_code'])

    def test_changing_original_goal_in_replan_is_rejected(self):
        goal={'x':10.,'y':0.,'body_heading_rad':0.,'reference_point':'ego_rear_axle'}
        self.observe(1,0.,goal=goal); self.now+=.1
        goal['x']=.2; reply=self.observe(2,.1,goal=goal)
        self.assertEqual('FAILED',reply['status']); self.assertEqual('PROGRESS_ORIGINAL_GOAL_CHANGED',reply['reason_code'])

    def test_zero_projection_bound_is_rejected_without_default(self):
        bounds=copy.deepcopy(self.bounds); bounds['position_tolerance_m']=0.
        reply=self.observe(1,0.,bounds=bounds)
        self.assertEqual('FAILED',reply['status']); self.assertEqual('PROGRESS_BOUNDS_REQUIRED',reply['reason_code'])
        self.assertFalse(self.transport._path_bounds)

    def brake_hold(self,receipt=None,speed=0.,change=None,safety=False):
        p=actual(1,self.now,x=1.,speed=speed);self.transport.prepare(p)
        d=DecisionTarget().bind(p);d.valid=True
        identity=dict(intent_id='emergency',stage='EXECUTE_BRAKE',revision=2)
        goal=dict(x=3.,y=0.,body_heading_rad=0.,reference_point='ego_rear_axle')
        d.behavior_request=dict(identity,maneuver='EMERGENCY',candidate_id='frozen-brake',goal_pose=goal,
            task_context=self.transport.context.to_dict(),produced_frame_id=1,valid_until_s=p.valid_until,
            stop_distance_m=2.)
        t=candidate(p,points=[TrajectoryPoint(1.,0.,0.,0.,0.),TrajectoryPoint(1.,0.,0.,0.,.1)])
        t.candidate_id='frozen-brake';t.behavior_identity=identity;t.stop_required=True;t.target_speed=0.
        t.planning_metrics=dict(emergency_outcome='verified_safe_brake_stop',original_goal_pose=copy.deepcopy(goal))
        c=ControlOut().bind(t);c.valid=True;c.brake=.4;c.hazard_signal=True
        c.execution_observation=dict(identity=identity,candidate_id='frozen-brake',actual_brake_stop_confirmed=True,
                                     goal_pose_arrived=False,hold_completed=False)
        if change:change(p,d,t,c)
        self.transport.observe(p,d,t,c,safety,receipt)
        return next((v for v in self.transport.feedbacks if v['producer']=='runtime'),None)

    def test_safe_early_brake_stop_is_distinct_from_original_goal_and_path_completion(self):
        reply=self.brake_hold({'attempted':True,'ok':True})
        self.assertTrue(reply['actual_brake_stop_confirmed']);self.assertTrue(reply['actual_standstill_confirmed'])
        self.assertEqual('verified_actual_brake_stop',reply['evidence_level'])
        self.assertEqual('safe_brake_stop_observed_not_goal',reply['reason_code'])
        self.assertEqual(0.,reply['progress']);self.assertFalse(reply['goal_pose_arrived'])
        self.assertFalse(reply['hold_completed']);self.assertEqual('EXECUTING',reply['status'])
        self.assertEqual(1.,reply['actual_pose']['x'])
        self.assertEqual(3.,reply['planning_metrics']['original_goal_pose']['x'])

    def test_early_brake_stop_without_successful_actual_send_is_unconfirmed(self):
        for receipt in (None,{'attempted':False,'ok':True},{'attempted':True,'ok':False}):
            with self.subTest(receipt=receipt):
                reply=self.brake_hold(receipt)
                self.assertFalse(reply['actual_brake_stop_confirmed'])
                self.assertFalse(reply['actual_standstill_confirmed'])

    def test_controller_claim_cannot_override_actual_moving_or_unusable_gps_or_safety(self):
        sent={'attempted':True,'ok':True}
        reply=self.brake_hold(sent,speed=.01);self.assertFalse(reply['actual_brake_stop_confirmed'])
        reply=self.brake_hold(sent,change=lambda p,d,t,c:p.source_status['gps'].update(usable=False))
        self.assertFalse(reply['actual_brake_stop_confirmed'])
        reply=self.brake_hold(sent,safety=True);self.assertFalse(reply['actual_brake_stop_confirmed'])
        self.assertEqual('FAILED',reply['status'])

    def test_safe_brake_stop_requires_original_candidate_and_stationary_hold_purpose(self):
        sent={'attempted':True,'ok':True}
        changes=(lambda p,d,t,c:c.execution_observation.update(candidate_id='another'),
                 lambda p,d,t,c:t.planning_metrics.update(emergency_outcome='nominal'),
                 lambda p,d,t,c:t.planning_metrics['original_goal_pose'].update(x=1.),
                 lambda p,d,t,c:setattr(t.points[-1],'x',2.))
        for change in changes:
            with self.subTest(change=change):
                reply=self.brake_hold(sent,change=change)
                self.assertFalse(reply['actual_brake_stop_confirmed'])


if __name__=='__main__':
    unittest.main()
