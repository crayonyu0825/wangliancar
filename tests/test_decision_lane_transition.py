"""D03 consumes the real producer across lane changes; all lamps/model are synthetic.

No live SDK, default capabilities or production action selection is changed.
"""
import copy
import json
import math
import unittest
from unittest.mock import patch

from core.behavior_contract import BehaviorFrame,TaskContext,Capabilities,BehaviorGoal,GoalPose
from core.behavior_channel import feedback_dict,feedback_from_dict
from core.interfaces import Trajectory
from core.serialization import perception_to_dict
from scripts.replay_decision import _assign
from core.interfaces import Perception
from members.decision.behaviors.environment import lane_candidates
from members.decision.behaviors.lane_change import LaneChangePolicy
from members.decision.behaviors.session import BehaviorSession
from members.decision.behaviors.observations import MotionObject
from members.control.controller import ControlEngine
from tests import test_original_lane_transition as producer_fixture
from tests import test_planning_behavior_contract as contract_fixture
from tests.test_decision_behavior_contract import feedback
from tests.test_control_maneuvers import BidirectionalPlant
from tests.control_benchmark import vehicle as control_vehicle
from tests.test_planning_candidate_validation import vehicle,limits


SOURCE,TARGET=producer_fixture.SOURCE,producer_fixture.TARGET


class DecisionLaneTransitionTests(unittest.TestCase):
    def setUp(self):
        self.producer=producer_fixture.OriginalLaneTransitionTests(); self.producer.setUp()
        self.addCleanup(self.producer.doCleanups)
        self.f=self.producer.f
        self.context=TaskContext(self.f.p.case_id,self.f.p.task_id,self.f.p.scene_id,'synthetic-d03-source-session')
        self.policy=LaneChangePolicy(3.9,.9,.9)

    def frame(self):
        p=self.f.p
        return BehaviorFrame(self.context,p.frame_id,self.f.now,p.valid_until,
            p.ego.x,p.ego.y,p.ego.heading,p.ego.speed,usable=True)

    def caps(self,f):
        # Explicit test prerequisite, never installed into runtime defaults.
        return Capabilities(f.context,('LANE_CHANGE','LIGHTS','PATH','FEEDBACK'),
                            f.observed_at_s,f.valid_until_s,usable=True)

    def candidates(self):
        return lane_candidates(self.f.p,self.frame(),self.f.clock,active_candidate=self.policy.selected)

    def advance(self,x=None,y=None,yaw=None,current=None,speed=None):
        self.f.now+=.049
        if speed is not None:
            self.f.p.ego.speed=speed
            self.f.p.ego.vx=speed*math.cos(yaw if yaw is not None else self.f.p.ego.heading)
            self.f.p.ego.vy=speed*math.sin(yaw if yaw is not None else self.f.p.ego.heading)
        self.producer.update(x,y,yaw,current)

    def evaluate(self,reply=None,cap=None,**changes):
        f=self.frame(); candidates,unused=self.candidates()
        return self.policy.evaluate(f,'lane-change:source-bound','LANE_CHANGE',candidates,self.caps(f),
                                    feedback=reply,speed_cap_mps=cap,**changes)

    def execute(self,cap=1.5,goal_pose=None):
        request=self.evaluate(cap=cap,goal_pose=goal_pose).request
        self.assertEqual('PREPARE',request.stage)
        self.advance()
        request=self.evaluate(feedback(request,self.frame(),producer='runtime',status='EXECUTING',
            lights_confirmed=True,lights_duration_s=1.)).request
        self.assertEqual('WAIT_GAP',request.stage)
        self.advance(); request=self.evaluate().request
        self.assertEqual('REQUEST_PATH',request.stage)
        self.advance(); request=self.evaluate(feedback(request,self.frame())).request
        self.assertEqual('EXECUTE',request.stage)
        return request

    def actual_reply(self,request=None,**changes):
        p=self.f.p
        values=dict(producer='runtime',status='EXECUTING',actual_lane_id=p.lane.lane_id,
                    actual_pose=GoalPose(p.ego.x,p.ego.y,p.ego.heading))
        values.update(changes)
        return feedback(request or self.policy.session.snapshot(self.frame()),self.frame(),**values)

    def test_selected_target_is_refreshed_as_current_with_original_direction(self):
        original=self.evaluate(cap=1.5).request
        self.advance(18.,2.2,.15,TARGET)
        candidates,failures=self.candidates()
        self.assertEqual([],failures)
        c=candidates[0]
        self.assertEqual((SOURCE,TARGET,'left',True),(c.source_lane_id,c.lane_id,c.side,c.is_current_lane))
        self.assertTrue(c.crossing_allowed)
        result=self.evaluate().request
        self.assertEqual(original.intent_id,result.intent_id)
        self.assertEqual(SOURCE,result.goal.source_lane_id)
        self.assertEqual(1.5,result.goal.speed_cap_mps)

    def test_active_pair_and_map_cannot_be_replaced_after_selection(self):
        self.evaluate(cap=1.5)
        original=copy.deepcopy(self.policy.selected)
        for field,value in (('source_lane_id','other'),('map_digest','a'*32)):
            active=copy.deepcopy(original); setattr(active,field,value)
            with self.assertRaises(ValueError):
                lane_candidates(self.f.p,self.frame(),self.f.clock,active)

    def test_missing_original_permission_keeps_fresh_target_but_blocks_straddling(self):
        self.execute(); self.advance(18.,2.2,.15,TARGET)
        self.f.route._maneuver_map.crossing_ranges.clear(); self.producer.update()
        candidates,failures=self.candidates()
        self.assertTrue(failures); self.assertTrue(candidates[0].is_current_lane)
        self.assertFalse(candidates[0].crossing_allowed)
        result=self.evaluate(self.actual_reply())
        self.assertTrue(result.hold_required)
        self.assertFalse(result.request.dispatch_allowed)
        self.assertEqual('RECOVER',result.phase)

    def test_settlement_does_not_request_another_crossing_permission(self):
        self.execute(); self.advance(29.,3.5,0.,TARGET)
        self.f.route._maneuver_map.crossing_ranges.clear(); self.producer.update()
        self.assertFalse(self.candidates()[0][0].crossing_allowed)
        for unused in range(3):
            result=self.evaluate(self.actual_reply())
            if result.request.status=='COMPLETED': break
            self.advance()
        self.assertEqual('COMPLETED',result.request.status)
        self.assertFalse(result.request.goal.light_intent.left_signal)

    def test_initial_window_is_not_reimposed_on_remaining_execution(self):
        self.execute(); self.advance(18.,2.2,.15,TARGET,speed=3.)
        candidates,unused=self.candidates(); c=candidates[0]
        c.crossing_ranges=[[(15.,1.75),(23.,1.75)]]
        self.assertFalse(self.policy.opportunity(self.frame(),c)[0])
        result=self.policy.evaluate(self.frame(),'lane-change:source-bound','LANE_CHANGE',candidates,self.caps(self.frame()))
        self.assertFalse(result.hold_required)
        self.assertEqual('EXECUTE',result.phase)
        # This preliminary decision check does not certify that a path can
        # cross this short opening; P02 still has to validate actual movement.

    def test_source_side_and_map_identity_changes_block_existing_intent(self):
        for field,value in (('source_lane_id','other'),('side','right'),('map_digest','a'*32)):
            self.policy=LaneChangePolicy(3.9,.9,.9)
            initial=self.evaluate(cap=1.5).request
            candidates,unused=self.candidates(); setattr(candidates[0],field,value)
            result=self.policy.evaluate(self.frame(),'lane-change:source-bound','LANE_CHANGE',candidates,self.caps(self.frame()))
            self.assertEqual(initial.intent_id,result.request.intent_id)
            self.assertFalse(result.request.dispatch_allowed)
            self.assertEqual('SELECTED_LANE_SOURCE_OR_MAP_CHANGED',result.reason)

    def test_stopped_ego_requires_explicit_cap_and_recovery_retains_it(self):
        self.assertIsNone(self.evaluate().request)
        self.assertIsNone(self.evaluate(cap=0.).request)
        initial=self.execute(cap=1.5)
        self.advance()
        candidates,unused=self.candidates(); candidates[0].objects=(MotionObject(9,15.,3.5,0.,0.,4.,2.),)
        blocked=self.policy.evaluate(self.frame(),'lane-change:source-bound','LANE_CHANGE',candidates,self.caps(self.frame()))
        self.assertEqual(0.,blocked.request.goal.speed_cap_mps)
        self.advance(speed=0.)
        recovered=self.evaluate().request
        self.assertEqual(initial.intent_id,recovered.intent_id)
        self.assertEqual('REQUEST_PATH',recovered.stage)
        self.assertEqual(1.5,recovered.goal.speed_cap_mps)
        self.assertEqual(SOURCE,recovered.goal.source_lane_id)

    def test_explicit_zero_cap_withdraws_motion_until_positive_authorization(self):
        self.execute(); self.advance()
        self.assertFalse(self.evaluate(cap=0.).request.dispatch_allowed)
        self.advance()
        blocked=self.evaluate().request
        self.assertFalse(blocked.dispatch_allowed); self.assertEqual(0.,blocked.goal.speed_cap_mps)
        self.advance()
        resumed=self.evaluate(cap=1.2).request
        self.assertEqual('REQUEST_PATH',resumed.stage)
        self.assertEqual(1.2,resumed.goal.speed_cap_mps)

    def test_body_occupancy_is_checked_even_when_completely_in_target(self):
        self.execute(); self.advance(29.,3.5,0.,TARGET)
        candidates,unused=self.candidates()
        candidates[0].crossing_allowed=False
        candidates[0].objects=(MotionObject(9,29.,3.5,0.,0.,4.,2.,None),)
        self.assertEqual((False,'TARGET_BODY_OCCUPIED'),self.policy.opportunity(self.frame(),candidates[0],True))

    def test_current_pose_and_feedback_body_must_both_be_settled(self):
        self.execute(); self.advance(18.,2.2,.15,TARGET)
        result=self.evaluate(self.actual_reply(actual_pose=GoalPose(29.,3.5,0.)))
        self.assertEqual('EXECUTE',result.phase)
        self.advance(29.,3.5,0.,TARGET)
        result=self.evaluate(self.actual_reply(actual_pose=GoalPose(18.,2.2,.15)))
        self.assertEqual('EXECUTE',result.phase)

    def test_stability_is_reset_by_missing_feedback_pause_and_safety_override(self):
        for interruption in ('missing','paused','safety'):
            self.policy=LaneChangePolicy(3.9,.9,.9)
            self.advance(10.,0.,0.,SOURCE)
            self.execute(); self.advance(29.,3.5,0.,TARGET)
            self.assertEqual('SETTLE',self.evaluate(self.actual_reply()).phase)
            self.advance(); self.evaluate(self.actual_reply())
            self.assertEqual(1,self.policy._settled)
            self.advance()
            if interruption=='missing': self.evaluate()
            elif interruption=='safety': self.evaluate(self.actual_reply(),safety_override=True)
            else:
                f=self.frame(); f.paused=True
                self.policy.evaluate(f,'lane-change:source-bound','LANE_CHANGE',self.candidates()[0],self.caps(f))
            self.assertEqual(0,self.policy._settled,interruption)
            self.assertNotEqual('COMPLETED',self.policy.session.status)

    def test_repeated_execution_feedback_cannot_finish_settlement(self):
        self.execute(); self.advance(29.,3.5,0.,TARGET)
        self.evaluate(self.actual_reply())
        self.advance(); reply=self.actual_reply(); self.evaluate(reply)
        self.assertEqual(1,self.policy._settled)
        self.advance(); result=self.evaluate(reply)
        self.assertNotEqual('COMPLETED',result.request.status)
        self.assertEqual(0,self.policy._settled)

    def test_active_intent_change_requires_cancellation(self):
        initial=self.execute(); self.advance()
        result=self.policy.evaluate(self.frame(),'different-goal','MERGE',self.candidates()[0],self.caps(self.frame()))
        self.assertEqual(initial.intent_id,result.request.intent_id)
        self.assertFalse(result.request.dispatch_allowed)
        self.assertEqual('ACTIVE_INTENT_CHANGE_REQUIRES_CANCEL',result.reason)

    def test_current_target_without_prior_selection_cannot_start_a_lane_change(self):
        self.evaluate(cap=1.5); active=self.policy.selected
        self.advance(29.,3.5,0.,TARGET)
        candidates,unused=lane_candidates(self.f.p,self.frame(),self.f.clock,active)
        fresh=LaneChangePolicy(3.9,.9,.9)
        result=fresh.evaluate(self.frame(),'new','LANE_CHANGE',candidates,self.caps(self.frame()),speed_cap_mps=1.5)
        self.assertIsNone(result.request)

    def test_json_perception_and_feedback_keep_original_source_and_deadlines(self):
        request=self.execute(); self.advance(18.,2.2,.15,TARGET)
        p=Perception(); _assign(p,json.loads(json.dumps(perception_to_dict(self.f.p))),'perception')
        candidates,unused=lane_candidates(p,self.frame(),self.f.clock,self.policy.selected)
        self.assertEqual(SOURCE,candidates[0].source_lane_id)
        payload=json.loads(json.dumps(request.to_dict('runtime_connected')))
        self.assertEqual(SOURCE,payload['source_lane_id'])
        reply=self.actual_reply(request)
        self.assertEqual(feedback_dict(reply),feedback_dict(feedback_from_dict(json.loads(json.dumps(feedback_dict(reply))))))
        self.f.now+=.21
        with self.assertRaises(ValueError): lane_candidates(p,self.frame(),self.f.clock,self.policy.selected)

    def test_aged_objects_are_not_treated_as_current_positions(self):
        self.evaluate(cap=1.5); candidates,unused=self.candidates(); c=candidates[0]
        c.objects=(MotionObject(9,50.,3.5,-20.,0.,4.,2.),)
        policy=LaneChangePolicy(3.9,.9,.9,acceleration_uncertainty_mps2=.01)
        # A short remaining model separates age from the ordinary horizon.
        policy.duration=.5
        self.assertTrue(policy.opportunity(self.frame(),c)[0])
        c.object_age_s=1.5
        self.assertFalse(policy.opportunity(self.frame(),c)[0])

    def test_expired_or_incomplete_sensor_cannot_be_used_for_settlement(self):
        self.execute(); self.advance(29.,3.5,0.,TARGET)
        self.f.p.maneuver_environment.coverage_regions=[]
        candidates,failures=self.candidates()
        self.assertEqual([],candidates); self.assertTrue(failures)

    def test_optional_public_source_is_semantic_and_preserved_on_block(self):
        f=self.frame(); session=BehaviorSession()
        old=BehaviorGoal(target_lane_id=TARGET,source_lane_id=SOURCE,speed_cap_mps=1.5)
        initial=session.start(f,'same','LANE_CHANGE','PREPARE',old)
        session.start(f,'same','LANE_CHANGE','PREPARE',BehaviorGoal(target_lane_id=TARGET,
            source_lane_id='other',speed_cap_mps=1.5))
        self.assertGreater(session.revision,initial.revision)
        self.assertEqual('other',session.block(f,'TEST_BLOCK').goal.source_lane_id)
        self.assertNotIn('source_lane_id',BehaviorGoal().to_dict())
        for invalid in ('',False,1):
            with self.assertRaises(ValueError): BehaviorGoal(source_lane_id=invalid)
        with self.assertRaises(ValueError): BehaviorGoal(target_lane_id=SOURCE,source_lane_id=SOURCE)

    def test_legacy_planning_consumer_rejects_explicit_source_instead_of_ignoring_it(self):
        c=contract_fixture.BehaviorContractTests(); c.setUp()
        for value,reason in ((SOURCE,'UNSUPPORTED_SOURCE_LANE'),('', 'SOURCE_LANE_INVALID')):
            c.request['source_lane_id']=value
            result=c.assess()
            self.assertFalse(result['eligible']); self.assertEqual(reason,result['reason_code'])

    def test_frozen_formal_goal_survives_recovery_and_rejects_moving_endpoint(self):
        goal=GoalPose(35.,3.5,0.)
        initial=self.execute(goal_pose=goal)
        self.advance(); candidates,unused=self.candidates()
        candidates[0].objects=(MotionObject(9,15.,3.5,0.,0.,4.,2.),)
        self.policy.evaluate(self.frame(),'lane-change:source-bound','LANE_CHANGE',candidates,self.caps(self.frame()))
        self.advance(); recovered=self.evaluate().request
        self.assertEqual(initial.goal.goal_pose.to_dict(),recovered.goal.goal_pose.to_dict())
        self.advance()
        moved=self.evaluate(goal_pose=GoalPose(36.,3.5,0.))
        self.assertFalse(moved.request.dispatch_allowed)
        self.assertEqual('ACTIVE_GOAL_CHANGE_REQUIRES_CANCEL',moved.reason)
        self.assertEqual(35.,moved.request.goal.goal_pose.x)

    def test_mutated_request_object_cannot_move_the_active_formal_target(self):
        initial=self.execute(goal_pose=GoalPose(35.,3.5,0.))
        initial.goal.goal_pose.x=40.; initial.goal.source_lane_id='other'
        self.advance(); result=self.evaluate()
        self.assertFalse(result.request.dispatch_allowed)
        self.assertEqual('ACTIVE_TARGET_BINDING_CHANGED',result.reason)
        self.assertEqual(SOURCE,result.request.goal.source_lane_id)
        self.assertEqual(35.,result.request.goal.goal_pose.x)

    def test_reversed_travel_right_signal_and_original_source_survive_lane_switch(self):
        self.f.p.ego.x,self.f.p.ego.heading=60.,math.pi
        self.producer.restrict('decrease','LHT')
        initial=self.execute(goal_pose=GoalPose(35.,3.5,math.pi))
        self.assertTrue(initial.goal.light_intent.right_signal)
        self.assertFalse(initial.goal.light_intent.left_signal)
        self.advance(50.,2.2,math.pi-.15,TARGET)
        self.assertEqual('right',self.candidates()[0][0].side)
        self.assertFalse(self.evaluate(self.actual_reply()).hold_required)
        self.advance(45.,3.5,math.pi,TARGET)
        for unused in range(3):
            result=self.evaluate(self.actual_reply())
            if result.request.status=='COMPLETED': break
            self.advance()
        self.assertEqual('COMPLETED',result.request.status)
        self.assertEqual(initial.intent_id,result.request.intent_id)

    def test_long_observation_gap_cannot_continue_stability_count(self):
        self.execute(); self.advance(29.,3.5,0.,TARGET)
        self.evaluate(self.actual_reply()); self.advance(); self.evaluate(self.actual_reply())
        self.assertEqual(1,self.policy._settled)
        self.f.now+=1.; self.advance()
        result=self.evaluate(self.actual_reply())
        self.assertEqual('SUSPENDED',result.request.status)
        self.assertEqual(0,self.policy._settled)

    def test_exhausted_recovery_never_restores_a_blocked_positive_speed(self):
        self.execute()
        for unused in range(3):
            self.advance(); candidates,ignored=self.candidates()
            candidates[0].objects=(MotionObject(9,15.,3.5,0.,0.,4.,2.),)
            self.policy.evaluate(self.frame(),'lane-change:source-bound','LANE_CHANGE',candidates,self.caps(self.frame()))
            self.advance(); result=self.evaluate()
            if result.request.dispatch_allowed or result.request.status=='REQUESTED':
                self.policy.session.advance(self.frame(),'EXECUTE')
        self.assertEqual('BLOCKED',result.request.status)
        self.assertEqual(0.,result.request.goal.speed_cap_mps)
        self.assertFalse(result.request.dispatch_allowed)

    def test_actual_producer_policy_generator_control_loop_completes_original_intent(self):
        clock_patch=patch('time.monotonic',self.f.clock)
        clock_patch.start(); self.addCleanup(clock_patch.stop)
        calibration=control_vehicle(); geometry=vehicle(3.9,.9,.9,calibration.wheelbase_m)
        motion=limits(max_front_steer_rad=calibration.front_steer_max_rad)
        first=self.producer.plan(speed_cap_mps=1.5,vehicle=geometry,limits=motion)
        self.assertEqual('safe',first['status'],first)
        goal=first['target_pose']; initial=self.execute(goal_pose=GoalPose(*goal))
        plant=BidirectionalPlant(x=10.,lag=.3); engine=ControlEngine(calibration,clock=self.f.clock)
        stages=set(); switched=False
        for unused in range(700):
            current=TARGET if plant.y>=1.75 else SOURCE
            self.advance(plant.x,plant.y,plant.heading,current,plant.speed)
            self.f.p.ego.yaw_rate=plant.yaw_rate
            result=self.evaluate(self.actual_reply(progress=min(1.,max(0.,(plant.x-10.)/25.))))
            stages.add(result.phase)
            if result.request.status=='COMPLETED': break
            self.assertFalse(result.hold_required,(plant.x,plant.y,result.reason))
            self.assertEqual(initial.intent_id,result.request.intent_id)
            self.assertEqual(SOURCE,result.request.goal.source_lane_id)
            switched=switched or current==TARGET
            curvature=math.tan(plant.steering*calibration.front_steer_max_rad/calibration.steering_sign)/calibration.wheelbase_m
            candidate=self.producer.plan(source_lane_id=result.request.goal.source_lane_id,
                fixed_goal_pose=(result.request.goal.goal_pose.x,result.request.goal.goal_pose.y,
                                 result.request.goal.goal_pose.body_heading_rad),
                speed_cap_mps=result.request.goal.speed_cap_mps,
                initial_curvature_m_inv=curvature,vehicle=geometry,limits=motion)
            self.assertEqual('safe',candidate['status'],candidate)
            self.assertEqual(goal,candidate['target_pose'])
            t=Trajectory().bind(self.f.p); t.valid=True; t.points=candidate['points']; t.target_speed=1.5
            t.target_lane_id=TARGET
            control=engine.compute(self.f.p,t)
            self.assertTrue(control.valid,control.errors)
            self.assertFalse(control.throttle>0 and control.brake>0)
            plant.step(control,.05,calibration)
        self.assertTrue(switched)
        self.assertIn('EXECUTE',stages); self.assertIn('SETTLE',stages)
        self.assertEqual('COMPLETED',result.request.status)
        self.assertEqual(initial.intent_id,result.request.intent_id)


if __name__=='__main__': unittest.main()
