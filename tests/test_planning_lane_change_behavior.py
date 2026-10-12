"""Actual fixed entry and transport, synthetic model/lamps, no SDK or sends."""
import copy
import json
import math
import unittest
from unittest.mock import patch

from core.behavior_channel import BehaviorTransport,feedback_from_dict,read_channel
from core.interfaces import DecisionTarget,DecisionMode,Trajectory,TrajectoryPoint,ControlOut
from core.validation import validate_output
from core.serialization import to_dict
from core.behavior_contract import GoalPose,Capabilities
from core.behavior_evidence import Evidence
from core.safety_supervisor import SafetySupervisor
from types import SimpleNamespace as NS
from members import planning_stub,control_stub
from members.planning.lane_change_behavior import LaneChangePlanningInputs
from members.planning.lane_planner import PlannerSettings
from members.planning.lane_change_generator import PredictionEnvelope
from members.planning.replay_bundle import _restore
from members.control.controller import ControlEngine
from tests import test_decision_lane_transition as decision_fixture
from tests.test_decision_behavior_contract import feedback
from tests.test_planning_candidate_validation import vehicle,limits,budget
from tests.test_planning_lane_change_generator import search
from tests.control_benchmark import vehicle as control_vehicle
from tests.test_control_maneuvers import BidirectionalPlant
from tests.test_maneuver_perception import target
from lane_change_integration import LaneChangeInputBridge,LaneChangeIntegrationInputs
from members.decision.behaviors.lane_change_runtime import LaneChangeDecisionInputs
from members.planning.candidate_validation import ExecutionEnvelope


SOURCE,TARGET=decision_fixture.SOURCE,decision_fixture.TARGET


class PlanningLaneChangeBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.h=decision_fixture.DecisionLaneTransitionTests(); self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.f=self.h.f
        manager=patch('time.monotonic',self.f.clock); manager.start(); self.addCleanup(manager.stop)
        self.calibration=control_vehicle()
        self.geometry=vehicle(3.9,.9,.9,self.calibration.wheelbase_m)
        self.motion=limits(max_front_steer_rad=self.calibration.front_steer_max_rad)
        self.settings=PlannerSettings(front_offset_m=3.9,rear_offset_m=.9,half_width_m=.9,
            wheelbase_m=self.geometry.wheelbase_m,front_steer_max_rad=self.motion.max_front_steer_rad)
        self.curvature=0.; self.input_changes={}
        self.goal=GoalPose(35.,3.5,0.)
        planning_stub.configure_planning(lane_change_inputs_provider=self.inputs)
        self.addCleanup(planning_stub.configure_planning)
        manager=patch('members.planning_stub._settings_with_control_geometry',lambda:self.settings)
        manager.start(); self.addCleanup(manager.stop)
        self.engine=ControlEngine(self.calibration,clock=self.f.clock)
        manager=patch('members.control_stub._engine',self.engine); manager.start(); self.addCleanup(manager.stop)
        self.transport=BehaviorTransport(('PATH','LANE_CHANGE','LIGHTS','FEEDBACK'),self.f.clock)
        self.transport.prepare(self.f.p)
        self.h.context=self.transport.context

    def inputs(self,p,context):
        value=LaneChangePlanningInputs(context,p.frame_id,self.f.now,p.valid_until,
            'synthetic-bicycle-and-empty-object-model-v1',self.curvature,self.geometry,self.motion,
            search(),budget(max_checks=1000000),{})
        for key,item in self.input_changes.items(): setattr(value,key,item)
        return value

    def decision(self,request):
        p=self.f.p
        self.transport.prepare(p)
        d=DecisionTarget().bind(p); d.valid=True
        d.target_lane_id=TARGET; d.target_speed=1.5
        d.behavior_request=request.to_dict('runtime_connected') if hasattr(request,'to_dict') else copy.deepcopy(request)
        d.behavior_active_identity={key:d.behavior_request[key] for key in
            ('intent_id','stage','revision','source_frame_id','issued_at_s')}
        return d

    def request_path(self):
        r=self.h.evaluate(cap=1.5,goal_pose=self.goal).request
        self.h.advance()
        r=self.h.evaluate(feedback(r,self.h.frame(),producer='runtime',status='EXECUTING',
            lights_confirmed=True,lights_duration_s=1.)).request
        self.h.advance(); r=self.h.evaluate().request
        self.assertEqual('REQUEST_PATH',r.stage)
        return r

    def prepare(self):
        r=self.request_path(); d=self.decision(r); t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        self.assertEqual('PLANNED',t.behavior_feedback['status'],t.errors)
        self.assertTrue(all(abs(v.y)<1e-8 for v in t.points))
        self.transport.observe(self.f.p,d,t,ControlOut().bind(t))
        return r,d,t

    def execute(self):
        r,d,t=self.prepare()
        self.h.advance()
        r=self.h.evaluate(feedback_from_dict(t.behavior_feedback)).request
        self.assertEqual('EXECUTE',r.stage)
        d=self.decision(r); t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        return r,d,t

    def reject(self,d,reason):
        t=planning_stub.plan(self.f.p,d)
        self.assertFalse(t.valid); self.assertFalse(t.points); self.assertIsNone(t.behavior_identity)
        self.assertIn(reason,t.reason)
        return t

    def monitored_supervisor(self):
        # This legacy component fixture supplies an explicit zero-error authored
        # candidate model. It does not arm execution monitoring or certify a car.
        def paired(p,context):
            e=Evidence(context,p.frame_id,self.f.now,p.valid_until,True,True,
                       'legacy-authored-component-model',source_kind='task')
            d=LaneChangeDecisionInputs(e,e,'synthetic-bicycle-and-empty-object-model-v1',
                'legacy-authored-lane-task',SOURCE,(3.9,.9,.9),3.)
            m=self.inputs(p,context); m.execution_envelope=ExecutionEnvelope(0.,0.,0.,0.)
            m.speed_error_mps=.5
            return LaneChangeIntegrationInputs(d,m,budget(max_checks=1000000))
        bridge=LaneChangeInputBridge(paired,self.f.clock); bridge.settings=self.settings
        def assess(p,d,t,c):
            bridge.prepare(p)
            return bridge.assess(p,d,t,c)
        return SafetySupervisor(NS(send_control=True),self.f.clock,planning_settings=self.settings,
                                lane_change_guard=assess)

    def test_request_path_ack_stays_in_current_lane_before_execute_without_fault_latch(self):
        r,d,t=self.prepare()
        self.assertEqual('LANE_CHANGE_PATH_PREPARED_WAIT_EXECUTE',t.reason)
        self.assertIsNone(t.behavior_identity)
        self.assertEqual(r.intent_id,self.transport.feedbacks[0]['intent_id'])
        c=control_stub.compute_control(self.f.p,t)
        self.assertTrue(c.valid,c.errors)
        supervisor=self.monitored_supervisor()
        safety=supervisor.evaluate(self.f.p,d,t,c)
        self.assertFalse(safety.active,safety.reason)

    def test_explicit_model_and_current_capabilities_both_required(self):
        r=self.request_path(); d=self.decision(r)
        planning_stub.configure_planning()
        self.reject(d,'LANE_CHANGE_MODEL_INPUTS_UNAVAILABLE')
        planning_stub.configure_planning(lane_change_inputs_provider=self.inputs)
        self.f.p.behavior_channel['capabilities']['actions']=['PATH_STOP','DWELL','FEEDBACK']
        t=self.reject(d,'CAPABILITY_ACTIONS_MISSING')
        self.assertEqual('REJECTED',t.behavior_feedback['status'])

    def test_model_snapshot_cannot_borrow_other_frame_context_clock_or_deadline(self):
        r=self.request_path(); d=self.decision(r)
        for key,value in (('frame_id',r.produced_frame_id-1),('context',None),
            ('clock_id','other'),('produced_at_s',self.f.now+.1),
            ('valid_until_s',self.f.now),('valid_until_s',self.f.p.valid_until+.1)):
            self.input_changes={key:value}
            self.reject(d,'LANE_CHANGE_MODEL_SOURCE_MISMATCH_OR_EXPIRED')

    def test_missing_curvature_or_model_revision_never_uses_gps_steering(self):
        r=self.request_path(); d=self.decision(r)
        self.f.p.ego.steering=0.
        for changes in ({'initial_curvature_m_inv':None},{'model_id':''}):
            self.input_changes=changes
            self.reject(d,'LANE_CHANGE_MOTION_MODEL_UNAVAILABLE')

    def test_provider_failure_is_explicit_and_default_action_set_is_unchanged(self):
        def failed(p,context): raise ValueError('SOURCE_READ_FAILED')
        planning_stub.configure_planning(lane_change_inputs_provider=failed)
        self.reject(self.decision(self.request_path()),'SOURCE_READ_FAILED')
        self.assertEqual((),BehaviorTransport().actions)

    def test_preparation_stays_current_and_terminal_stages_cannot_dispatch(self):
        r=self.request_path(); payload=r.to_dict('runtime_connected')
        for stage in ('COMPLETE','RECOVER'):
            payload['stage']=stage
            self.reject(self.decision(payload),'LANE_CHANGE_STAGE_NOT_PLANNABLE')
        for stage in ('PREPARE','WAIT_GAP'):
            payload['stage']=stage; payload['revision']+=1
            d=self.decision(payload); d.target_lane_id=SOURCE
            d.stop_distance=8.; d.precision_stop=True
            t=planning_stub.plan(self.f.p,d)
            self.assertTrue(t.valid,t.errors)
            self.assertIsNone(t.behavior_identity); self.assertIsNone(t.behavior_feedback)
            self.assertIsNone(t.candidate_id)
            self.assertTrue(all(abs(v.y)<1e-9 for v in t.points))
            self.assertLessEqual(t.points[-1].x-self.f.p.ego.x,8.+1e-9)

    def test_cold_execute_requires_successful_request_path_for_same_intent(self):
        r=self.request_path(); payload=r.to_dict('runtime_connected'); payload['stage']='EXECUTE'
        self.reject(self.decision(payload),'LANE_CHANGE_REQUEST_PATH_REQUIRED')

    def test_stop_emergency_and_red_keep_authority_over_lane_change(self):
        r,d,t=self.execute()
        for mode,distance,precision in ((DecisionMode.STOP,0.,False),
            (DecisionMode.EMERGENCY_BRAKE,-1.,False),(DecisionMode.KEEP_LANE,20.,True)):
            altered=copy.deepcopy(d); altered.mode=mode; altered.stop_distance=distance
            altered.precision_stop=precision
            self.reject(altered,'LANE_CHANGE_OVERRIDDEN_BY_STOP')
        self.f.p.traffic.required=True
        self.reject(d,'LANE_CHANGE_SIGNAL_STOP_REQUIRED')

    def test_target_pair_pose_direction_stop_and_lights_are_not_lowered_to_forward(self):
        r=self.request_path(); original=r.to_dict('runtime_connected')
        for changes,reason in (({'source_lane_id':None},'LANE_CHANGE_LANE_PAIR_INVALID'),
            ({'target_lane_id':SOURCE},'LANE_CHANGE_LANE_PAIR_INVALID'),
            ({'goal_pose':None},'LANE_CHANGE_FIXED_POSE_REQUIRED'),
            ({'motion_direction':-1},'UNSUPPORTED_DIRECTION'),
            ({'minimum_standstill_duration_s':10.,'stop_distance_m':0.},'LANE_CHANGE_GOAL_EXTENSIONS_UNSUPPORTED'),
            ({'light_intent':dict(left_signal=False,right_signal=False,hazard_signal=False)},'LANE_CHANGE_INDICATOR_REQUIRED')):
            payload=copy.deepcopy(original); payload.update(changes)
            self.reject(self.decision(payload),reason)

    def test_indicator_is_bound_to_actual_original_crossing_side(self):
        r=self.request_path(); payload=r.to_dict('runtime_connected')
        payload['light_intent']=dict(left_signal=False,right_signal=True,hazard_signal=False)
        self.reject(self.decision(payload),'LANE_CHANGE_INDICATOR_SIDE_MISMATCH')

    def test_active_target_map_or_model_cannot_move_under_new_revision(self):
        r,d,t=self.execute(); payload=copy.deepcopy(d.behavior_request)
        payload['revision']+=1; payload['goal_pose']['x']+=1.
        self.reject(self.decision(payload),'LANE_CHANGE_ACTIVE_BINDING_CHANGED')
        self.input_changes={'model_id':'another-model'}
        current=copy.deepcopy(d.behavior_request); current['revision']=payload['revision']+1
        self.reject(self.decision(current),'LANE_CHANGE_ACTIVE_BINDING_CHANGED')

    def test_reused_revision_cannot_change_stage_or_issued_identity(self):
        r,d,t=self.execute()
        for key,value in (('stage','SETTLE'),('issued_at_s',d.behavior_request['issued_at_s']-.01)):
            payload=copy.deepcopy(d.behavior_request); payload[key]=value
            self.reject(self.decision(payload),'REQUEST_REVISION_REGRESSED_OR_CHANGED')
        payload=copy.deepcopy(d.behavior_request)
        payload['revision']+=1; payload['source_frame_id']-=1
        self.reject(self.decision(payload),'REQUEST_REVISION_REGRESSED_OR_CHANGED')

    def test_model_cannot_disagree_with_control_geometry(self):
        r=self.request_path(); self.settings.wheelbase_m+=.2
        self.reject(self.decision(r),'LANE_CHANGE_CONTROL_GEOMETRY_MISMATCH')

    def test_bounded_work_and_incomplete_sensor_withdraw_path(self):
        r=self.request_path(); d=self.decision(r)
        self.input_changes={'budget':budget(max_checks=1)}
        self.reject(d,'BUDGET_EXHAUSTED')
        self.input_changes={}; self.f.p.maneuver_environment.coverage_regions=[]
        t=planning_stub.plan(self.f.p,d)
        self.assertFalse(t.valid); self.assertFalse(t.points)

    def test_source_that_expires_during_generation_cannot_be_published(self):
        from members.planning.lane_change_behavior import plan_lane_change_candidate
        r=self.request_path(); d=self.decision(r)
        def delayed(*args,**kwargs):
            value=plan_lane_change_candidate(*args,**kwargs)
            self.f.now+=.3
            return value
        with patch('members.planning.lane_change_behavior.plan_lane_change_candidate',delayed):
            t=self.reject(d,'LANE_CHANGE_SOURCE_EXPIRED_DURING_DISPATCH')
        self.assertFalse(t.behavior_feedback['usable'])

    def test_output_deadline_preserves_short_model_capability_and_source_lifetimes(self):
        r,d,t=self.execute()
        self.input_changes={'valid_until_s':self.f.now+.04}
        self.f.p.behavior_channel['capabilities']['valid_until_s']=self.f.now+.03
        t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        self.assertLessEqual(t.valid_until,self.f.now+.03)
        self.assertEqual(t.valid_until,t.behavior_feedback['valid_until_s'])

    def test_moving_identity_roundtrip_and_actual_tracking_feedback_do_not_claim_arrival(self):
        r,d,t=self.execute()
        replay=_restore(Trajectory(),json.loads(json.dumps(to_dict(t))),'trajectory')
        replay.points=[TrajectoryPoint(**point) for point in replay.points]
        validate_output(replay,Trajectory,d)
        c=control_stub.compute_control(self.f.p,replay)
        self.assertTrue(c.valid,c.errors)
        self.assertEqual(t.behavior_identity,c.execution_observation['identity'])
        self.assertFalse(c.execution_observation['goal_pose_arrived'])
        self.assertFalse(c.execution_observation['hold_completed'])
        self.transport.observe(self.f.p,d,replay,c)
        observed=feedback_from_dict(self.transport.feedbacks[-1])
        self.assertEqual('runtime',observed.producer)
        self.assertEqual(SOURCE,observed.actual_lane_id)
        self.assertEqual(self.f.p.ego.x,observed.actual_pose.x)
        self.assertFalse(observed.lights_confirmed)
        self.assertFalse(observed.goal_pose_arrived)

    def test_unacknowledged_or_mismatched_moving_identity_is_rejected(self):
        r,d,t=self.execute()
        for changes in ({'behavior_feedback':None},{'hold_duration_s':1.},
            {'behavior_identity':dict(t.behavior_identity,revision=r.revision+1)}):
            altered=copy.deepcopy(t)
            for key,value in changes.items(): setattr(altered,key,value)
            with self.assertRaises(ValueError): validate_output(altered,Trajectory,d)

    def test_new_stage_rechecks_its_new_capabilities_without_replaying_feedback(self):
        r=self.h.evaluate(cap=1.5,goal_pose=self.goal).request
        self.h.advance()
        r=self.h.evaluate(feedback(r,self.h.frame(),producer='runtime',status='EXECUTING',
            lights_confirmed=True,lights_duration_s=1.)).request
        self.assertEqual('WAIT_GAP',r.stage); self.assertTrue(r.dispatch_allowed)
        self.assertIsNone(self.h.policy.session.last_feedback)
        self.h.advance(); f=self.h.frame(); candidates,unused=self.h.candidates()
        no_path=Capabilities(f.context,('LANE_CHANGE','LIGHTS','FEEDBACK'),
            f.observed_at_s,f.valid_until_s,usable=True)
        result=self.h.policy.evaluate(f,'lane-change:source-bound','LANE_CHANGE',candidates,no_path)
        self.assertEqual('REQUEST_PATH',result.request.stage)
        self.assertFalse(result.request.dispatch_allowed); self.assertTrue(result.hold_required)
        self.assertEqual('CAPABILITY_OR_CONTRACT_UNAVAILABLE',result.reason)
        self.assertEqual(0.,self.h.policy.session._elapsed)

    def test_current_sdk_target_keeps_original_source_and_fixed_endpoint(self):
        r,d,t=self.execute()
        self.h.advance(18.,2.2,.15,TARGET)
        r=self.h.policy.session.snapshot(self.h.frame())
        t=planning_stub.plan(self.f.p,self.decision(r))
        self.assertTrue(t.valid,t.errors)
        self.assertEqual((35.,3.5),(t.points[-1].x,t.points[-1].y))
        self.assertEqual(SOURCE,r.goal.source_lane_id)

    def test_monitor_target_only_continuation_still_checks_actual_body_and_remaining_path(self):
        r,d,t=self.execute()
        self.h.advance(25.,3.5,0.,TARGET,1.5)
        # The original neighbor is genuinely absent in this formal snapshot.
        # No old crossing certificate is borrowed once the whole body fits B.
        self.f.p.maneuver_environment.neighbor_lanes=[]
        r=self.h.policy.session.snapshot(self.h.frame());d=self.decision(r)
        t=planning_stub.plan(self.f.p,d);self.assertTrue(t.valid,t.errors)
        c=control_stub.compute_control(self.f.p,t);self.assertTrue(c.valid,c.errors)
        self.assertFalse(self.monitored_supervisor().evaluate(self.f.p,d,t,c).active)
        altered=copy.deepcopy(t);altered.points[len(altered.points)//2].y=1.5
        self.assertTrue(self.monitored_supervisor().evaluate(self.f.p,d,altered,c).active)

    def test_actual_map_replacement_with_same_lane_ids_rejects_active_binding(self):
        r,d,t=self.execute(); self.h.producer.restrict('decrease')
        r=self.h.policy.session.snapshot(self.h.frame())
        self.reject(self.decision(r),'LANE_CHANGE_ACTIVE_BINDING_CHANGED')

    def test_reversed_travel_produces_right_indicator_with_frozen_original_goal(self):
        self.f.p.ego.x,self.f.p.ego.heading=60.,math.pi
        self.h.producer.restrict('decrease','LHT')
        self.goal=GoalPose(35.,3.5,math.pi)
        r,d,t=self.execute()
        self.assertTrue(t.right_signal); self.assertFalse(t.left_signal)
        self.assertEqual(math.pi,t.points[0].heading)
        self.assertEqual(SOURCE,d.behavior_request['source_lane_id'])

    def test_new_formal_object_needs_its_own_prediction_and_rejects_collision(self):
        r,d,t=self.execute()
        self.f.p.targets=[target(24.,2.,heading=None)]; self.h.advance()
        r=self.h.policy.session.snapshot(self.h.frame()); d=self.decision(r)
        self.reject(d,'PREDICTION_OBJECT_SET_MISMATCH')
        self.input_changes={'prediction_envelopes':{9:PredictionEnvelope(60.,.1,.1)}}
        t=planning_stub.plan(self.f.p,d)
        self.assertFalse(t.valid); self.assertFalse(t.points)

    def test_actual_gps_distance_progress_survives_each_replan_but_never_arrives_by_itself(self):
        r,d,t=self.execute()
        c=control_stub.compute_control(self.f.p,t)
        self.transport.observe(self.f.p,d,t,c)
        self.h.advance(12.,.05,.01,SOURCE,.5)
        r=self.h.policy.session.snapshot(self.h.frame()); d=self.decision(r)
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        c=control_stub.compute_control(self.f.p,t)
        self.transport.observe(self.f.p,d,t,c)
        actual=feedback_from_dict(self.transport.feedbacks[-1])
        self.assertGreater(actual.progress,0.)
        self.assertFalse(actual.goal_pose_arrived); self.assertFalse(actual.hold_completed)
        self.assertFalse(actual.lights_confirmed)

    def test_staging_preserves_the_lower_request_speed_cap(self):
        r=self.request_path(); r.goal.speed_cap_mps=1.
        t=planning_stub.plan(self.f.p,self.decision(r))
        self.assertTrue(t.valid,t.errors)
        self.assertLessEqual(t.target_speed,1.)
        self.assertTrue(all(v.speed<=1.+1e-9 for v in t.points))

    def test_straddling_recovery_cannot_snap_to_a_lane_as_an_acknowledgement(self):
        r,d,t=self.execute(); self.h.advance(18.,2.2,.15,TARGET)
        payload=self.h.policy.session.snapshot(self.h.frame()).to_dict('runtime_connected')
        payload['revision']+=1; payload['stage']='REQUEST_PATH'
        t=self.reject(self.decision(payload),'LANE_CHANGE_STAGING_')
        self.assertEqual('REJECTED',t.behavior_feedback['status'])

    def test_staging_checks_current_curvature_to_first_segment_steering_rate(self):
        self.f.p.ego.speed=self.f.p.ego.vx=2.
        self.curvature=.1; self.motion.max_steer_rate_rad_s=.2
        r=self.request_path(); d=self.decision(r); model=self.inputs(self.f.p,self.h.context)
        with self.assertRaisesRegex(ValueError,'LANE_CHANGE_STAGING_INITIAL_STEERING_RATE_LIMIT'):
            planning_stub._LANE_CHANGE_PLANNER._staging(self.f.p,d,self.settings,model,0,self.f.p.valid_until,2.5)

    def test_actual_fixed_entries_transport_and_policy_complete_original_intent(self):
        self.assert_continuous_completion()

    def assert_continuous_completion(self,withdraw_original=False):
        r,d,t=self.execute(); initial=r.intent_id
        plant=BidirectionalPlant(x=10.,lag=.3); stages=set(); switched=False; withdrawn=False
        supervisor=self.monitored_supervisor()
        for unused in range(700):
            c=control_stub.compute_control(self.f.p,t)
            self.assertTrue(c.valid,c.errors); validate_output(c,ControlOut,t)
            safety=supervisor.evaluate(self.f.p,d,t,c)
            self.assertFalse(safety.active,safety.reason)
            self.transport.observe(self.f.p,d,t,c)
            plant.step(c,.05,self.calibration)
            current=TARGET if plant.y>=1.75 else SOURCE
            self.h.advance(plant.x,plant.y,plant.heading,current,plant.speed)
            corners=[plant.y+along*math.sin(plant.heading)+across*math.cos(plant.heading)
                     for along in (-.9,3.9) for across in (-.9,.9)]
            if withdraw_original and current==TARGET and min(corners)>1.8 and max(corners)<5.2:
                withdrawn=True
            if withdrawn:
                # The current formal B geometry/objects/view remain. The
                # original neighbor and its permission are no longer supplied.
                self.f.p.maneuver_environment.neighbor_lanes=[]
                self.f.p.maneuver_environment.road_regions=[v for v in
                    self.f.p.maneuver_environment.road_regions if v['lane_id']==TARGET]
            self.f.p.ego.yaw_rate=plant.yaw_rate
            self.curvature=math.tan(plant.steering*self.calibration.front_steer_max_rad/
                                    self.calibration.steering_sign)/self.calibration.wheelbase_m
            self.transport.prepare(self.f.p)
            context,caps,replies=read_channel(self.f.p,self.f.now)
            matched=[v for v in replies if v.intent_id==r.intent_id and v.revision==r.revision and v.stage==r.stage]
            actual=max(matched,key=lambda v:(v.produced_at_s,v.producer!='planning')) if matched else None
            result=self.h.evaluate(actual); r=result.request
            stages.add(result.phase); switched=switched or current==TARGET
            self.assertEqual(initial,r.intent_id)
            if r.status=='COMPLETED': break
            self.assertFalse(result.hold_required,result.reason)
            d=self.decision(r); t=planning_stub.plan(self.f.p,d)
            self.assertTrue(t.valid,(plant.x,plant.y,t.errors,result.reason))
            if withdrawn: self.assertEqual('CURRENT_TARGET_NOMINAL_CONTINUATION_PATH',t.reason)
            self.assertEqual(GoalPose(35.,3.5,0.).to_dict(),d.behavior_request['goal_pose'])
        self.assertTrue(switched)
        self.assertIn('EXECUTE',stages); self.assertIn('SETTLE',stages)
        self.assertEqual('COMPLETED',r.status)
        if withdraw_original: self.assertTrue(withdrawn)


if __name__=='__main__': unittest.main()
