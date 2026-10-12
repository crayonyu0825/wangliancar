"""Real fixed entries, formal synthetic sources and actual stopped segments."""
import copy
import json
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.behavior_channel import read_channel
from core.behavior_contract import BehaviorGoal,GoalPose,ExecutionFeedback
from core.serialization import to_dict
from core.safety_supervisor import SafetySupervisor
from members import decision_stub,planning_stub,control_stub
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from members.planning.parking_generator import ParkingSearch
from parking_integration import ParkingInputBridge,ParkingIntegrationInputs
from simone_platform.map_observations import MapObservationReader
from tests.test_map_observations import parking,pt,Vector,map_api
from tests.test_planning_candidate_validation import budget
from tests.test_control_maneuvers import BidirectionalPlant
from tests import test_planning_parking_behavior as existing
from tests.test_planning_parking_lattice import choices
from members.planning.candidate_validation import ExecutionEnvelope


class ParkingSegmentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.h=existing.ParkingFixedEntryTests(); self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.f=self.h.f
        self.execution_envelope=None; self.speed_error_mps=None
        original_build=self.f.build
        def same_frame_build():
            # This fixture authors genuine matching increasing synthetic
            # headers; raw SDK milliseconds are never a process clock.
            p=self.f.p; p.timestamp=p.frame_id*50
            p.ego.timestamp=p.targets_timestamp=p.timestamp
            p.source_status['gps'].update(frame_id=p.frame_id,timestamp=p.timestamp,
                                           observed_at_s=self.f.now,quality='fresh')
            p.source_status['targets'].update(frame_id=p.frame_id,timestamp=p.timestamp)
            return original_build()
        self.f.build=same_frame_build
        native=parking(7); native.pt=pt(20.,0.); native.heading=pt(1.,0.)
        native.boundaryKnots=Vector([pt(10.,-1.5),pt(30.,-1.5),pt(30.,1.5),pt(10.,1.5)])
        catalogue=MapObservationReader(map_api(parks=[native])).catalog()
        self.f.p.parking_spaces=catalogue['parking_spaces']
        self.f.p.ego.x=12.
        self.f.p.map_observation_status=catalogue['map_observation_status']; self.f.build()
        original=self.h.h.mission
        self.h.h.mission=lambda **changes:original(**dict(dict(body_heading_rad=0.,access_edge_index=1,
            approach_pose=GoalPose(15.,0.,0.),position_pose=GoalPose(16.,0.,0.),exit_goal=GoalPose(34.,0.,0.)),**changes))
        def planning_inputs(p,context):
            value=self.h.inputs(p,context)
            value.search=ParkingSearch((1.,),(1.,),.2,1000,lattice_search=choices(
                step_lengths_m=(.8,),curvature_fractions=(0.,),connector_distance_m=4.,
                max_nodes=64,max_expansions=32,max_direction_changes=1,max_path_length_m=6.))
            value.execution_envelope=copy.deepcopy(self.execution_envelope)
            value.speed_error_mps=self.speed_error_mps
            return value
        def paired(p,context):
            return ParkingIntegrationInputs(self.h.decision_inputs(p,context),planning_inputs(p,context),
                                           budget(max_checks=1000000))
        self.bridge=ParkingInputBridge(paired,self.f.clock); self.bridge.settings=self.h.settings
        self.engine=DecisionEngine(DecisionSettings(front_offset_m=3.9,half_width_m=.9),
                                  self.f.clock,self.bridge.decision_inputs)
        manager=patch('members.decision_stub._ENGINE',self.engine); manager.start(); self.addCleanup(manager.stop)
        planning_stub.configure_planning(parking_inputs_provider=self.bridge.planning_inputs)
        self.safety=SafetySupervisor(SimpleNamespace(),self.f.clock,self.h.settings,self.engine.settings,
                                     parking_guard=self.bridge.assess,execution_guard=self.bridge.assess_execution)

    def cycle(self):
        p=self.f.p; self.bridge.prepare(p)
        d=decision_stub.decide(p)
        t=planning_stub.plan(p,d); self.assertTrue(t.valid,t.errors)
        c=control_stub.compute_control(p,t); self.assertTrue(c.valid,c.errors)
        assessment=self.safety.evaluate(p,d,t,c)
        self.assertFalse(assessment.active,assessment.reason)
        self.h.transport.observe(p,d,t,c,assessment.active)
        return d,t,c

    def test_reverse_then_forward_segments_actual_feedback_dwell_and_exit(self):
        self.run_closed_loop()

    def test_execution_bounded_sent_reverse_forward_dwell_and_exit(self):
        # Explicit synthetic engineering bounds, not vehicle calibration data.
        self.execution_envelope=ExecutionEnvelope(.45,.02,.6,1.5)
        self.speed_error_mps=1.
        self.run_closed_loop(robust=True)

    def run_closed_loop(self,robust=False):
        p=self.f.p
        plant=BidirectionalPlant(p.ego.x,p.ego.y,p.ego.heading,lag=.3)
        directions=[]; segment_ids=[]; proposed=False; completed=False
        parked_at=None; exited_at=None
        receipts=0; tracked=0; bounded_progress=0
        trace=[]
        artifact_dir=os.environ.get('WANGLIANCAR_PARKING_EVIDENCE_DIR') if robust else None
        artifact=None
        if artifact_dir:
            os.makedirs(artifact_dir,exist_ok=True)
            artifact=open(os.path.join(artifact_dir,'parking_multisegment_closed_loop.jsonl'),'w',encoding='utf-8')
            self.addCleanup(artifact.close)
            manifest=dict(recording_kind='synthetic_closed_loop',data_kind='synthetic',
                send_kind='synthetic_success_receipt_no_SDK',capture_clock='process_monotonic_seconds',
                execution_envelope=vars(self.execution_envelope),speed_error_mps=self.speed_error_mps,
                geometry=vars(self.h.geometry),limits=vars(self.h.motion),plant=dict(lag_s=.3,dt_s=.05),
                claims=dict(simone_acceptance='not_evaluated',score='not_evaluated'))
            manifest['data_source']='authored_native_shapes_and_sensor_headers'
            mission=self.h.h.mission()
            manifest['mission']=dict(mission_id=mission.mission_id,space_id=mission.space_id,
                road_lane_ids=list(mission.road_lane_ids),access_edge_index=mission.access_edge_index,
                body_heading_rad=mission.body_heading_rad,approach_pose=mission.approach_pose.to_dict(),
                position_pose=mission.position_pose.to_dict(),exit_goal=mission.exit_goal.to_dict())
            manifest['arrival_tolerances']=dict(position_m=.15,heading_rad=.15,standstill_speed_mps=.05)
            context=read_channel(p,self.f.now)[0]
            model=self.bridge.provider(p,context).planning
            manifest['model_id']=model.model_id
            manifest['search']=dict(vars(model.search))
            manifest['search']['lattice_search']=dict(vars(model.search.lattice_search))
            manifest['search']['execution_envelope']=None
            manifest['search']['dynamic_search']=None
            with open(os.path.join(artifact_dir,'parking_multisegment_manifest.json'),'w',encoding='utf-8') as stream:
                json.dump(manifest,stream,indent=2,sort_keys=True)
        for tick in range(1600):
            self.bridge.prepare(p)
            d=decision_stub.decide(p)
            state=decision_stub.decision_behavior_info().get('parking',{})
            if state.get('status')=='COMPLETED': completed=True; break
            self.assertIsNotNone(d.behavior_request,(tick,state,d.reason,plant.x,plant.speed,
                self.engine._behavior.parking.policy.session.goal.to_dict(),self.h.transport.feedbacks))
            t=planning_stub.plan(p,d)
            self.assertTrue(t.valid,(tick,d.behavior_request['stage'],t.errors,plant.x,plant.speed))
            c=control_stub.compute_control(p,t); self.assertTrue(c.valid,c.errors)
            trace.append((tick,plant.x,plant.speed,c.throttle,c.brake,c.handbrake,
                c.diagnostics.get('reference_speed_mps'),c.diagnostics.get('reference_acceleration_mps2'),
                c.diagnostics.get('integral_m')))
            del trace[:-12]
            assessed=self.safety.evaluate(p,d,t,c); self.assertFalse(assessed.active,(tick,assessed.reason))
            receipt=dict(attempted=True,ok=True,reason='synthetic_plant_command_applied') if robust else None
            sent=self.bridge.record_sent(p,t,receipt,safety_active=assessed.active) if robust else None
            if robust:
                self.assertFalse(sent['active'],(tick,sent))
                receipts+=1
                tracked+=int(self.bridge.execution_tracking['reason_code']=='EXECUTION_WITHIN_BOUNDS')
                if t.behavior_identity is not None:
                    self.assertTrue(t.planning_metrics['execution_robust'])
                    self.assertEqual(.45,t.planning_metrics['progress_bounds']['position_tolerance_m'])
            self.h.transport.observe(p,d,t,c,assessed.active,receipt)
            for reply in self.h.transport.feedbacks:
                bounded_progress+=int(reply.get('evidence_level')=='gps_reachable_path_progress')
            if artifact:
                row=dict(record_index=tick,capture_monotonic_s=self.f.now,relative_time_s=tick*.05,
                    perception=to_dict(p),decision=to_dict(d),trajectory=to_dict(t),control=to_dict(c),
                    safety=to_dict(assessed),send=receipt,execution_tracking=self.bridge.execution_tracking,
                    sent_monitor=sent,feedback=copy.deepcopy(self.h.transport.feedbacks))
                artifact.write(json.dumps(row,sort_keys=True)+'\n')
            if t.planning_metrics.get('parking_segment_proposal'):
                proposed=True
                self.assertEqual(0.,c.throttle); self.assertGreater(c.brake,0.)
                self.assertIsNone(t.behavior_identity)
                self.assertEqual('REJECTED',t.behavior_feedback['status'])
                self.assertTrue(t.behavior_feedback['candidate_options'])
            if t.planning_metrics.get('parking_segment') and t.candidate_id not in segment_ids:
                segment_ids.append(t.candidate_id); directions.append(t.motion_direction)
                self.assertEqual('ALIGN',d.behavior_request['stage'])
                self.assertEqual(t.candidate_id,d.behavior_request['candidate_id'])
                self.assertLessEqual(plant.speed,.05,'segment change before real standstill')
            if self.h.engine.standstill.anchor is not None and parked_at is None: parked_at=self.f.now
            if d.behavior_request['stage']=='EXIT' and c.throttle>0 and exited_at is None: exited_at=self.f.now
            if parked_at is not None and self.f.now-parked_at<10.:
                self.assertEqual(0.,c.throttle); self.assertTrue(c.handbrake)
            plant.step(c,.05,self.h.calibration)
            self.h.curvature=math.tan(plant.steering*self.h.calibration.front_steer_max_rad/
                                     self.h.calibration.steering_sign)/self.h.calibration.wheelbase_m
            self.h.advance(GoalPose(plant.x,plant.y,plant.heading),plant.speed,plant.direction)
        self.assertTrue(completed,(plant.x,plant.speed,decision_stub.decision_behavior_info(),
            d.behavior_request,t.stop_distance,[(v.x,v.speed,v.relative_time) for v in t.points],
            c.diagnostics,c.execution_observation,self.h.transport.feedbacks,trace))
        self.assertTrue(proposed); self.assertEqual([-1,1],directions)
        self.assertEqual(2,len(segment_ids)); self.assertIsNotNone(parked_at); self.assertIsNotNone(exited_at)
        self.assertGreaterEqual(exited_at-parked_at,10.)
        if robust:
            self.assertGreater(receipts,200); self.assertGreater(tracked,100); self.assertGreater(bounded_progress,100)
        if artifact:
            artifact.flush()
            with open(os.path.join(artifact_dir,'parking_multisegment_summary.json'),'w',encoding='utf-8') as stream:
                json.dump(dict(completed=completed,records=tick,segment_ids=segment_ids,directions=directions,
                    actual_parked_at_s=parked_at,actual_exit_started_at_s=exited_at,
                    actual_dwell_before_exit_s=exited_at-parked_at,confirmed_synthetic_receipts=receipts,
                    tracking_within_bounds_frames=tracked,gps_reachable_progress_frames=bounded_progress,
                    final_pose=dict(x=plant.x,y=plant.y,heading=plant.heading,speed=plant.speed),
                    final_decision_state=decision_stub.decision_behavior_info(),
                    claims=dict(simone_acceptance='not_evaluated',score='not_evaluated')),
                    stream,indent=2,sort_keys=True)

    def test_planning_suggestion_does_not_allow_a_moving_initial_segment(self):
        # This verifies the actual-stop gate independently of search geometry.
        self.h.evaluate()
        policy=self.h.policy
        self.h.advance(policy.space.position_pose,speed=.3)
        frame=self.h.frame()
        policy.session.advance(frame,'REVERSE_ENTRY',policy._goal(policy.park_goal,-1,True))
        reply=self.proposal(policy,frame,'segment-0',GoalPose(15.2,0.,0.),-1)
        policy.offer_candidate_options(frame,reply)
        self.assertIsNotNone(policy._proposed)
        self.assertGreater(frame.ego_speed,policy.standstill_speed_mps)
        # No goal or revision is changed by merely accepting the proposal.
        revision=policy.session.revision
        result=policy.evaluate(frame,self.h.h.read(),self.h.h.caps(),None)
        self.assertEqual('REVERSE_ENTRY',result.request.stage); self.assertEqual(revision,result.request.revision)
        self.assertIsNone(policy.session.goal.candidate_id)

    @staticmethod
    def proposal(policy,frame,identifier,pose,direction):
        goal=BehaviorGoal(parking_space_id=7,goal_pose=pose,motion_direction=direction,
            speed_cap_mps=1.,precision_stop=True,candidate_id=identifier)
        option=dict(candidate_id=identifier,segment_id=identifier,segment_count=2,action='PARK',
            goal=goal.to_dict(),feasible=True,risk_cost=1.,rule_cost=.8,validation_reason='test-proof',
            source_valid_until_s=frame.valid_until_s)
        return ExecutionFeedback(frame.context,policy.session.intent_id,policy.session.stage,
            policy.session.revision,'planning',frame.frame_id,frame.frame_id,frame.observed_at_s,
            frame.valid_until_s,'REJECTED',usable=True,candidate_options=[option])

    def selected_first_segment(self):
        self.h.evaluate(); policy=self.h.policy
        self.h.advance(policy.space.position_pose)
        frame=self.h.frame()
        policy.session.advance(frame,'REVERSE_ENTRY',policy._goal(policy.park_goal,-1,True))
        first=self.proposal(policy,frame,'segment-0',GoalPose(15.2,0.,0.),-1)
        policy.offer_candidate_options(frame,first)
        chosen=policy.evaluate(frame,self.h.h.read(),self.h.h.caps(),None)
        self.assertEqual('ALIGN',chosen.request.stage)
        self.assertEqual('segment-0',chosen.request.goal.candidate_id)
        return policy

    def test_next_segment_requires_matching_actual_arrival_not_planning_ack(self):
        policy=self.selected_first_segment()
        self.h.advance(GoalPose(15.2,0.,0.)); frame=self.h.frame()
        planned=self.proposal(policy,frame,'segment-1',policy.park_goal,1)
        planned.status='PLANNED'; policy.offer_candidate_options(frame,planned)
        revision=policy.session.revision
        result=policy.evaluate(frame,self.h.h.read(),self.h.h.caps(),planned)
        self.assertEqual(revision,result.request.revision)
        self.assertEqual('segment-0',result.request.goal.candidate_id)
        self.h.advance(GoalPose(15.2,0.,0.)); frame=self.h.frame()
        planned=self.proposal(policy,frame,'segment-1',policy.park_goal,1)
        planned.status='PLANNED'; policy.offer_candidate_options(frame,planned)
        actual=ExecutionFeedback(frame.context,policy.session.intent_id,'ALIGN',revision,'runtime',
            frame.frame_id,frame.frame_id,frame.observed_at_s,frame.valid_until_s,'DWELLING',
            usable=True,progress=1.,actual_standstill_confirmed=True,goal_pose_arrived=True,
            actual_pose=GoalPose(15.2,0.,0.),motion_direction=-1,candidate_id='wrong-segment')
        result=policy.evaluate(frame,self.h.h.read(),self.h.h.caps(),actual)
        self.assertEqual(revision,result.request.revision)
        self.assertEqual('segment-0',result.request.goal.candidate_id)
        self.h.advance(GoalPose(15.2,0.,0.)); frame=self.h.frame()
        planned=self.proposal(policy,frame,'segment-1',policy.park_goal,1)
        planned.status='PLANNED'; policy.offer_candidate_options(frame,planned)
        actual=ExecutionFeedback(frame.context,policy.session.intent_id,'ALIGN',revision,'runtime',
            frame.frame_id,frame.frame_id,frame.observed_at_s,frame.valid_until_s,'DWELLING',
            usable=True,progress=1.,actual_standstill_confirmed=True,goal_pose_arrived=True,
            actual_pose=GoalPose(15.2,0.,0.),motion_direction=-1,candidate_id='segment-0')
        result=policy.evaluate(frame,self.h.h.read(),self.h.h.caps(),actual)
        self.assertEqual(revision+1,result.request.revision)
        self.assertEqual('segment-1',result.request.goal.candidate_id)
        self.assertEqual(1,result.request.goal.motion_direction)
        self.assertEqual(0.,result.request.goal.minimum_standstill_duration_s)

    def test_expired_or_wrong_session_segment_offer_is_not_adopted(self):
        policy=self.selected_first_segment()
        self.h.advance(GoalPose(15.2,0.,0.)); frame=self.h.frame()
        offer=self.proposal(policy,frame,'segment-1',policy.park_goal,1)
        offer.candidate_options[0]['source_valid_until_s']=frame.observed_at_s
        policy.offer_candidate_options(frame,offer); self.assertIsNone(policy._proposed)
        offer=self.proposal(policy,frame,'segment-1',policy.park_goal,1)
        from core.behavior_contract import TaskContext
        offer.context=TaskContext(frame.context.case_id,frame.context.task_id,frame.context.scene_id,'other-run')
        policy.offer_candidate_options(frame,offer); self.assertIsNone(policy._proposed)

    def test_cold_alignment_candidate_cannot_create_a_cached_authorization(self):
        self.bridge.prepare(self.f.p)
        request=self.h.evaluate().request.to_dict('runtime_connected')
        request.update(stage='ALIGN',motion_direction=-1,goal_pose=GoalPose(15.2,0.,0.).to_dict(),
                       candidate_id='invented-segment',minimum_standstill_duration_s=0.,
                       parking_brake_at_stop=False,stop_obligation_id=None)
        self.h.reject(self.h.decision(request),'PARKING_SEGMENT_PLAN_UNAVAILABLE')

    def test_sent_monitor_needs_receipt_and_cannot_refresh_same_frame(self):
        self.execution_envelope=ExecutionEnvelope(.45,.02,.6,1.5); self.speed_error_mps=1.
        d,t,c=self.cycle()
        self.assertFalse(self.bridge.assess_execution(self.f.p,t).active)
        self.assertFalse(self.bridge.execution_monitor.history)
        self.bridge.record_sent(self.f.p,t,dict(attempted=False,ok=True))
        self.bridge.record_sent(self.f.p,t,dict(attempted=True,ok=False))
        self.bridge.record_sent(self.f.p,t,dict(attempted=True,ok=True),safety_active=True)
        self.assertFalse(self.bridge.execution_monitor.history)
        report=self.bridge.record_sent(self.f.p,t,dict(attempted=True,ok=True))
        self.assertFalse(report['active']); self.assertEqual(1,len(self.bridge.execution_monitor.history))
        report=self.bridge.record_sent(self.f.p,t,dict(attempted=True,ok=True))
        self.assertTrue(report['active']); self.assertIn('DUPLICATE_SEND',report['reason_code'])
        self.assertEqual(1,len(self.bridge.execution_monitor.history))

    def test_execution_envelope_does_not_invent_a_speed_tracking_error_bound(self):
        self.execution_envelope=ExecutionEnvelope(.45,.02,.6,1.5)
        self.bridge.prepare(self.f.p); d=decision_stub.decide(self.f.p)
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        result=self.bridge.assess_execution(self.f.p,t)
        self.assertTrue(result.active); self.assertIn('UNVERIFIED_SPEED_ERROR',result.reason)

    def test_same_frame_memo_cannot_hide_withdrawn_gps_source(self):
        self.execution_envelope=ExecutionEnvelope(.45,.02,.6,1.5); self.speed_error_mps=1.
        d,t,c=self.cycle(); self.assertFalse(self.bridge.assess_execution(self.f.p,t).active)
        self.f.p.source_status['gps']['usable']=False
        result=self.bridge.assess_execution(self.f.p,t)
        self.assertTrue(result.active); self.assertIn('GPS_UNAVAILABLE',result.reason)

    def test_short_retained_tail_keeps_launch_station_and_actual_initial_speed(self):
        self.cycle()
        self.h.advance(GoalPose(14.8,0.,0.),speed=.001)
        self.bridge.prepare(self.f.p)
        d=decision_stub.decide(self.f.p); t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        self.assertEqual(.001,t.points[0].speed); self.assertEqual(.001,self.f.p.ego.speed)
        self.assertGreaterEqual(len(t.points),3)
        self.assertGreater(max(v.speed for v in t.points),.05)
        self.assertEqual(0.,t.points[-1].speed)
        self.assertEqual((15.,0.),(t.points[-1].x,t.points[-1].y))

    def test_segment_identity_is_part_of_authorization_and_actual_stop_gate(self):
        from members.planning.parking_behavior import ParkingBehaviorPlanner
        planner=ParkingBehaviorPlanner(clock=self.f.clock)
        pose=GoalPose(15.2,0.,0.)
        planner.segment_plans['intent']=dict(binding='frozen',authorized_index=0,offered_index=1,
            offered_until=self.f.now+.2,sequence=(dict(goal=(15.2,0.,0.),direction=-1,candidate_id='old'),
                                                dict(goal=(15.2,0.,0.),direction=-1,candidate_id='next')))
        request=dict(intent_id='intent',stage='ALIGN',revision=2,source_frame_id=self.f.p.frame_id,
            issued_at_s=self.f.now,goal_pose=pose.to_dict(),motion_direction=-1,candidate_id='next')
        unused,index,unused=planner._authorized_segment(request,'frozen',self.f.now)
        self.assertEqual(1,index)
        previous=dict(revision=(1,'ALIGN',self.f.p.frame_id,self.f.now),pose=pose,direction=-1,
            accepted=True,candidate_id='old',segment_candidate=True,tolerances=(.15,.15,.05))
        self.f.p.ego.x=pose.x; self.f.p.ego.speed=.2
        with self.assertRaisesRegex(ValueError,'PRIOR_GOAL_NOT_ACTUALLY_STOPPED'):
            planner._transition(self.f.p,request,previous,[],self.h.transport.context)


if __name__=='__main__': unittest.main()
