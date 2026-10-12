"""Actual D04/fixed P04/control/transport with explicit synthetic sources."""
import copy
import math
import unittest
from unittest.mock import patch
from core.behavior_contract import BehaviorFrame,GoalPose
from core.behavior_channel import BehaviorTransport,read_channel,feedback_from_dict
from core.interfaces import DecisionTarget,DecisionMode,Trajectory
from core.validation import validate_output
from core.behavior_evidence import Evidence
from core.parking_mission import ParkingMission
from core.maneuver_facts import ManeuverFacts
from members.decision.behaviors.observations import Evidence as LegacyEvidence
from members.decision.behaviors.parking_environment import ParkingMission as LegacyMission
from members.decision.behaviors.parking_runtime import ParkingDecisionInputs
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from members.planning.parking_behavior import ParkingPlanningInputs
from members.planning.lane_planner import PlannerSettings
from members.planning.maneuver_scene import ManeuverGeometryCache
from members.planning.lane_change_generator import PredictionEnvelope
from members.control.controller import ControlEngine
from members import decision_stub,planning_stub,control_stub
from tests import test_decision_parking_environment as fixture
from tests.test_planning_candidate_validation import vehicle,limits,budget
from tests.test_planning_parking_generator import search
from tests.test_control_maneuvers import BidirectionalPlant
from tests.control_benchmark import vehicle as control_vehicle
from tests.test_maneuver_perception import target,profile


class ParkingFixedEntryTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.FormalParkingDecisionTests(); self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.f=self.h.f; self.policy=self.h.policy; self.curvature=0.; self.input_changes={}
        self.calibration=control_vehicle()
        self.geometry=vehicle(3.9,.9,.9,self.calibration.wheelbase_m)
        self.motion=limits(max_front_steer_rad=self.calibration.front_steer_max_rad)
        self.cache=ManeuverGeometryCache(3,1000000)
        self.settings=PlannerSettings(front_offset_m=3.9,rear_offset_m=.9,half_width_m=.9,
            wheelbase_m=self.geometry.wheelbase_m,front_steer_max_rad=self.motion.max_front_steer_rad)
        self.engine=ControlEngine(self.calibration,clock=self.f.clock)
        self.transport=BehaviorTransport(('PARK','PATH','REVERSE','DWELL','FEEDBACK'),self.f.clock)
        self.transport.prepare(self.f.p); self.h.frame=self.frame
        # Facts-only poses do not prove a trajectory. Author an explicit
        # feasible execution task for this steering model, separately testing
        # an infeasible short lateral task without relaxing its limits.
        original_mission=self.h.mission
        self.h.mission=lambda **changes:original_mission(**dict({'approach_pose':GoalPose(15.,0.,0.),
            'position_pose':GoalPose(33.,2.,0.),'exit_goal':GoalPose(33.,2.,0.)},**changes))
        planning_stub.configure_planning(parking_inputs_provider=self.inputs)
        self.addCleanup(planning_stub.configure_planning)
        for name,value in (('members.planning_stub._settings_with_control_geometry',lambda:self.settings),
                           ('members.control_stub._engine',self.engine)):
            manager=patch(name,value); manager.start(); self.addCleanup(manager.stop)

    def frame(self):
        p=self.f.p
        return BehaviorFrame(self.transport.context,p.frame_id,self.f.now,p.valid_until,
                             p.ego.x,p.ego.y,p.ego.heading,p.ego.speed,usable=True)

    def inputs(self,p,context):
        value=ParkingPlanningInputs(self.h.mission(),(.15,.15,.05),context,p.frame_id,self.f.now,p.valid_until,
            'synthetic-parking-model-v1',self.curvature,self.geometry,self.motion,
            search(tangent_scales=(.7,1.,1.4,1.8,2.),end_tangent_scales=(.7,.8,1.,2.),
                   speed_scales=(1.,),terminal_straight_m=(1.,0.)),
            budget(max_checks=1000000),{},geometry_cache=self.cache)
        for key,item in self.input_changes.items(): setattr(value,key,item)
        return value

    def decision_inputs(self,p,context):
        model=Evidence(context,p.frame_id,self.f.now,p.valid_until,usable=True,coverage_verified=True,
                       source='synthetic-parking-model-snapshot',source_kind='task')
        return ParkingDecisionInputs(model,self.h.mission(),'synthetic-parking-model-v1',
                                     (3.9,.9,.9),.1,.1,1000000,(.15,.15,.05),1.)

    def advance(self,pose=None,speed=0.,direction=1):
        self.h.advance(pose,speed)
        self.f.p.ego.vx=direction*speed*math.cos(self.f.p.ego.heading)
        self.f.p.ego.vy=direction*speed*math.sin(self.f.p.ego.heading)
        self.f.build(); self.transport.prepare(self.f.p)

    def evaluate(self):
        context,caps,replies=read_channel(self.f.p,self.f.now)
        current=[r for r in replies if r.intent_id==self.policy.session.intent_id
                 and r.revision==self.policy.session.revision]
        reply=max(current,key=lambda r:(r.produced_at_s,r.producer!='planning')) if current else None
        return self.policy.evaluate(self.frame(),self.h.read(),caps,reply)

    def decision(self,request):
        p=self.f.p; d=DecisionTarget().bind(p); d.valid=True; d.target_speed=1.
        d.target_lane_id=p.lane.lane_id
        d.behavior_request=request.to_dict('runtime_connected') if hasattr(request,'to_dict') else copy.deepcopy(request)
        d.behavior_active_identity={k:d.behavior_request[k] for k in
                                   ('intent_id','stage','revision','source_frame_id','issued_at_s')}
        return d

    def cycle(self):
        result=self.evaluate(); self.assertIsNotNone(result.request,result.reason)
        self.assertTrue(result.request.dispatch_allowed,(result.phase,result.reason))
        d=self.decision(result.request); t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        c=control_stub.compute_control(self.f.p,t); self.assertTrue(c.valid,c.errors)
        self.transport.observe(self.f.p,d,t,c)
        return result.request,d,t,c

    def reject(self,d,reason):
        t=planning_stub.plan(self.f.p,d)
        self.assertFalse(t.valid,t.reason); self.assertFalse(t.points); self.assertIsNone(t.behavior_identity)
        self.assertIn(reason,t.reason)
        return t

    def test_shared_evidence_and_task_keep_legacy_import_identity(self):
        self.assertIs(Evidence,LegacyEvidence); self.assertIs(ParkingMission,LegacyMission)

    def test_fixed_entry_approach_preserves_actual_start_and_original_goal(self):
        r,d,t,c=self.cycle()
        self.assertEqual('APPROACH',r.stage)
        self.assertEqual((self.f.p.ego.x,self.f.p.ego.y,self.f.p.ego.heading,self.f.p.ego.speed),
                         (t.points[0].x,t.points[0].y,t.points[0].heading,t.points[0].speed))
        self.assertEqual((15.,0.,0.),(t.points[-1].x,t.points[-1].y,t.points[-1].heading))
        self.assertEqual(0.,t.points[-1].speed); self.assertTrue(t.precision_stop)
        self.assertEqual(r.intent_id,t.behavior_identity['intent_id'])
        self.assertFalse(c.execution_observation['goal_pose_arrived'])

    def test_provider_and_current_capabilities_are_both_required(self):
        d=self.decision(self.evaluate().request)
        planning_stub.configure_planning(); self.reject(d,'PARKING_MODEL_INPUTS_UNAVAILABLE')
        planning_stub.configure_planning(parking_inputs_provider=self.inputs)
        self.f.p.behavior_channel['capabilities']['actions']=['PATH_STOP','DWELL','FEEDBACK']
        self.reject(d,'CAPABILITY_ACTIONS_MISSING')
        self.assertEqual((),BehaviorTransport().actions)

    def test_model_snapshot_and_actual_curvature_are_never_inferred(self):
        d=self.decision(self.evaluate().request)
        for changes,reason in (({'frame_id':-1},'MODEL_SOURCE_MISMATCH'),
            ({'context':None},'MODEL_SOURCE_MISMATCH'),({'clock_id':'other'},'MODEL_SOURCE_MISMATCH'),
            ({'valid_until_s':self.f.now},'MODEL_SOURCE_MISMATCH'),
            ({'initial_curvature_m_inv':None},'MOTION_MODEL_UNAVAILABLE'),
            ({'tolerances':(0.,.15,.05)},'ARRIVAL_MODEL_UNAVAILABLE')):
            self.input_changes=changes; self.reject(d,reason)

    def test_original_goal_direction_and_dwell_extensions_are_checked(self):
        payload=self.evaluate().request.to_dict('runtime_connected')
        for changes,reason in (({'motion_direction':-1},'STAGE_DIRECTION_MISMATCH'),
            ({'stop_distance_m':0.},'GOAL_EXTENSIONS_UNSUPPORTED'),
            ({'goal_pose':GoalPose(16.,1.,0.).to_dict()},'ORIGINAL_STAGE_GOAL_MISMATCH'),
            ({'minimum_standstill_duration_s':10.},'DWELL_OBLIGATION_INVALID')):
            value=copy.deepcopy(payload); value.update(changes); self.reject(self.decision(value),reason)

    def test_cold_position_cannot_skip_original_approach(self):
        payload=self.evaluate().request.to_dict('runtime_connected')
        payload.update(stage='POSITION',goal_pose=self.h.mission().position_pose.to_dict())
        self.reject(self.decision(payload),'PARKING_APPROACH_REQUIRED')

    def test_planning_ack_cannot_replace_actual_stage_feedback(self):
        r,d,t,c=self.cycle(); self.transport.feedbacks=[t.behavior_feedback]
        self.advance(self.h.mission().approach_pose)
        payload=self.policy.session.snapshot(self.frame()).to_dict('runtime_connected')
        payload.update(stage='POSITION',revision=payload['revision']+1,source_frame_id=self.f.p.frame_id,
                       issued_at_s=self.f.now,goal_pose=self.h.mission().position_pose.to_dict())
        self.reject(self.decision(payload),'MATCHED_ACTUAL_STAGE_FEEDBACK_REQUIRED')

    def test_actual_controller_arrival_advances_position_with_same_intent(self):
        r,d,t,c=self.cycle(); identity=r.intent_id
        self.advance(self.h.mission().approach_pose)
        r,d,t,c=self.cycle()
        self.assertTrue(c.execution_observation['goal_pose_arrived']); self.assertFalse(c.execution_observation['hold_completed'])
        self.advance(self.h.mission().approach_pose)
        r,d,t,c=self.cycle(); self.assertEqual('POSITION',r.stage); self.assertEqual(identity,r.intent_id)

    def test_active_task_access_and_model_bindings_cannot_move(self):
        r,d,t,c=self.cycle()
        for changes in ({'mission':self.h.mission(access_edge_index=0)},
                        {'mission':self.h.mission(exit_goal=GoalPose(34.,2.,0.))},
                        {'model_id':'different'},{'tolerances':(.2,.15,.05)}):
            self.input_changes=changes; self.reject(d,'ACTIVE_BINDING_CHANGED')

    def test_stop_and_red_retain_authority_over_parking(self):
        d=self.decision(self.evaluate().request)
        for mode in (DecisionMode.STOP,DecisionMode.EMERGENCY_BRAKE):
            changed=copy.deepcopy(d); changed.mode=mode; self.reject(changed,'OVERRIDDEN_BY_STOP')
        self.f.p.traffic.required=True; self.reject(d,'SIGNAL_STOP_REQUIRED')

    def test_short_model_deadline_and_tiny_work_budget_are_preserved(self):
        d=self.decision(self.evaluate().request)
        self.input_changes={'valid_until_s':self.f.now+.03}
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        self.assertEqual(self.f.now+.03,t.valid_until)
        self.input_changes={'budget':budget(max_checks=1)}; self.reject(d,'BUDGET_EXHAUSTED')

    def test_current_unknown_yaw_object_and_missing_view_withdraw_path(self):
        d=self.decision(self.evaluate().request)
        self.f.p.maneuver_environment.coverage_regions=[]
        self.reject(d,'COVERAGE_UNKNOWN')
        self.f.p.targets=[target(self.f.p.ego.x,self.f.p.ego.y,heading=None)]; self.advance()
        d=self.decision(self.policy.session.snapshot(self.frame()))
        self.reject(d,'PREDICTION_OBJECT_SET_MISMATCH')

    def test_unacknowledged_parking_execution_identity_is_rejected(self):
        r,d,t,c=self.cycle()
        for changes in ({'behavior_feedback':None},{'precision_stop':False},
                        {'behavior_identity':dict(t.behavior_identity,revision=r.revision+1)}):
            value=copy.deepcopy(t)
            for key,item in changes.items(): setattr(value,key,item)
            with self.assertRaises(ValueError): validate_output(value,Trajectory,d)

    def test_infeasible_short_lateral_approach_does_not_relax_steering_limits(self):
        self.h.mission=lambda **changes:fixture.FormalParkingDecisionTests.mission(
            self.h,**dict({'approach_pose':GoalPose(12.,1.,0.)},**changes))
        d=self.decision(self.evaluate().request)
        self.reject(d,'NO_CERTIFIED_PARKING_SEGMENT')

    def test_actual_gps_progress_does_not_claim_arrival(self):
        r,d,t,c=self.cycle(); self.advance(GoalPose(10.5,.1,0.))
        r,d,t,c=self.cycle()
        reply=feedback_from_dict(self.transport.feedbacks[-1])
        self.assertGreater(reply.progress,0.); self.assertFalse(reply.goal_pose_arrived)
        self.assertEqual('APPROACH',r.stage)

    def test_actual_overspeed_is_preserved_and_brakes_to_authorized_cap(self):
        self.advance(speed=1.1)
        r,d,t,c=self.cycle()
        self.assertEqual(1.1,t.points[0].speed); self.assertEqual(1.,t.target_speed)
        distance=0.
        for a,b in zip(t.points,t.points[1:]):
            distance+=math.hypot(b.x-a.x,b.y-a.y)
            self.assertLessEqual(b.speed*b.speed,max(1.,1.1**2-2*self.motion.max_deceleration_mps2*distance)+1e-9)
        self.assertEqual(0.,t.points[-1].speed)

    def test_task_evidence_and_mutable_task_pose_are_revalidated(self):
        d=self.decision(self.evaluate().request)
        mission=self.h.mission(); mission.evidence.source_kind='diagnostic_ground_truth'
        self.input_changes={'mission':mission}; self.reject(d,'TASK_MISSION_UNVERIFIED')
        mission=self.h.mission(); mission.approach_pose.x=float('nan')
        self.input_changes={'mission':mission}; self.reject(d,'invalid goal pose')

    def test_source_expiring_during_generation_cannot_publish_a_candidate(self):
        from members.planning import parking_behavior
        original=parking_behavior.plan_parking_candidate
        def late(*args,**kwargs):
            result=original(*args,**kwargs); self.f.now+=.21; return result
        with patch.object(parking_behavior,'plan_parking_candidate',side_effect=late):
            self.reject(self.decision(self.evaluate().request),'EXPIRED')

    def test_goal_tolerance_search_preserves_original_position_and_heading_contract(self):
        r,d,t,c=self.cycle()
        # A small heading mismatch near a stop remains within the task's
        # explicit arrival region; it cannot shift the original goal position.
        self.advance(GoalPose(14.65,-.01,.03),speed=.42)
        d=self.decision(self.policy.session.snapshot(self.frame()))
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        self.assertEqual((15.,0.),(t.points[-1].x,t.points[-1].y))
        self.assertLessEqual(abs(t.points[-1].heading),.15)
        self.assertEqual(self.h.mission().approach_pose.to_dict(),d.behavior_request['goal_pose'])

    def test_stationary_goal_still_requires_visibility_and_work_budget(self):
        self.cycle(); self.advance(self.h.mission().approach_pose)
        d=self.decision(self.evaluate().request)
        self.input_changes={'budget':budget(max_checks=1)}
        self.reject(d,'BUDGET_EXHAUSTED')
        self.input_changes={}; self.f.p.maneuver_environment.coverage_regions=[]
        self.reject(d,'COVERAGE_UNKNOWN')

    def test_reverse_stage_requires_fresh_reverse_and_dwell_capabilities(self):
        self.cycle()
        for pose in (self.h.mission().approach_pose,self.h.mission().approach_pose,
                     self.h.mission().position_pose):
            self.advance(pose); self.cycle()
        self.advance(self.h.mission().position_pose)
        r=self.evaluate().request; self.assertEqual('REVERSE_ENTRY',r.stage)
        d=self.decision(r)
        self.f.p.behavior_channel['capabilities']['actions']=['PARK','PATH','FEEDBACK']
        self.reject(d,'CAPABILITY_ACTIONS_MISSING')

    def test_accepted_revision_cannot_change_its_creation_identity(self):
        r,d,t,c=self.cycle(); payload=copy.deepcopy(d.behavior_request)
        payload['issued_at_s']-=.001
        self.reject(self.decision(payload),'REQUEST_REVISION_REGRESSED_OR_CHANGED')

    def test_controller_does_not_claim_arrival_from_planning_ack_or_nearby_stop(self):
        r,d,t,c=self.cycle()
        self.assertFalse(c.execution_observation['goal_pose_arrived'])
        self.advance(GoalPose(14.,0.,0.))
        d=self.decision(self.policy.session.snapshot(self.frame()))
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        t.stop_distance=0.; t.target_speed=0.
        c=control_stub.compute_control(self.f.p,t)
        self.assertFalse(c.execution_observation['goal_pose_arrived'])

    def test_stopped_steering_preparation_does_not_acknowledge_a_parking_path_or_dwell(self):
        self.cycle()
        for pose in (self.h.mission().approach_pose,self.h.mission().approach_pose,
                     self.h.mission().position_pose):
            self.advance(pose); self.cycle()
        self.advance(GoalPose(32.904914049164624,1.9973166151544746,.029375020035250676))
        self.curvature=-.020592610287030473
        r=self.evaluate().request; self.assertEqual('REVERSE_ENTRY',r.stage)
        d=self.decision(r)
        declined=dict(status='unsafe',reason_code='NO_CERTIFIED_PARKING_SEGMENT',checks=0,points=[])
        # Isolate the declined-connector branch; all current body/object/view
        # proof and the controller/transport below still execute normally.
        with patch('members.planning.parking_behavior.ParkingBehaviorPlanner._moving',return_value=declined):
            t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors); self.assertEqual('PARKING_STOPPED_STEERING_PREPARATION',t.reason)
        self.assertIsNone(t.behavior_feedback); self.assertIsNone(t.behavior_identity)
        self.assertEqual('',t.stop_obligation_id); self.assertEqual(0.,t.hold_duration_s)
        c=control_stub.compute_control(self.f.p,t)
        self.assertEqual(0.,c.throttle); self.assertEqual(0.,c.steering)
        self.assertIsNone(c.execution_observation)
        self.transport.observe(self.f.p,d,t,c); self.assertEqual([],self.transport.feedbacks)
        self.curvature=0.; self.advance(GoalPose(self.f.p.ego.x,self.f.p.ego.y,self.f.p.ego.heading))
        r=self.evaluate().request; t=planning_stub.plan(self.f.p,self.decision(r))
        self.assertTrue(t.valid,t.errors); self.assertIsNotNone(t.behavior_feedback)

    def test_retained_segment_starts_at_actual_gps_and_does_not_reuse_old_speed_time(self):
        r,d,t,c=self.cycle()
        original=planning_stub._PARKING_PLANNER.records[r.intent_id]['reference']
        self.advance(GoalPose(12.,0.,0.),speed=.5)
        d=self.decision(self.policy.session.snapshot(self.frame()))
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        self.assertEqual((12.,0.,0.,.5),(t.points[0].x,t.points[0].y,t.points[0].heading,t.points[0].speed))
        self.assertEqual(0.,t.points[0].relative_time)
        self.assertAlmostEqual(3.,t.stop_distance)
        self.assertEqual(original,planning_stub._PARKING_PLANNER.records[r.intent_id]['reference'])

    def test_retained_path_checks_fresh_objects_only_along_actual_remaining_body(self):
        self.cycle(); self.f.p.targets=[target(8.,0.,heading=None)]
        self.input_changes={'prediction_envelopes':{9:PredictionEnvelope(60.,.1,.1)}}
        self.advance(GoalPose(12.,0.,0.),speed=.5)
        d=self.decision(self.policy.session.snapshot(self.frame()))
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        self.f.p.maneuver_environment.coverage_regions=[]
        self.reject(d,'COVERAGE_UNKNOWN')

    def test_retained_or_stationary_paths_cannot_bypass_mutated_search_or_actual_direction(self):
        r,d,t,c=self.cycle(); self.advance(GoalPose(12.,0.,0.),speed=.5)
        d=self.decision(self.policy.session.snapshot(self.frame()))
        for value in (search(spacing_m=0.),search(spacing_m=float('nan')),search(max_points=True)):
            self.input_changes={'search':value}; self.reject(d,'PARKING_SEARCH_INVALID')
        self.input_changes={'search':search(max_points=3)}
        self.reject(d,'NO_CERTIFIED_PARKING_SEGMENT')
        self.input_changes={}; self.advance(GoalPose(12.,0.,0.),speed=.5,direction=-1)
        self.reject(self.decision(self.policy.session.snapshot(self.frame())),'ACTUAL_MOTION_OPPOSES_PARKING_STAGE')

    def capture(self,pose=None,**changes):
        from members.planning.parking_behavior import ParkingBehaviorPlanner
        self.advance(pose or GoalPose(21.71286196045203,-6.368926586420488,1.6533378734070732),
                     speed=.4961725428465529,direction=-1)
        self.curvature=-.07525969924848207
        model=self.inputs(self.f.p,self.transport.context)
        for key,value in changes.items(): setattr(model,key,value)
        goal=model.mission.stage_pose('REVERSE_ENTRY',ManeuverFacts(self.f.p,self.f.clock).parking(7),
            model.vehicle.front_offset_m,model.vehicle.rear_offset_m)
        result=ParkingBehaviorPlanner(clock=self.f.clock)._capture(self.f.p,model.mission,model,
            dict(stage='REVERSE_ENTRY',motion_direction=-1),goal,1.,0)
        return result,goal

    def test_arrival_capture_uses_actual_curvature_and_original_pose_region(self):
        result,goal=self.capture()
        self.assertEqual('safe',result['status'],result)
        points=result['points']; first=points[0]; end=points[-1]
        self.assertEqual((self.f.p.ego.x,self.f.p.ego.y,self.f.p.ego.heading,self.f.p.ego.speed),
                         (first.x,first.y,first.heading,first.speed))
        self.assertLessEqual(math.hypot(end.x-goal.x,end.y-goal.y),.15)
        self.assertLessEqual(abs(end.heading-goal.body_heading_rad),.15)
        self.assertEqual(0.,end.speed)
        for a,b in zip(points,points[1:]):
            ds=math.hypot(b.x-a.x,b.y-a.y)
            self.assertAlmostEqual(self.curvature,(b.heading-a.heading)/(-ds),places=4)

    def test_arrival_capture_rejects_outside_heading_region_and_expired_work(self):
        result,goal=self.capture(GoalPose(21.71286196045203,-6.368926586420488,1.9))
        self.assertFalse(result['points']); self.assertNotEqual('safe',result['status'])
        result,goal=self.capture(budget=budget(max_checks=1))
        self.assertEqual('BUDGET_EXHAUSTED',result['reason_code']); self.assertFalse(result['points'])

    def test_arrival_capture_checks_current_whole_body_and_visibility(self):
        result,goal=self.capture(); self.assertEqual('safe',result['status'],result)
        self.f.p.maneuver_environment.coverage_regions=[]
        # Rebuild only the current model; the withdrawn source is not replaced.
        from members.planning.parking_behavior import ParkingBehaviorPlanner
        model=self.inputs(self.f.p,self.transport.context)
        result=ParkingBehaviorPlanner(clock=self.f.clock)._capture(self.f.p,model.mission,model,
            dict(stage='REVERSE_ENTRY',motion_direction=-1),goal,1.,0)
        self.assertFalse(result['points']); self.assertIn('COVERAGE',result['reason_code'])

    def test_formal_decision_fixed_entry_control_transport_complete_original_stages(self):
        from parking_integration import ParkingInputBridge,ParkingIntegrationInputs
        from core.safety_supervisor import SafetySupervisor
        from types import SimpleNamespace
        p=self.f.p; plant=BidirectionalPlant(x=p.ego.x,y=p.ego.y,heading=p.ego.heading,lag=.3)
        def paired_inputs(perception,context):
            return ParkingIntegrationInputs(self.decision_inputs(perception,context),
                self.inputs(perception,context),budget(max_checks=1000000))
        bridge=ParkingInputBridge(paired_inputs,self.f.clock); bridge.settings=self.settings
        planning_stub.configure_planning(parking_inputs_provider=bridge.planning_inputs)
        decision_engine=DecisionEngine(DecisionSettings(front_offset_m=3.9,half_width_m=.9),
                                      self.f.clock,bridge.decision_inputs)
        safety=SafetySupervisor(SimpleNamespace(),self.f.clock,self.settings,decision_engine.settings,
                                parking_guard=bridge.assess)
        manager=patch('members.decision_stub._ENGINE',decision_engine)
        manager.start(); self.addCleanup(manager.stop)
        # One original packet has two explicitly verified convex view pieces.
        # Their true union is the old field; the rear axle lies on their seam.
        # Every stage must consume both without inventing a fused Sensor source.
        model=profile(p); first=model['profiles'][0]
        first['body_region_m']=[[-100.,-20.],[0.,-20.],[0.,20.],[-100.,20.]]
        second=copy.deepcopy(first)
        second['body_region_m']=[[0.,-20.],[150.,-20.],[150.,20.],[0.,20.]]
        model['profiles'].append(second); self.f.write_profile(model); self.f.build()
        self.assertEqual(2,len(p.maneuver_environment.coverage_regions))
        seen=[]; identity=None; parked_at=None; exited_at=None; completed=False
        for tick in range(3000):
            bridge.prepare(p)
            d=decision_stub.decide(p); validate_output(d,DecisionTarget,p)
            state=decision_stub.decision_behavior_info().get('parking',{})
            if state.get('status')=='COMPLETED': completed=True; break
            r=d.behavior_request
            self.assertIsNotNone(r,(tick,state,d.reason,plant.x,plant.y,plant.heading))
            if identity is None: identity=r['intent_id']
            self.assertEqual(identity,r['intent_id'])
            if not seen or seen[-1]!=r['stage']: seen.append(r['stage'])
            t=planning_stub.plan(p,d)
            self.assertTrue(t.valid,(tick,r['stage'],plant.x,plant.y,plant.heading,plant.speed,self.curvature,t.errors))
            c=control_stub.compute_control(p,t); self.assertTrue(c.valid,(tick,c.errors))
            assessment=safety.evaluate(p,d,t,c)
            self.assertFalse(assessment.active,(tick,r['stage'],assessment.reason))
            self.transport.observe(p,d,t,c,assessment.active)
            if self.engine.standstill.anchor is not None and parked_at is None: parked_at=self.f.now
            if r['stage']=='EXIT' and c.throttle>0 and exited_at is None: exited_at=self.f.now
            if parked_at is not None and self.f.now-parked_at<10.:
                self.assertEqual(0.,c.throttle); self.assertTrue(c.handbrake)
            plant.step(c,.05,self.calibration)
            self.curvature=math.tan(plant.steering*self.calibration.front_steer_max_rad/
                self.calibration.steering_sign)/self.calibration.wheelbase_m
            self.advance(GoalPose(plant.x,plant.y,plant.heading),plant.speed,plant.direction)
            p.ego.yaw_rate=plant.yaw_rate
        self.assertTrue(completed,(seen,plant.x,plant.y,plant.heading))
        self.assertEqual(['APPROACH','POSITION','REVERSE_ENTRY','PARKED_DWELL','EXIT_PREPARE','EXIT'],seen)
        self.assertIsNotNone(parked_at); self.assertIsNotNone(exited_at)
        self.assertGreaterEqual(exited_at-parked_at,10.)
        self.advance(GoalPose(plant.x,plant.y,plant.heading),plant.speed,plant.direction)
        bridge.prepare(p)
        finished=decision_stub.decide(p)
        self.assertIsNone(finished.behavior_request)
        self.assertEqual('COMPLETED',decision_stub.decision_behavior_info()['parking']['status'])


if __name__=='__main__': unittest.main()
