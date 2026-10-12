"""Real fixed chain with authored Sensor/model worlds; no SDK or live sends.

The model bounds, visibility and successful send receipts here are synthetic
test prerequisites, not production calibration or actual lamp observations.
"""
import copy
import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.behavior_channel import BehaviorTransport,read_channel
from core.behavior_evidence import Evidence
from core.safety_supervisor import SafetySupervisor
from core.interfaces import DecisionMode
from members import decision_stub,planning_stub,control_stub
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from members.decision.behaviors.lane_change_runtime import LaneChangeDecisionInputs
from members.planning.candidate_validation import ExecutionEnvelope
from members.planning.maneuver_scene import ManeuverGeometryCache
from members.planning.lane_change_generator import PredictionEnvelope
from lane_change_integration import LaneChangeInputBridge,LaneChangeIntegrationInputs,LANE_CHANGE_CAPABILITIES
from tests import test_planning_lane_change_behavior as fixture
from tests.test_planning_candidate_validation import budget
from tests.test_control_maneuvers import BidirectionalPlant
from tests.test_maneuver_perception import target


SOURCE,TARGET=fixture.SOURCE,fixture.TARGET


class LaneChangeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.PlanningLaneChangeBehaviorTests(); self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.f=self.h.f; self.calls=0; self.enabled=True; self.change=None
        self.cache=ManeuverGeometryCache(12,1000000)
        self.lead=target(40.,0.); self.lead.vx=1.
        self.lead.lane_id=SOURCE; self.lead.same_lane_valid=True; self.lead.same_lane=True
        self.f.p.targets=[self.lead]
        self.f.p.ego.timestamp=self.f.p.timestamp
        self.f.p.ego.speed=self.f.p.ego.vx=1.; self.f.build()
        self.bridge=LaneChangeInputBridge(self.inputs,self.f.clock); self.bridge.settings=self.h.settings
        self.transport=BehaviorTransport(LANE_CHANGE_CAPABILITIES,self.f.clock)
        self.engine=DecisionEngine(DecisionSettings(cruise_speed=3.,front_offset_m=3.9,half_width_m=.9),
            self.f.clock,lane_change_inputs_provider=self.bridge.decision_inputs)
        manager=patch('members.decision_stub._ENGINE',self.engine); manager.start(); self.addCleanup(manager.stop)
        planning_stub.configure_planning(lane_change_inputs_provider=self.bridge.planning_inputs)
        self.safety=SafetySupervisor(NS(),self.f.clock,self.h.settings,self.engine.settings,
                                     lane_change_guard=self.bridge.assess,
                                     execution_guard=self.bridge.assess_execution)

    def inputs(self,p,context):
        self.calls+=1
        if not self.enabled: return None
        evidence=Evidence(context,p.frame_id,self.f.now,p.valid_until,True,True,
                          'authored-test-model',source_kind='task')
        route=Evidence(context,p.frame_id,self.f.now,p.valid_until,True,True,
                       'authored-test-task-return-corridor',source_kind='task')
        d=LaneChangeDecisionInputs(evidence,route,'synthetic-bicycle-and-empty-object-model-v1',
            'authored-overtake-mission',SOURCE,(3.9,.9,.9),3.,allow_command_lights=True,
            acceleration_uncertainty_mps2=.1)
        m=self.h.inputs(p,context)
        m.geometry_cache=self.cache
        # Authored controller-lag bounds for this bicycle plant, not SimOne.
        m.execution_envelope=ExecutionEnvelope(.25,.03,.2,.6)
        m.speed_error_mps=.5
        m.prediction_envelopes={9:PredictionEnvelope(90.,.02,.002)}
        pair=LaneChangeIntegrationInputs(d,m,budget(max_checks=1000000))
        if self.change: self.change(pair)
        return pair

    def cycle(self,sent=True,check=True):
        p=self.f.p
        self.transport.prepare(p); self.bridge.prepare(p)
        d=decision_stub.decide(p); t=planning_stub.plan(p,d)
        if check: self.assertTrue(t.valid,(d.reason,t.errors,self.bridge.reason,p.ego.x,p.ego.y,
            p.ego.heading,p.ego.speed,self.lead.x))
        c=control_stub.compute_control(p,t)
        if check: self.assertTrue(c.valid,c.errors)
        safety=self.safety.evaluate(p,d,t,c)
        if check: self.assertFalse(safety.active,(d.reason,t.reason,safety.reason,p.ego.x,p.ego.y,
            p.ego.heading,p.ego.speed,self.lead.x,self.engine._behavior.lane_changes.snapshot()))
        receipt=dict(attempted=sent,ok=sent,reason='synthetic_send_receipt_only')
        self.transport.observe(p,d,t,c,safety.active,receipt)
        self.bridge.record_sent(p,t,receipt,safety.active)
        self.bridge.observe_lights(p,d,t,c,self.transport,receipt,safety.active)
        return d,t,c

    def advance(self,x=None,y=None,heading=None,current=None,speed=1.):
        self.lead.x+=.05
        self.h.h.advance(x,y,heading,current,speed)
        self.f.p.timestamp+=50; self.f.p.ego.timestamp=self.f.p.timestamp
        self.lead.same_lane=self.f.p.lane.lane_id==SOURCE
        self.f.build()

    def stage(self,wanted,limit=110):
        result=None
        for unused in range(limit):
            result=self.cycle()
            if result[0].behavior_request and result[0].behavior_request['stage']==wanted:
                return result
            self.advance(self.f.p.ego.x+.05)
        self.fail(('missing stage',wanted,self.engine._behavior.lane_changes.snapshot()))

    def test_shared_pair_is_detached_and_not_refreshed_on_repeated_packet(self):
        self.cycle(); self.assertEqual(1,self.calls)
        context=self.transport.context
        d=self.bridge.decision_inputs(self.f.p,context); m=self.bridge.planning_inputs(self.f.p,context)
        d.extents=(999.,.9,.9); m.vehicle.front_offset_m=999.
        self.assertEqual((3.9,.9,.9),self.bridge.decision_inputs(self.f.p,context).extents)
        self.assertEqual(3.9,self.bridge.planning_inputs(self.f.p,context).vehicle.front_offset_m)
        self.f.now+=.01; self.bridge.prepare(self.f.p); self.assertEqual(1,self.calls)
        self.f.now=self.f.p.valid_until
        with self.assertRaises(ValueError): self.bridge.decision_inputs(self.f.p,context)

    def test_only_sent_timeline_arms_and_actual_tracking_error_vetoes(self):
        d,t,c=self.cycle(sent=False)
        self.assertFalse(self.bridge.execution.history)
        self.advance(self.f.p.ego.x+.05)
        d,t,c=self.cycle()
        self.assertEqual(1,len(self.bridge.execution.history))
        self.advance(self.f.p.ego.x+5.)
        self.transport.prepare(self.f.p); self.bridge.prepare(self.f.p)
        d=decision_stub.decide(self.f.p); t=planning_stub.plan(self.f.p,d)
        report=self.bridge.assess_execution(self.f.p,t)
        self.assertTrue(report.active); self.assertIn('EXECUTION_POSITION_OR_TIME_ERROR',report.reason)

    def test_missing_bounds_or_unknown_rear_does_not_create_an_overtake(self):
        self.change=lambda v:setattr(v.planning,'execution_envelope',None)
        for unused in range(45):
            d,t,c=self.cycle(check=False); self.assertIsNone(d.behavior_request)
            self.assertTrue(self.bridge.assess_execution(self.f.p,t).active)
            self.advance(self.f.p.ego.x+.05)
        self.assertIn('EXECUTION_ERROR_UNAVAILABLE',self.bridge.reason)
        self.change=None
        for unused in range(50):
            # Remove only the rear Sensor certificate, preserve observed lead.
            self.f.p.maneuver_environment.coverage_regions=[]
            d=decision_stub.decide(self.f.p); self.assertIsNone(d.behavior_request)
            self.advance(self.f.p.ego.x+.05)
            self.transport.prepare(self.f.p); self.bridge.prepare(self.f.p)

    def test_staging_command_evidence_never_claims_observed_lamps_or_path(self):
        d,t,c=self.stage('PREPARE')
        self.assertIsNone(t.behavior_identity); self.assertIsNone(t.candidate_id)
        self.assertTrue(all(abs(v.y)<1e-9 for v in t.points))
        replies=self.transport.feedbacks
        lamps=[v for v in replies if v.get('reason_code')=='LAMP_COMMAND_SENT']
        self.assertEqual(1,len(lamps)); self.assertFalse(lamps[0]['lights_confirmed'])
        self.assertEqual(0.,lamps[0]['lights_duration_s'])
        self.assertEqual('command_sent',lamps[0]['lights_evidence_level'])
        self.assertTrue(all(v['status']!='PLANNED' for v in replies))
        for unused in range(2):
            self.advance(self.f.p.ego.x+.05); d,t,c=self.cycle(sent=False)
        self.assertEqual('PREPARE',d.behavior_request['stage'])
        self.advance(self.f.p.ego.x+.05); self.f.now+=.1
        d,t,c=self.cycle(sent=False,check=False)
        self.assertTrue(self.safety.evaluate(self.f.p,d,t,c).active)

    def test_monitor_checks_actual_path_and_adopted_candidate(self):
        d,t,c=self.stage('EXECUTE')
        self.assertEqual(t.candidate_id,d.behavior_request['candidate_id'])
        altered=copy.deepcopy(t); altered.points[0].x+=.1
        self.assertIn('ACTUAL_START_MISMATCH',self.bridge.assess(self.f.p,d,altered,c).reason)
        altered=copy.deepcopy(t);altered.candidate_id='unaccepted-candidate'
        altered.behavior_feedback['candidate_id']=altered.candidate_id
        self.assertIn('CANDIDATE_IDENTITY',self.bridge.assess(self.f.p,d,altered,c).reason)
        altered=copy.deepcopy(d); altered.behavior_request['candidate_id']='another-candidate'
        rejected=planning_stub.plan(self.f.p,altered)
        self.assertFalse(rejected.valid); self.assertIn('ADOPTED_CANDIDATE_MISMATCH',rejected.reason)
        self.f.p.maneuver_environment.coverage_regions=[]
        self.assertTrue(self.bridge.assess(self.f.p,d,t,c).active)

    def test_active_intent_is_preserved_on_label_or_frame_regression(self):
        d,t,c=self.stage('EXECUTE')
        state=self.engine._behavior.lane_changes
        original=state.policy.session.intent_id
        self.f.p.task_id='untrusted-label-change'
        self.transport.prepare(self.f.p); self.bridge.prepare(self.f.p)
        d=decision_stub.decide(self.f.p)
        self.assertIs(state,self.engine._behavior.lane_changes)
        self.assertEqual(original,state.policy.session.intent_id)
        self.assertEqual(0.,d.target_speed); self.assertIsNone(d.behavior_request)

    def test_real_chain_changes_passes_and_returns_on_actual_motion(self):
        plant=BidirectionalPlant(x=10.,lag=.3); plant.speed=1.
        phases=set(); stages=set(); candidates={}; side_reached=False; actual_passed=False; passing_checks=0
        for step in range(1800):
            d,t,c=self.cycle()
            state=self.engine._behavior.lane_changes
            phases.add(state.phase)
            if state.phase=='PASS':
                tracking=self.bridge.execution_tracking
                self.assertFalse(tracking['active'],tracking)
                self.assertEqual('EXECUTION_WITHIN_BOUNDS',tracking['reason_code'])
                self.assertLess(tracking['sent_source_frame_id'],self.f.p.frame_id)
                passing_checks+=1
            request=d.behavior_request
            if request:
                stages.add(request['stage'])
                self.assertEqual('OVERTAKE',request['maneuver'])
                if t.candidate_id:
                    old=candidates.setdefault(request['intent_id'],t.candidate_id)
                    self.assertEqual(old,t.candidate_id)
            side_reached=side_reached or (self.f.p.lane.lane_id==TARGET and plant.y>3.)
            actual_passed=actual_passed or (plant.x-.9>self.lead.x+2.25+10.5)
            if state.phase=='COMPLETE': break
            plant.step(c,.05,self.h.calibration)
            current=TARGET if plant.y>=1.75 else SOURCE
            self.h.curvature=math.tan(plant.steering*self.h.calibration.front_steer_max_rad/
                                     self.h.calibration.steering_sign)/self.h.calibration.wheelbase_m
            self.advance(plant.x,plant.y,plant.heading,current,plant.speed)
            self.f.p.ego.yaw_rate=plant.yaw_rate
        self.assertEqual('COMPLETE',state.phase,(step,plant.x,plant.y,state.snapshot()))
        self.assertTrue(side_reached); self.assertTrue(actual_passed)
        self.assertEqual(SOURCE,self.f.p.lane.lane_id)
        self.assertLess(abs(plant.y),.3); self.assertLess(abs(plant.heading),.15)
        self.assertTrue({'OUT','PASS','RETURN','COMPLETE'}<=phases)
        self.assertTrue({'PREPARE','WAIT_GAP','REQUEST_PATH','EXECUTE','SETTLE'}<=stages)
        self.assertEqual(2,len(candidates)); self.assertEqual(2,len(set(candidates.values())))
        self.assertGreater(passing_checks,20)


if __name__=='__main__': unittest.main()
