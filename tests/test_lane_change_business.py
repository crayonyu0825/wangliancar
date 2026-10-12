"""AVOID/MERGE use the actual fixed chain with authored bounds, never the SDK.

The Sensor-shaped producer, controller plant and successful receipt model are
synthetic. These tests establish component integration, not field calibration.
"""
import copy
import math
import unittest

from members.decision.behaviors.lane_change_runtime import LaneChangeDecisionInputs
from core.interfaces import TrajectoryPoint
from members.planning.maneuver_scene import read_maneuver_scene
from members.planning.candidate_validation import validate_candidate
from tests import test_lane_change_integration as fixture
from tests.test_control_maneuvers import BidirectionalPlant
from tests.test_planning_candidate_validation import budget


class LaneChangeBusinessTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.LaneChangeIntegrationTests(); self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.options=dict(maneuver='MERGE',target_lane_id=fixture.TARGET,
                          return_required=False,confirmation_s=.05,indicator_lead_s=.1)
        self.h.change=self.configure

    def configure(self,pair):
        old=pair.decision
        pair.decision=LaneChangeDecisionInputs(old.model_evidence,old.route_evidence,
            old.model_id,'authored-lane-business',fixture.SOURCE,old.extents,old.speed_cap,
            acceleration_uncertainty_mps2=.1,allow_command_lights=True,**self.options)
        if not self.h.f.p.targets:
            pair.planning.prediction_envelopes={}

    def empty_road(self):
        self.h.f.p.targets=[]; self.h.f.build()

    def advance(self,x=None,y=None,heading=None,current=None,speed=1.):
        # A static object has a fixed observed pose as well as zero velocity.
        original=self.h.lead.x
        self.h.advance(x,y,heading,current,speed)
        if self.options['maneuver']=='AVOID':
            self.h.lead.x=original; self.h.f.build()

    def stage(self,wanted):
        for unused in range(40):
            d,t,c=self.h.cycle()
            if d.behavior_request:
                self.assertEqual(self.options['maneuver'],d.behavior_request['maneuver'])
                if d.behavior_request['stage']==wanted:
                    return d,t,c
            self.advance(self.h.f.p.ego.x+.05)
        self.fail(self.h.engine._behavior.lane_changes.snapshot())

    def test_actual_merge_without_lead_steers_and_completes_in_target_lane(self):
        self.empty_road()
        plant=BidirectionalPlant(x=10.,lag=.3); plant.speed=1.
        stages=set(); steered=False; candidate=None
        for step in range(650):
            d,t,c=self.h.cycle()
            state=self.h.engine._behavior.lane_changes
            if d.behavior_request:
                self.assertEqual('MERGE',d.behavior_request['maneuver'])
                stages.add(d.behavior_request['stage'])
                if t.candidate_id:
                    if candidate is None: candidate=t.candidate_id
                    self.assertEqual(candidate,t.candidate_id)
            steered=steered or abs(c.steering)>.001
            if state.phase=='COMPLETE': break
            plant.step(c,.05,self.h.h.calibration)
            current=fixture.TARGET if plant.y>=1.75 else fixture.SOURCE
            self.h.h.curvature=math.tan(plant.steering*self.h.h.calibration.front_steer_max_rad/
                self.h.h.calibration.steering_sign)/self.h.h.calibration.wheelbase_m
            self.advance(plant.x,plant.y,plant.heading,current,plant.speed)
            self.h.f.p.ego.yaw_rate=plant.yaw_rate
        self.assertEqual('COMPLETE',state.phase,(step,state.snapshot(),plant.x,plant.y))
        self.assertEqual(fixture.TARGET,self.h.f.p.lane.lane_id)
        self.assertGreater(plant.y,3.)
        self.assertTrue(steered); self.assertIsNotNone(candidate)
        self.assertTrue({'PREPARE','WAIT_GAP','REQUEST_PATH','EXECUTE','SETTLE'}<=stages)
        self.assertIsNone(state.lead_id)
        # Completion is retained for this one-way task, rather than initiating
        # another crossing or inventing a return to the original corridor.
        self.advance(plant.x,plant.y,plant.heading,fixture.TARGET,plant.speed)
        self.h.cycle(); self.assertEqual('COMPLETE',state.phase)

    def test_static_avoidance_keeps_preparation_stop_and_executes_positive_speed(self):
        self.options.update(maneuver='AVOID',return_required=True)
        self.h.lead.vx=0.; self.h.f.build()
        d,t,c=self.stage('REQUEST_PATH')
        self.assertGreaterEqual(d.stop_distance,0.)
        self.assertTrue(d.precision_stop)
        self.assertIsNone(t.behavior_identity)
        self.assertEqual('PLANNED',t.behavior_feedback['status'])
        self.advance(self.h.f.p.ego.x+.05)
        d,t,c=self.stage('EXECUTE')
        self.assertEqual(-1.,d.stop_distance)
        self.assertGreater(d.target_speed,0.)
        self.assertEqual(0.,self.h.lead.vx)
        self.assertEqual(9,self.h.engine._behavior.lane_changes.lead_id)
        self.assertGreater(max(v.y for v in t.points),3.)
        self.assertGreater(abs(c.steering),.001)
        self.assertTrue(t.planning_metrics['execution_robust'])
        # The independent monitor still sees the real static object; moving
        # its current formal observation onto the delivered path must veto.
        self.h.lead.x,self.h.lead.y=t.points[4].x,t.points[4].y
        self.h.f.build()
        self.assertTrue(self.h.bridge.assess(self.h.f.p,d,t,c).active)

    def test_actual_static_avoidance_passes_body_and_authorizes_return(self):
        self.options.update(maneuver='AVOID',return_required=True)
        self.h.lead.vx=0.; self.h.f.build()
        plant=BidirectionalPlant(x=10.,lag=.3); plant.speed=1.
        phases=set(); stages=set(); candidates={}; passed=False; passing_checks=0; handoff_checked=False
        for step in range(1400):
            state=self.h.engine._behavior.lane_changes
            if (not handoff_checked and state.phase=='OUT'
                    and self.h.f.p.lane.lane_id==fixture.TARGET):
                p=self.h.f.p
                self.h.transport.prepare(p);self.h.bridge.prepare(p)
                self.assertTrue(state.observed_avoidance_handoff(p))
                missing=copy.deepcopy(p)
                missing.targets=[];missing.maneuver_environment.objects=[]
                self.assertFalse(state.observed_avoidance_handoff(missing))
                label_only=copy.deepcopy(p);label_only.ego.y=0.
                self.assertFalse(state.observed_avoidance_handoff(label_only))
                another_lane=copy.deepcopy(p);another_lane.lane.lane_id=fixture.SOURCE
                self.assertFalse(state.observed_avoidance_handoff(another_lane))
                changed=copy.deepcopy(p);changed.target_source='sensor:other-source'
                self.assertFalse(state.observed_avoidance_handoff(changed))
                accepted=state.policy.session.goal.candidate_id
                state.policy.session.goal.candidate_id=None
                self.assertFalse(state.observed_avoidance_handoff(p))
                state.policy.session.goal.candidate_id=accepted
                self.h.bridge.prepare(p)
                handoff_checked=True
            try:
                d,t,c=self.h.cycle()
            except AssertionError as error:
                facts=self.h.engine._last_constraints
                self.fail((str(error),[(v.kind,v.source,v.identifier,v.reason)
                                      for v in facts.items] if facts is not None else []))
            state=self.h.engine._behavior.lane_changes
            phases.add(state.phase)
            if state.phase=='PASS':
                tracking=self.h.bridge.execution_tracking
                self.assertEqual('EXECUTION_WITHIN_BOUNDS',tracking['reason_code'],tracking)
                self.assertLess(tracking['sent_source_frame_id'],self.h.f.p.frame_id)
                passing_checks+=1
            passed=passed or plant.x-.9>self.h.lead.x+2.25+10.5
            if state.phase=='RETURN': self.assertTrue(passed)
            request=d.behavior_request
            if request:
                self.assertEqual('AVOID',request['maneuver'])
                stages.add(request['stage'])
                if t.candidate_id:
                    old=candidates.setdefault(request['intent_id'],t.candidate_id)
                    self.assertEqual(old,t.candidate_id)
            if state.phase=='COMPLETE': break
            plant.step(c,.05,self.h.h.calibration)
            current=fixture.TARGET if plant.y>=1.75 else fixture.SOURCE
            self.h.h.curvature=math.tan(plant.steering*self.h.h.calibration.front_steer_max_rad/
                self.h.h.calibration.steering_sign)/self.h.h.calibration.wheelbase_m
            self.advance(plant.x,plant.y,plant.heading,current,plant.speed)
            self.h.f.p.ego.yaw_rate=plant.yaw_rate
        self.assertEqual('COMPLETE',state.phase,(step,state.snapshot(),plant.x,plant.y))
        self.assertEqual((40.,0.),(self.h.lead.x,self.h.lead.vx))
        self.assertTrue(passed);self.assertGreater(passing_checks,5)
        self.assertTrue(handoff_checked)
        self.assertEqual(fixture.SOURCE,self.h.f.p.lane.lane_id)
        self.assertLess(abs(plant.heading),.15)
        # Completion promises the actual whole car is stable in the original
        # corridor, not an arbitrary 0.3 m center-line error. This stationary
        # probe checks spatial enclosure only, never an actual stop claim.
        scene=read_maneuver_scene(self.h.f.p,(fixture.SOURCE,),self.h.f.clock,self.h.cache)
        model=self.h.inputs(self.h.f.p,self.h.transport.context).planning
        pose=[TrajectoryPoint(plant.x,plant.y,0.,plant.heading,at) for at in (0.,1.)]
        report=validate_candidate(pose,1,model.vehicle,model.limits,scene['corridor'],[],
            budget(max_checks=1000000),execution_envelope=model.execution_envelope)
        self.assertEqual('safe',report['status'],report)
        self.assertTrue({'OUT','PASS','RETURN','COMPLETE'}<=phases)
        self.assertTrue({'PREPARE','WAIT_GAP','REQUEST_PATH','EXECUTE','SETTLE'}<=stages)
        self.assertEqual(2,len(candidates));self.assertEqual(2,len(set(candidates.values())))

    def test_merge_requires_named_target_and_explicit_no_return_task(self):
        self.h.transport.prepare(self.h.f.p)
        pair=self.h.inputs(self.h.f.p,self.h.transport.context); old=pair.decision
        for target,returning in ((None,False),(fixture.TARGET,True),('',False)):
            with self.assertRaisesRegex(ValueError,'TASK_ACTION_OR_TARGET_INVALID'):
                LaneChangeDecisionInputs(old.model_evidence,old.route_evidence,old.model_id,
                    old.mission_id,fixture.SOURCE,old.extents,3.,maneuver='MERGE',
                    target_lane_id=target,return_required=returning)
        for changed in ({'maneuver':'AVOID'},{'target_lane_id':'another-lane'},
                        {'return_required':True}):
            value=old.snapshot()
            for key,item in changed.items():
                setattr(value,key,item)
            self.assertNotEqual(old.binding(),value.binding())

    def test_avoidance_needs_current_static_target_and_active_task_cannot_change(self):
        self.options.update(maneuver='AVOID',return_required=True)
        for unused in range(5):
            d,t,c=self.h.cycle()
            self.assertIsNone(d.behavior_request)
            self.advance(self.h.f.p.ego.x+.05)
        self.assertEqual('STATIC_BLOCKING_TARGET_UNOBSERVED',self.h.engine._behavior.lane_changes.reason)
        self.h.lead.vx=0.; self.h.f.build()
        self.stage('EXECUTE')
        self.options.update(maneuver='MERGE',return_required=False)
        self.advance(self.h.f.p.ego.x+.05)
        d,t,c=self.h.cycle(check=False)
        self.assertIsNone(d.behavior_request)
        self.assertEqual(0.,d.target_speed)
        self.assertIn('ACTIVE_TASK_OR_MODEL_CHANGED',d.reason)

    def one_way_pass(self,action):
        # This is a short business-transition regression with an explicit
        # completed-crossing precondition. The full preceding crossing is
        # covered separately; no arrival/progress feedback is fabricated here.
        state=self.h.engine._behavior.lane_changes
        self.options.update(maneuver=action,return_required=False)
        self.h.lead.x=40.; self.h.lead.vx=0. if action=='AVOID' else 1.
        x=40.+2.25+10.5+.9-.1
        self.advance(x,3.5,0.,fixture.TARGET,2.)
        self.h.transport.prepare(self.h.f.p); self.h.bridge.prepare(self.h.f.p)
        context=self.h.transport.context
        value=self.h.bridge.decision_inputs(self.h.f.p,context)
        state.phase,state.context,state.binding='PASS',context.key(),value.binding()
        state.origin_lane,state.target_lane,state.lead_id=fixture.SOURCE,fixture.TARGET,9
        state.policy=None
        state.last_frame,state.last_time=None,None
        # Each actual observation advances only 0.1 m in 50 ms at
        # measured 2 m/s, using the real ordinary output and monitor.
        d,t,c=self.h.cycle()
        self.assertEqual('PASS',state.phase)
        self.assertIsNone(d.behavior_request)
        for unused in range(5):
            self.advance(self.h.f.p.ego.x+.1,3.5,0.,fixture.TARGET,2.)
            self.h.cycle()
            if state.phase=='COMPLETE': break
        self.assertEqual('COMPLETE',state.phase,state.snapshot())
        self.assertGreater(self.h.f.p.ego.x-.9,self.h.lead.x+2.25+10.5)
        self.assertEqual(fixture.TARGET,self.h.f.p.lane.lane_id)

    def test_one_way_avoidance_needs_actual_retained_body_clearance(self):
        self.one_way_pass('AVOID')

    def test_one_way_overtake_needs_actual_retained_body_clearance(self):
        self.one_way_pass('OVERTAKE')

    def test_unknown_rear_or_short_geometry_never_dispatches_merge(self):
        self.empty_road(); self.h.transport.prepare(self.h.f.p)
        self.h.f.p.maneuver_environment.coverage_regions=[]
        d,t,c=self.h.cycle(check=False)
        self.assertIsNone(d.behavior_request)
        self.assertIn('NO_LEGAL_NEIGHBOR',self.h.engine._behavior.lane_changes.reason)
        self.advance(self.h.f.p.ego.x+.05)
        self.options['transition_distance_m']=1000.
        d,t,c=self.h.cycle(check=False)
        self.assertIsNone(d.behavior_request)
        self.assertIn('OUTSIDE_OBSERVED_GEOMETRY',self.h.engine._behavior.lane_changes.reason)

    def test_merge_budget_failure_never_authorizes_execute(self):
        self.empty_road(); original=self.configure
        def small_budget(pair):
            original(pair); pair.planning.budget.max_checks=1
        self.h.change=small_budget
        d,t,c=self.h.cycle(sent=False,check=False)
        self.assertFalse(t.valid)
        self.assertIn('BUDGET',t.reason)
        self.assertIsNone(t.behavior_identity)
        self.advance(self.h.f.p.ego.x+.05)
        d,t,c=self.h.cycle(sent=False,check=False)
        self.assertTrue(d.behavior_request is None or d.behavior_request['stage']!='EXECUTE')

    def test_missing_prediction_model_is_structured_rejection_not_an_exception(self):
        self.options.update(maneuver='AVOID',return_required=True)
        self.h.lead.vx=0.; self.h.f.build()
        original=self.configure
        def missing_prediction(pair):
            original(pair); pair.planning.prediction_envelopes={}
        self.h.change=missing_prediction
        self.h.cycle(sent=False,check=False)
        self.advance(self.h.f.p.ego.x+.05)
        d,t,c=self.h.cycle(sent=False,check=False)
        self.assertFalse(t.valid)
        self.assertIn('PREDICTION_OBJECT_SET_MISMATCH',t.reason)
        self.assertEqual('REJECTED',t.behavior_feedback['status'])
        self.assertIsNone(t.behavior_identity)


if __name__=='__main__': unittest.main()
