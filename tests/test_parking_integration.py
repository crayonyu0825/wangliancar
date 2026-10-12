"""Offline captain integration; real fixed entries, synthetic original sources."""
import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.safety_supervisor import SafetySupervisor
from members import decision_stub,planning_stub,control_stub
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from parking_integration import ParkingInputBridge,ParkingIntegrationInputs,PARKING_ACTIONS
from runtime import CaptainRuntime
from tests import test_planning_parking_behavior as fixture
from tests.test_planning_candidate_validation import budget
from tests.test_safety_runtime import FakeAdapter
from tests.test_maneuver_perception import target
from members.planning.lane_change_generator import PredictionEnvelope


class ParkingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.ParkingFixedEntryTests(); self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.f=self.h.f; self.calls=0; self.change=None; self.enabled=True
        self.bridge=ParkingInputBridge(self.inputs,self.f.clock); self.bridge.settings=self.h.settings
        self.engine=DecisionEngine(DecisionSettings(front_offset_m=3.9,half_width_m=.9),
                                  self.f.clock,self.bridge.decision_inputs)
        manager=patch('members.decision_stub._ENGINE',self.engine)
        manager.start(); self.addCleanup(manager.stop)
        planning_stub.configure_planning(parking_inputs_provider=self.bridge.planning_inputs)
        self.safety=SafetySupervisor(SimpleNamespace(),self.f.clock,self.h.settings,self.engine.settings,
                                     parking_guard=self.bridge.assess)

    def inputs(self,p,context):
        self.calls+=1
        if not self.enabled: return None
        pair=ParkingIntegrationInputs(self.h.decision_inputs(p,context),self.h.inputs(p,context),
                                      budget(max_checks=1000000))
        if self.change: self.change(pair)
        return pair

    def cycle(self):
        self.bridge.prepare(self.f.p)
        d=decision_stub.decide(self.f.p); t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        c=control_stub.compute_control(self.f.p,t); self.assertTrue(c.valid,c.errors)
        return d,t,c

    def test_one_original_pair_feeds_both_members_and_independent_monitor(self):
        d,t,c=self.cycle()
        self.assertEqual(1,self.calls)
        self.assertEqual('APPROACH',d.behavior_request['stage'])
        self.assertEqual('normal',self.safety.evaluate(self.f.p,d,t,c).mode)
        self.assertEqual(1,self.calls)
        self.assertEqual('CURRENT_PARKING_PATH_RECHECKED',self.bridge.reason)

    def test_returned_models_are_detached_and_repeated_frame_does_not_refresh(self):
        self.bridge.prepare(self.f.p); context=self.h.transport.context
        a=self.bridge.decision_inputs(self.f.p,context)
        b=self.bridge.planning_inputs(self.f.p,context)
        a.mission.approach_pose.x=999.; b.vehicle.front_offset_m=999.
        self.assertEqual(15.,self.bridge.decision_inputs(self.f.p,context).mission.approach_pose.x)
        self.assertEqual(3.9,self.bridge.planning_inputs(self.f.p,context).vehicle.front_offset_m)
        self.f.now+=.01; self.bridge.prepare(self.f.p)
        self.assertEqual(1,self.calls)
        self.f.now=self.f.p.valid_until
        with self.assertRaises(ValueError): self.bridge.decision_inputs(self.f.p,context)

    def test_mismatched_task_model_geometry_tolerance_and_source_hold_decision(self):
        changes=(lambda v:setattr(v.planning,'model_id','other'),
            lambda v:setattr(v.planning.mission.position_pose,'x',999.),
            lambda v:setattr(v.planning.vehicle,'rear_offset_m',1.),
            lambda v:setattr(v.planning,'tolerances',(.2,.15,.05)),
            lambda v:setattr(v.decision.model_evidence,'observed_at_s',self.f.now-.01),
            lambda v:setattr(v.planning,'initial_curvature_m_inv',None),
            lambda v:setattr(v.monitor_budget,'max_checks',0))
        for change in changes:
            self.h.advance(); self.change=change; self.bridge.prepare(self.f.p)
            d=decision_stub.decide(self.f.p)
            self.assertIsNone(d.behavior_request,d.reason); self.assertEqual(0.,d.target_speed,d.reason)

    def test_configured_control_geometry_mismatch_is_unavailable(self):
        self.bridge.settings=copy.copy(self.h.settings); self.bridge.settings.wheelbase_m+=.1
        self.bridge.prepare(self.f.p)
        d=decision_stub.decide(self.f.p)
        self.assertEqual(0.,d.target_speed); self.assertIn('CONTROL_GEOMETRY_MISMATCH',d.reason)

    def test_unprepared_new_frame_and_source_withdrawal_do_not_reuse_pair(self):
        d,t,c=self.cycle(); self.h.advance()
        with self.assertRaises(ValueError): self.bridge.planning_inputs(self.f.p,self.h.transport.context)
        self.enabled=False; self.bridge.prepare(self.f.p)
        held=decision_stub.decide(self.f.p)
        self.assertIsNone(held.behavior_request); self.assertEqual(0.,held.target_speed)
        self.assertEqual('SUSPENDED',self.engine._behavior.parking.policy.session.status)

    def test_provider_error_is_cached_for_the_frame(self):
        def failed(*args): self.calls+=1; raise RuntimeError('synthetic source lost')
        self.bridge.provider=failed
        for unused in range(2): self.bridge.prepare(self.f.p)
        self.assertEqual(1,self.calls)
        d=decision_stub.decide(self.f.p); self.assertEqual(0.,d.target_speed)
        self.assertIn('synthetic source lost',d.reason)

    def test_monitor_rechecks_withdrawn_view_and_fresh_obstacle_after_planning(self):
        d,t,c=self.cycle()
        self.f.p.maneuver_environment.coverage_regions=[]
        result=self.safety.evaluate(self.f.p,d,t,c)
        self.assertTrue(result.active); self.assertIn('COVERAGE',result.reason)
        self.f.build()
        self.f.p.targets=[target(13.,0.)]; self.f.build()
        result=self.bridge.assess(self.f.p,d,t,c)
        self.assertTrue(result.active); self.assertIn('PREDICTION_OBJECT_SET_MISMATCH',result.reason)

    def test_monitor_budget_exhaustion_and_source_expiry_withdraw_motion(self):
        self.change=lambda v:setattr(v.monitor_budget,'max_checks',1)
        d,t,c=self.cycle(); result=self.safety.evaluate(self.f.p,d,t,c)
        self.assertTrue(result.active); self.assertIn('BUDGET',result.reason)
        self.f.now=self.f.p.valid_until
        self.assertTrue(self.bridge.assess(self.f.p,d,t,c).active)

    def test_stationary_hold_uses_existing_standstill_bound_and_requires_actual_brake(self):
        self.h.advance(self.h.h.mission().approach_pose,speed=.03)
        d,t,c=self.cycle()
        self.assertEqual(0.,t.points[0].speed); self.assertEqual(.03,self.f.p.ego.speed)
        self.assertEqual('normal',self.bridge.assess(self.f.p,d,t,c).mode)
        altered=copy.deepcopy(c); altered.throttle=.1
        self.assertIn('CONTROL_NOT_NORMALIZED',self.bridge.assess(self.f.p,d,t,altered).reason)
        self.assertEqual(.1,altered.throttle)
        altered.brake=0.
        self.assertIn('ACTUAL_START_MISMATCH',self.bridge.assess(self.f.p,d,t,altered).reason)
        self.f.p.ego.vx=.06
        self.assertIn('ACTUAL_START_MISMATCH',self.bridge.assess(self.f.p,d,t,c).reason)

    def test_monitor_checks_actual_start_identity_direction_and_original_endpoint(self):
        d,t,c=self.cycle()
        for field,value in (('motion_direction',-1),('behavior_identity',None)):
            altered=copy.deepcopy(t); setattr(altered,field,value)
            self.assertTrue(self.bridge.assess(self.f.p,d,altered,c).active)
        altered=copy.deepcopy(t); altered.points[0].x+=.1
        self.assertIn('ACTUAL_START_MISMATCH',self.bridge.assess(self.f.p,d,altered,c).reason)
        altered=copy.deepcopy(t); altered.points[-1].x+=.3
        self.assertIn('ARRIVAL_REGION_MISMATCH',self.bridge.assess(self.f.p,d,altered,c).reason)
        self.f.p.ego.speed=1.; self.f.p.ego.vx=-1.
        self.assertIn('ACTUAL_DIRECTION_MISMATCH',self.bridge.assess(self.f.p,d,t,c).reason)

    def test_monitor_does_not_apply_forward_route_ttc_to_valid_parking_path(self):
        # A deliberately obsolete derived forward TTC is irrelevant to the
        # formal objects/path. A real current obstacle is separately rechecked.
        self.f.p.targets=[target(40.,0.)]; self.f.build()
        self.h.input_changes={'prediction_envelopes':{9:PredictionEnvelope(60.,.1,.1)}}
        d,t,c=self.cycle()
        with patch.object(SafetySupervisor,'_imminent_collision',return_value=True) as forward:
            result=self.safety.evaluate(self.f.p,d,t,c)
        self.assertEqual('normal',result.mode,result.reason); self.assertFalse(forward.called)

    def test_new_collision_on_certified_path_has_independent_emergency_veto(self):
        self.f.p.targets=[target(40.,0.)]; self.f.build()
        self.h.input_changes={'prediction_envelopes':{9:PredictionEnvelope(60.,.1,.1)}}
        d,t,c=self.cycle()
        self.f.p.targets[0].x=13.; self.f.build()
        result=self.safety.evaluate(self.f.p,d,t,c)
        self.assertEqual('emergency_stop',result.mode,result.reason)
        self.assertIn('OBSTACLE_COLLISION',result.reason)
        self.assertEqual(0.,result.control.throttle); self.assertEqual(1.,result.control.brake)

    def test_authorized_reverse_stage_is_monitored_in_its_actual_direction(self):
        d,t,c=self.cycle(); self.h.transport.observe(self.f.p,d,t,c)
        for pose in (self.h.h.mission().approach_pose,self.h.h.mission().approach_pose,
                     self.h.h.mission().position_pose,self.h.h.mission().position_pose):
            self.h.advance(pose); d,t,c=self.cycle(); self.h.transport.observe(self.f.p,d,t,c)
        self.assertEqual('REVERSE_ENTRY',d.behavior_request['stage'])
        # The just-issued revision must first consume matching acceptance.
        # A physical gear handoff also holds still before reverse propulsion.
        self.h.advance(self.h.h.mission().position_pose)
        d,t,c=self.cycle(); self.h.transport.observe(self.f.p,d,t,c)
        self.h.advance(self.h.h.mission().position_pose,speed=.2,direction=-1)
        d,t,c=self.cycle()
        self.assertEqual(-1,t.motion_direction)
        self.assertEqual('normal',self.safety.evaluate(self.f.p,d,t,c).mode)
        self.f.p.ego.vx=.2
        result=self.bridge.assess(self.f.p,d,t,c)
        self.assertTrue(result.active); self.assertIn('ACTUAL_DIRECTION_MISMATCH',result.reason)

    def test_sources_expiring_or_withdrawing_during_recheck_cannot_remain_clear(self):
        from members.planning import parking_scene
        original=parking_scene.validate_candidate
        d,t,c=self.cycle()
        def withdraw(*args,**kwargs):
            result=original(*args,**kwargs)
            self.f.p.maneuver_environment.coverage_regions=[]
            return result
        with patch.object(parking_scene,'validate_candidate',side_effect=withdraw):
            result=self.bridge.assess(self.f.p,d,t,c)
        self.assertTrue(result.active); self.assertIn('COVERAGE',result.reason)
        self.f.build()
        def expire(*args,**kwargs):
            result=original(*args,**kwargs); self.f.now+=.3; return result
        with patch.object(parking_scene,'validate_candidate',side_effect=expire):
            self.assertTrue(self.bridge.assess(self.f.p,d,t,c).active)

    def test_supervisor_never_accepts_reverse_flag_without_parking_monitor(self):
        d,t,c=self.cycle(); safety=SafetySupervisor(SimpleNamespace(),self.f.clock)
        self.assertEqual('parking_monitor_unavailable',safety.evaluate(self.f.p,d,t,c).reason)
        d.behavior_request=None; t.motion_direction=-1
        self.assertEqual('reverse_maneuver_unverified',safety.evaluate(self.f.p,d,t,c).reason)
        t.motion_direction=1; self.f.p.ego.speed=1.; self.f.p.ego.vx=-1.
        self.assertEqual('actual_motion_opposes_forward_path',safety.evaluate(self.f.p,d,t,c).reason)

    def config(self,directory):
        values=dict(runtime_dir=directory,loop_hz=20,send_control=False,safety_brake_enabled=False,
                    publish_json=True,control_calibrated=True)
        values.update(('control_'+k,v) for k,v in vars(self.h.calibration).items())
        return SimpleNamespace(**values)

    def test_actual_captain_loop_uses_paired_provider_and_never_sends_in_observe_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            config=self.config(directory)
            with patch.dict(os.environ,{'NEVC_VEHICLE_FRONT_OFFSET_M':'3.9','NEVC_VEHICLE_HALF_WIDTH_M':'.9'}):
                runtime=CaptainRuntime(config,SimpleNamespace(info=lambda *a:None,warning=lambda *a:None),
                                       parking_inputs_provider=self.inputs,parking_actions=PARKING_ACTIONS)
            self.h.transport=runtime.behaviors
            runtime.adapter=FakeAdapter(1)
            runtime._loop(SimpleNamespace(build=lambda:self.f.p),True)
            self.assertEqual([],runtime.adapter.sent)
            with open(runtime.pipeline_path,encoding='utf-8') as stream: pipeline=json.load(stream)
            self.assertEqual('PARK',pipeline['decision']['behavior_request']['maneuver'])
            self.assertTrue(pipeline['trajectory']['valid'],pipeline['trajectory']['errors'])
            self.assertTrue(pipeline['control']['valid'],pipeline['control']['errors'])
            self.assertEqual('normal',pipeline['safety']['mode'],pipeline['safety']['reason'])
            self.assertEqual('observe_mode',pipeline['send']['reason'])
            self.assertEqual(1,self.calls)

    def test_runtime_capabilities_require_explicit_complete_opt_in(self):
        logger=SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)
        with tempfile.TemporaryDirectory() as directory:
            config=self.config(directory)
            for provider,actions in ((None,PARKING_ACTIONS),(self.inputs,('PARK',)),
                    (self.inputs,PARKING_ACTIONS+('PARK',))):
                with self.assertRaises(ValueError): CaptainRuntime(config,logger,provider,actions)
            config.control_calibrated=False
            with self.assertRaises(ValueError): CaptainRuntime(config,logger,self.inputs,PARKING_ACTIONS)


if __name__=='__main__': unittest.main()
