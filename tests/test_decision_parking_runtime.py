"""Fixed decide entry with formal facts and explicit synthetic task/model."""
import copy
import unittest
from unittest.mock import patch

from core.behavior_channel import BehaviorTransport
from core.behavior_contract import TaskContext
from core.interfaces import DecisionTarget,DecisionMode,ControlOut
from core.validation import validate_output
from members import decision_stub,planning_stub
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from tests import test_planning_parking_behavior as fixture
from tests.test_maneuver_perception import target


class FixedParkingDecisionTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.ParkingFixedEntryTests(); self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.f=self.h.f; self.enabled=True; self.changes={}
        settings=DecisionSettings(front_offset_m=3.9,half_width_m=.9)
        self.engine=DecisionEngine(settings,self.f.clock,self.inputs)
        manager=patch('members.decision_stub._ENGINE',self.engine)
        manager.start(); self.addCleanup(manager.stop)

    def inputs(self,p,context):
        if not self.enabled: return None
        value=self.h.decision_inputs(p,context)
        for key,item in self.changes.items(): setattr(value,key,item)
        return value

    def decide(self):
        self.h.transport.prepare(self.f.p)
        d=decision_stub.decide(self.f.p); validate_output(d,DecisionTarget,self.f.p)
        return d

    def hold(self,d):
        self.assertIn(d.mode,(DecisionMode.STOP,DecisionMode.EMERGENCY_BRAKE),d.reason)
        self.assertEqual(0.,d.target_speed); self.assertIsNone(d.behavior_request)
        self.assertIsNone(d.behavior_active_identity)

    def test_fixed_decide_emits_formal_approach_consumed_by_fixed_plan(self):
        d=self.decide(); self.assertEqual('PARK',d.behavior_request['maneuver'])
        self.assertEqual('APPROACH',d.behavior_request['stage'])
        self.assertEqual(dict(x=15.,y=0.,body_heading_rad=0.,reference_point='ego_rear_axle'),
                         d.behavior_request['goal_pose'])
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        self.assertEqual(d.behavior_request['intent_id'],t.behavior_identity['intent_id'])

    def test_absent_task_keeps_legacy_forward_and_reverse_rejection(self):
        self.enabled=False; d=self.decide()
        self.assertIsNone(d.behavior_request); self.assertGreater(d.target_speed,0.,d.reason)
        self.f.p.ego.speed=1.; self.f.p.ego.vx=-1.
        d=self.decide(); self.hold(d); self.assertIn('reverse_unsupported',d.reason)

    def test_task_withdrawal_suspends_and_recovery_requires_new_revision(self):
        first=self.decide().behavior_request
        self.h.advance(); self.enabled=False; self.hold(self.decide())
        self.assertEqual('SUSPENDED',self.engine._behavior.parking.policy.session.status)
        self.h.advance(); self.enabled=True; resumed=self.decide().behavior_request
        self.assertEqual(first['intent_id'],resumed['intent_id'])
        self.assertGreater(resumed['revision'],first['revision'])

    def test_provider_error_and_malformed_snapshot_fail_closed(self):
        self.engine._behavior.parking.provider=lambda *args:object()
        self.hold(self.decide())
        def failed(*args): raise RuntimeError('synthetic source unavailable')
        self.engine._behavior.parking.provider=failed
        self.hold(self.decide())

    def test_model_frame_context_source_and_expiry_are_rechecked(self):
        evidence=self.h.decision_inputs(self.f.p,self.h.transport.context).model_evidence
        for key,value in (('frame_id',0),('context',TaskContext('other','task',7,'other')),
                          ('source_kind','diagnostic_ground_truth'),('valid_until_s',self.f.now),
                          ('coverage_verified',False)):
            altered=copy.deepcopy(evidence); setattr(altered,key,value)
            self.changes={'model_evidence':altered}; self.hold(self.decide())
        for changes in ({'extents':(3.9,.9,False)},{'tolerances':None},{'max_checks':0},
                        {'position_error':None},{'speed_cap':0.},{'dwell_s':9.}):
            self.changes=changes; self.hold(self.decide())

    def test_model_or_task_cannot_move_an_active_intent(self):
        first=self.decide().behavior_request; self.h.advance()
        self.changes={'model_id':'different-model'}; d=self.decide(); self.hold(d)
        self.assertIn('ACTIVE_MODEL_OR_TASK_CHANGED',d.reason)
        self.assertEqual(first['intent_id'],self.engine._behavior.parking.policy.session.intent_id)
        self.changes={}; self.h.advance(); restored=self.decide().behavior_request
        self.assertEqual(first['intent_id'],restored['intent_id'])

    def test_empty_capabilities_cannot_start_parking(self):
        self.h.transport=BehaviorTransport((),self.f.clock)
        self.h.transport.prepare(self.f.p)
        d=self.decide(); self.hold(d)
        self.assertIn('CAPABILITY',d.reason)

    def test_lost_runtime_channel_withdraws_an_existing_intent(self):
        self.decide(); self.h.advance(); self.f.p.behavior_channel={}
        d=decision_stub.decide(self.f.p); self.hold(d)
        self.assertIn('channel unavailable',d.reason)
        self.assertEqual('SUSPENDED',self.engine._behavior.parking.policy.session.status)

    def test_existing_signal_dwell_has_priority_even_after_green(self):
        from tests.test_decision import add_red_light
        self.enabled=False; p=self.f.p; p.scene_id=20
        # This test explicitly starts a new authored scene. A changed scene
        # cannot silently keep the parking fixture's previous runtime session.
        self.h.transport.reset_session(); self.f.build()
        self.h.transport.actions+=('PATH_STOP',)
        add_red_light(p,20.); p.traffic.signal_id=42
        first=self.decide(); self.assertIsNotNone(first.behavior_request,first.reason)
        self.assertEqual('SIGNAL_STOP',first.behavior_request['maneuver'])
        duty=first.behavior_request['stop_obligation_id']
        self.h.advance(); p.traffic.signal_state='GREEN'; self.enabled=True
        d=self.decide(); self.assertIsNotNone(d.behavior_request,d.reason)
        self.assertEqual('SIGNAL_STOP',d.behavior_request['maneuver'])
        self.assertEqual(duty,d.behavior_request['stop_obligation_id'])
        self.assertIsNone(self.engine._behavior.parking.policy.session.intent_id)

    def test_new_mission_cannot_replace_an_unfinished_task(self):
        first=self.decide().behavior_request; self.h.advance()
        self.changes={'mission':self.h.h.mission(mission_id='other-verified-task')}
        d=self.decide(); self.hold(d)
        self.assertIn('ACTIVE_MODEL_OR_TASK_CHANGED',d.reason)
        self.assertEqual(first['intent_id'],self.engine._behavior.parking.policy.session.intent_id)

    def test_red_interrupts_an_already_requested_parking_stage(self):
        from tests.test_decision import add_red_light
        self.decide(); self.h.advance(); self.h.transport.actions+=('PATH_STOP',)
        add_red_light(self.f.p,20.); self.f.p.traffic.signal_id=42
        d=self.decide(); self.hold(d)
        self.assertEqual('SUSPENDED',self.engine._behavior.parking.policy.session.status)

    def test_positive_published_limit_caps_the_fixed_planning_output(self):
        self.f.p.lane.speed_limit=.4
        d=self.decide(); self.assertEqual(.4,d.target_speed)
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        self.assertLessEqual(max(v.speed for v in t.points),.4+1e-9)

    def test_clock_expiry_or_regression_after_prepare_cannot_publish_motion(self):
        original=self.engine._behavior.observe_legacy
        def shifted(*args,**kwargs):
            original(*args,**kwargs); self.f.now+=self.shift
        for delta in (.3,-.01):
            self.shift=delta
            with patch.object(self.engine._behavior,'observe_legacy',side_effect=shifted):
                d=decision_stub.decide(self.f.p)
            self.hold(d); self.f.now-=delta
            self.h.advance()

    def test_planning_ack_does_not_advance_the_parking_stage(self):
        d=self.decide(); t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        self.h.transport.observe(self.f.p,d,t,ControlOut().bind(t))
        self.h.advance(); following=self.decide().behavior_request
        self.assertEqual('APPROACH',following['stage'])

    def test_unknown_signal_stop_without_a_signal_request_still_blocks_parking(self):
        p=self.f.p; p.traffic.required=True; p.traffic.observed=True
        p.traffic.valid=False; p.traffic.signal_state='RED'; p.traffic.stop_line_distance=-1.
        d=self.decide(); self.hold(d)
        self.assertIn('SAFETY_OR_SIGNAL',d.reason)

    def test_current_object_invasion_or_unknown_coverage_blocks_formal_dispatch(self):
        self.decide(); self.h.advance()
        self.f.p.targets=[target(self.f.p.ego.x,self.f.p.ego.y)]
        self.f.build(); self.hold(self.decide())
        self.f.p.targets=[]; self.f.build()
        self.f.p.maneuver_environment.coverage_regions=[]; self.hold(self.decide())

    def test_short_model_deadline_binds_both_request_and_decision(self):
        value=self.h.decision_inputs(self.f.p,self.h.transport.context)
        value.model_evidence.valid_until_s=self.f.now+.03
        self.changes={'model_evidence':value.model_evidence}
        d=self.decide(); self.assertEqual(self.f.now+.03,d.valid_until)
        self.assertEqual(d.valid_until,d.behavior_request['valid_until_s'])
        t=planning_stub.plan(self.f.p,d); self.assertTrue(t.valid,t.errors)
        self.h.advance(); self.hold(self.decide())

    def test_unexpected_reverse_is_not_authorized_by_an_approach_task(self):
        self.f.p.ego.speed=.3; self.f.p.ego.vx=-.3
        d=self.decide(); self.hold(d); self.assertIn('UNEXPECTED_REVERSE_MOTION',d.reason)

    def test_invalid_gps_suspends_existing_intent_before_legacy_protection(self):
        first=self.decide().behavior_request; self.f.p.source_status['gps']['usable']=False
        self.hold(self.decide()); self.assertEqual('SUSPENDED',self.engine._behavior.parking.policy.session.status)
        self.f.p.source_status['gps']['usable']=True; self.h.advance()
        resumed=self.decide().behavior_request
        self.assertEqual(first['intent_id'],resumed['intent_id']); self.assertGreater(resumed['revision'],first['revision'])

    def test_source_expiry_during_formal_read_withdraws_dispatch(self):
        from members.decision.behaviors.parking_runtime import parking_facts
        def delayed(*args,**kwargs):
            result=parking_facts(*args,**kwargs); self.f.now+=.3; return result
        with patch('members.decision.behaviors.parking_runtime.parking_facts',delayed):
            d=decision_stub.decide(self.f.p)
        self.hold(d)

    def test_zero_published_speed_limit_suspends_a_formal_task(self):
        self.decide(); self.f.p.lane.speed_limit=0.
        d=self.decide(); self.hold(d); self.assertIn('SPEED_LIMIT_ZERO',d.reason)


if __name__=='__main__': unittest.main()
