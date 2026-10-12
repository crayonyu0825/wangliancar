"""Fixed-entry model identity regressions; all sources and bounds are synthetic.

The formal fixtures use the actual decision/planning/control, input bridge and
old-sent monitor. These tests stop at model rejection, without a full mission
loop or any live vehicle sends. The None cases cover only the legacy component.
"""
import copy
import unittest

from members import planning_stub
from members.planning.candidate_validation import ExecutionEnvelope
from tests import test_lane_change_integration as lane_fixture
from tests import test_driving_integration as driving_fixture
from tests import test_planning_lane_change_behavior as legacy_fixture


BOUND_FIELDS=('position_error_m','heading_error_rad','early_time_s','late_time_s','speed_error_mps')


def widen_model(pair,field):
    model=pair.planning
    owner=model if field=='speed_error_mps' else model.execution_envelope
    setattr(owner,field,getattr(owner,field)+.01)


class PlanningModelBindingTests(unittest.TestCase):
    def test_lane_fixed_chain_rejects_same_id_bound_changes_after_adoption(self):
        for field in BOUND_FIELDS:
            with self.subTest(field=field):
                h=lane_fixture.LaneChangeIntegrationTests()
                try:
                    h.setUp()
                    d,t,c=h.stage('EXECUTE')
                    intent=d.behavior_request['intent_id'];cid=t.candidate_id
                    original_goal=copy.deepcopy(d.behavior_request['goal_pose'])
                    original_model=h.bridge.value.planning.model_id
                    binding=planning_stub._LANE_CHANGE_PLANNER.bindings[intent]
                    h.change=lambda pair,field=field:widen_model(pair,field)
                    h.advance(h.f.p.ego.x+.05)
                    d,t,c=h.cycle(check=False)
                    self.assertEqual(intent,d.behavior_request['intent_id'])
                    self.assertEqual(original_goal,d.behavior_request['goal_pose'])
                    self.assertEqual(cid,d.behavior_request['candidate_id'])
                    self.assertEqual(original_model,h.bridge.value.planning.model_id)
                    self.assertFalse(t.valid)
                    self.assertEqual('REJECTED',t.behavior_feedback['status'])
                    self.assertIn('LANE_CHANGE_ACTIVE_BINDING_CHANGED',t.behavior_feedback['reason_code'])
                    self.assertEqual([],t.points)
                    self.assertEqual(binding,planning_stub._LANE_CHANGE_PLANNER.bindings[intent])
                    self.assertEqual(cid,planning_stub._LANE_CHANGE_PLANNER.candidate_ids[intent])
                finally:
                    h.doCleanups()

    def test_driving_fixed_chain_rejects_same_id_bound_changes_for_same_intent(self):
        for field in BOUND_FIELDS:
            with self.subTest(field=field):
                h=driving_fixture.DrivingIntegrationTests()
                try:
                    h.setUp()
                    d,t,c,unused=h.cycle()
                    intent=d.behavior_request['intent_id']
                    original_model=h.bridge.value.planning.model_id
                    binding=planning_stub._DRIVING_PLANNER.bindings[intent]
                    h.change=lambda pair,field=field:widen_model(pair,field)
                    h.advance(c)
                    d,t,c,unused=h.cycle(check=False)
                    self.assertEqual(intent,d.behavior_request['intent_id'])
                    self.assertEqual(original_model,h.bridge.value.planning.model_id)
                    self.assertFalse(t.valid)
                    self.assertEqual('REJECTED',t.behavior_feedback['status'])
                    self.assertIn('DRIVING_ACTIVE_MODEL_OR_MISSION_CHANGED',t.behavior_feedback['reason_code'])
                    self.assertEqual([],t.points)
                    self.assertEqual(binding,planning_stub._DRIVING_PLANNER.bindings[intent])
                finally:
                    h.doCleanups()

    def test_legacy_none_bounds_remain_usable_and_cannot_change_under_same_intent(self):
        cases=({'execution_envelope':ExecutionEnvelope(0.,0.,0.,0.)},
               {'speed_error_mps':0.})
        for changed in cases:
            with self.subTest(field=next(iter(changed))):
                h=legacy_fixture.PlanningLaneChangeBehaviorTests()
                try:
                    h.setUp()
                    r,d,t=h.execute()
                    self.assertTrue(t.valid)
                    self.assertIsNone(h.inputs(h.f.p,h.transport.context).execution_envelope)
                    self.assertIsNone(h.inputs(h.f.p,h.transport.context).speed_error_mps)
                    original=planning_stub._LANE_CHANGE_PLANNER.bindings[r.intent_id]
                    h.input_changes=changed
                    rejected=planning_stub.plan(h.f.p,d)
                    self.assertFalse(rejected.valid)
                    self.assertEqual('REJECTED',rejected.behavior_feedback['status'])
                    self.assertIn('ACTIVE_BINDING_CHANGED',rejected.behavior_feedback['reason_code'])
                    self.assertEqual(original,planning_stub._LANE_CHANGE_PLANNER.bindings[r.intent_id])
                finally:
                    h.doCleanups()


if __name__=='__main__':unittest.main()
