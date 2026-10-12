"""Remaining-lane proofs after an accepted crossing, no SDK or live control."""
import copy
import math
import unittest
from unittest.mock import patch

from core.behavior_contract import GoalPose
from core.maneuver_facts import ManeuverFacts
from members import planning_stub
from members.planning.lane_change_generator import plan_lane_change_candidate,PredictionEnvelope
from members.planning.maneuver_scene import ManeuverGeometryCache
from tests import test_planning_lane_change_behavior as fixture
from tests.test_planning_candidate_validation import budget
from tests.test_maneuver_perception import target


SOURCE,TARGET=fixture.SOURCE,fixture.TARGET


class TargetLaneContinuationTests(unittest.TestCase):
    def setUp(self):
        self.b=fixture.PlanningLaneChangeBehaviorTests(); self.b.setUp()
        self.addCleanup(self.b.doCleanups)
        self.h,self.f=self.b.h,self.b.f

    def enter(self,x=25.,y=3.5,yaw=0.,speed=1.,remove_neighbor=False):
        self.b.execute()
        self.h.advance(x,y,yaw,TARGET,speed)
        self.f.route._maneuver_map.crossing_ranges.clear(); self.h.producer.update()
        if remove_neighbor:
            self.f.p.maneuver_environment.neighbor_lanes=[]
            self.f.p.maneuver_environment.road_regions=[v for v in
                self.f.p.maneuver_environment.road_regions if v['lane_id']==TARGET]
        with self.assertRaises(ValueError): ManeuverFacts(self.f.p,self.f.clock).crossing(TARGET,SOURCE)
        r=self.h.policy.session.snapshot(self.h.frame())
        return r,self.b.decision(r)

    def candidate(self,**changes):
        values=dict(perception=self.f.p,target_lane_id=TARGET,initial_curvature_m_inv=self.b.curvature,
            speed_cap_mps=1.5,vehicle=self.b.geometry,limits=self.b.motion,search=fixture.search(),
            budget=budget(max_checks=1000000),prediction_envelopes={},clock=self.f.clock,
            source_lane_id=SOURCE,fixed_goal_pose=(35.,3.5,0.),allow_target_lane_continuation=True)
        values.update(changes)
        return plan_lane_change_candidate(**values)

    def test_inside_target_retains_accepted_identity_and_goal_without_crossing_proof(self):
        r,d=self.enter(); t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        self.assertEqual('CURRENT_TARGET_NOMINAL_CONTINUATION_PATH',t.reason)
        self.assertEqual((r.intent_id,r.stage,r.revision),tuple(t.behavior_identity[k]
                         for k in ('intent_id','stage','revision')))
        self.assertEqual((25.,3.5,1.,0.),(t.points[0].x,t.points[0].y,t.points[0].speed,t.points[0].heading))
        self.assertEqual((35.,3.5),(t.points[-1].x,t.points[-1].y))
        self.assertTrue(t.left_signal); self.assertFalse(t.right_signal)
        self.assertEqual('PLANNED',t.behavior_feedback['status'])

    def test_original_neighbor_geometry_is_unneeded_after_whole_body_enters(self):
        r,d=self.enter(remove_neighbor=True)
        candidates,failures=self.h.candidates()
        self.assertTrue(failures); self.assertEqual(TARGET,candidates[0].lane_id)
        self.assertFalse(candidates[0].crossing_allowed)
        t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        result=self.candidate(geometry_cache=ManeuverGeometryCache(2,1000000))
        self.assertEqual('safe',result['status'],result)
        self.assertEqual('current_target_lane',result['path_scope'])
        self.assertEqual('safe',result['current_body_report']['status'])
        self.assertEqual('safe',result['legal_report']['status'])
        self.assertEqual('safe',result['visibility_report']['status'])
        self.assertEqual((SOURCE,TARGET),(result['source_lane_id'],result['target_lane_id']))

    def test_settle_stage_uses_same_remaining_lane_proof(self):
        r,d=self.enter()
        result=self.h.evaluate(self.h.actual_reply(r,candidate_id=self.h.policy.session.goal.candidate_id))
        self.assertEqual('SETTLE',result.request.stage)
        t=planning_stub.plan(self.f.p,self.b.decision(result.request))
        self.assertTrue(t.valid,t.errors)
        self.assertEqual('CURRENT_TARGET_NOMINAL_CONTINUATION_PATH',t.reason)

    def test_whole_body_in_target_may_align_but_cannot_complete_while_heading_is_unstable(self):
        r,d=self.enter(x=27.4,y=2.86,yaw=.17)
        result=self.h.evaluate(self.h.actual_reply(r,candidate_id=self.h.policy.session.goal.candidate_id))
        self.assertFalse(result.hold_required,result.reason)
        self.assertEqual('EXECUTE',result.phase)
        self.assertNotEqual('COMPLETED',result.request.status)
        self.assertEqual(0,self.h.policy._settled)
        t=planning_stub.plan(self.f.p,self.b.decision(result.request))
        self.assertTrue(t.valid,t.errors)
        self.assertEqual('CURRENT_TARGET_NOMINAL_CONTINUATION_PATH',t.reason)

    def test_sdk_id_switch_cannot_authorize_straddling_without_original_permission(self):
        for y,yaw in ((2.2,.15),(3.3,.4)):
            with self.subTest(y=y,yaw=yaw):
                if y==2.2: r,d=self.enter(y=y,yaw=yaw)
                else:
                    self.f.p.ego.y,self.f.p.ego.heading=y,yaw
                    r=self.h.policy.session.snapshot(self.h.frame()); d=self.b.decision(r)
                t=planning_stub.plan(self.f.p,d)
                self.assertFalse(t.valid); self.assertFalse(t.points)
                self.assertIsNone(t.behavior_identity)

    def test_straddling_with_original_permission_uses_original_corridor(self):
        self.b.execute(); self.h.advance(18.,2.2,.15,TARGET,1.)
        result=self.candidate()
        self.assertEqual('safe',result['status'],result)
        self.assertEqual('source_bound_crossing',result['path_scope'])
        self.assertEqual('unsafe',result['current_body_report']['status'])
        self.assertEqual('left',result['crossing_side'])
        self.assertLessEqual(result['checks'],1000000)

    def test_no_second_crossing_fallback_when_a_remaining_path_leaves_target(self):
        self.b.execute(); self.h.advance(10.,3.5,0.,TARGET,1.)
        result=self.candidate(initial_curvature_m_inv=.2)
        self.assertEqual('current_target_lane',result['path_scope'])
        self.assertEqual('safe',result['current_body_report']['status'])
        self.assertFalse(result['points'],result)
        self.assertTrue(any(v.get('legal_report',{}).get('reason_code')=='CORRIDOR_COLLISION'
                            for v in result['attempts']),result)

    def test_default_component_api_still_requires_original_crossing_proof(self):
        self.enter()
        self.assertFalse(self.candidate(allow_target_lane_continuation=False)['points'])
        self.assertFalse(self.candidate(allow_target_lane_continuation='yes')['points'])

    def test_cold_execute_cannot_use_target_continuation_without_accepted_binding(self):
        r,d=self.enter()
        planning_stub.configure_planning(lane_change_inputs_provider=self.b.inputs)
        self.b.reject(d,'LANE_CHANGE_REQUEST_PATH_REQUIRED')

    def test_request_path_cannot_use_target_continuation_to_create_an_intent(self):
        r,d=self.enter()
        payload=copy.deepcopy(d.behavior_request)
        payload['stage']='REQUEST_PATH'; payload['revision']+=1
        t=planning_stub.plan(self.f.p,self.b.decision(payload))
        self.assertFalse(t.valid); self.assertFalse(t.points)

    def test_current_target_model_and_original_goal_cannot_be_replaced(self):
        r,d=self.enter()
        self.b.input_changes={'model_id':'different-model'}
        self.b.reject(d,'LANE_CHANGE_ACTIVE_BINDING_CHANGED')
        self.b.input_changes={}
        payload=copy.deepcopy(d.behavior_request)
        payload['goal_pose']['x']+=1.; payload['revision']+=1
        self.b.reject(self.b.decision(payload),'LANE_CHANGE_ACTIVE_BINDING_CHANGED')

    def test_whole_body_probe_and_remaining_path_share_the_original_budget(self):
        self.enter()
        tiny=self.candidate(budget=budget(max_checks=1))
        self.assertFalse(tiny['points']); self.assertEqual('BUDGET_EXHAUSTED',tiny['reason_code'])
        self.assertLessEqual(tiny['checks'],2)
        self.b.input_changes={'budget':budget(max_checks=1)}
        r=self.h.policy.session.snapshot(self.h.frame())
        self.b.reject(self.b.decision(r),'BUDGET_EXHAUSTED')

    def test_unknown_or_wrong_height_visibility_cannot_become_a_remaining_path(self):
        r,d=self.enter()
        original=copy.deepcopy(self.f.p.maneuver_environment.coverage_regions)
        wrong_height=copy.deepcopy(original)
        for view in wrong_height: view['reference_z_m']+=1.
        for views in ([],wrong_height):
            with self.subTest(views=views):
                self.f.p.maneuver_environment.coverage_regions=views
                self.b.reject(d,'MANEUVER_PLANNING_COVERAGE_UNAVAILABLE')

    def test_new_current_target_object_requires_prediction_and_rejects_intrusion(self):
        self.b.execute(); self.f.p.targets=[target(29.,3.5,heading=None)]
        self.h.advance(25.,3.5,0.,TARGET,1.)
        self.f.route._maneuver_map.crossing_ranges.clear(); self.h.producer.update()
        r=self.h.policy.session.snapshot(self.h.frame()); d=self.b.decision(r)
        self.b.reject(d,'PREDICTION_OBJECT_SET_MISMATCH')
        self.b.input_changes={'prediction_envelopes':{9:PredictionEnvelope(60.,.1,.1)}}
        t=planning_stub.plan(self.f.p,d)
        self.assertFalse(t.valid); self.assertFalse(t.points)

    def test_expiry_during_target_continuation_cannot_publish_old_proof(self):
        r,d=self.enter()
        def delayed(*args,**kwargs):
            value=plan_lane_change_candidate(*args,**kwargs)
            self.f.now+=.3
            return value
        with patch('members.planning.lane_change_behavior.plan_lane_change_candidate',delayed):
            t=self.b.reject(d,'LANE_CHANGE_SOURCE_EXPIRED_DURING_DISPATCH')
        self.assertFalse(t.behavior_feedback['usable'])

    def test_reversed_direction_keeps_original_right_indicator_without_original_neighbor(self):
        self.f.p.ego.x,self.f.p.ego.heading=60.,math.pi
        self.h.producer.restrict('decrease','LHT')
        self.b.goal=GoalPose(35.,3.5,math.pi)
        r,d=self.enter(x=45.,yaw=math.pi,remove_neighbor=True)
        t=planning_stub.plan(self.f.p,d)
        self.assertTrue(t.valid,t.errors)
        self.assertTrue(t.right_signal); self.assertFalse(t.left_signal)
        self.assertEqual((35.,3.5,math.pi),(t.points[-1].x,t.points[-1].y,t.points[-1].heading))

    def test_actual_fixed_entries_complete_after_original_neighbor_is_withdrawn(self):
        self.b.assert_continuous_completion(withdraw_original=True)


if __name__=='__main__': unittest.main()
