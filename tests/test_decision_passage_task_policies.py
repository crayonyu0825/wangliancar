"""Independent D05–D09 observations and planning feedback are synthetic."""

import math
import unittest

from members.decision.behaviors.contract import Capabilities, BehaviorGoal, GoalPose
from members.decision.behaviors.observations import MotionObject
from members.decision.behaviors.intersection import OccupancyTracker, IntersectionFacts, IntersectionPolicy
from members.decision.behaviors.emergency import EmergencyPolicy, EmergencyCandidate
from members.decision.behaviors.task_progress import TaskRouteFacts, TaskProgressPolicy
from tests.test_decision_behavior_contract import frame, feedback, context
from tests.test_decision_maneuver_policies import evidence

REGION = [(10,-3),(14,-3),(14,3),(10,3)]
EXIT_REGION = [(18,-2),(24,-2),(24,2),(18,2)]


def caps(f):
    return Capabilities(f.context,("INTERSECTION","PATH","PATH_STOP","FEEDBACK","TASK_ROUTE","EMERGENCY_CANDIDATES"),
                        f.observed_at_s,f.observed_at_s+10,usable=True)


def junction(f, objects=(), distance=5.0, green=True, exit_space=15.0, exit_covered=True,
             inside=False, exit_path=False, rule=True, window=(0.5,2.0)):
    return IntersectionFacts("junction:1",evidence(f),REGION,objects,distance,
                              green,rule,"YIELD" if rule else "UNKNOWN",exit_space,
                              exit_covered,window,inside,EXIT_REGION,GoalPose(20,0,0),exit_path)


def task(f, progress=2.0, segment="out", end="window_end", destination=False):
    return TaskRouteFacts(evidence(f),"route:1",[("out","lane-a",0,20),("return","lane-b",20,40)],
                          segment,progress,end,
                          [(-1,4),(1,4),(1,6),(-1,6)] if destination else None,
                          GoalPose(0,5,math.pi) if destination else None)


class PassageTaskPolicyTests(unittest.TestCase):
    def test_group_crossing_predicted_sweep_and_green_independent_yield(self):
        policy=IntersectionPolicy(4.4)
        f=frame()
        people=[MotionObject(1,12,-4,0,2,0.5,0.5),MotionObject(2,11,4,0,-2,0.7,0.5)]
        result=policy.evaluate(f,junction(f,people),caps(f),purpose="CROSSWALK")
        self.assertTrue(result.hold_required)
        self.assertEqual("WAIT_ENTRY",result.phase)
        self.assertEqual(5.0,result.request.goal.stop_distance_m)
        self.assertIn("OCCUPANCY",result.reason)
        self.assertFalse(policy.occupancy.evaluate(f,evidence(f),people,REGION).state == "CLEAR")

    def test_unknown_coverage_and_short_loss_do_not_mean_crosswalk_clear(self):
        tracker=OccupancyTracker(return_guard_s=0.5)
        f=frame()
        occupied=tracker.evaluate(f,evidence(f),[MotionObject(1,12,0,0,0,0.5,0.5)],REGION)
        self.assertEqual("OCCUPIED",occupied.state)
        f=frame(2,0.1)
        self.assertEqual("UNKNOWN",tracker.evaluate(f,evidence(f,False),[],REGION).state)
        self.assertEqual("OCCUPIED",tracker.evaluate(f,evidence(f),[],REGION).state)
        f=frame(3,0.6)
        self.assertEqual("UNKNOWN",tracker.evaluate(f,evidence(f),[],REGION).state)
        self.assertEqual("UNKNOWN",tracker.evaluate(f,evidence(f),[],REGION).state)
        f=frame(4,0.7)
        self.assertEqual("CLEAR",tracker.evaluate(f,evidence(f),[],REGION).state)

    def test_foldback_and_new_id_keep_region_occupied_but_sidewalk_does_not(self):
        tracker=OccupancyTracker()
        f=frame()
        tracker.evaluate(f,evidence(f),[MotionObject(1,12,2,0,-1,0.5,0.5)],REGION)
        f=frame(2,0.1)
        returning=tracker.evaluate(f,evidence(f),[MotionObject(9,12,4,0,-2,0.5,0.5)],REGION)
        self.assertEqual("OCCUPIED",returning.state)
        self.assertIn(9,returning.identifiers)
        clean=OccupancyTracker()
        for index in (1,2):
            f=frame(index,(index-1)*0.1)
            result=clean.evaluate(f,evidence(f),[MotionObject(3,30,8,0,0,0.5,0.5)],REGION)
        self.assertEqual("CLEAR",result.state)

    def test_green_exit_blocked_and_unknown_do_not_admit_vehicle(self):
        for value,covered in ((5.0,True),(None,False)):
            policy=IntersectionPolicy(4.4)
            for index in (1,2):
                f=frame(index,(index-1)*0.1)
                result=policy.evaluate(f,junction(f,exit_space=value,exit_covered=covered),caps(f),purpose="JUNCTION_ENTRY")
            self.assertEqual("WAIT_ENTRY",result.phase)
            self.assertTrue(result.hold_required)
            self.assertIn("EXIT",result.reason)
            f=frame(3,0.2)
            released=policy.evaluate(f,junction(f,exit_space=15),caps(f),purpose="JUNCTION_ENTRY")
            self.assertEqual("ENTER",released.phase)
            self.assertFalse(released.hold_required)

    def test_missing_stop_boundary_and_unknown_rules_are_explicit(self):
        f=frame()
        result=IntersectionPolicy(4.4).evaluate(f,junction(f,distance=None,objects=[MotionObject(1,12,0,0,0,1,1)]),caps(f))
        self.assertEqual("ENTRY_STOP_BOUNDARY_UNKNOWN",result.reason)
        self.assertFalse(result.request.dispatch_allowed)
        unknown=IntersectionPolicy(4.4).evaluate(f,junction(f,rule=False),caps(f))
        self.assertEqual("RIGHT_OF_WAY_UNKNOWN",unknown.reason)
        self.assertTrue(unknown.hold_required)

    def test_already_inside_uses_verified_exit_or_minimum_risk_not_entrance_stop(self):
        policy=IntersectionPolicy(4.4)
        f=frame()
        unavailable=policy.evaluate(f,junction(f,inside=True),caps(f))
        self.assertEqual("INSIDE_RECOVER",unavailable.phase)
        self.assertEqual(-1.0,unavailable.request.goal.stop_distance_m)
        self.assertFalse(unavailable.request.dispatch_allowed)
        policy=IntersectionPolicy(4.4)
        result=policy.evaluate(f,junction(f,inside=True,exit_path=True),caps(f))
        self.assertEqual("EXIT",result.phase)
        self.assertEqual(20,result.request.goal.goal_pose.x)
        f=frame(2,0.1)
        complete=policy.evaluate(f,junction(f,inside=True,exit_path=True),caps(f),feedback(
            result.request,f,producer="control",status="EXECUTING",progress=1,actual_pose=GoalPose(20,0,0)))
        self.assertEqual("COMPLETE",complete.phase)

    def test_clear_conflict_releases_quickly_and_new_conflict_cancels_entry_window(self):
        policy=IntersectionPolicy(4.4)
        for index in (1,2):
            f=frame(index,(index-1)*0.1)
            result=policy.evaluate(f,junction(f),caps(f))
        self.assertEqual("ENTER",result.phase)
        self.assertLess(f.observed_at_s,5.0)
        f=frame(3,0.2)
        blocked=policy.evaluate(f,junction(f,[MotionObject(1,12,0,0,0,1,1)]),caps(f))
        self.assertEqual("WAIT_ENTRY",blocked.phase)
        self.assertTrue(blocked.hold_required)
        self.assertGreater(blocked.request.revision,result.request.revision)

    def test_emergency_brake_is_immediate_without_any_candidate_channel(self):
        f=frame()
        result=EmergencyPolicy().evaluate(f,evidence(f),"danger:1",True)
        self.assertTrue(result.immediate_brake_required)
        self.assertTrue(result.hold_required)
        self.assertFalse(result.request.dispatch_allowed)

    def test_emergency_selects_only_matching_feasible_legal_side_candidate(self):
        policy=EmergencyPolicy()
        f=frame()
        initial=policy.evaluate(f,evidence(f),"d",True,capabilities=caps(f))
        f=frame(2,0.1)
        options=[EmergencyCandidate("left","LEFT",feedback(initial.request,f,candidate_id="left"),True,0.1,0,side_coverage_verified=True,crossing_allowed=False),
                 EmergencyCandidate("right","BRAKE_RIGHT",feedback(initial.request,f,candidate_id="right"),True,0.2,1,
                                    BehaviorGoal(target_lane_id="right"),True,True),
                 EmergencyCandidate("brake","BRAKE",feedback(initial.request,f,candidate_id="brake"),True,0.8,0)]
        selected=policy.evaluate(f,evidence(f),"d",True,options,caps(f))
        self.assertEqual("right",selected.selected)
        self.assertTrue(selected.immediate_brake_required)
        self.assertEqual("EXECUTE_BRAKE_RIGHT",selected.phase)
        f=frame(3,0.2)
        changed=policy.evaluate(f,evidence(f),"new-danger",True,options,caps(f))
        self.assertIsNone(changed.selected)
        self.assertTrue(changed.immediate_brake_required)
        self.assertGreater(changed.request.revision,initial.request.revision)

    def test_emergency_unknown_infeasible_and_future_feedback_all_fallback(self):
        policy=EmergencyPolicy()
        f=frame()
        initial=policy.evaluate(f,evidence(f),"d",True,capabilities=caps(f))
        f=frame(2,0.1)
        bad=feedback(initial.request,f)
        bad.produced_frame_id=50
        choices=[EmergencyCandidate("unknown","RIGHT",feedback(initial.request,f),None,0,0),
                 EmergencyCandidate("bad","LEFT",bad,True,0,0,side_coverage_verified=True,crossing_allowed=True)]
        result=policy.evaluate(f,evidence(f),"d",True,choices,caps(f))
        self.assertIsNone(result.selected)
        self.assertTrue(result.immediate_brake_required)

    def test_new_danger_while_selecting_invalidates_old_candidate_and_accepts_new_one(self):
        policy = EmergencyPolicy()
        f = frame()
        initial = policy.evaluate(f, evidence(f), "danger-A", True, capabilities=caps(f))
        old_choice = EmergencyCandidate("left-for-A", "LEFT",
            feedback(initial.request, f, candidate_id="left-for-A"), True, 0, 0,
            BehaviorGoal(target_lane_id="left", speed_cap_mps=1), True, True)
        f = frame(2, 0.1)
        changed = policy.evaluate(f, evidence(f), "danger-B", True, [old_choice], caps(f))
        self.assertIsNone(changed.selected)
        self.assertEqual("SELECT", changed.request.stage)
        self.assertEqual(initial.request.intent_id, changed.request.intent_id)
        self.assertGreater(changed.request.revision, initial.request.revision)
        self.assertEqual(f.frame_id, changed.request.source_frame_id)
        self.assertEqual(f.observed_at_s, changed.request.issued_at_s)
        self.assertTrue(changed.immediate_brake_required)
        f = frame(3, 0.2)
        new_choice = EmergencyCandidate("right-for-B", "BRAKE_RIGHT",
            feedback(changed.request, f, candidate_id="right-for-B"), True, 0, 0,
            BehaviorGoal(target_lane_id="right"), True, True)
        selected = policy.evaluate(f, evidence(f), "danger-B", True, [new_choice], caps(f))
        self.assertEqual("right-for-B", selected.selected)
        self.assertEqual("EXECUTE_BRAKE_RIGHT", selected.phase)

    def test_new_danger_with_no_capabilities_invalidates_feedback_before_recovery(self):
        policy = EmergencyPolicy()
        f = frame()
        initial = policy.evaluate(f, evidence(f), "danger-A", True)
        f = frame(2, 0.1)
        changed = policy.evaluate(f, evidence(f), "danger-B", True)
        self.assertGreater(changed.request.revision, initial.request.revision)
        self.assertFalse(changed.request.dispatch_allowed)
        f = frame(3, 0.2)
        old_choice = EmergencyCandidate("old", "BRAKE",
            feedback(initial.request, f, candidate_id="old"), True, 0, 0)
        recovered = policy.evaluate(f, evidence(f), "danger-B", True, [old_choice], caps(f))
        self.assertIsNone(recovered.selected)
        self.assertEqual("SELECT", recovered.request.stage)
        self.assertTrue(recovered.immediate_brake_required)

    def test_emergency_requires_verified_recovery_then_can_handle_new_danger(self):
        policy=EmergencyPolicy()
        f=frame()
        initial=policy.evaluate(f,evidence(f),"d",True,capabilities=caps(f))
        f=frame(2,0.1)
        choice=EmergencyCandidate("brake","BRAKE",feedback(initial.request,f,candidate_id="brake"),True,0,0)
        selected=policy.evaluate(f,evidence(f),"d",True,[choice],caps(f))
        for index in (3,4):
            f=frame(index,(index-1)*0.1)
            result=policy.evaluate(f,evidence(f),"d",False,capabilities=caps(f),
                execution_feedback=feedback(policy.session.snapshot(f),f,producer="control",status="COMPLETED",
                                            progress=1,actual_standstill_confirmed=True,
                                            goal_pose_arrived=True,actual_pose=GoalPose(f.ego_x,f.ego_y,f.ego_heading)),
                verified_task_route_available=True)
        self.assertEqual("COMPLETE",result.phase)
        self.assertFalse(result.immediate_brake_required)
        f=frame(5,0.4)
        same=policy.evaluate(f,evidence(f),"d",False,capabilities=caps(f))
        self.assertFalse(same.immediate_brake_required)
        f=frame(6,0.5)
        new=policy.evaluate(f,evidence(f),"d",True,capabilities=caps(f))
        self.assertTrue(new.immediate_brake_required)
        self.assertNotEqual(selected.request.intent_id,new.request.intent_id)

    def test_rotated_translated_crosswalk_has_same_occupancy(self):
        outcomes=[]
        for angle,shift in ((0.0,(0,0)),(0.9,(80,-20))):
            def turn(p):
                return (shift[0]+p[0]*math.cos(angle)-p[1]*math.sin(angle),
                        shift[1]+p[0]*math.sin(angle)+p[1]*math.cos(angle))
            x,y=turn((12,-4))
            vx,vy=-2*math.sin(angle),2*math.cos(angle)
            obj=MotionObject(1,x,y,vx,vy,0.5,0.5,angle)
            f=frame()
            result=OccupancyTracker().evaluate(f,evidence(f),[obj],[turn(p) for p in REGION])
            outcomes.append((result.state,result.identifiers))
        self.assertEqual(outcomes[0],outcomes[1])

    def test_empty_emergency_and_unusable_new_inputs_do_not_dispatch_old_actions(self):
        f=frame()
        empty=EmergencyPolicy().evaluate(f,evidence(f),"none",False,capabilities=caps(f))
        self.assertIsNone(empty.request)
        self.assertFalse(empty.immediate_brake_required)
        policy=IntersectionPolicy(4.4)
        first=policy.evaluate(f,junction(f),caps(f))
        f=frame(2,0.1)
        unknown=policy.evaluate(f,None,caps(f))
        self.assertEqual(first.request.intent_id,unknown.request.intent_id)
        self.assertFalse(unknown.request.dispatch_allowed)
        self.assertTrue(unknown.hold_required)

    def test_unidentified_candidate_and_diagnostic_ground_truth_cannot_admit_behavior(self):
        f=frame()
        e=evidence(f)
        e.source_kind="diagnostic_ground_truth"
        unknown=IntersectionPolicy(4.4).evaluate(f,junction(f),caps(f))
        facts=junction(f)
        facts.evidence=e
        rejected=IntersectionPolicy(4.4).evaluate(f,facts,caps(f))
        self.assertIsNone(rejected.request)
        self.assertTrue(rejected.hold_required)
        policy=EmergencyPolicy()
        first=policy.evaluate(f,evidence(f),"d",True,capabilities=caps(f))
        f=frame(2,0.1)
        no_candidate_id=EmergencyCandidate("left","LEFT",feedback(first.request,f),True,0,0,
                                           side_coverage_verified=True,crossing_allowed=True)
        result=policy.evaluate(f,evidence(f),"d",True,[no_candidate_id],caps(f))
        self.assertIsNone(result.selected)
        self.assertTrue(result.immediate_brake_required)

    def test_task_window_end_is_not_destination_and_progress_cannot_jump_return_leg(self):
        policy=TaskProgressPolicy()
        f=frame(speed=2)
        initial=policy.evaluate(f,task(f),caps(f))
        self.assertEqual("FOLLOW_ROUTE",initial.phase)
        f=frame(2,0.1,speed=2)
        jumped=policy.evaluate(f,task(f,progress=35,segment="return"),caps(f))
        self.assertEqual("RECOVER",jumped.phase)
        self.assertFalse(jumped.request.dispatch_allowed)
        window=TaskProgressPolicy().evaluate(f,task(f,progress=20,end="window_end"),caps(f))
        self.assertEqual("WAIT_ROUTE",window.phase)
        self.assertNotEqual("COMPLETE",window.phase)

    def test_task_completion_requires_actual_destination_pose_and_stop(self):
        policy=TaskProgressPolicy()
        f=frame(speed=1)
        initial=policy.evaluate(f,task(f,39.9,"return","task_destination",True),caps(f))
        f=frame(2,0.1,speed=1)
        planned=policy.evaluate(f,task(f,40,"return","task_destination",True),caps(f),feedback(initial.request,f))
        self.assertNotEqual("COMPLETE",planned.phase)
        f=frame(3,0.2)
        near=policy.evaluate(f,task(f,40,"return","task_destination",True),caps(f),feedback(planned.request,f,
            producer="control",status="COMPLETED",goal_pose_arrived=True,actual_standstill_confirmed=True,
            actual_pose=GoalPose(0,5.3,math.pi),progress=1))
        self.assertNotEqual("COMPLETE",near.phase)
        f=frame(4,0.3)
        done=policy.evaluate(f,task(f,40,"return","task_destination",True),caps(f),feedback(near.request,f,
            producer="control",status="COMPLETED",goal_pose_arrived=True,actual_standstill_confirmed=True,
            actual_pose=GoalPose(0,5,math.pi),progress=1))
        self.assertEqual("COMPLETE",done.phase)

    def test_temporary_maneuver_preserves_original_route_and_task_change_resets(self):
        policy=TaskProgressPolicy()
        f=frame(speed=2)
        original=policy.evaluate(f,task(f),caps(f))
        f=frame(2,0.1,speed=2)
        waiting=policy.evaluate(f,task(f,2.2),caps(f),temporary_maneuver_active=True)
        self.assertFalse(waiting.request.dispatch_allowed)
        self.assertEqual("lane-a",waiting.request.goal.target_lane_id)
        f=frame(3,0.2,speed=2)
        resumed=policy.evaluate(f,task(f,2.4),caps(f))
        self.assertEqual(original.request.intent_id,resumed.request.intent_id)
        self.assertEqual("lane-a",resumed.request.goal.target_lane_id)
        self.assertGreater(resumed.request.revision,original.request.revision)
        f=frame(4,0.3,context("new"),speed=2)
        new=policy.evaluate(f,task(f),caps(f))
        self.assertNotEqual(original.request.intent_id,new.request.intent_id)


if __name__ == "__main__":
    unittest.main()
