"""Independent D03/D04 behavior with synthetic geometry and explicit feedback."""

import math
import unittest

from members.decision.behaviors.contract import Capabilities, GoalPose
from members.decision.behaviors.observations import Evidence, MotionObject
from members.decision.behaviors.lane_change import LaneCandidate, LaneChangePolicy
from members.decision.behaviors.parking import ParkingSpace, ParkingFacts, ParkingPolicy
from tests.test_decision_behavior_contract import frame, feedback as contract_feedback


def feedback(request, f, **changes):
    # These are authored synthetic stopped-pose observations. Parking cases
    # explicitly report the direction of the stage they model; the generic
    # contract fixture's forward default is not reverse execution evidence.
    if request.maneuver == 'PARK':
        changes.setdefault('motion_direction', request.goal.motion_direction)
    return contract_feedback(request, f, **changes)


def evidence(f, covered=True):
    return Evidence(f.context, f.frame_id, f.observed_at_s, f.observed_at_s + 0.2,
                    usable=True, coverage_verified=covered, source="synthetic_authored", source_kind="synthetic")


def caps(f):
    return Capabilities(f.context, ("LANE_CHANGE", "PARK", "PATH", "LIGHTS", "REVERSE", "DWELL", "FEEDBACK"),
                        f.observed_at_s, f.observed_at_s + 10.0, usable=True)


def lane(f, objects=(), marking="DASHED", allowed=True, forward=True, rear=True):
    return LaneCandidate("left", "left", [(-150.0, 3.5), (200.0, 3.5)], 3.5,
                         marking, allowed, forward, evidence(f), rear, objects,
                         [(-150.0,5.25),(200.0,5.25)],[(-150.0,1.75),(200.0,1.75)],
                         is_current_lane=f.ego_y>=1.75)


def parking(f, occupancy="empty", boundary=None, objects=(), rear=True, signed=0.0):
    boundary = boundary or [(7,-1.5),(13,-1.5),(13,1.5),(7,1.5)]
    space = ParkingSpace(1, boundary, 0.0, occupancy, evidence(f),
                         approach_pose=GoalPose(0,0,0), position_pose=GoalPose(15,0,0))
    return ParkingFacts(evidence(f), [space], [(-20,-20),(40,-20),(40,20),(-20,20)],
                        objects, rear, True, GoalPose(25,0,0), signed)


class ManeuverPolicyTests(unittest.TestCase):
    def test_lane_gap_checks_front_rear_closing_and_unknown_coverage(self):
        policy = LaneChangePolicy(3.5,0.9,0.9)
        f = frame(speed=5.0)
        self.assertTrue(policy.opportunity(f,lane(f))[0])
        rear = MotionObject(1,-80,3.5,20,0,4,2)
        front = MotionObject(2,15,3.5,0,0,4,2)
        self.assertIn("REAR", policy.opportunity(f,lane(f,[rear]))[1])
        self.assertIn("FRONT", policy.opportunity(f,lane(f,[front]))[1])
        unknown = policy.evaluate(f,"avoid:1","AVOID",[lane(f,rear=False)],caps(f),speed_cap_mps=5.)
        self.assertIsNone(unknown.request)
        self.assertEqual("BLOCKED",unknown.phase)

    def test_initial_selection_prefers_safe_legal_opportunity_then_stays_stable(self):
        f=frame(speed=5.0)
        blocked=lane(f,[MotionObject(1,15,3.5,0,0,4,2)])
        safe=LaneCandidate("right","right",[(-150,-3.5),(200,-3.5)],3.5,"DASHED",True,True,evidence(f),True,(),
                           [(-150,-1.75),(200,-1.75)],[(-150,-5.25),(200,-5.25)])
        policy=LaneChangePolicy(3.5,0.9,0.9)
        result=policy.evaluate(f,"avoid","AVOID",[blocked,safe],caps(f),speed_cap_mps=5.)
        self.assertEqual("right",result.request.goal.target_lane_id)
        f=frame(2,0.1,speed=5.0)
        refreshed=lane(f)
        safe.evidence=evidence(f)
        again=policy.evaluate(f,"avoid","AVOID",[refreshed,safe],caps(f))
        self.assertEqual("right",again.request.goal.target_lane_id)

    def test_solid_and_opposite_candidates_never_dispatch(self):
        for candidate_args in ({"marking":"SOLID"},{"forward":False},{"allowed":False}):
            f=frame()
            result=LaneChangePolicy(3.5,0.9,0.9).evaluate(f,"merge","MERGE",[lane(f,**candidate_args)],caps(f),speed_cap_mps=5.)
            self.assertIsNone(result.request)

    def test_lane_phases_need_actual_indicator_path_and_settlement_feedback(self):
        policy=LaneChangePolicy(3.5,0.9,0.9)
        f=frame(speed=5.0)
        r=policy.evaluate(f,"avoid:1","AVOID",[lane(f)],caps(f),speed_cap_mps=5.).request
        self.assertEqual("PREPARE",r.stage)
        f=frame(2,0.1,speed=5.0)
        r=policy.evaluate(f,"avoid:1","AVOID",[lane(f)],caps(f),feedback(r,f,producer="control",status="EXECUTING",
             lights_confirmed=True,lights_duration_s=1.0)).request
        self.assertEqual("WAIT_GAP",r.stage)
        f=frame(3,0.2,speed=5.0)
        r=policy.evaluate(f,"avoid:1","AVOID",[lane(f)],caps(f)).request
        self.assertEqual("REQUEST_PATH",r.stage)
        f=frame(4,0.3,speed=5.0)
        r=policy.evaluate(f,"avoid:1","AVOID",[lane(f)],caps(f),feedback(r,f)).request
        self.assertEqual("EXECUTE",r.stage)
        f=frame(5,0.4,speed=5.0)
        unchanged=policy.evaluate(f,"avoid:1","AVOID",[lane(f)],caps(f),feedback(r,f)).request
        self.assertEqual("EXECUTE",unchanged.stage)
        for index in (6,7,8):
            f=frame(index,(index-1)*0.1,speed=5.0)
            f.ego_x,f.ego_y=5.,3.5
            result=policy.evaluate(f,"avoid:1","AVOID",[lane(f)],caps(f),feedback(policy.session.snapshot(f),f,
                 producer="control",status="EXECUTING",progress=0.9,actual_lane_id="left",actual_pose=GoalPose(5,3.5,0)))
        self.assertEqual("COMPLETED",result.request.status)
        self.assertFalse(result.request.goal.light_intent.left_signal)
        same=policy.evaluate(frame(9,0.8,speed=5.0),"avoid:1","AVOID",[lane(frame(9,0.8))],caps(frame(9,0.8)))
        self.assertEqual("COMPLETE",same.phase)

    def test_execution_obstruction_requires_recovery_path_not_snap_back(self):
        policy=LaneChangePolicy(3.5,0.9,0.9)
        f=frame()
        policy.evaluate(f,"change","LANE_CHANGE",[lane(f)],caps(f),speed_cap_mps=2.)
        policy.session.advance(f,"EXECUTE")
        f=frame(2,0.1)
        blocked=policy.evaluate(f,"change","LANE_CHANGE",[lane(f,[MotionObject(1,5,3.5,0,0,4,2)])],caps(f))
        self.assertEqual("RECOVER",blocked.phase)
        self.assertTrue(blocked.hold_required)
        self.assertEqual("left",blocked.request.goal.target_lane_id)
        self.assertFalse(blocked.request.dispatch_allowed)
        self.assertFalse(blocked.request.goal.light_intent.left_signal)

    def test_execution_recovery_requests_new_path_and_restores_indicator_with_finite_retries(self):
        policy=LaneChangePolicy(3.5,0.9,0.9)
        f=frame()
        initial=policy.evaluate(f,"change","LANE_CHANGE",[lane(f)],caps(f),speed_cap_mps=2.)
        policy.session.advance(f,"EXECUTE")
        for cycle in range(3):
            blocked_frame=frame(2+cycle*2,0.1+cycle*0.2)
            policy.evaluate(blocked_frame,"change","LANE_CHANGE",[lane(blocked_frame,
                 [MotionObject(1,5,3.5,0,0,4,2)])],caps(blocked_frame))
            recovered_frame=frame(3+cycle*2,0.2+cycle*0.2)
            result=policy.evaluate(recovered_frame,"change","LANE_CHANGE",[lane(recovered_frame)],caps(recovered_frame))
            if cycle < 2:
                self.assertEqual(initial.request.intent_id,result.request.intent_id)
                self.assertEqual("REQUEST_PATH",result.phase)
                self.assertTrue(result.request.goal.light_intent.left_signal)
                policy.session.advance(recovered_frame,"EXECUTE")
            else:
                self.assertEqual("BLOCKED",result.request.status)
                self.assertFalse(result.request.dispatch_allowed)

    def test_parking_evidence_recovery_does_not_leave_a_zero_speed_goal(self):
        policy=ParkingPolicy(3.5,0.9,0.9)
        f=frame()
        policy.evaluate(f,parking(f),caps(f))
        f=frame(2,0.1)
        blocked=policy.evaluate(f,parking(f,rear=False),caps(f))
        self.assertEqual("BLOCKED",blocked.phase)
        f=frame(3,0.2)
        recovered=policy.evaluate(f,parking(f),caps(f))
        self.assertGreater(recovered.request.goal.speed_cap_mps,0.0)
        self.assertEqual("APPROACH",recovered.request.stage)
        self.assertGreater(recovered.request.revision,blocked.request.revision)

    def test_lane_settlement_requires_entire_body_inside_not_only_rear_axle(self):
        policy=LaneChangePolicy(3.5,0.9,0.9)
        f=frame()
        policy.evaluate(f,"change","LANE_CHANGE",[lane(f)],caps(f),speed_cap_mps=2.)
        r=policy.session.advance(f,"EXECUTE")
        f=frame(2,0.1)
        result=policy.evaluate(f,"change","LANE_CHANGE",[lane(f)],caps(f),feedback(r,f,
            producer="control",status="EXECUTING",actual_lane_id="left",actual_pose=GoalPose(5,4.0,0.15),progress=1))
        self.assertEqual("EXECUTE",result.phase)
        self.assertNotEqual("COMPLETED",result.request.status)

    def test_lane_boundary_geometry_overrides_an_optimistic_width_field(self):
        policy = LaneChangePolicy(3.5, 0.9, 0.9)
        f = frame()
        candidate = lane(f)
        candidate.left_boundary = [(-150, 4.0), (200, 4.0)]
        candidate.right_boundary = [(-150, 3.0), (200, 3.0)]
        safe, reason = policy.opportunity(f, candidate)
        self.assertFalse(safe)
        self.assertEqual("TARGET_BOUNDARY_SPACE_INSUFFICIENT", reason)
        # A concavity can intersect the body even with all corners inside.
        from members.decision.behaviors.observations import corridor_contains
        self.assertFalse(corridor_contains([(-2, 1), (2, 1), (2, -1), (-2, -1)],
            [(-5, 2), (0, 0), (5, 2)], [(-5, -2), (5, -2)]))

    def test_perpendicular_and_parallel_space_goal_transform_consistently(self):
        for angle in (0.0,math.pi/2):
            f=frame()
            original=parking(f)
            def turn(p):
                return (100+p[0]*math.cos(angle)-p[1]*math.sin(angle),
                        50+p[0]*math.sin(angle)+p[1]*math.cos(angle))
            s=ParkingSpace(1,[turn(p) for p in original.spaces[0].boundary],angle,"empty",evidence(f),
                GoalPose(*turn((0,0)),angle),GoalPose(*turn((15,0)),angle))
            facts=ParkingFacts(evidence(f),[s],[turn(p) for p in original.free_area],(),True,True,
                               GoalPose(*turn((25,0)),angle))
            policy=ParkingPolicy(3.5,0.9,0.9)
            result=policy.evaluate(f,facts,caps(f))
            self.assertIsNotNone(result.request)
            expected=turn((8.7,0))
            self.assertAlmostEqual(expected[0],policy.park_goal.x)
            self.assertAlmostEqual(expected[1],policy.park_goal.y)
            self.assertAlmostEqual(angle,policy.park_goal.body_heading_rad)

    def test_parking_selects_rear_axle_goal_and_rejects_unknown_or_small_space(self):
        f=frame()
        policy=ParkingPolicy(3.5,0.9,0.9)
        result=policy.evaluate(f,parking(f),caps(f))
        self.assertIsNotNone(result.request)
        self.assertAlmostEqual(8.7,policy.park_goal.x)
        self.assertNotEqual(10.0,policy.park_goal.x)
        for args in ({"occupancy":"unknown"},{"rear":False},
                     {"boundary":[(8.5,-1),(11.5,-1),(11.5,1),(8.5,1)]},
                     {"objects":[MotionObject(3,10,0,0,0,4,2)]}):
            rejected=ParkingPolicy(3.5,0.9,0.9).evaluate(f,parking(f,**args),caps(f))
            self.assertFalse(rejected.request is not None and rejected.request.dispatch_allowed)

    def test_parking_requires_actual_stop_pose_ten_seconds_and_exit(self):
        policy=ParkingPolicy(3.5,0.9,0.9)
        f=frame()
        r=policy.evaluate(f,parking(f),caps(f)).request
        identity=r.intent_id
        f=frame(2,0.1)
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="ARRIVED",
             goal_pose_arrived=True,actual_standstill_confirmed=True,actual_pose=GoalPose(0,0,0),progress=1)).request
        self.assertEqual("POSITION",r.stage)
        f=frame(3,0.2)
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f)).request
        self.assertEqual("POSITION",r.stage)
        f=frame(4,0.3)
        f.ego_x=15.
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="ARRIVED",
             goal_pose_arrived=True,actual_standstill_confirmed=True,actual_pose=GoalPose(15,0,0),progress=1)).request
        self.assertEqual("REVERSE_ENTRY",r.stage)
        self.assertEqual(-1,r.goal.motion_direction)
        obligation=r.goal.stop_obligation_id
        f=frame(5,0.4)
        f.ego_x=policy.park_goal.x
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="ARRIVED",
             goal_pose_arrived=True,actual_standstill_confirmed=True,actual_pose=policy.park_goal,progress=1)).request
        self.assertEqual("PARKED_DWELL",r.stage)
        self.assertEqual(obligation,r.goal.stop_obligation_id)
        f=frame(6,0.5)
        f.ego_x=policy.park_goal.x
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="COMPLETED",
             goal_pose_arrived=True,actual_standstill_confirmed=True,actual_pose=policy.park_goal,
             progress=1,hold_completed=True,standstill_duration_s=9)).request
        self.assertEqual("PARKED_DWELL",r.stage)
        f=frame(7,0.6)
        f.ego_x=policy.park_goal.x
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="COMPLETED",
             goal_pose_arrived=True,actual_standstill_confirmed=True,actual_pose=policy.park_goal,
             progress=1,hold_completed=True,standstill_duration_s=10)).request
        self.assertEqual("EXIT_PREPARE",r.stage)
        f=frame(8,0.7)
        f.ego_x=policy.park_goal.x
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="ARRIVED",
             goal_pose_arrived=True,actual_standstill_confirmed=True,actual_pose=policy.park_goal)).request
        self.assertEqual("EXIT",r.stage)
        self.assertEqual(1,r.goal.motion_direction)
        f=frame(9,0.8)
        f.ego_x=25.
        r=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="ARRIVED",
             goal_pose_arrived=True,actual_standstill_confirmed=True,actual_pose=GoalPose(25,0,0),progress=1)).request
        self.assertEqual("COMPLETED",r.status)
        self.assertEqual(identity,r.intent_id)
        for index, kind in ((10, "normal"), (11, "missing"), (12, "unknown_rear")):
            f = frame(index, (index - 1) * 0.1)
            facts = None if kind == "missing" else parking(f, rear=kind != "unknown_rear")
            same = policy.evaluate(f, facts, caps(f))
            self.assertEqual("COMPLETE", same.phase)
            self.assertEqual("COMPLETED", same.request.status)
            self.assertEqual(identity, same.request.intent_id)
            self.assertFalse(same.request.dispatch_allowed)

    def test_updated_parking_boundary_blocks_old_goal_and_recovery_requires_new_feedback(self):
        policy = ParkingPolicy(3.5, 0.9, 0.9)
        f = frame()
        policy.evaluate(f, parking(f), caps(f))
        original_goal = policy.park_goal
        request = policy.session.advance(f, "REVERSE_ENTRY", policy._goal(original_goal, -1, True))
        f = frame(2, 0.1)
        small = parking(f, boundary=[(8.5,-1), (11.5,-1), (11.5,1), (8.5,1)])
        blocked = policy.evaluate(f, small, caps(f), feedback(request, f,
            producer="control", status="ARRIVED", actual_standstill_confirmed=True,
            goal_pose_arrived=True, actual_pose=original_goal, progress=1))
        self.assertEqual("BLOCKED", blocked.request.status)
        self.assertTrue(blocked.hold_required)
        self.assertFalse(blocked.request.dispatch_allowed)
        self.assertNotEqual("PARKED_DWELL", blocked.request.stage)
        f = frame(3, 0.2)
        recovered = policy.evaluate(f, parking(f), caps(f), feedback(request, f,
            producer="control", status="ARRIVED", actual_standstill_confirmed=True,
            goal_pose_arrived=True, actual_pose=original_goal, progress=1))
        self.assertEqual("REVERSE_ENTRY", recovered.request.stage)
        self.assertGreater(recovered.request.revision, request.revision)
        self.assertIsNone(policy.session.last_feedback)
        self.assertIs(original_goal, policy.park_goal)
        self.assertGreater(recovered.request.goal.speed_cap_mps, 0)
        f = frame(4, 0.3)
        f.ego_x=original_goal.x
        arrived = policy.evaluate(f, parking(f), caps(f), feedback(recovered.request, f,
            producer="control", status="ARRIVED", actual_standstill_confirmed=True,
            goal_pose_arrived=True, actual_pose=original_goal, progress=1))
        self.assertEqual("PARKED_DWELL", arrived.request.stage)

    def test_updated_parking_boundary_checks_actual_body_without_moving_selected_goal(self):
        policy = ParkingPolicy(3.5, 0.9, 0.9)
        f = frame()
        policy.evaluate(f, parking(f), caps(f))
        goal = policy.park_goal
        request = policy.session.advance(f, "REVERSE_ENTRY", policy._goal(goal, -1, True))
        f = frame(2, 0.1)
        refined = parking(f, boundary=[(7.75,-1.5), (13,-1.5), (13,1.5), (7.75,1.5)])
        actual = GoalPose(goal.x - 0.1, goal.y, goal.body_heading_rad)
        self.assertTrue(policy._body_clear(goal, refined, refined.spaces[0]))
        self.assertFalse(policy._body_clear(actual, refined, refined.spaces[0]))
        result = policy.evaluate(f, refined, caps(f), feedback(request, f,
            producer="control", status="ARRIVED", actual_standstill_confirmed=True,
            goal_pose_arrived=True, actual_pose=actual, progress=1))
        self.assertEqual("ALIGN", result.request.stage)
        self.assertIs(goal, policy.park_goal)

    def test_cancelled_parking_keeps_terminal_state_on_next_frame(self):
        policy = ParkingPolicy(3.5, 0.9, 0.9)
        f = frame()
        initial = policy.evaluate(f, parking(f), caps(f))
        policy.session.finish(f, False, "TASK_CANCELLED")
        f = frame(2, 0.1)
        result = policy.evaluate(f, parking(f), caps(f))
        self.assertEqual("CANCELLED", result.request.status)
        self.assertEqual(initial.request.intent_id, result.request.intent_id)
        self.assertFalse(result.request.dispatch_allowed)

    def test_dwell_completion_requires_actual_body_inside_latest_boundary(self):
        policy = ParkingPolicy(3.5, 0.9, 0.9)
        f = frame()
        policy.evaluate(f, parking(f), caps(f))
        goal = policy.park_goal
        request = policy.session.advance(f, "PARKED_DWELL", policy._goal(goal, -1, True))
        f = frame(2, 0.1)
        refined = parking(f, boundary=[(7.75,-1.5), (13,-1.5), (13,1.5), (7.75,1.5)])
        actual = GoalPose(goal.x - 0.1, goal.y, goal.body_heading_rad)
        held = policy.evaluate(f, refined, caps(f), feedback(request, f,
            producer="control", status="COMPLETED", actual_standstill_confirmed=True,
            goal_pose_arrived=True, actual_pose=actual, progress=1,
            hold_completed=True, standstill_duration_s=10))
        self.assertEqual("PARKED_DWELL", held.request.stage)
        self.assertTrue(held.hold_required)
        f = frame(3, 0.2)
        f.ego_x=goal.x
        completed = policy.evaluate(f, parking(f), caps(f), feedback(held.request, f,
            producer="control", status="COMPLETED", actual_standstill_confirmed=True,
            goal_pose_arrived=True, actual_pose=goal, progress=1,
            hold_completed=True, standstill_duration_s=10))
        self.assertEqual("EXIT_PREPARE", completed.request.stage)

    def test_wrong_parking_pose_cannot_start_dwell(self):
        policy=ParkingPolicy(3.5,0.9,0.9)
        f=frame()
        policy.evaluate(f,parking(f),caps(f))
        r=policy.session.advance(f,"REVERSE_ENTRY",policy._goal(policy.park_goal,-1,True))
        f=frame(2,0.1)
        result=policy.evaluate(f,parking(f),caps(f),feedback(r,f,producer="control",status="ARRIVED",
             actual_standstill_confirmed=True,goal_pose_arrived=True,actual_pose=GoalPose(10,0,0),progress=1))
        self.assertEqual("ALIGN",result.request.stage)
        self.assertNotEqual("PARKED_DWELL",result.request.stage)

    def test_unexpected_reverse_and_new_obstacle_never_release_protection(self):
        policy=ParkingPolicy(3.5,0.9,0.9)
        f=frame()
        policy.evaluate(f,parking(f),caps(f))
        f=frame(2,0.1,speed=0.5)
        slipped=policy.evaluate(f,parking(f,signed=-0.5),caps(f))
        self.assertEqual("UNEXPECTED_REVERSE_MOTION",slipped.reason)
        self.assertFalse(slipped.request.dispatch_allowed)
        f=frame(3,0.2)
        blocked=policy.evaluate(f,parking(f,objects=[MotionObject(2,0,0,0,0,4,2)]),caps(f))
        self.assertEqual("BLOCKED",blocked.phase)
        self.assertTrue(blocked.hold_required)


if __name__ == "__main__":
    unittest.main()
