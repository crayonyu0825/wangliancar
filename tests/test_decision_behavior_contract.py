"""Synthetic R05/R07 feedback: no public runtime channel or SDK is mocked into existence."""

import copy
import json
import unittest

from members.decision.behaviors.contract import (
    TaskContext, BehaviorFrame, BehaviorGoal, LightIntent, GoalPose, Capabilities,
    ExecutionFeedback, FeedbackValidator,
)
from members.decision.behaviors.session import BehaviorSession


def context(task="task"):
    return TaskContext("case", task, 20, "session")


def frame(index=1, time_s=0.0, ctx=None, speed=0.0, paused=False, usable=True):
    return BehaviorFrame(ctx or context(), index, time_s, time_s + 0.2,
                         ego_speed=speed, paused=paused, usable=usable)


def capabilities(f, actions=("DWELL", "LIGHTS", "PATH")):
    return Capabilities(f.context, actions, f.observed_at_s, f.observed_at_s + 20.0,
                        usable=True)


def feedback(request, f, producer="planning", status="PLANNED", **values):
    values.setdefault("usable", True)
    return ExecutionFeedback(f.context, request.intent_id, request.stage,
                             request.revision, producer, request.source_frame_id,
                             f.frame_id, f.observed_at_s, f.observed_at_s + 0.2,
                             status, **values)


class BehaviorContractTests(unittest.TestCase):
    def start(self, session=None, goal=None):
        session = session or BehaviorSession()
        f = frame()
        session.start(f, "stop:1", "SIGNAL_STOP", "APPROACH",
                      goal or BehaviorGoal(stop_distance_m=10.0), ("DWELL",))
        request = session.tick(f, capabilities(f))
        return session, request

    def test_same_stage_resend_keeps_identity_and_revision(self):
        session, initial = self.start()
        f = frame(2, 0.1)
        session.start(f, "stop:1", "SIGNAL_STOP", "APPROACH",
                      BehaviorGoal(stop_distance_m=9.0, speed_cap_mps=1.0), ("DWELL",))
        repeated = session.tick(f, capabilities(f))
        self.assertEqual((initial.intent_id, initial.stage, initial.revision),
                         (repeated.intent_id, repeated.stage, repeated.revision))
        self.assertEqual(9.0, repeated.goal.stop_distance_m)
        json.dumps(repeated.to_dict(), allow_nan=False)

    def test_independent_or_replaced_policies_cannot_alias_intent_or_feedback(self):
        first, initial = self.start()
        second, other = self.start()
        self.assertNotEqual(initial.intent_id, other.intent_id)
        f = frame(2, 0.1)
        second.tick(f, capabilities(f), feedback(initial, f))
        self.assertIsNone(second.last_feedback)
        second.reset()
        unused, restarted = self.start(second)
        self.assertNotEqual(initial.intent_id, restarted.intent_id)
        self.assertNotEqual(other.intent_id, restarted.intent_id)

    def test_semantic_target_or_direction_change_revises_request(self):
        session, initial = self.start()
        f = frame(2, 0.1)
        session.start(f, "stop:1", "SIGNAL_STOP", "APPROACH",
                      BehaviorGoal(motion_direction=-1, goal_pose=GoalPose(1, 2, 0)), ("DWELL",))
        request = session.tick(f, capabilities(f))
        self.assertEqual(initial.intent_id, request.intent_id)
        self.assertGreater(request.revision, initial.revision)
        old = feedback(initial, f)
        session.tick(f, capabilities(f), old)
        self.assertIsNone(session.last_feedback)

    def test_missing_capability_produces_only_an_inactive_proposal(self):
        session = BehaviorSession()
        f = frame()
        session.start(f, "park", "PARK", "REVERSE_ENTRY",
                      BehaviorGoal(motion_direction=-1), ("PARK",))
        request = session.tick(f)
        self.assertFalse(request.dispatch_allowed)
        self.assertEqual("CAPABILITY_OR_CONTRACT_UNAVAILABLE", request.reason_code)

    def test_forced_same_stage_revision_rejects_old_feedback_and_preserves_retry_budget(self):
        session, initial = self.start()
        session.block(frame(), "RECOVERABLE_EVIDENCE")
        f = frame(2, 0.1)
        self.assertTrue(session.retry_after_evidence(f, "EVIDENCE_RECOVERED"))
        request = session.tick(f, capabilities(f))
        f = frame(3, 0.2)
        revised = session.advance(f, "APPROACH", force_revision=True)
        self.assertEqual(initial.intent_id, revised.intent_id)
        self.assertGreater(revised.revision, request.revision)
        self.assertEqual(request.attempt, revised.attempt)
        self.assertEqual(1, revised.attempt)
        self.assertEqual(f.frame_id, revised.source_frame_id)
        session.tick(f, capabilities(f), feedback(request, f))
        self.assertIsNone(session.last_feedback)
        f = frame(4, 0.3)
        session.tick(f, capabilities(f), feedback(revised, f))
        self.assertEqual("ACCEPTED", session.status)

    def test_feedback_from_previous_source_frame_is_accepted_when_fresh(self):
        session, request = self.start()
        produced = frame(2, 0.1)
        reply = feedback(request, produced)
        now = frame(3, 0.15)
        session.tick(now, capabilities(now), reply)
        self.assertIs(reply, session.last_feedback)
        self.assertEqual("ACCEPTED", session.status)

    def test_transition_records_include_goal_constraints_and_verified_feedback(self):
        session, request = self.start()
        f = frame(2, 0.1)
        session.tick(f, capabilities(f), feedback(request, f))
        event = session.events[-1]
        self.assertEqual("ACCEPTED", event["status"])
        self.assertEqual(request.context.to_dict(), event["task_context"])
        self.assertEqual(10.0, event["goal"]["stop_distance_m"])
        self.assertEqual(["DWELL"], event["required_capabilities"])
        self.assertEqual("PLANNED", event["feedback"]["status"])
        self.assertEqual(f.frame_id, event["feedback"]["produced_frame_id"])
        json.dumps(event, allow_nan=False)

    def test_wrong_context_intent_revision_future_and_expired_feedback_cannot_progress(self):
        malformed_pose = GoalPose(0, 0, 0)
        malformed_pose.x = float("nan")
        mutations = (("actual_pose", malformed_pose), ("context", context("wrong")), ("intent_id", "wrong"),
                     ("revision", 99), ("stage", "EXIT"), ("produced_frame_id", 50),
                     ("produced_at_s", 1.0), ("valid_until_s", 0.0),
                     ("clock_id", "sdk_unknown_units"), ("progress", float("nan")),
                     ("actual_standstill_confirmed", 1))
        for field, value in mutations:
            session, request = self.start()
            now = frame(2, 0.1)
            reply = feedback(request, now, producer="control", status="COMPLETED",
                             actual_standstill_confirmed=True, goal_pose_arrived=True)
            setattr(reply, field, value)
            session.tick(now, capabilities(now), reply)
            self.assertIsNone(session.last_feedback, field)
            self.assertNotEqual("COMPLETED", session.status, field)

    def test_accepted_path_and_control_send_are_not_behavior_completion(self):
        session, request = self.start()
        f = frame(2, 0.1)
        session.tick(f, capabilities(f), feedback(request, f))
        self.assertEqual("APPROACH", session.stage)
        self.assertEqual("ACCEPTED", session.status)
        f = frame(3, 0.2)
        request = session.snapshot(f)
        session.tick(f, capabilities(f), feedback(request, f, producer="control", status="EXECUTING"))
        self.assertEqual("APPROACH", session.stage)
        self.assertEqual("EXECUTING", session.status)

    def test_duplicate_and_out_of_order_feedback_do_not_advance(self):
        session, request = self.start()
        f = frame(2, 0.1)
        reply = feedback(request, f)
        session.tick(f, capabilities(f), reply)
        self.assertTrue(session.feedback_distinct)
        session.tick(f, capabilities(f), reply)
        self.assertFalse(session.feedback_distinct)
        older = feedback(request, frame(1, 0.05))
        session.tick(frame(3, 0.15), capabilities(frame(3, 0.15)), older)
        self.assertIsNone(session.last_feedback)
        self.assertIn("OUT_OF_ORDER", session.reason_code)

    def test_safety_pause_preserves_stage_but_invalidates_old_revision(self):
        session, request = self.start()
        stopped = frame(2, 0.1)
        suspended = session.tick(stopped, capabilities(stopped), safety_override=True)
        self.assertEqual("APPROACH", suspended.stage)
        self.assertFalse(suspended.dispatch_allowed)
        resumed = frame(3, 0.2)
        revised = session.tick(resumed, capabilities(resumed), feedback(request, resumed))
        self.assertEqual(request.intent_id, revised.intent_id)
        self.assertGreater(revised.revision, request.revision)
        self.assertIsNone(session.last_feedback)

    def test_pause_repeated_frames_and_gap_do_not_consume_stage_time(self):
        session, request = self.start(BehaviorSession(acknowledgement_timeout_s=0.3))
        for time_s in (0.1, 0.2, 0.3):
            f = frame(1, time_s, paused=True)
            session.tick(f, capabilities(f))
        f = frame(2, 1.0)
        request = session.tick(f, capabilities(f))
        self.assertFalse(request.dispatch_allowed)
        f = frame(3, 1.1)
        request = session.tick(f, capabilities(f))
        self.assertTrue(request.dispatch_allowed)
        self.assertEqual("REQUESTED", request.status)

    def test_retries_are_bounded_and_old_rejection_cannot_repeat_them(self):
        session, request = self.start(BehaviorSession(max_retries=1, retry_delay_s=0.1))
        f = frame(2, 0.1)
        rejection = feedback(request, f, status="REJECTED", retryable=True)
        first = session.tick(f, capabilities(f), rejection)
        self.assertEqual("RETRY_WAIT", first.status)
        session.tick(f, capabilities(f), rejection)
        self.assertEqual(1, session.attempt)
        f = frame(3, 0.2)
        second = session.tick(f, capabilities(f))
        self.assertGreater(second.revision, request.revision)
        f = frame(4, 0.3)
        failed = session.tick(f, capabilities(f), feedback(second, f, status="REJECTED", retryable=True))
        self.assertEqual("BLOCKED", failed.status)
        self.assertFalse(failed.dispatch_allowed)

    def test_repeated_planning_ack_cannot_hide_lack_of_actual_progress(self):
        session, request = self.start(BehaviorSession(progress_timeout_s=0.2))
        for index in (2, 3, 4, 5):
            f = frame(index, (index - 1) * 0.1)
            request = session.tick(f, capabilities(f), feedback(session.snapshot(f), f))
        self.assertEqual("BLOCKED", request.status)
        self.assertEqual("EXECUTION_PROGRESS_TIMEOUT", request.reason_code)

    def test_nonretryable_rejection_cannot_be_erased_by_capability_or_safety_toggle(self):
        session,request=self.start()
        f=frame(2,0.1)
        rejected=session.tick(f,capabilities(f),feedback(request,f,status="REJECTED",retryable=False,
                                                       reason_code="NO_SAFE_PATH"))
        self.assertEqual("BLOCKED",rejected.status)
        f=frame(3,0.2)
        session.tick(f,None,safety_override=True)
        f=frame(4,0.3)
        still=session.tick(f,capabilities(f))
        self.assertEqual("BLOCKED",still.status)
        self.assertEqual("NO_SAFE_PATH",still.reason_code)
        self.assertFalse(still.dispatch_allowed)

    def test_terminal_rejection_cannot_be_erased_by_goal_stage_or_invalid_start(self):
        session, request = self.start()
        f = frame(2, 0.1)
        session.tick(f, capabilities(f), feedback(request, f, status="REJECTED", retryable=False,
                                                reason_code="NO_SAFE_PATH"))
        session.start(frame(3, 0.2, usable=False), "stop:1", "SIGNAL_STOP", "EXIT", BehaviorGoal())
        session.start(frame(4, 0.3), "stop:1", "SIGNAL_STOP", "APPROACH",
                      BehaviorGoal(goal_pose=GoalPose(5, 0, 0)))
        session.advance(frame(4, 0.3), "EXIT", BehaviorGoal())
        session.block(frame(4, 0.3), "NEW_OBSTRUCTION")
        still = session.tick(frame(5, 0.4), capabilities(frame(5, 0.4)))
        self.assertEqual("BLOCKED", still.status)
        self.assertEqual("NO_SAFE_PATH", still.reason_code)
        self.assertEqual(request.stage, still.stage)
        self.assertFalse(still.dispatch_allowed)

    def test_completed_or_cancelled_intent_cannot_restart_on_bad_frame_and_recovery(self):
        for completed in (True, False):
            session, request = self.start()
            terminal = session.finish(frame(2, 0.1), completed)
            bad = frame(3, 0.2, usable=False)
            session.tick(bad)
            session.start(bad, "stop:1", "SIGNAL_STOP", "EXIT", BehaviorGoal())
            good = frame(4, 0.3)
            session.advance(good, "EXIT", BehaviorGoal())
            session.block(good, "NEW_OBSTRUCTION")
            still = session.tick(good, capabilities(good))
            self.assertEqual(terminal.status, still.status)
            self.assertFalse(still.dispatch_allowed)
            self.assertEqual(request.intent_id, still.intent_id)

    def test_invalid_context_or_paused_start_cannot_replace_active_intent(self):
        session, request = self.start()
        bad = frame(2,0.1,context("bad"),usable=False)
        preserved = session.start(bad,"other","PARK","EXIT",BehaviorGoal(motion_direction=-1))
        self.assertEqual(request.intent_id,preserved.intent_id)
        self.assertEqual(request.context.key(),preserved.context.key())
        self.assertFalse(preserved.dispatch_allowed)
        with self.assertRaises(ValueError):
            session.advance(frame(3,0.2,paused=True),"EXIT")

    def test_task_change_cancel_clears_lights_and_retains_old_context(self):
        session, request = self.start(goal=BehaviorGoal(light_intent=LightIntent(left_signal=True)))
        changed = frame(2, 0.1, context("new"))
        cancelled = session.tick(changed, capabilities(changed))
        self.assertEqual("CANCELLED", cancelled.status)
        self.assertFalse(cancelled.dispatch_allowed)
        self.assertFalse(cancelled.goal.light_intent.left_signal)
        self.assertEqual(request.context.key(), cancelled.context.key())


if __name__ == "__main__":
    unittest.main()
