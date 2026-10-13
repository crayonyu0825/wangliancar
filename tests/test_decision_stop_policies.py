"""D01 request policies using explicit synthetic R05 execution evidence."""

import unittest

from members.decision.behaviors.contract import Capabilities
from members.decision.behaviors.stop_policies import (
    SignalDwellPolicy, SignalStopObservation, BlockedRoadPolicy, BlockingObservation,
)
from tests.test_decision_behavior_contract import frame, feedback, context


def caps(f):
    return Capabilities(f.context, ("PATH_STOP", "DWELL", "LIGHTS", "FEEDBACK"),
                        f.observed_at_s, f.observed_at_s + 10.0, usable=True)


def signal(red=True, distance=0.0, identity="line:1"):
    return SignalStopObservation(identity, distance, red, speed_cap_mps=3.0)


class StopPolicyTests(unittest.TestCase):
    def test_signal_twenty_request_exceeds_five_seconds_and_ten_has_no_dwell(self):
        policy = SignalDwellPolicy()
        result = policy.evaluate(frame(), signal(distance=20.0))
        self.assertGreater(result.request.goal.minimum_standstill_duration_s, 5.0)
        self.assertFalse(result.request.dispatch_allowed)
        self.assertFalse(result.request.goal.light_intent.hazard_signal)
        ten = SignalDwellPolicy().evaluate(frame(), signal(), dwell_required=False)
        self.assertIsNone(ten.request)
        self.assertFalse(ten.hold_required)

    def test_time_before_actual_arrival_never_completes_dwell(self):
        policy = SignalDwellPolicy()
        f = frame()
        request = policy.evaluate(f, signal(distance=20), caps(f)).request
        f = frame(2, 0.1)
        reply = feedback(request, f, producer="control", status="COMPLETED",
                         hold_completed=True, standstill_duration_s=100.0)
        result = policy.evaluate(f, signal(distance=20), caps(f), reply)
        self.assertEqual("APPROACH", result.phase)
        self.assertNotEqual("COMPLETED", result.request.status)

    def test_early_green_cannot_shorten_actual_stop_obligation(self):
        policy = SignalDwellPolicy()
        f = frame()
        request = policy.evaluate(f, signal(), caps(f)).request
        f = frame(2, 0.1)
        arrival = feedback(request, f, producer="control", status="ARRIVED",
                           actual_standstill_confirmed=True, goal_pose_arrived=True)
        arrived = policy.evaluate(f, signal(), caps(f), arrival)
        self.assertTrue(arrived.hold_required)
        identity = (arrived.request.intent_id, arrived.request.revision)
        f = frame(3, 0.2)
        short = feedback(arrived.request, f, producer="control", status="COMPLETED",
                         actual_standstill_confirmed=True, goal_pose_arrived=True,
                         standstill_duration_s=5.0, hold_completed=True)
        early = policy.evaluate(f, signal(False), caps(f), short)
        self.assertTrue(early.hold_required)
        self.assertNotEqual("COMPLETE", early.phase)
        self.assertEqual(identity, (early.request.intent_id, early.request.revision))
        f = frame(4, 0.3)
        complete = feedback(early.request, f, producer="control", status="COMPLETED",
                            actual_standstill_confirmed=True, goal_pose_arrived=True,
                            standstill_duration_s=5.1, hold_completed=True)
        done = policy.evaluate(f, signal(False), caps(f), complete)
        self.assertFalse(done.hold_required)
        self.assertEqual("COMPLETE", done.phase)
        self.assertFalse(done.request.dispatch_allowed)

    def test_completed_same_stop_does_not_restart_and_red_still_holds(self):
        policy = SignalDwellPolicy()
        f = frame()
        request = policy.evaluate(f, signal(), caps(f)).request
        f = frame(2, 0.1)
        complete = feedback(request, f, producer="control", status="COMPLETED",
                            actual_standstill_confirmed=True, goal_pose_arrived=True,
                            standstill_duration_s=6.0, hold_completed=True)
        done = policy.evaluate(f, signal(), caps(f), complete)
        self.assertTrue(done.hold_required)
        f = frame(3, 0.2)
        repeated = policy.evaluate(f, signal(False), caps(f))
        self.assertEqual(done.request.intent_id, repeated.request.intent_id)
        self.assertEqual("COMPLETED", repeated.request.status)
        self.assertFalse(repeated.hold_required)

    def test_drift_or_wrong_revision_cannot_finish_hold(self):
        policy = SignalDwellPolicy()
        f = frame()
        request = policy.evaluate(f, signal(), caps(f)).request
        f = frame(2, 0.1)
        arrived = policy.evaluate(f, signal(), caps(f), feedback(
            request, f, producer="control", status="ARRIVED",
            actual_standstill_confirmed=True, goal_pose_arrived=True))
        f = frame(3, 0.2, speed=0.5)
        drifting = policy.evaluate(f, signal(False), caps(f), feedback(
            arrived.request, f, producer="control", status="COMPLETED",
            goal_pose_arrived=False, actual_standstill_confirmed=False,
            hold_completed=True, standstill_duration_s=10.0))
        self.assertTrue(drifting.hold_required)
        self.assertEqual("DWELL_REVALIDATE", drifting.phase)
        f = frame(4, 0.3)
        forged = feedback(drifting.request, f, producer="control", status="COMPLETED",
                          actual_standstill_confirmed=True, goal_pose_arrived=True,
                          hold_completed=True, standstill_duration_s=6.0)
        forged.revision += 1
        rejected = policy.evaluate(f, signal(False), caps(f), forged)
        self.assertTrue(rejected.hold_required)
        self.assertNotEqual("COMPLETE", rejected.phase)

    def test_repeated_gps_and_pause_cannot_complete_dwell(self):
        policy = SignalDwellPolicy()
        f = frame()
        request = policy.evaluate(f, signal(), caps(f)).request
        f = frame(2, 0.1)
        arrival = feedback(request, f, producer="control", status="ARRIVED",
                           actual_standstill_confirmed=True, goal_pose_arrived=True)
        arrived = policy.evaluate(f, signal(), caps(f), arrival)
        duplicate_frame = frame(2, 0.15)
        fake = feedback(arrived.request, duplicate_frame, producer="control", status="COMPLETED",
                        actual_standstill_confirmed=True, goal_pose_arrived=True,
                        hold_completed=True, standstill_duration_s=8.0)
        repeated = policy.evaluate(duplicate_frame, signal(False), caps(duplicate_frame), fake)
        self.assertTrue(repeated.hold_required)
        paused = frame(3, 0.2, paused=True)
        same = policy.evaluate(paused, signal(False), caps(paused), feedback(
            repeated.request, paused, producer="control", status="COMPLETED",
            actual_standstill_confirmed=True, goal_pose_arrived=True,
            hold_completed=True, standstill_duration_s=8.0))
        self.assertTrue(same.hold_required)
        self.assertFalse(same.request.dispatch_allowed)

    def test_same_session_rollback_keeps_completed_ledger_and_new_task_resets_it(self):
        policy = SignalDwellPolicy()
        f = frame(5, 0.0)
        r = policy.evaluate(f, signal(), caps(f)).request
        f = frame(6, 0.1)
        policy.evaluate(f, signal(), caps(f), feedback(r, f, producer="control", status="COMPLETED",
                        actual_standstill_confirmed=True, goal_pose_arrived=True,
                        hold_completed=True, standstill_duration_s=6.0))
        rolled = frame(1, 0.2)
        reset = policy.evaluate(rolled, signal(False), caps(rolled))
        self.assertEqual("RECOVER", reset.phase)
        fresh = frame(2, 0.3)
        new = policy.evaluate(fresh, signal(False), caps(fresh))
        self.assertEqual("COMPLETED", new.request.status)
        changed = frame(3, 0.4, context("new"))
        another = policy.evaluate(changed, signal(), caps(changed))
        self.assertNotEqual(new.request.intent_id, another.request.intent_id)

    def test_blocked_route_requests_hazards_but_red_and_follow_do_not(self):
        observation = BlockingObservation("object:1", 10.0, True, "unavailable", speed_cap_mps=2.0)
        proposed = BlockedRoadPolicy().evaluate(frame(), observation)
        self.assertTrue(proposed.request.goal.light_intent.hazard_signal)
        self.assertFalse(proposed.request.dispatch_allowed)
        for flags in ({"ordinary_follow": True}, {"signal_only": True}):
            policy = BlockedRoadPolicy()
            excluded = policy.evaluate(frame(), BlockingObservation("x", 0.0, True, **flags))
            self.assertIsNone(excluded.request)

    def test_blockage_release_requires_new_clear_frames_and_clears_lights(self):
        policy = BlockedRoadPolicy()
        f = frame()
        request = policy.evaluate(f, BlockingObservation("b", 0.0, True), caps(f)).request
        clear = BlockingObservation("b", 0.0, False, verified_clear=True)
        f = frame(2, 0.1)
        first = policy.evaluate(f, clear, caps(f))
        self.assertTrue(first.request.goal.light_intent.hazard_signal)
        policy.evaluate(f, clear, caps(f))
        f = frame(3, 0.2)
        done = policy.evaluate(f, clear, caps(f))
        self.assertEqual("COMPLETE", done.phase)
        self.assertFalse(done.request.goal.light_intent.hazard_signal)
        self.assertFalse(done.hold_required)

    def test_blockage_frame_rollback_keeps_unfinished_intent_and_requires_revalidation(self):
        policy = BlockedRoadPolicy()
        f = frame(5, .5)
        observation = BlockingObservation('b', 0., True)
        first = policy.evaluate(f, observation, caps(f)).request
        f = frame(1, .6)
        rolled = policy.evaluate(f, None, caps(f))
        self.assertEqual(first.intent_id, rolled.request.intent_id)
        self.assertFalse(rolled.request.dispatch_allowed)
        self.assertTrue(rolled.request.goal.light_intent.hazard_signal)
        f = frame(2, .7)
        restored = policy.evaluate(f, observation, caps(f)).request
        self.assertEqual(first.intent_id, restored.intent_id)
        self.assertGreater(restored.revision, first.revision)
        self.assertTrue(restored.dispatch_allowed)


if __name__ == "__main__":
    unittest.main()
