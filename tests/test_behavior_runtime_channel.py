"""Fixed entries in a synthetic bicycle loop; never connects/sends to SimOne."""
import copy
import unittest
from unittest.mock import patch

from core.behavior_channel import BehaviorTransport,read_channel,feedback_dict
from core.behavior_contract import ExecutionFeedback,TaskContext
from core.interfaces import DecisionTarget,Trajectory,ControlOut
from core.serialization import to_dict
from core.validation import validate_output
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from members.planning.lane_planner import PlannerSettings
from members.planning.stop_behavior import StopBehaviorPlanner
from members.control.controller import ControlEngine
from members import decision_stub,planning_stub,control_stub
from tests.test_decision import perception,add_red_light,add_target
from tests.test_control_maneuvers import BidirectionalPlant
from tests.control_benchmark import vehicle


class BehaviorRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        clock = lambda:self.now
        self.transport = BehaviorTransport(("PATH_STOP","DWELL","FEEDBACK"),clock)
        self.engine = DecisionEngine(DecisionSettings(front_offset_m=3.5,half_width_m=.9),clock)
        self.planner = StopBehaviorPlanner(clock)
        self.settings = PlannerSettings(front_offset_m=3.5,half_width_m=.9,rear_offset_m=1.,
            wheelbase_m=2.9,front_steer_max_rad=.5,max_lateral_acceleration_mps2=1.5)
        self.control = ControlEngine(vehicle(),clock=clock)
        for target,value in (("time.monotonic",clock),("members.decision_stub._ENGINE",self.engine),
                             ("members.planning_stub._BEHAVIOR_PLANNER",self.planner),
                             ("members.planning_stub._settings_with_control_geometry",lambda:self.settings),
                             ("members.control_stub._engine",self.control)):
            manager = patch(target,value)
            manager.start()
            self.addCleanup(manager.stop)

    def frame(self,number=1,x=0.,speed=0.,green=False):
        p = perception(speed=speed,frame_id=number,ttl=.2)
        p.case_id,p.task_id,p.scene_id = "synthetic","trial",20
        p.ego.x,p.ego.frame_id,p.ego.age_ms = x,number,0
        add_red_light(p,max(0.,30.-x))
        p.traffic.signal_id = 42
        if green: p.traffic.signal_state = "GREEN"
        self.transport.prepare(p)
        return p

    def pipeline(self,p,receipt=None):
        decision = decision_stub.decide(p)
        validate_output(decision,DecisionTarget,p)
        trajectory = planning_stub.plan(p,decision)
        self.assertTrue(trajectory.valid,(trajectory.errors,trajectory.reason,decision.reason,decision.behavior_request))
        validate_output(trajectory,Trajectory,decision)
        control = control_stub.compute_control(p,trajectory)
        validate_output(control,ControlOut,trajectory)
        self.transport.observe(p,decision,trajectory,control,False,receipt)
        return decision,trajectory,control

    def test_early_green_does_not_skip_actual_stop_and_continuous_dwell(self):
        plant = BidirectionalPlant(speed=2.,lag=.2)
        arrived_at,departed_at,identity = None,None,None
        for frame in range(1,601):
            p = self.frame(frame,plant.x,plant.speed,green=frame>=5)
            p.ego.y,p.ego.heading,p.ego.vy,p.ego.yaw_rate = plant.y,plant.heading,0.,plant.yaw_rate
            decision,trajectory,control = self.pipeline(p)
            if decision.behavior_request is not None:
                request = decision.behavior_request
                identity = identity or request["intent_id"]
                self.assertEqual(identity,request["intent_id"])
            observed = control.execution_observation
            if observed and observed["goal_pose_arrived"] and arrived_at is None:
                arrived_at = self.now
                self.assertLess(abs(plant.x-26.2),.2)
            if arrived_at is not None and control.throttle>0:
                departed_at = self.now
                break
            plant.step(control,.05,vehicle())
            self.now += .05
        self.assertIsNotNone(arrived_at,(plant.x,plant.speed,decision.reason,trajectory.reason,control.diagnostics,
                                       self.engine.behavior_diagnostics()))
        self.assertIsNotNone(departed_at)
        self.assertGreaterEqual(departed_at-arrived_at,5.1)
        self.assertEqual("COMPLETE",self.engine.behavior_diagnostics()["policies"][0]["phase"])

    def test_complete_dwell_still_waits_for_red_then_green_releases(self):
        for frame in range(1,113):
            p = self.frame(frame,x=26.2)
            decision,trajectory,control = self.pipeline(p)
            self.assertEqual(0.,control.throttle)
            self.now += .05
        self.assertEqual("WAIT_SIGNAL",self.engine.behavior_diagnostics()["policies"][0]["phase"])
        for frame in range(113,139):
            decision,trajectory,control = self.pipeline(self.frame(frame,x=26.2,green=True))
            self.now += .05
        self.assertGreater(control.throttle,0.)

    def test_capability_default_and_expired_channel_never_dispatch(self):
        p = self.frame()
        p.behavior_channel["capabilities"]["actions"] = []
        self.assertIsNone(self.engine.run(p).behavior_request)
        self.now += .25
        with self.assertRaises(ValueError): read_channel(p,self.now)

    def test_task_and_frame_rollback_preserve_scope_until_captain_reset(self):
        p = self.frame(10,x=26.2)
        self.pipeline(p)
        old = p.behavior_channel["context"]
        self.now += .05
        rolled = self.frame(1,x=26.2)
        self.assertFalse(rolled.behavior_channel)
        self.assertEqual(old,self.transport.context.to_dict())
        rolled.task_id = "different"
        self.transport.prepare(rolled)
        self.assertFalse(rolled.behavior_channel)
        self.assertEqual(old,self.transport.context.to_dict())
        self.transport.reset_session()
        self.transport.prepare(rolled)
        self.assertNotEqual(old,rolled.behavior_channel["context"])
        self.assertFalse(rolled.behavior_channel["feedbacks"])

    def test_lost_or_disabled_channel_cannot_erase_unfinished_signal_obligation(self):
        for change in ("missing","disabled","bad_context","diagnostic_error"):
            self.setUp()
            first = self.frame(speed=2.)
            self.pipeline(first)
            self.now += .05
            p = self.frame(2,speed=2.,green=True)
            if change=="missing": p.behavior_channel = {}
            if change=="disabled": p.behavior_channel["capabilities"]["usable"] = False
            if change=="bad_context": p.behavior_channel["context"]["task_id"] = "untrusted"
            if change=="diagnostic_error":
                with patch.object(self.engine._behavior,"observe_legacy",side_effect=ValueError("bad observation")):
                    output = self.engine.run(p)
            else:
                output = self.engine.run(p)
            self.assertEqual("EMERGENCY_BRAKE",output.mode,change)
            self.assertEqual(0.,output.target_speed,change)
            self.assertIsNone(output.behavior_request)

    def test_untrusted_new_task_does_not_reset_transport_context(self):
        first = self.frame()
        context = copy.deepcopy(first.behavior_channel["context"])
        bad = copy.deepcopy(first)
        bad.task_id,bad.valid = "untrusted",False
        self.transport.prepare(bad)
        self.assertFalse(bad.behavior_channel)
        self.now += .05
        self.assertEqual(context,self.frame(2).behavior_channel["context"])

    def test_receipt_success_and_planned_result_cannot_claim_arrival(self):
        p = self.frame(x=0.,speed=2.)
        decision,trajectory,control = self.pipeline(p,dict(attempted=True,ok=True))
        replies = self.transport.feedbacks
        self.assertTrue(replies)
        self.assertTrue(all(not r["goal_pose_arrived"] and not r["hold_completed"] for r in replies))
        self.assertEqual("PLANNED",replies[0]["status"])
        saved = to_dict(p)
        self.assertEqual("behavior-channel-v1",saved["behavior_channel"]["contract_version"])

    def test_send_failure_produces_failed_feedback_without_completion(self):
        p = self.frame(x=26.2)
        self.pipeline(p,dict(attempted=True,ok=False))
        reply = self.transport.feedbacks[-1]
        self.assertEqual("FAILED",reply["status"])
        self.assertFalse(reply["hold_completed"])
        self.assertFalse(reply["actual_standstill_confirmed"])

    def test_future_feedback_cannot_finish_stop(self):
        p = self.frame(x=26.2)
        decision,trajectory,control = self.pipeline(p)
        request = decision.behavior_request
        context = TaskContext(**request["task_context"])
        false = ExecutionFeedback(context,request["intent_id"],request["stage"],request["revision"],
            "runtime",99,99,self.now+1,self.now+2,"COMPLETED",usable=True,
            actual_standstill_confirmed=True,goal_pose_arrived=True,standstill_duration_s=100.,hold_completed=True)
        self.transport.feedbacks = [feedback_dict(false)]
        self.now += .05
        later = self.engine.run(self.frame(2,x=26.2,green=True))
        self.assertIsNotNone(later.behavior_request)
        self.assertNotEqual("COMPLETED",later.behavior_request["status"])

    def test_earlier_obstacle_stop_does_not_start_signal_dwell(self):
        p = self.frame(speed=2.)
        decision = self.engine.run(p)
        decision.stop_distance = 5.
        decision.target_speed = 2.
        trajectory = self.planner.plan(p,decision,self.settings)
        self.assertTrue(trajectory.valid,trajectory.errors)
        self.assertEqual(0.,trajectory.hold_duration_s)
        self.assertIsNone(trajectory.behavior_identity)
        self.assertEqual("EARLIER_SAFETY_STOP",trajectory.behavior_feedback["reason_code"])

    def test_planner_reference_end_cannot_discharge_distant_signal_obligation(self):
        p = self.frame(speed=2.)
        decision = self.engine.run(p)
        p.lane.center_line = [(0.,0.),(10.,0.)]
        trajectory = self.planner.plan(p,decision,self.settings)
        self.assertTrue(trajectory.valid,trajectory.errors)
        self.assertEqual(10.,trajectory.stop_distance)
        self.assertEqual(0.,trajectory.hold_duration_s)
        self.assertIsNone(trajectory.behavior_identity)
        self.assertEqual("EARLIER_PLANNING_STOP",trajectory.behavior_feedback["reason_code"])

    def test_revision_regression_expiry_and_unsupported_action_fail_closed(self):
        for change in ("revision","expiry","maneuver"):
            p = self.frame(speed=2.)
            decision = self.engine.run(p)
            good = self.planner.plan(p,decision,self.settings)
            self.assertTrue(good.valid,good.errors)
            bad = copy.deepcopy(decision)
            if change=="revision": bad.behavior_request["revision"] = 0
            if change=="expiry": bad.behavior_request["valid_until_s"] = self.now
            if change=="maneuver": bad.behavior_request["maneuver"] = "PARK_REVERSE"
            self.assertFalse(self.planner.plan(p,bad,self.settings).valid,change)

    def test_changed_unfinished_obligation_is_rejected_and_larger_dwell_extends(self):
        p = self.frame(x=26.2)
        decision,trajectory,control = self.pipeline(p)
        self.now += .05
        second = self.frame(2,x=26.2)
        extended = copy.deepcopy(trajectory).bind(second)
        extended.hold_duration_s = 10.
        output = self.control.compute(second,extended)
        self.assertTrue(output.valid,output.errors)
        self.assertEqual(10.,self.control.standstill.duration)
        self.now += .05
        changed = copy.deepcopy(extended).bind(self.frame(3,x=26.2))
        changed.stop_obligation_id = "different-stop"
        self.assertFalse(self.control.compute(self.frame(3,x=26.2),changed).valid)


if __name__=="__main__": unittest.main()
