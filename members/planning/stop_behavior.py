"""P07: validated signal stop+dwell requests reuse the forward planner."""
import copy
import time

from core.behavior_channel import read_channel,capabilities_dict,feedback_dict
from core.behavior_contract import ExecutionFeedback
from core.interfaces import Trajectory,DecisionMode
from members.planning.behavior_contract import assess_behavior_request
from members.planning.lane_planner import build_trajectory


class StopBehaviorPlanner(object):
    def __init__(self,clock=None):
        self.clock = clock or time.monotonic
        self.context,self.revisions = None,{}

    def plan(self,p,decision,settings):
        now = self.clock()
        request = decision.behavior_request
        output = Trajectory().bind(decision)
        context = None
        try:
            context,caps,unused = read_channel(p,now)
            assessment = assess_behavior_request(request,context.to_dict(),p.frame_id,now,
                p.valid_until,"process_monotonic",decision.behavior_active_identity,
                capabilities=capabilities_dict(caps),supported_actions=("SIGNAL_STOP",),
                current_lane_id=p.lane.lane_id,frame_usable=p.valid,frame_paused=False,runtime_stop=True)
            if not assessment["eligible"]:
                raise ValueError(assessment["reason_code"])
            if self.context!=context.key():
                self.context,self.revisions = context.key(),{}
            previous = self.revisions.get(request["intent_id"])
            revision = (request["revision"],request["stage"],request["source_frame_id"],request["issued_at_s"])
            if previous is not None and (revision[0]<previous[0] or (revision[0]==previous[0] and revision!=previous)):
                raise ValueError("REQUEST_REVISION_REGRESSED_OR_CHANGED")
            if decision.mode==DecisionMode.EMERGENCY_BRAKE:
                raise ValueError("SAFETY_OVERRIDE")
            if decision.stop_distance>=0 and decision.stop_distance+1e-6<request["stop_distance_m"]:
                # An obstacle stop before the lamp cannot discharge its dwell.
                output = build_trajectory(p,decision,settings)
                raise ValueError("EARLIER_SAFETY_STOP")
            compiled = copy.deepcopy(decision)
            # The existing STOP mode prohibits any acceleration. A finite
            # stop with a positive approach cap uses the existing bounded
            # forward mode, so a replan can still reach its stop from rest.
            compiled.mode = (DecisionMode.KEEP_LANE if request["stop_distance_m"]>1e-6
                             and request["speed_cap_mps"]>0 else DecisionMode.STOP)
            compiled.stop_distance = request["stop_distance_m"]
            compiled.target_speed = min(decision.target_speed,request["speed_cap_mps"])
            compiled.precision_stop = True
            output = build_trajectory(p,compiled,settings)
            if not output.valid or output.emergency_stop or not output.stop_required:
                raise ValueError("STOP_PATH_UNAVAILABLE_OR_PROTECTED")
            if output.stop_distance+1e-6<request["stop_distance_m"]:
                # The planner can independently find a nearer obstacle,
                # curvature limit or reference end than decision observed.
                output.behavior_feedback = self._reply(context,request,p,now,"REJECTED","EARLIER_PLANNING_STOP")
                return output
            output.hold_duration_s = request["minimum_standstill_duration_s"]
            output.stop_obligation_id = request["stop_obligation_id"]
            output.behavior_identity = {key:request[key] for key in ("intent_id","stage","revision")}
            self.revisions[request["intent_id"]] = revision
            if len(self.revisions)>128:
                self.revisions = {request["intent_id"]:revision}
            output.behavior_feedback = self._reply(context,request,p,now,"PLANNED","STOP_DWELL_PATH_READY")
            return output
        except (ValueError,TypeError,AttributeError,KeyError) as exc:
            reason = str(exc)
            if reason!="EARLIER_SAFETY_STOP":
                output = Trajectory().bind(decision)
                output.errors.append("BEHAVIOR_REQUEST:"+reason)
                output.reason = "BEHAVIOR_REQUEST:"+reason
            if context is not None and isinstance(request,dict):
                try:
                    output.behavior_feedback = self._reply(context,request,p,now,"REJECTED",reason)
                except (KeyError,TypeError,ValueError):
                    pass
            return output

    @staticmethod
    def _reply(context,request,p,now,status,reason):
        return feedback_dict(ExecutionFeedback(context,request["intent_id"],request["stage"],request["revision"],
            "planning",p.frame_id,p.frame_id,now,min(p.valid_until,request["valid_until_s"]),status,
            usable=True,reason_code=reason,retryable=reason in ("EARLIER_SAFETY_STOP","EARLIER_PLANNING_STOP","STOP_PATH_UNAVAILABLE_OR_PROTECTED")))
