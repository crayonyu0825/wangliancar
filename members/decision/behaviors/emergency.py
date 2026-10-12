"""D08 immediate brake fallback and validated planning-candidate selection."""

from members.decision.behaviors.contract import BehaviorGoal, ExecutionFeedback, finite, require
from members.decision.behaviors.observations import Evidence
from members.decision.behaviors.session import BehaviorSession
from members.decision.behaviors.stop_policies import PolicyResult


class EmergencyCandidate(object):
    def __init__(self, identifier, action, feedback, feasible, risk_cost, rule_cost,
                 goal=None, side_coverage_verified=False, crossing_allowed=False):
        require(isinstance(identifier, str) and action in ("BRAKE", "LEFT", "RIGHT", "BRAKE_LEFT", "BRAKE_RIGHT"),
                "invalid emergency action")
        require(isinstance(feedback, ExecutionFeedback) and (feasible is None or type(feasible) is bool), "invalid candidate feasibility")
        require(finite(risk_cost) and risk_cost >= 0 and finite(rule_cost) and rule_cost >= 0,
                "invalid candidate risk/cost")
        require(goal is None or isinstance(goal, BehaviorGoal), "invalid emergency candidate goal")
        require(type(side_coverage_verified) is bool and type(crossing_allowed) is bool, "invalid side proof")
        self.identifier, self.action, self.feedback, self.feasible = identifier, action, feedback, feasible
        self.risk_cost, self.rule_cost, self.goal = risk_cost, rule_cost, goal or BehaviorGoal()
        self.side_coverage_verified, self.crossing_allowed = side_coverage_verified, crossing_allowed


class EmergencyResult(PolicyResult):
    def __init__(self, request=None, reason="", phase="BRAKE_FALLBACK", selected=None):
        super(EmergencyResult, self).__init__(request, True, reason, phase)
        self.immediate_brake_required = True
        self.selected = selected


class EmergencyPolicy(object):
    def __init__(self, clear_frames=2, session=None):
        require(type(clear_frames) is int and clear_frames > 0, "invalid emergency clear frames")
        self.clear_frames, self.session = clear_frames, session or BehaviorSession()
        self.reset()

    def reset(self):
        self.session.reset()
        self._context, self._danger_id, self.selected, self._clear, self._last_frame = None, None, None, 0, None

    def evaluate(self, frame, evidence, danger_id, dangerous, candidates=(), capabilities=None,
                 execution_feedback=None, verified_task_route_available=False):
        require(type(dangerous) is bool and isinstance(danger_id, str), "invalid emergency fact")
        require(all(isinstance(c, EmergencyCandidate) for c in candidates), "invalid emergency candidates")
        if not frame.current(frame.observed_at_s) or not isinstance(evidence, Evidence) or not evidence.verified(frame, allowed_source_kinds=("sensor", "verified_fusion", "synthetic", "task")):
            request = self.session.tick(frame, capabilities, safety_override=True) if self.session.intent_id else None
            return EmergencyResult(request, "EMERGENCY_OBSERVATION_UNKNOWN")
        if self._context != frame.context.key():
            self.reset()
            self._context = frame.context.key()
        distinct = self._last_frame is None or frame.frame_id > self._last_frame
        self._last_frame = frame.frame_id
        if not dangerous and (self.session.intent_id is None or self.session.status == "COMPLETED"):
            result = EmergencyResult(self.session.snapshot(frame), "NO_ACTIVE_EMERGENCY", "IDLE" if self.session.intent_id is None else "COMPLETE")
            result.immediate_brake_required, result.hold_required = False, False
            return result
        if self.session.intent_id is None or self.session.status in ("COMPLETED", "CANCELLED"):
            self.selected = None
            self.session.start(frame, "emergency:" + danger_id, "EMERGENCY", "SELECT",
                               BehaviorGoal(), ("EMERGENCY_CANDIDATES", "PATH", "FEEDBACK"))
            self._danger_id = danger_id
        elif dangerous and danger_id != self._danger_id:
            self._danger_id, self.selected = danger_id, None
            # A new danger invalidates candidates even while still selecting.
            self.session.advance(frame, "SELECT", BehaviorGoal(), reason="NEW_DANGER_REPLAN",
                                 force_revision=True)
        fallback=False
        if (self.selected is not None and self.session.stage!='BRAKE_FALLBACK'
                and execution_feedback is not None and execution_feedback.producer=='planning'
                and execution_feedback.status=='REJECTED'
                and execution_feedback.candidate_id==self.selected.identifier
                and execution_feedback.planning_metrics.get('fallback_brake_required') is True):
            valid,unused,unused2=self.session.validator.check(execution_feedback,self.session.snapshot(frame),frame)
            fallback=(valid and execution_feedback.producer=='planning' and execution_feedback.status=='REJECTED'
                and execution_feedback.candidate_id==self.selected.identifier
                and execution_feedback.planning_metrics.get('fallback_brake_required') is True)
        if fallback:
            self.session.advance(frame,'BRAKE_FALLBACK',self.selected.goal,
                reason='PRIMARY_MOTION_REJECTED_AUTHORIZE_BRAKING_FALLBACK')
            request=self.session.tick(frame,capabilities)
        else:
            request = self.session.tick(frame, capabilities, execution_feedback)
        if dangerous:
            self._clear = 0
            if not request.dispatch_allowed:
                return EmergencyResult(request, "BRAKE_NOW_CANDIDATE_CHANNEL_UNAVAILABLE")
            if self.session.stage == "SELECT":
                feasible = []
                for candidate in candidates:
                    valid, _, _ = self.session.validator.check(candidate.feedback, request, frame)
                    if (not valid or candidate.feedback.producer != "planning"
                            or candidate.feedback.status != "PLANNED" or candidate.feasible is not True
                            or candidate.feedback.candidate_id != candidate.identifier):
                        continue
                    sideways = candidate.action != "BRAKE"
                    if sideways and not (candidate.side_coverage_verified and candidate.crossing_allowed):
                        continue
                    feasible.append(candidate)
                if not feasible:
                    return EmergencyResult(request, "ALL_CANDIDATES_UNKNOWN_OR_INFEASIBLE_BRAKE_FALLBACK")
                self.selected = min(feasible, key=lambda c: (c.risk_cost, c.rule_cost, c.action, c.identifier))
                request = self.session.advance(frame, "EXECUTE_" + self.selected.action,
                                               self.selected.goal, reason="VALIDATED_EMERGENCY_CANDIDATE_SELECTED")
                request = self.session.tick(frame, capabilities)
            return EmergencyResult(request, request.reason_code, self.session.stage,
                                   None if self.selected is None else self.selected.identifier)
        if distinct:
            self._clear += 1
        reply = self.session.last_feedback
        actual_stop=(reply is not None and reply.producer=='runtime'
            and reply.actual_brake_stop_confirmed and reply.actual_standstill_confirmed
            and reply.command_send_status=='sent' and self.selected is not None
            and reply.candidate_id==self.selected.identifier and reply.actual_pose is not None)
        executed = (actual_stop or reply is not None and reply.producer in ("control", "runtime")
                    and reply.actual_standstill_confirmed and reply.goal_pose_arrived
                    and reply.actual_pose is not None)
        if self._clear >= self.clear_frames and executed and verified_task_route_available:
            reason=('DANGER_CLEARED_ACTUAL_BRAKE_STOP_RETURN_TO_TASK' if actual_stop
                    else 'DANGER_CLEARED_EXECUTION_VERIFIED_RETURN_TO_TASK')
            request = self.session.finish(frame, True, reason)
            result = EmergencyResult(request, request.reason_code, "COMPLETE",
                                     None if self.selected is None else self.selected.identifier)
            result.immediate_brake_required, result.hold_required = False, False
            return result
        return EmergencyResult(request, "DANGER_CLEAR_OR_TASK_ROUTE_CONFIRMATION_PENDING", "RECOVER",
                               None if self.selected is None else self.selected.identifier)
