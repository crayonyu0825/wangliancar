"""Runtime-owned previous-cycle exchange; no disk feedback or member internals."""
import copy
import time

from core.behavior_protocol import ExecutionFeedback, RuntimeRequest
from core.validation import current


class BehaviorExchange(object):
    def __init__(self):
        self.context=None; self.previous=[]
        self._lamp_key=None; self._lamp_elapsed=0.0
        self._last_time=None; self._last_frame=None

    def prepare(self,p,capabilities):
        context=(p.case_id,p.task_id,p.session_id)
        if context!=self.context:
            self.__init__(); self.context=context
        p.behavior_feedback=copy.deepcopy(self.previous)
        p.behavior_capabilities=capabilities

    def finish(self,p,trajectory,control,receipt,safety_active=False):
        self.previous=list(trajectory.planning_feedback)
        reply=control.execution_feedback
        request=trajectory.behavior_request
        if not isinstance(request,RuntimeRequest) or not isinstance(reply,ExecutionFeedback):
            self._lamp_elapsed=0.0
            return
        reply=copy.deepcopy(reply)
        now=time.monotonic()
        distinct=self._last_frame is None or p.frame_id>self._last_frame
        dt=0.0 if self._last_time is None else now-self._last_time
        lights=(control.left_signal,control.right_signal,control.hazard_signal)
        key=(self.context,request.intent_id,lights)
        sent=receipt.get('ok') is True and receipt.get('attempted') is True and not safety_active
        if key!=self._lamp_key or dt<0 or dt>0.5 or not sent or not current(control,trajectory):
            self._lamp_elapsed=0.0
        elif distinct:
            self._lamp_elapsed+=dt
        self._lamp_key=key
        if distinct:
            self._last_frame,self._last_time=p.frame_id,now
        wanted=request.goal.light_intent
        reply.lights_confirmed=sent and lights==(wanted.left_signal,wanted.right_signal,wanted.hazard_signal)
        reply.lights_duration_s=self._lamp_elapsed if reply.lights_confirmed else 0.0
        reply.lights_evidence_level='command_sent' if reply.lights_confirmed else 'unknown'
        if safety_active:
            reply.usable=False
            reply.reason_code='SAFETY_OVERRIDE_NOT_EXECUTED'
        self.previous.append(reply)
