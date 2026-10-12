"""Reuse D01 policies and read local R06 facts through the runtime channel."""

import time

from core.traffic_quality import signal_stop_requirement, traffic_usable
from core.validation import number
from members.decision.behaviors.contract import TaskContext, BehaviorFrame
from members.decision.behaviors.session import BehaviorSession
from core.behavior_channel import read_channel
from core.maneuver_facts import ManeuverFacts
from members.decision.behaviors.environment import lane_candidates
from members.decision.behaviors.parking_runtime import ParkingRuntime
from members.decision.behaviors.lane_change_runtime import LaneChangeRuntime
from members.decision.behaviors.driving_runtime import DrivingRuntime
from members.decision.behaviors.stop_policies import (
    SignalDwellPolicy, SignalStopObservation, BlockedRoadPolicy, BlockingObservation,
)


class BehaviorCoordinator(object):
    def __init__(self, settings, clock=None,parking_inputs_provider=None,lane_change_inputs_provider=None,
                 driving_inputs_provider=None):
        self.settings, self.clock = settings, clock or time.monotonic
        self.parking_inputs_provider=parking_inputs_provider
        self.lane_change_inputs_provider=lane_change_inputs_provider
        self.driving_inputs_provider=driving_inputs_provider
        self._generation = 0
        self.reset()

    def _session(self):
        s = self.settings
        return BehaviorSession(s.behavior_feedback_max_age_s, s.behavior_ack_timeout_s,
                               s.behavior_progress_timeout_s, s.behavior_retry_delay_s,
                               s.behavior_max_retries, s.behavior_observation_gap_s)

    def reset(self):
        self._generation += 1
        self.signal = SignalDwellPolicy(self.settings.signal_dwell_duration_s, self._session())
        self.blocked = BlockedRoadPolicy(self.settings.release_frames, self._session())
        self.parking=ParkingRuntime(self.parking_inputs_provider,self._session,self.clock)
        self.lane_changes=LaneChangeRuntime(self.lane_change_inputs_provider,self._session,self.clock,self.settings)
        self.driving=DrivingRuntime(self.driving_inputs_provider,self._session,self.clock,self.settings)
        self._signal_seen = set()
        self._runtime_context = None
        self._blockage_anchor = None
        self._latest = dict(interface_status="proposal_not_runtime_connected", requests=[],
                            dependencies=["R05_EXECUTION_FEEDBACK", "R07_BEHAVIOR_GOAL_CHANNEL"],
                            frame_id=-1)

    def _signal_observation(self, perception, output):
        traffic = perception.traffic
        candidates = []
        if traffic.signal_groups_valid:
            for group in traffic.signal_groups:
                if (not isinstance(group, dict) or group.get("boundary_kind") != "stop_line"
                        or group.get("association_valid") is not True
                        or not number(group.get("stop_line_distance")) or group["stop_line_distance"] < 0
                        or not group.get("signal_ids") or not all(type(i) is int and i >= 0 for i in group["signal_ids"])):
                    continue
                key = "signals:" + ",".join(str(i) for i in sorted(group["signal_ids"]))
                known = group.get("valid") is True and group.get("ambiguous") is False
                red = not known or group.get("signal_state") != "GREEN"
                candidates.append((group["stop_line_distance"], key, red, known))
        elif (traffic.observed and traffic.signal_id >= 0 and traffic_usable(perception)
              and number(traffic.stop_line_distance) and traffic.stop_line_distance >= 0):
            candidates.append((traffic.stop_line_distance, "signal:" + str(traffic.signal_id),
                               traffic.signal_state != "GREEN", True))
        if not candidates or self.settings.front_offset_m is None:
            return None
        distance, key, red, known = min(candidates)
        if red and known:
            self._signal_seen.add(key)
        if key not in self._signal_seen:
            return None
        stop = max(0.0, distance - self.settings.front_offset_m - self.settings.traffic_stop_margin)
        if stop<1e-6:
            stop = 0.
        return SignalStopObservation(key, stop, red, usable=known,
                                     speed_cap_mps=output.target_speed)

    def observe_legacy(self, perception, output, constraints, now_s):
        context = TaskContext(perception.case_id, perception.task_id, perception.scene_id,
                              "decision-session:" + str(self._generation))
        capabilities,feedbacks,connected = None,[],False
        try:
            context,capabilities,feedbacks = read_channel(perception,now_s)
            connected = True
            self._runtime_context = context
        except (ValueError,TypeError,AttributeError,KeyError):
            if self._runtime_context is not None:
                context = self._runtime_context
        def reply(session):
            current = [f for f in feedbacks if f.intent_id==session.intent_id and f.revision==session.revision]
            return max(current,key=lambda f:(f.produced_at_s,f.producer!="planning")) if current else None
        frame = BehaviorFrame(context, perception.frame_id, now_s, perception.valid_until,
                              perception.ego.x, perception.ego.y, perception.ego.heading,
                              perception.ego.speed, usable=True)
        requests, policy_states = [], []
        environment = dict(usable=False, lane_candidates=[], reason='MANEUVER_ENVIRONMENT_UNAVAILABLE')
        try:
            candidates, failures = lane_candidates(perception, frame, self.clock)
            environment = dict(usable=True, lane_candidates=[c.lane_id for c in candidates], failures=failures)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as error:
            environment['reason'] = str(error)
        # Scene IDs select the supplied stop/light duty only; geometry and
        # collision/stop determination are still the existing shared facts.
        signal = self.signal.evaluate(frame, self._signal_observation(perception, output),
                                      capabilities=capabilities,feedback=reply(self.signal.session),
                                      dwell_required=perception.scene_id == 20,
                                      safety_override=output.mode == "EMERGENCY_BRAKE")
        policy_states.append(dict(policy="SIGNAL_DWELL", phase=signal.phase, reason_code=signal.reason))
        if signal.request is not None:
            requests.append(signal.request.to_dict("runtime_connected" if connected else "proposal_not_runtime_connected"))
            if connected and signal.request.dispatch_allowed and output.mode!="EMERGENCY_BRAKE":
                output.behavior_request = signal.request.to_dict("runtime_connected")
                output.behavior_active_identity = {key:output.behavior_request[key] for key in
                    ("intent_id","stage","revision","source_frame_id","issued_at_s")}
        if connected and signal.hold_required and output.mode!="EMERGENCY_BRAKE":
            output.mode,output.target_speed,output.stop_distance = "STOP",0.,0.
            output.precision_stop = True
            output.reason = "SIGNAL_DWELL:"+signal.reason
        if self.signal_pending() and (not connected or signal.request is None
                                         or not signal.request.dispatch_allowed):
            output.mode = "EMERGENCY_BRAKE" if perception.ego.speed>self.settings.standstill_speed else "STOP"
            output.target_speed,output.stop_distance = 0.,0.
            output.behavior_request,output.behavior_active_identity = None,None
            output.reason = "SIGNAL_OBLIGATION_PENDING:"+signal.reason
        obstruction = None
        if perception.scene_id == 11 and constraints is not None and constraints.blockage_candidates:
            identifier, distance = min(constraints.blockage_candidates, key=lambda item: (item[1], item[0]))
            obstruction = BlockingObservation("target:" + str(identifier), distance, True,
                                               speed_cap_mps=output.target_speed)
            target = next((v for v in perception.targets if v.id == identifier and v.valid), None)
            digest = None
            try:
                digest = ManeuverFacts(perception, self.clock).map_digest()
            except (ValueError, TypeError, AttributeError, KeyError):
                pass
            self._blockage_anchor = ((frame.context.key(), perception.lane.lane_id, (target.x, target.y, target.z), digest)
                                     if target is not None else None)
        if obstruction is None and self.blocked.session.intent_id:
            obstruction = BlockingObservation("prior", None, False, usable=False)
            try:
                if (self._blockage_anchor is not None and self._blockage_anchor[0] == frame.context.key()
                        and self._blockage_anchor[3] is not None):
                    facts = ManeuverFacts(perception, self.clock)
                    if facts.map_digest() == self._blockage_anchor[3]:
                        clear = facts.road_clear_at(*self._blockage_anchor[1:3])
                        obstruction = BlockingObservation('prior', None, False, verified_clear=clear,
                            source=perception.target_source, source_frame_id=perception.targets_frame_id,
                            source_observed_at_s=facts.env.dynamic_observed_at_s)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
                pass
        blockage = self.blocked.evaluate(frame, obstruction,
                                         capabilities=capabilities,feedback=reply(self.blocked.session),
                                         safety_override=output.mode == "EMERGENCY_BRAKE")
        policy_states.append(dict(policy="BLOCKED_ROAD", phase=blockage.phase, reason_code=blockage.reason))
        if blockage.request is not None:
            requests.append(blockage.request.to_dict())
        self._latest = dict(interface_status="runtime_connected" if connected else "proposal_not_runtime_connected", frame_id=perception.frame_id,
                            requests=requests, policies=policy_states,
                            maneuver_environment=environment,
                            dependencies=([] if connected else ["R05_EXECUTION_FEEDBACK", "R07_BEHAVIOR_GOAL_CHANNEL",
                                          "DOWNSTREAM_CAPABILITIES"]))

    def signal_pending(self):
        session = self.signal.session
        return bool(self._runtime_context is not None and session.intent_id
                    and session.status not in ("COMPLETED","CANCELLED"))

    def execution_pending(self):
        return (self.signal_pending() or self.driving.pending() or self.parking.pending()
                or self.lane_changes.pending())

    def apply_driving(self,perception,output):
        self.driving.apply(perception,output,priority=self.signal_pending())
        if self.driving.provider is not None or self.driving.pending():
            self._latest['driving']=self.driving.snapshot()

    def apply_parking(self,perception,output):
        priority=(self.signal_pending() or signal_stop_requirement(perception) is not None
                  or self.driving.pending() or self.lane_changes.pending()
                  or output.mode=='EMERGENCY_BRAKE' or output.behavior_request is not None)
        self.parking.apply(perception,output,priority)
        if self.parking.provider is not None or self.parking.pending():
            self._latest['parking']=self.parking.snapshot()

    def apply_lane_change(self,perception,output,constraints):
        priority=(self.signal_pending() or self.driving.pending() or self.parking.claimed or self.parking.pending()
                  or output.behavior_request is not None)
        self.lane_changes.apply(perception,output,constraints,priority)
        if self.lane_changes.provider is not None:
            self._latest['lane_change']=self.lane_changes.snapshot()

    def diagnostic_error(self, frame_id, reason):
        self._latest = dict(interface_status="proposal_not_runtime_connected", frame_id=frame_id,
                            requests=[], reason_code="BEHAVIOR_DIAGNOSTIC_ERROR:" + reason,
                            dependencies=["R05_EXECUTION_FEEDBACK", "R07_BEHAVIOR_GOAL_CHANNEL"])

    def snapshot(self):
        import copy
        return copy.deepcopy(self._latest)
