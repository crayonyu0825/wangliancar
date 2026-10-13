"""PR23 task progress/obligations adapted to main's one formal channel."""
import copy
import math

from core.behavior_channel import read_channel
from core.behavior_contract import BehaviorFrame, require
from core.geometry import project_polyline, projection_within_polyline, normalize_angle
from core.maneuver_facts import ManeuverFacts
from core.region_geometry import rectangle
from core.corridor_region import CorridorRegion
from core.task_route import TaskDecisionInputs
from members.decision.behaviors.task_progress import TaskProgressPolicy


class TaskRuntime(object):
    def __init__(self, provider, session_factory, clock):
        self.provider, self.clock = provider, clock
        self.policy = TaskProgressPolicy(session=session_factory())
        self.context = self.binding = self.anchor = None
        self.completed = set()
        self._input_key, self._input_value = None, None
        self.phase, self.reason = 'IDLE', 'TASK_PROGRESS_NOT_SUPPLIED'

    def snapshot(self):
        return dict(configured=self.provider is not None, phase=self.phase, reason_code=self.reason,
                    recovery_anchor=self.anchor, completed_obligations=sorted(self.completed))

    def _hold(self, p, out, reason):
        self.reason, self.phase = reason, 'RECOVER'
        out.mode = 'STOP' if p.ego.speed <= .05 else 'EMERGENCY_BRAKE'
        out.target_speed, out.stop_distance = 0., 0.
        out.behavior_request = out.behavior_active_identity = None
        out.reason = 'TASK_PROTECTION:' + reason

    def current_inputs(self, p, context, now):
        key = (id(p), context.key(), p.frame_id, p.timestamp)
        if key != self._input_key:
            value = self.provider(p, context) if self.provider is not None else None
            self._input_key = key
            self._input_value = value.snapshot() if isinstance(value, TaskDecisionInputs) else value
        value = self._input_value
        if value is None:
            return None
        require(isinstance(value, TaskDecisionInputs), 'TASK_INPUTS_INVALID')
        frame = BehaviorFrame(context, p.frame_id, now, p.valid_until, p.ego.x, p.ego.y,
                              p.ego.heading, p.ego.speed, usable=p.valid)
        proof = value.facts.evidence
        require(proof.frame_id == p.frame_id and proof.valid_until_s <= p.valid_until
                and proof.verified(frame, allowed_source_kinds=('task', 'verified_fusion')),
                'TASK_PROGRESS_SOURCE_UNVERIFIED')
        require(ManeuverFacts(p, self.clock).map_digest() == value.map_digest, 'TASK_MAP_CHANGED')
        return value.snapshot()

    def apply(self, p, out, temporary=False, completed_obligations=()):
        if self.provider is None:
            return
        try:
            now = self.clock()
            context, caps, replies = read_channel(p, now)
            value = self.current_inputs(p, context, now)
            require(isinstance(value, TaskDecisionInputs), 'FORMAL_TASK_PROGRESS_LOST')
            value = value.snapshot()
            facts = value.facts
            frame = BehaviorFrame(context, p.frame_id, now, p.valid_until, p.ego.x, p.ego.y,
                                  p.ego.heading, p.ego.speed, usable=p.valid)
            require(facts.evidence.frame_id == p.frame_id and facts.evidence.valid_until_s <= p.valid_until
                    and facts.evidence.verified(frame, allowed_source_kinds=('task', 'verified_fusion')),
                    'TASK_PROGRESS_SOURCE_UNVERIFIED')
            require(ManeuverFacts(p, self.clock).map_digest() == value.map_digest, 'TASK_MAP_CHANGED')
            if self.context is not None and context.key() != self.context:
                require(context.session_id != self.context[-1], 'TASK_CONTEXT_CHANGED_WITHOUT_NEW_SESSION')
                self.policy.reset()
                self.binding, self.anchor, self.completed = None, None, set()
            self.context = context.key()
            binding = (facts.binding(), value.vehicle_extents, value.map_digest)
            require(self.binding is None or binding == self.binding, 'TASK_ROUTE_DEFINITION_CHANGED')
            self.binding = binding
            self.completed.update(completed_obligations)
            if temporary and self.anchor is None:
                self.anchor = (facts.route_id, facts.segment_id, facts.progress_s_m)
            session = self.policy.session
            matching = [r for r in replies if r.intent_id == session.intent_id
                        and r.stage == session.stage and r.revision == session.revision
                        and r.producer in ('control', 'runtime')]
            reply = copy.deepcopy(max(matching, key=lambda r: r.produced_at_s)) if matching else None
            if reply is not None and reply.goal_pose_arrived:
                complete = p.ego.speed <= .05 and set(facts.required_obligations) <= self.completed
                complete = complete and facts.destination_region is not None and reply.actual_pose is not None
                if complete:
                    region = CorridorRegion((facts.destination_region,))
                    for pose in ((p.ego.x, p.ego.y, p.ego.heading),
                                 (reply.actual_pose.x, reply.actual_pose.y, reply.actual_pose.body_heading_rad)):
                        body = rectangle(*pose, *value.vehicle_extents)
                        complete = complete and region.body_clearance(body, lambda: None, 100000) > 0
                reply.goal_pose_arrived = bool(complete)
            require(frame.current(self.clock()) and self.clock() < facts.evidence.valid_until_s,
                    'TASK_SOURCE_EXPIRED_DURING_COMPLETION_CHECK')
            result = self.policy.evaluate(frame, facts, caps, reply,
                safety_override=out.mode == 'EMERGENCY_BRAKE', temporary_maneuver_active=temporary)
            self.phase, self.reason = result.phase, result.reason
            if temporary or out.mode == 'EMERGENCY_BRAKE':
                return
            require(result.phase == 'COMPLETE' or not result.hold_required, result.reason)
            request = result.request
            require(request is not None and request.dispatch_allowed or result.phase == 'COMPLETE',
                    'TASK_REQUEST_NOT_DISPATCHABLE')
            index = next(i for i, segment in enumerate(facts.segments) if segment[0] == facts.segment_id)
            require(facts.segments[index][1] == p.lane.lane_id, 'TASK_CURRENT_SEGMENT_LANE_MISMATCH')
            final = (facts.reference_end_reason == 'task_destination' and index == len(facts.segments)-1
                     and facts.destination_goal is not None)
            if final:
                reference = p.lane.forward_reference if p.lane.forward_reference_valid else p.lane.center_line
                here = project_polyline(reference, p.ego.x, p.ego.y)
                goal = project_polyline(reference, facts.destination_goal.x, facts.destination_goal.y)
                require(here is not None and projection_within_polyline(goal, len(reference))
                        and goal['distance'] <= .15 and goal['s'] >= here['s']-.15
                        and abs(normalize_angle(facts.destination_goal.body_heading_rad-goal['heading'])) <= .15,
                        'DESTINATION_OUTSIDE_SELECTED_FINITE_REFERENCE')
                bound = max(0., goal['s']-here['s'])
            else:
                bound = max(0., facts.segments[index][3]-facts.progress_s_m-value.vehicle_extents[0]-.3)
            out.stop_distance = min(out.stop_distance, bound) if out.stop_distance >= 0 else bound
            out.target_speed = min(out.target_speed, facts.speed_cap_mps)
            out.precision_stop = True
            if result.phase == 'COMPLETE' or bound <= .05:
                out.mode, out.target_speed = 'STOP', 0.
            session.goal.stop_distance_m = bound
            session.goal.precision_stop = True
            session.goal.speed_cap_mps = facts.speed_cap_mps
            # Terminal status keeps a normal current-pose hold, without asking
            # planning to treat a completed request as fresh motion authority.
            if result.phase == 'COMPLETE':
                out.behavior_request = out.behavior_active_identity = None
                out.reason = 'TASK_DESTINATION_ACTUALLY_REACHED_AND_STOPPED'
            else:
                request = session.snapshot(frame)
                out.valid_until = min(out.valid_until, facts.evidence.valid_until_s)
                out.behavior_request = request.to_dict('runtime_connected')
                out.behavior_request['valid_until_s'] = out.valid_until
                out.behavior_active_identity = {k: out.behavior_request[k] for k in
                    ('intent_id', 'stage', 'revision', 'source_frame_id', 'issued_at_s')}
            self.anchor = None
        except (ValueError, TypeError, AttributeError, KeyError, StopIteration, OverflowError) as error:
            if not temporary:
                self._hold(p, out, str(error))
            else:
                self.phase, self.reason = 'RECOVER', str(error)
