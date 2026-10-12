"""D03 decision strategy: legal corridor, front/rear gap and actual settlement."""

import math
from core.geometry import project_polyline, normalize_angle, projection_within_polyline
from members.decision.behaviors.contract import BehaviorGoal, GoalPose, LightIntent, finite, require
from members.decision.behaviors.observations import Evidence, MotionObject, vehicle_footprint, corridor_contains, intersects
from members.decision.behaviors.session import BehaviorSession
from members.decision.behaviors.stop_policies import PolicyResult


class LaneCandidate(object):
    def __init__(self, lane_id, side, center_line, width_m, shared_marking,
                 crossing_allowed, same_direction, evidence, rear_coverage_verified,
                 objects=(), left_boundary=None, right_boundary=None,
                 crossing_boundary=None, crossing_ranges=None, source_lane_id=None,
                 map_digest=None, is_current_lane=False, object_age_s=0.):
        require(isinstance(lane_id, str) and bool(lane_id) and side in ("left", "right"), "invalid candidate lane")
        require(isinstance(center_line, (list, tuple)) and 2 <= len(center_line) <= 20000
                and all(isinstance(p, (list, tuple)) and len(p) == 2
                        and all(finite(v) for v in p) for p in center_line), "candidate lane geometry missing")
        require(finite(width_m) and width_m > 0 and isinstance(evidence, Evidence), "invalid candidate coverage")
        require(all(type(v) is bool for v in (crossing_allowed, same_direction, rear_coverage_verified)), "invalid crossing flags")
        require(all(isinstance(o, MotionObject) for o in objects), "invalid lane objects")
        self.lane_id, self.side, self.center_line, self.width_m = lane_id, side, list(center_line), width_m
        self.shared_marking, self.crossing_allowed, self.same_direction = shared_marking, crossing_allowed, same_direction
        for boundary in (left_boundary, right_boundary):
            require(boundary is None or (isinstance(boundary, (list, tuple)) and len(boundary) >= 2
                    and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(finite(v) for v in p)
                            for p in boundary)), "invalid candidate boundary")
        self.left_boundary, self.right_boundary = left_boundary, right_boundary
        require((crossing_boundary is None) == (crossing_ranges is None), 'crossing scope pair required')
        if crossing_boundary is not None:
            require(isinstance(crossing_ranges, (list, tuple)), 'invalid crossing ranges')
            for line in [crossing_boundary] + list(crossing_ranges):
                require(isinstance(line, (list, tuple)) and 2 <= len(line) <= 20000
                        and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(finite(v) for v in p)
                                for p in line), 'invalid crossing scope geometry')
        self.crossing_boundary, self.crossing_ranges = crossing_boundary, crossing_ranges
        self.evidence, self.rear_coverage_verified, self.objects = evidence, rear_coverage_verified, tuple(objects)
        require(source_lane_id is None or isinstance(source_lane_id,str) and bool(source_lane_id)
                and source_lane_id!=lane_id,'invalid original candidate source')
        require(map_digest is None or isinstance(map_digest,str) and len(map_digest)==32
                and all(c in '0123456789abcdef' for c in map_digest),'invalid candidate map identity')
        require(type(is_current_lane) is bool and finite(object_age_s) and object_age_s>=0,
                'invalid current lane or object age')
        self.source_lane_id,self.map_digest=source_lane_id,map_digest
        self.is_current_lane,self.object_age_s=is_current_lane,object_age_s


class LaneChangePolicy(object):
    def __init__(self, front_m, rear_m, half_width_m, min_front_gap_m=10.5,
                 min_rear_gap_m=10.5, time_headway_s=1.2, maneuver_time_s=4.0,
                 acceleration_uncertainty_mps2=2.0, indicator_lead_s=1.0,
                 settle_frames=2, session=None, allow_command_lights=False):
        require(all(finite(v) and v > 0 for v in (front_m, rear_m, half_width_m,
                min_front_gap_m, min_rear_gap_m, time_headway_s, maneuver_time_s,
                acceleration_uncertainty_mps2, indicator_lead_s)), "invalid lane-change limits")
        require(type(settle_frames) is int and settle_frames > 0, "invalid settlement count")
        require(type(allow_command_lights) is bool, 'invalid command-lamp trial policy')
        self.allow_command_lights=allow_command_lights
        self.front_m, self.rear_m, self.half_width_m = front_m, rear_m, half_width_m
        self.front_gap, self.rear_gap, self.headway = min_front_gap_m, min_rear_gap_m, time_headway_s
        self.duration, self.uncertainty, self.indicator_lead_s = maneuver_time_s, acceleration_uncertainty_mps2, indicator_lead_s
        self.settle_frames, self.session = settle_frames, session or BehaviorSession()
        self.reset()

    def reset(self):
        self.session.reset()
        self.selected, self._context, self._settled = None, None, 0
        self._selection_identity,self._speed_cap=None,None
        self._execution_started=False
        self._goal_identity=None

    @staticmethod
    def _identity(candidate):
        return candidate.source_lane_id,candidate.lane_id,candidate.side,candidate.map_digest

    def _body_inside(self,pose,candidate,require_alignment=True):
        projection=project_polyline(candidate.center_line,pose.x,pose.y)
        corners=vehicle_footprint(pose,self.front_m,self.rear_m,self.half_width_m)
        return (projection_within_polyline(projection,len(candidate.center_line))
                and (abs(normalize_angle(pose.body_heading_rad-projection['heading']))<=.15
                     if require_alignment else abs(normalize_angle(
                         pose.body_heading_rad-projection['heading']))<math.pi/2)
                and all(projection_within_polyline(project_polyline(candidate.center_line,*p),
                                                   len(candidate.center_line)) for p in corners)
                and corridor_contains(corners,candidate.left_boundary,candidate.right_boundary))

    def opportunity(self, frame, candidate, continuing=False):
        if not candidate.evidence.verified(frame, allowed_source_kinds=("sensor", "verified_fusion", "synthetic")) or not candidate.rear_coverage_verified:
            return False, "SIDE_REAR_COVERAGE_UNKNOWN"
        if candidate.left_boundary is None or candidate.right_boundary is None:
            return False, "CANDIDATE_BOUNDARIES_UNAVAILABLE"
        if not candidate.same_direction:
            return False,'CROSSING_ILLEGAL_OR_OPPOSITE'
        if candidate.width_m<2*self.half_width_m:
            return False,'TARGET_LANE_TOO_NARROW'
        actual=GoalPose(frame.ego_x,frame.ego_y,frame.ego_heading)
        if continuing and candidate.is_current_lane and self._body_inside(actual,candidate,require_alignment=False):
            # Completing a crossing does not require permission to start a
            # second crossing. Current occupancy still constrains the body;
            # Future path safety remains the planner's independent obligation.
            # The body may still need to align inside B; completion below
            # retains its separate stricter heading/stability requirement.
            body=vehicle_footprint(actual,self.front_m,self.rear_m,self.half_width_m)
            age=candidate.object_age_s
            for obj in candidate.objects:
                if intersects(body,obj.footprint(age,.5*self.uncertainty*age*age)):
                    return False,'TARGET_BODY_OCCUPIED'
            return True,'TARGET_BODY_CLEAR_PATH_REVALIDATION_REQUIRED'
        if (not candidate.crossing_allowed
                or candidate.shared_marking not in ("DASHED", "BROKEN")):
            return False, "CROSSING_ILLEGAL_OR_OPPOSITE"
        if candidate.crossing_boundary is not None and not continuing:
            boundary = candidate.crossing_boundary
            anchor = project_polyline(boundary, frame.ego_x, frame.ego_y)
            if not projection_within_polyline(anchor, len(boundary)):
                return False, 'CROSSING_LOCAL_WINDOW_UNAVAILABLE'
            reach = math.hypot(max(self.front_m, self.rear_m), self.half_width_m)
            low, high = anchor['s'] - reach, anchor['s'] + frame.ego_speed * self.duration + reach
            intervals = []
            for line in candidate.crossing_ranges:
                start, end = project_polyline(boundary, *line[0]), project_polyline(boundary, *line[-1])
                if start is not None and end is not None:
                    intervals.append((start['s'], end['s']))
            # This preliminary opportunity window never substitutes for the
            # planner's actual continuous whole-body boundary crossing test.
            if not any(a <= low and high <= b for a,b in intervals):
                return False, 'CROSSING_MANEUVER_WINDOW_UNVERIFIED'
        ego = project_polyline(candidate.center_line, frame.ego_x, frame.ego_y)
        if ego is None or abs(normalize_angle(ego["heading"] - frame.ego_heading)) >= math.pi / 2:
            return False, "CANDIDATE_DIRECTION_UNVERIFIED"
        at_target = GoalPose(ego["point"][0], ego["point"][1], ego["heading"])
        if not corridor_contains(vehicle_footprint(at_target, self.front_m, self.rear_m, self.half_width_m),
                                 candidate.left_boundary, candidate.right_boundary):
            return False, "TARGET_BOUNDARY_SPACE_INSUFFICIENT"
        travel = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(candidate.center_line, candidate.center_line[1:]))
        if not continuing and travel - ego["s"] < frame.ego_speed * self.duration + self.front_m:
            return False, "CANDIDATE_COVERAGE_TOO_SHORT"
        for obj in candidate.objects:
            age=candidate.object_age_s
            proj = project_polyline(candidate.center_line, obj.x+obj.vx*age, obj.y+obj.vy*age)
            if proj is None:
                return False, "TARGET_LANE_OBJECT_PROJECTION_UNKNOWN"
            age_margin=.5*self.uncertainty*age*age
            extent = 0.5 * math.hypot(obj.length, obj.width)+age_margin
            angle = None if obj.heading is None else obj.heading - proj['heading']
            lateral = extent if angle is None else .5*(obj.length*abs(math.sin(angle))+obj.width*abs(math.cos(angle)))+age_margin
            # A clamped endpoint distance also contains longitudinal distance.
            # It cannot prove that a rear/front object is in another lane. Keep
            # the clamped station as a conservative gap bound in that case.
            if (projection_within_polyline(proj,len(candidate.center_line))
                    and proj["distance"] > candidate.width_m / 2 + lateral):
                continue
            rel = proj["s"] - ego["s"]
            velocity = math.cos(proj["heading"]) * obj.vx + math.sin(proj["heading"]) * obj.vy
            uncertainty = .5*self.uncertainty*(age+self.duration)**2-age_margin
            if rel >= 0:
                available = rel - extent - self.front_m
                future = available + (velocity - frame.ego_speed) * self.duration - uncertainty
                if min(available, future) < self.front_gap + self.headway * frame.ego_speed:
                    return False, "TARGET_LANE_FRONT_GAP"
            else:
                available = -rel - extent - self.rear_m
                future = available - max(0.0, velocity - frame.ego_speed) * self.duration - uncertainty
                if min(available, future) < self.rear_gap:
                    return False, "TARGET_LANE_REAR_CLOSING"
        return True, "GAP_VERIFIED_WITH_CONSTANT_VELOCITY_UNCERTAINTY"

    def evaluate(self, frame, intent_key, purpose, candidates, capabilities=None,
                 feedback=None, safety_override=False, preferred_lane_id=None,speed_cap_mps=None,
                 goal_pose=None):
        if not frame.current(frame.observed_at_s) or frame.paused or safety_override:
            self._settled=0
            request = self.session.tick(frame, capabilities, safety_override=safety_override) if self.session.intent_id else None
            return PolicyResult(request, True, "OBSERVATION_UNUSABLE", "RECOVER")
        if self._context != frame.context.key():
            self.reset()
            self._context = frame.context.key()
        if self.session.status in ("COMPLETED","CANCELLED"):
            if self.session.key == intent_key:
                complete=self.session.status=='COMPLETED'
                return PolicyResult(self.session.snapshot(frame),not complete,self.session.reason_code,
                                    'COMPLETE' if complete else 'CANCELLED')
            self.reset()
            self._context = frame.context.key()
        require(purpose in ("LANE_CHANGE", "AVOID", "OVERTAKE", "MERGE"), "unsupported lane-change purpose")
        require(all(isinstance(c, LaneCandidate) for c in candidates), "invalid neighbor observations")
        require(speed_cap_mps is None or finite(speed_cap_mps) and speed_cap_mps>=0,
                'invalid authorized lane-change speed cap')
        require(goal_pose is None or isinstance(goal_pose,GoalPose)
                and all(finite(v) for v in (goal_pose.x,goal_pose.y,goal_pose.body_heading_rad)),
                'invalid lane-change goal pose')
        goal_key=None if goal_pose is None else (goal_pose.x,goal_pose.y,goal_pose.body_heading_rad)
        if self.selected is not None and (self.session.key!=intent_key or self.session.maneuver!=purpose):
            self._settled=0
            request=self.session.block(frame,'ACTIVE_INTENT_CHANGE_REQUIRES_CANCEL')
            return PolicyResult(request,True,request.reason_code,'RECOVER')
        if self.selected is not None and goal_key is not None and goal_key!=self._goal_identity:
            self._settled=0
            request=self.session.block(frame,'ACTIVE_GOAL_CHANGE_REQUIRES_CANCEL')
            return PolicyResult(request,True,request.reason_code,'RECOVER')
        if self.selected is not None:
            old=self.session.goal
            pose_key=None if old.goal_pose is None else (old.goal_pose.x,old.goal_pose.y,old.goal_pose.body_heading_rad)
            if (old.source_lane_id,old.target_lane_id,pose_key)!=(self._selection_identity[0],
                    self._selection_identity[1],self._goal_identity):
                # Public goal instances are mutable for speed/distance refresh;
                # target/source/pose are owned by this active intent.
                old.source_lane_id,old.target_lane_id=self._selection_identity[:2]
                old.goal_pose=None if self._goal_identity is None else GoalPose(*self._goal_identity)
                self._settled=0
                request=self.session.block(frame,'ACTIVE_TARGET_BINDING_CHANGED')
                return PolicyResult(request,True,request.reason_code,'RECOVER')
        if speed_cap_mps is not None:
            self._speed_cap=speed_cap_mps
        if self.selected is None:
            if self._speed_cap is None or self._speed_cap<=0:
                return PolicyResult(reason='LANE_CHANGE_SPEED_CAP_UNAVAILABLE',phase='BLOCKED')
            usable = [c for c in candidates if c.evidence.verified(frame, allowed_source_kinds=("sensor", "verified_fusion", "synthetic")) and c.same_direction
                      and c.crossing_allowed and c.shared_marking in ("DASHED", "BROKEN")
                      and c.rear_coverage_verified and c.width_m >= 2 * self.half_width_m
                      and c.left_boundary is not None and c.right_boundary is not None
                      and not c.is_current_lane and (preferred_lane_id is None or c.lane_id == preferred_lane_id)]
            if not usable:
                return PolicyResult(reason="LEGAL_CANDIDATE_GEOMETRY_UNAVAILABLE", phase="BLOCKED")
            self.selected = min(usable, key=lambda c: (not self.opportunity(frame, c)[0], c.lane_id))
            self._selection_identity=self._identity(self.selected)
            self._goal_identity=goal_key
            lights = LightIntent(left_signal=self.selected.side == "left", right_signal=self.selected.side == "right")
            self.session.start(frame, intent_key, purpose, "PREPARE",
                               BehaviorGoal(target_lane_id=self.selected.lane_id,
                                            source_lane_id=self.selected.source_lane_id,
                                            goal_pose=None if goal_key is None else GoalPose(*goal_key),
                                            speed_cap_mps=self._speed_cap, light_intent=lights),
                               ("LANE_CHANGE", "LIGHTS", "FEEDBACK"))
        current = next((c for c in candidates if c.lane_id == self.selected.lane_id), None)
        if current is None:
            self._settled=0
            request = self.session.block(frame, "SELECTED_CORRIDOR_UNAVAILABLE")
            return PolicyResult(request, True, request.reason_code, "BLOCKED")
        if self._identity(current)!=self._selection_identity:
            self._settled=0
            request=self.session.block(frame,'SELECTED_LANE_SOURCE_OR_MAP_CHANGED')
            return PolicyResult(request,True,request.reason_code,'RECOVER')
        self.selected = current
        self._execution_started=self._execution_started or self.session.stage in ('EXECUTE','SETTLE')
        continuing=self._execution_started
        safe, reason = self.opportunity(frame, current,continuing)
        if self._speed_cap<=0:
            safe,reason=False,'LANE_CHANGE_SPEED_CAP_UNAVAILABLE'
        if not safe:
            self._settled=0
        if self.session.stage in ("EXECUTE", "SETTLE") and not safe:
            request = self.session.block(frame, "EXECUTION_CORRIDOR_OBSTRUCTED_NEEDS_RECOVERY_PATH")
            return PolicyResult(request, True, request.reason_code, "RECOVER")
        if (self.session.status == "BLOCKED" and safe and self.session.reason_code in
                ("SELECTED_CORRIDOR_UNAVAILABLE", "EXECUTION_CORRIDOR_OBSTRUCTED_NEEDS_RECOVERY_PATH")):
            old_stage = self.session.stage
            if self.session.retry_after_evidence(frame, "CORRIDOR_RECOVERED_REPLAN"):
                recovery_goal = BehaviorGoal(target_lane_id=current.lane_id,source_lane_id=current.source_lane_id,
                    goal_pose=self.session.goal.goal_pose,speed_cap_mps=self._speed_cap,
                    candidate_id=self.session.goal.candidate_id,
                    light_intent=LightIntent(left_signal=current.side == "left", right_signal=current.side == "right"))
                self.session.advance(frame, "REQUEST_PATH" if old_stage in ("EXECUTE", "SETTLE") else "PREPARE",
                                     recovery_goal, reason="RECOVERY_PATH_REVALIDATION_REQUIRED")
        request = self.session.tick(frame, capabilities, feedback, safety_override)
        if request.status in ("BLOCKED", "SUSPENDED", "CANCELLED"):
            self._settled=0
            return PolicyResult(request, True, request.reason_code, request.status)
        if safe:
            self.session.goal.speed_cap_mps=self._speed_cap
        reply = self.session.last_feedback
        checked_revision=request.revision
        if self.session.stage == "PREPARE":
            if (reply is not None and reply.producer in ("control", "runtime")
                    and ((reply.lights_confirmed and reply.lights_duration_s >= self.indicator_lead_s)
                         or (self.allow_command_lights and reply.lights_evidence_level=='command_sent'
                             and reply.lights_command_duration_s>=self.indicator_lead_s))):
                request = self.session.advance(frame, "WAIT_GAP", reason="INDICATOR_CONFIRMED")
        elif self.session.stage == "WAIT_GAP":
            if safe:
                request = self.session.advance(frame, "REQUEST_PATH", required=("LANE_CHANGE", "PATH", "LIGHTS", "FEEDBACK"),
                                               reason="GAP_AVAILABLE_REQUEST_PATH")
        elif self.session.stage == "REQUEST_PATH":
            if not safe:
                request = self.session.advance(frame, "WAIT_GAP", reason=reason)
            elif reply is not None and reply.producer == "planning" and reply.status == "PLANNED":
                self.session.goal.candidate_id=reply.candidate_id
                self._execution_started=True
                request = self.session.advance(frame, "EXECUTE", reason="MATCHING_PATH_ACCEPTED")
        elif self.session.stage in ("EXECUTE", "SETTLE"):
            reached = False
            if reply is not None and reply.producer in ("control", "runtime") and reply.actual_pose is not None:
                reached = (current.is_current_lane and reply.actual_lane_id==current.lane_id
                           and (self.session.goal.candidate_id is None
                                or reply.candidate_id==self.session.goal.candidate_id)
                           and reply.motion_direction==1
                           and self._body_inside(reply.actual_pose,current)
                           and self._body_inside(GoalPose(frame.ego_x,frame.ego_y,frame.ego_heading),current))
            if reached:
                if self.session.stage == "EXECUTE":
                    self._settled = 0
                    request = self.session.advance(frame, "SETTLE", reason="TARGET_LANE_ACTUALLY_REACHED")
                else:
                    self._settled += 1
                    if self._settled >= self.settle_frames:
                        request = self.session.finish(frame, True, "TARGET_LANE_SETTLED")
            else:
                self._settled = 0
        if self.session.revision!=checked_revision and self.session.status=='REQUESTED':
            # advance() deliberately withdraws the old dispatch. Check the
            # new stage's capabilities in this same frame; never replay the
            # old acknowledgement against the new revision or count time twice.
            request=self.session.tick(frame,capabilities,safety_override=safety_override)
            if not request.dispatch_allowed:
                self._settled=0
                return PolicyResult(request,True,request.reason_code,request.status)
        return PolicyResult(request, not safe, reason, self.session.stage)
