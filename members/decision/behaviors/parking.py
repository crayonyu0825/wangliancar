"""D04 parking goal selection and actual-feedback stage progression."""

import math
import copy
import time
from core.corridor_region import CorridorRegion,NumericResolution,PreparationLimit
from core.geometry import normalize_angle
from core.parking_mission import rear_axle_bay_goal
from members.decision.behaviors.contract import BehaviorGoal, GoalPose, finite, require
from core.behavior_contract import FeedbackValidator,LightIntent,validate_candidate_option
from members.decision.behaviors.observations import Evidence, MotionObject, polygon, inside, intersects, vehicle_footprint
from members.decision.behaviors.session import BehaviorSession
from members.decision.behaviors.stop_policies import PolicyResult


class ParkingSpace(object):
    def __init__(self, identifier, boundary, body_heading_rad, occupancy, evidence,
                 approach_pose=None, position_pose=None):
        require(type(identifier) is int and identifier >= 0 and finite(body_heading_rad), "invalid parking space")
        require(occupancy in ("empty", "occupied", "unknown") and isinstance(evidence, Evidence), "invalid occupancy")
        require(all(p is None or isinstance(p, GoalPose) for p in (approach_pose, position_pose)), "invalid entry pose")
        self.identifier, self.boundary, self.body_heading_rad = identifier, polygon(boundary), body_heading_rad
        self.occupancy, self.evidence = occupancy, evidence
        self.approach_pose, self.position_pose = approach_pose, position_pose


class ParkingFacts(object):
    def __init__(self, evidence, spaces, free_area, objects=(), rear_coverage_verified=False,
                 front_coverage_verified=False, exit_goal=None, signed_body_speed_mps=0.0,
                 object_age_s=0.,position_uncertainty_m=0.,velocity_uncertainty_mps=0.,
                 mission_key=None,vehicle_extents=None):
        require(isinstance(evidence, Evidence) and all(isinstance(s, ParkingSpace) for s in spaces), "invalid parking facts")
        require(all(isinstance(o, MotionObject) for o in objects) and finite(signed_body_speed_mps), "invalid parking objects/motion")
        require(type(rear_coverage_verified) is bool and type(front_coverage_verified) is bool,
                "invalid parking coverage flags")
        require(exit_goal is None or isinstance(exit_goal, GoalPose), "invalid exit goal")
        require(all(finite(v) and v>=0 for v in (object_age_s,position_uncertainty_m,velocity_uncertainty_mps)),
                'invalid parking object observation model')
        self.evidence, self.spaces, self.free_area, self.objects = evidence, tuple(spaces), (
            free_area if isinstance(free_area,ParkingArea) else polygon(free_area)), tuple(objects)
        self.rear_coverage_verified, self.front_coverage_verified = rear_coverage_verified, front_coverage_verified
        self.exit_goal, self.signed_body_speed_mps = exit_goal, signed_body_speed_mps
        self.object_age_s,self.position_uncertainty_m,self.velocity_uncertainty_mps=(
            object_age_s,position_uncertainty_m,velocity_uncertainty_mps)
        self.mission_key,self.vehicle_extents=mission_key,vehicle_extents


class ParkingArea(object):
    """Source-reader-owned road/bay and visibility unions with one frame budget.

    Geometry only: evidence and original deadlines remain in ParkingFacts.
    The same callback bounds preparation plus every body query in this frame.
    Contact, holes, disconnected gaps and unresolved geometry are not clear.
    """
    def __init__(self,regions,views,max_checks,valid_until_s,clock=None):
        require(type(max_checks) is int and max_checks>0 and finite(valid_until_s),
                'explicit parking geometry budget required')
        self.region=CorridorRegion(tuple(tuple(tuple(p) for p in v) for v in regions))
        self.visibility=CorridorRegion(tuple(tuple(tuple(p) for p in v) for v in views))
        self.max_checks,self.valid_until_s,self.clock=max_checks,valid_until_s,clock or time.monotonic
        self.checks,self.reason=0,'NOT_CHECKED'

    def _check(self):
        self.checks+=1
        now=self.clock()
        if self.checks>self.max_checks or not finite(now) or not 0<=now<self.valid_until_s:
            raise ValueError('PARKING_GEOMETRY_BUDGET_OR_DEADLINE')

    def contains(self,body):
        try:
            for region in (self.region,self.visibility):
                if region.body_clearance(body,self._check,self.max_checks)<=0:
                    self.reason='PARKING_BODY_OUTSIDE_ROAD_OR_VIEW'
                    return False
            self.reason='NOMINAL_BODY_CONTAINED'
            return True
        except (ValueError,TypeError,OverflowError,NumericResolution,PreparationLimit) as error:
            self.reason=str(error)
            return False


class ParkingPolicy(object):
    def __init__(self, front_m, rear_m, half_width_m, dwell_s=10.0,
                 position_tolerance_m=0.15, heading_tolerance_rad=0.15,
                 speed_cap_mps=1.0, session=None,standstill_speed_mps=.05):
        require(all(finite(v) and v > 0 for v in (front_m, rear_m, half_width_m, dwell_s,
                position_tolerance_m, heading_tolerance_rad, speed_cap_mps,standstill_speed_mps)) and dwell_s >= 10.0,
                "invalid parking limits")
        self.front_m, self.rear_m, self.half_width_m = front_m, rear_m, half_width_m
        self.dwell_s, self.position_tolerance_m, self.heading_tolerance_rad = dwell_s, position_tolerance_m, heading_tolerance_rad
        self.speed_cap_mps, self.session = speed_cap_mps, session or BehaviorSession()
        self.standstill_speed_mps=standstill_speed_mps
        self.reset()

    def reset(self):
        self.session.reset()
        self._context, self.space, self.park_goal = None, None, None
        self._mission_key=None
        self._selected_space,self._park_pose,self._exit_pose=None,None,None
        self._last_good_goal=None
        self._segment_goal,self._segment_id,self._segment_count=None,None,None
        self._park_direction=-1
        self._segment_seen,self._proposed=set(),None
        self._proposal_validator=FeedbackValidator()

    @staticmethod
    def _space_key(space):
        poses=tuple(None if p is None else (p.x,p.y,p.body_heading_rad)
                    for p in (space.approach_pose,space.position_pose))
        return space.identifier,tuple(space.boundary),space.body_heading_rad,poses

    def _expected_goal(self):
        stage=self.session.stage
        if stage=='ALIGN' and self._segment_goal is not None:
            return copy.deepcopy(self._segment_goal)
        if stage=='APPROACH': return self._goal(self._selected_space.approach_pose)
        if stage=='POSITION': return self._goal(self._selected_space.position_pose)
        if stage=='EXIT':
            require(self._exit_pose is not None,'unbound parking exit')
            return self._goal(self._exit_pose)
        require(stage in ('REVERSE_ENTRY','ALIGN','PARKED_DWELL','EXIT_PREPARE'),'unknown parking stage')
        return self._goal(GoalPose(*self._park_pose),self._park_direction,stage!='EXIT_PREPARE')

    def rear_axle_goal(self, space):
        return rear_axle_bay_goal(space.boundary,space.body_heading_rad,self.front_m,self.rear_m)

    def _body_clear(self, pose, facts, space=None):
        body = vehicle_footprint(pose, self.front_m, self.rear_m, self.half_width_m)
        if not (facts.free_area.contains(body) if isinstance(facts.free_area,ParkingArea)
                else all(inside(point, facts.free_area) for point in body)):
            return False
        if space is not None and not all(inside(point, space.boundary) for point in body):
            return False
        margin=facts.position_uncertainty_m+facts.velocity_uncertainty_mps*facts.object_age_s
        return not any(intersects(body,obj.footprint(facts.object_age_s,margin)) for obj in facts.objects)

    def _arrived(self, reply, goal,frame):
        return (reply is not None and reply.producer in ("control", "runtime")
                and reply.actual_pose is not None and reply.goal_pose_arrived
                and reply.actual_standstill_confirmed
                and (self.session.goal.candidate_id is None or reply.candidate_id==self.session.goal.candidate_id)
                and reply.motion_direction==self.session.goal.motion_direction
                and math.hypot(reply.actual_pose.x - goal.x, reply.actual_pose.y - goal.y) <= self.position_tolerance_m
                and abs(normalize_angle(reply.actual_pose.body_heading_rad - goal.body_heading_rad)) <= self.heading_tolerance_rad
                and frame.ego_speed<=self.standstill_speed_mps
                and math.hypot(frame.ego_x-goal.x,frame.ego_y-goal.y)<=self.position_tolerance_m
                and abs(normalize_angle(frame.ego_heading-goal.body_heading_rad))<=self.heading_tolerance_rad)

    def _goal(self, pose, direction=1, dwell=False):
        return BehaviorGoal(parking_space_id=self.space.identifier, goal_pose=copy.deepcopy(pose),
                            motion_direction=direction, speed_cap_mps=self.speed_cap_mps,
                            precision_stop=True, minimum_standstill_duration_s=self.dwell_s if dwell else 0.0,
                            parking_brake_at_stop=dwell,
                            stop_obligation_id="parking:" + str(self.space.identifier) if dwell else None)

    def offer_candidate_options(self,frame,reply):
        """Planning suggests a path; actual stop proof decides stage movement."""
        self._proposed=None
        if self.session.intent_id is None or reply is None or reply.producer!='planning': return
        usable,unused,unused=self._proposal_validator.check(reply,self.session.snapshot(frame),frame)
        if not usable: return
        candidates=[]
        for option in reply.candidate_options:
            try:
                validate_candidate_option(option,reply.valid_until_s)
                goal=option['goal']
                require(option['action']=='PARK' and option['feasible']
                    and frame.observed_at_s<option['source_valid_until_s']
                    and option['candidate_id']==option['segment_id']==goal.get('candidate_id')
                    and goal['parking_space_id']==self.space.identifier and goal['goal_pose'] is not None
                    and goal['precision_stop'] and goal['minimum_standstill_duration_s']==0.
                    and not goal['parking_brake_at_stop'] and goal['stop_obligation_id'] is None
                    and goal['speed_cap_mps']>0 and goal['speed_cap_mps']<=self.speed_cap_mps
                    and option['candidate_id'] not in self._segment_seen
                    and (self._segment_count is None or option['segment_count']==self._segment_count),
                    'PARKING_SEGMENT_PROPOSAL_NOT_CURRENT')
                candidates.append(copy.deepcopy(option))
            except (ValueError,TypeError,KeyError): continue
        if candidates:
            self._proposed=min(candidates,key=lambda item:(item['risk_cost']+item['rule_cost'],item['candidate_id']))

    def _select_segment(self,frame,option):
        data=copy.deepcopy(option['goal'])
        raw=data.pop('goal_pose'); lights=data.pop('light_intent')
        data['goal_pose']=GoalPose(raw['x'],raw['y'],raw['body_heading_rad'])
        data['light_intent']=LightIntent(**lights)
        goal=BehaviorGoal(**data)
        self._segment_goal,self._segment_id=copy.deepcopy(goal),option['candidate_id']
        self._segment_count=option['segment_count']; self._segment_seen.add(self._segment_id)
        require(len(self._segment_seen)<=self._segment_count<=128,'PARKING_SEGMENT_SEQUENCE_LIMIT')
        required=('PARK','PATH','FEEDBACK')+ (('REVERSE',) if goal.motion_direction==-1 else ())
        return self.session.advance(frame,'ALIGN',goal,required=required,
                                    reason='ACTUAL_STOP_AUTHORIZES_PARKING_SEGMENT',force_revision=True)

    def evaluate(self, frame, facts, capabilities=None, feedback=None, safety_override=False, intent_key="park"):
        if not frame.current(frame.observed_at_s):
            request = self.session.tick(frame, capabilities, safety_override=True) if self.session.intent_id else None
            return PolicyResult(request, True, "OBSERVATION_UNUSABLE", "RECOVER")
        if self._context != frame.context.key():
            self.reset()
            self._context = frame.context.key()
        if self.session.status in ("COMPLETED", "CANCELLED"):
            request = self.session.snapshot(frame)
            return PolicyResult(request, request.status == "CANCELLED", request.reason_code,
                                "COMPLETE" if request.status == "COMPLETED" else "CANCELLED")
        if not isinstance(facts, ParkingFacts) or not facts.evidence.verified(frame, allowed_source_kinds=("sensor", "verified_fusion", "synthetic")):
            request = self.session.tick(frame, capabilities, safety_override=True) if self.session.intent_id else None
            return PolicyResult(request, bool(request), "FREE_SPACE_OCCUPANCY_COVERAGE_UNKNOWN", "RECOVER" if request else "SEARCH")
        if not facts.rear_coverage_verified or not facts.front_coverage_verified:
            request = self.session.block(frame, "PARKING_SURROUND_COVERAGE_UNKNOWN") if self.session.intent_id else None
            return PolicyResult(request, True, "PARKING_SURROUND_COVERAGE_UNKNOWN", "BLOCKED")
        if facts.vehicle_extents is not None and facts.vehicle_extents!=(self.front_m,self.rear_m,self.half_width_m):
            request=self.session.block(frame,'PARKING_VEHICLE_GEOMETRY_MISMATCH') if self.session.intent_id else None
            return PolicyResult(request,True,'PARKING_VEHICLE_GEOMETRY_MISMATCH','BLOCKED')
        if self.space is not None and facts.mission_key!=self._mission_key:
            request=self.session.block(frame,'PARKING_ACTIVE_MISSION_OR_MAP_CHANGED')
            return PolicyResult(request,True,request.reason_code,'BLOCKED')
        if isinstance(facts.free_area,ParkingArea) and not self._body_clear(
                GoalPose(frame.ego_x,frame.ego_y,frame.ego_heading),facts):
            request=self.session.block(frame,'PARKING_CURRENT_BODY_UNOBSERVED_OR_OCCUPIED') if self.session.intent_id else None
            return PolicyResult(request,True,'PARKING_CURRENT_BODY_UNOBSERVED_OR_OCCUPIED','BLOCKED')
        if self.space is None:
            options = []
            fitting_but_entry_unknown = False
            for space in facts.spaces:
                if not space.evidence.verified(frame) or space.occupancy != "empty":
                    continue
                goal = self.rear_axle_goal(space)
                if self._body_clear(goal, facts, space):
                    if (space.approach_pose is not None and space.position_pose is not None
                            and self._body_clear(space.approach_pose, facts)
                            and self._body_clear(space.position_pose, facts)):
                        options.append((space.identifier, space, goal))
                    else:
                        fitting_but_entry_unknown = True
            if not options:
                return PolicyResult(reason="VERIFIED_APPROACH_AND_ENTRY_POSES_UNAVAILABLE" if fitting_but_entry_unknown
                                    else "NO_VERIFIED_FITTING_FREE_PARKING_SPACE", phase="SELECT")
            _, self.space, self.park_goal = min(options, key=lambda item: item[0])
            self.space=copy.deepcopy(self.space)
            self._mission_key=facts.mission_key
            self._selected_space=copy.deepcopy(self.space)
            self._park_pose=(self.park_goal.x,self.park_goal.y,self.park_goal.body_heading_rad)
            self._exit_pose=copy.deepcopy(facts.exit_goal)
            if self.space.approach_pose is None or self.space.position_pose is None:
                self.space, self.park_goal = None, None
                return PolicyResult(reason="VERIFIED_APPROACH_AND_ENTRY_POSES_UNAVAILABLE", phase="BLOCKED")
            self.session.start(frame, intent_key, "PARK", "APPROACH", self._goal(self.space.approach_pose),
                               ("PARK", "PATH", "FEEDBACK"))
            self._last_good_goal=copy.deepcopy(self.session.goal)
        try:
            selected=(self._space_key(self.space)==self._space_key(self._selected_space)
                      and (self.park_goal.x,self.park_goal.y,self.park_goal.body_heading_rad)==self._park_pose)
            expected=self._expected_goal()
            if self.session.status=='BLOCKED': expected.speed_cap_mps=0.
            selected=selected and self.session._goal_key(self.session.goal)==self.session._goal_key(expected)
        except (ValueError,TypeError,AttributeError):
            selected=False
        if not selected:
            self.space=copy.deepcopy(self._selected_space); self.park_goal=GoalPose(*self._park_pose)
            try: self.session.goal=self._expected_goal()
            except (ValueError,TypeError,AttributeError): self.session.goal=copy.deepcopy(self._last_good_goal)
            self.session.goal.speed_cap_mps=0.
            request=self.session.block(frame,'PARKING_ACTIVE_STAGE_GOAL_CHANGED')
            return PolicyResult(request,True,request.reason_code,'BLOCKED')
        observed_space = next((s for s in facts.spaces if s.identifier == self.space.identifier), None)
        if (observed_space is None or not observed_space.evidence.verified(frame)
                or observed_space.occupancy != "empty"):
            request = self.session.block(frame, "SELECTED_PARKING_OCCUPANCY_UNKNOWN_OR_OCCUPIED")
            return PolicyResult(request, True, request.reason_code, "BLOCKED")
        # Keep the selected poses stable, but validate them against fresh space
        # boundaries. A refined observation must not certify the old geometry.
        if ((self.session.stage != "EXIT" and not self._body_clear(self.park_goal, facts, observed_space))
                or not self._body_clear(self.session.goal.goal_pose, facts,
                    observed_space if self.session.stage in ("REVERSE_ENTRY", "ALIGN", "PARKED_DWELL")
                    and self._segment_goal is None else None)):
            request = self.session.block(frame, "NEW_PARKING_GOAL_OBSTRUCTION")
            return PolicyResult(request, True, request.reason_code, "BLOCKED")
        if (self.session.status == "BLOCKED" and self.session.reason_code in
                ("PARKING_SURROUND_COVERAGE_UNKNOWN", "SELECTED_PARKING_OCCUPANCY_UNKNOWN_OR_OCCUPIED",
                 "NEW_PARKING_GOAL_OBSTRUCTION","PARKING_CURRENT_BODY_UNOBSERVED_OR_OCCUPIED")):
            if self.session.retry_after_evidence(frame, "PARKING_EVIDENCE_RECOVERED_REPLAN"):
                old = self.session.goal
                self.session.goal = self._goal(old.goal_pose, old.motion_direction,
                                              old.minimum_standstill_duration_s > 0)
        authorized_reverse = (self.session.stage in ("REVERSE_ENTRY", "ALIGN")
            and self.session.goal.motion_direction==-1 and self.session.status in ("ACCEPTED", "EXECUTING"))
        if facts.signed_body_speed_mps < -0.05 and not authorized_reverse:
            request = self.session.tick(frame, capabilities, safety_override=True)
            return PolicyResult(request, True, "UNEXPECTED_REVERSE_MOTION", "RECOVER")
        proposal=self._proposed
        initial_segment=(proposal is not None and self._segment_id is None
            and self.session.stage=='REVERSE_ENTRY' and self.session.status not in ('BLOCKED','SUSPENDED','CANCELLED')
            and frame.ego_speed<=self.standstill_speed_mps
            and math.hypot(frame.ego_x-self.space.position_pose.x,frame.ego_y-self.space.position_pose.y)<=self.position_tolerance_m
            and abs(normalize_angle(frame.ego_heading-self.space.position_pose.body_heading_rad))<=self.heading_tolerance_rad)
        if initial_segment and not safety_override:
            self._select_segment(frame,proposal)
            feedback=None
        request = self.session.tick(frame, capabilities, feedback, safety_override)
        if request.status == "COMPLETED":
            return PolicyResult(request, False, request.reason_code, "COMPLETE")
        if request.status in ("BLOCKED", "SUSPENDED", "CANCELLED"):
            return PolicyResult(request, True, request.reason_code, request.status)
        reply = self.session.last_feedback
        stage = self.session.stage
        revision=request.revision
        parked = (self._arrived(reply, self.park_goal,frame)
                  and self._body_clear(reply.actual_pose, facts, observed_space)
                  and self._body_clear(GoalPose(frame.ego_x,frame.ego_y,frame.ego_heading),facts,observed_space))
        if stage == "APPROACH" and self._arrived(reply, self.space.approach_pose,frame):
            request = self.session.advance(frame, "POSITION", self._goal(self.space.position_pose), reason="APPROACH_ACTUALLY_STOPPED")
        elif stage == "POSITION" and self._arrived(reply, self.space.position_pose,frame):
            request = self.session.advance(frame, "REVERSE_ENTRY", self._goal(self.park_goal, -1, True),
                                           required=("PARK", "PATH", "REVERSE", "DWELL", "FEEDBACK"),
                                           reason="ENTRY_POSITION_ACTUALLY_STOPPED")
        elif stage in ("REVERSE_ENTRY", "ALIGN"):
            if parked:
                self._park_direction=self.session.goal.motion_direction
                request = self.session.advance(frame, "PARKED_DWELL", self._goal(self.park_goal,
                    self.session.goal.motion_direction, True),
                    required=('PARK','PATH','DWELL','FEEDBACK')+(('REVERSE',) if self._park_direction==-1 else ()),
                    reason="WHOLE_BODY_ACTUALLY_PARKED")
                self._segment_goal=None
            elif (stage=='ALIGN' and proposal is not None and not safety_override
                    and self._arrived(reply,self.session.goal.goal_pose,frame)
                    and reply.candidate_id==self._segment_id):
                request=self._select_segment(frame,proposal)
            elif (reply is not None and reply.producer in ("control", "runtime")
                  and reply.actual_standstill_confirmed and reply.progress >= 1.0 and stage == "REVERSE_ENTRY"):
                request = self.session.advance(frame, "ALIGN", self._goal(self.park_goal, -1, True), reason="PARKING_POSE_ALIGNMENT_REQUIRED")
        elif stage == "PARKED_DWELL":
            if parked and reply.hold_completed and reply.standstill_duration_s >= self.dwell_s:
                request = self.session.advance(frame, "EXIT_PREPARE", self._goal(self.park_goal,
                    self.session.goal.motion_direction), reason="TEN_SECOND_DWELL_ACTUALLY_COMPLETED")
        elif stage == "EXIT_PREPARE":
            if (parked and facts.exit_goal is not None
                    and self._body_clear(facts.exit_goal, facts)):
                request = self.session.advance(frame, "EXIT", self._goal(facts.exit_goal, 1), reason="STOPPED_AND_EXIT_GOAL_VERIFIED")
                self._exit_pose=copy.deepcopy(facts.exit_goal)
        elif stage == "EXIT" and self._arrived(reply, self.session.goal.goal_pose,frame):
            request = self.session.finish(frame, True, "PARKING_EXIT_ACTUALLY_COMPLETED")
        if request.status=='REQUESTED' and request.revision!=revision:
            request=self.session.tick(frame,capabilities,safety_override=safety_override)
        self._last_good_goal=copy.deepcopy(self.session.goal)
        return PolicyResult(request,request.stage in ('PARKED_DWELL','EXIT_PREPARE')
            or request.status in ('BLOCKED','SUSPENDED','CANCELLED'),request.reason_code,self.session.stage)
