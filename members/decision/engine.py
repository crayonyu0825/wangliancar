"""Decision facts -> route conflicts -> constraints -> behavior state -> target.

This member never reads the SDK or produces trajectory points or actuators.
The four public modes remain unchanged; state names below are private.
"""

import math
import time

from core.interfaces import DecisionMode, DecisionTarget
from core.geometry import opposes_direction
from core.traffic_quality import signal_stop_requirement, signal_stop_bound
from core.scene_requirements import requires_targets, FOLLOW_SCENES, AEB_SCENES

from members.decision import candidates, protocol, speed_policy
from members.decision.constraints import ConstraintSet
from members.decision.settings import DecisionSettings
from members.decision.target_history import TargetHistory
from members.decision.behaviors.coordinator import BehaviorCoordinator
from members.decision.behaviors.runtime_coordinator import RuntimeCoordinator
from members.decision.behaviors.parking_runtime import ParkingRuntime
from members.decision.behaviors.signal_runtime import SignalRuntime
from members.decision.behaviors.intersection_runtime import IntersectionRuntime
from members.decision.behaviors.emergency_runtime import EmergencyRuntime
from members.decision.behaviors.task_runtime import TaskRuntime
from members.decision.behaviors.blocked_runtime import BlockedRuntime


def _link_frame(perception, output):
    frame = getattr(perception, "frame_id", None)
    timestamp = getattr(perception, "timestamp", None)
    deadline = getattr(perception, "valid_until", None)
    output.frame_id = frame if type(frame) is int else -1
    output.timestamp = timestamp if type(timestamp) is int else 0
    output.valid_until = deadline if type(deadline) in (int, float) else 0.0


class DecisionEngine(object):
    def __init__(self, settings=None, clock=None):
        self._clock = clock or time.monotonic
        self.settings = (settings if settings is not None
                         else DecisionSettings.from_environment()).validate()
        self.reset()

    def reset(self, preserve_runtime=False):
        self._history = TargetHistory(self.settings)
        if not preserve_runtime or not hasattr(self, "_runtime_behavior"):
            self._runtime_behavior = RuntimeCoordinator(self.settings)
            self._parking_runtime = ParkingRuntime()
            self._signal_runtime = SignalRuntime()
            self._intersection_runtime = IntersectionRuntime()
            self._emergency_runtime = EmergencyRuntime()
            self._task_runtime = TaskRuntime()
            self._blocked_runtime = BlockedRuntime()
        if hasattr(self, "_behavior"):
            self._behavior.reset()
        else:
            self._behavior = BehaviorCoordinator(self.settings)
        self._last_constraints = None
        self._blind_fault_count = 0
        self._blind_stop = False
        self._recovery_count = 0
        self._fault_state = "NORMAL"
        self._normal_state = "CRUISE"
        self._clear_count = 0
        self._static_targets = {}
        self._moving_confirm = {}
        self._follow_targets = set()
        self._emergency_static_targets = set()
        self._had_target_conflict = False
        self._target_clear_count = 0
        self._target_hazard_ids = set()
        self._context = None
        self._last_frame = None

    def _observe_frame(self, perception):
        context = (getattr(perception, "case_id", ""),
                   getattr(perception, "task_id", ""),
                   getattr(perception, "scene_id", 0), getattr(perception,"session_id",""))
        frame = (getattr(perception, "frame_id", -1),
                 getattr(perception, "timestamp", 0))
        frame_id = frame[0]
        rolled_back = (self._last_frame is not None
                       and type(frame_id) is int and frame_id >= 0
                       and type(self._last_frame[0]) is int
                       and frame_id < self._last_frame[0])
        trusted = protocol.perception_usable(perception)
        if self._context is None:
            self._context = context
        if (context != self._context or rolled_back) and trusted:
            # An active parking/signal obligation may precede any lane intent.
            # The trusted run identity, rather than one member's local state,
            # owns whether a frame rollback is a new task.
            preserve = (bool(getattr(perception,"session_id","")) and self._context is not None
                        and perception.session_id == self._context[-1])
            self.reset(preserve_runtime=preserve)
            self._context = context
        distinct = self._last_frame is None or frame_id != self._last_frame[0]
        if trusted:
            self._last_frame = frame
        return distinct

    def run(self, perception):
        output = DecisionTarget()
        try:
            _link_frame(perception, output)
            distinct = self._observe_frame(perception)
            self._last_constraints = None
            parking=getattr(getattr(perception,'behavior_environment',None),'parking_task',None)
            handled=False
            emergency_task=getattr(getattr(perception,'behavior_environment',None),'emergency_task',None)
            interrupting_emergency=(self._emergency_runtime.active() or emergency_task is not None and emergency_task.dangerous)
            if parking is None and self._parking_runtime.active():
                self._protect(output,perception.ego.speed,'PARKING_ENVIRONMENT_LOST',force_emergency=True)
                handled=True
            if parking is not None and not interrupting_emergency and protocol.perception_usable(perception) and protocol.targets_usable(perception):
                output.valid=True;output.mode=DecisionMode.STOP;output.stop_distance=0.0
                output.target_lane_id=perception.lane.lane_id
                signed=perception.ego.vx*math.cos(perception.ego.heading)+perception.ego.vy*math.sin(perception.ego.heading)
                authorized_reverse=(self._parking_runtime.phase.startswith('EXECUTE') and self._parking_runtime.directions
                    and self._parking_runtime.directions[self._parking_runtime.index]==-1)
                if signed < -0.05 and not authorized_reverse:
                    self._protect(output,perception.ego.speed,'UNAUTHORIZED_PARKING_REVERSE',force_emergency=True)
                    handled=True
                else:handled=self._parking_runtime.update(perception,output)
            if not handled:self._arbitrate(perception, output, distinct)
            if output.valid and protocol.perception_usable(perception):
                if interrupting_emergency:
                    self._parking_runtime.suspend_for_emergency(perception)
                    self._runtime_behavior.suspend_for_emergency(perception)
                emergency_handled=self._emergency_runtime.update(perception,output)
                if not handled and not emergency_handled:
                    self._signal_runtime.update(perception,output,self._behavior._signal_observation(perception,output))
                    junction_handled=False
                    if output.behavior_request is None:
                        junction_handled=self._intersection_runtime.update(perception,output)
                    if output.behavior_request is None and not junction_handled:
                        self._runtime_behavior.update(perception,output,self._last_constraints,self._history,self._clock())
                if not handled and not emergency_handled:
                    self._blocked_runtime.update(perception,output,self._last_constraints)
                completed=[]
                if self._signal_runtime.policy is not None:
                    completed.extend(self._signal_runtime.policy._completed)
                if self._parking_runtime.phase=='COMPLETE' and self._parking_runtime.task is not None:
                    completed.append('parking:'+self._parking_runtime.task.identity)
                temporary=(self._parking_runtime.active() or self._emergency_runtime.active()
                    or self._intersection_runtime.active() or self._runtime_behavior.phase not in ('IDLE','COMPLETE')
                    or output.behavior_request is not None and output.behavior_request.maneuver in ('SIGNAL_STOP','BLOCKED_ROAD'))
                self._task_runtime.update(perception,output,temporary,completed)
                try:
                    self._behavior.observe_legacy(perception, output, self._last_constraints, self._clock())
                except (ValueError, TypeError, AttributeError, KeyError) as exc:
                    self._behavior.diagnostic_error(output.frame_id, type(exc).__name__)
        except Exception as exc:
            output.valid = False
            output.mode = DecisionMode.STOP
            output.target_speed = 0.0
            output.stop_distance = -1.0
            output.reason = "DECISION_EXCEPTION:" + type(exc).__name__ + ":" + str(exc)
            output.errors.append("DECISION_EXCEPTION:" + type(exc).__name__)
        return output

    def behavior_diagnostics(self):
        result = self._behavior.snapshot()
        result["target_history"] = self._history.snapshot()
        parking=self._parking_runtime
        result['parking_runtime']=dict(phase=parking.phase,reason_code=parking.reason,segment_index=parking.index,
                                     intent_id=parking.session.intent_id,revision=parking.session.revision)
        task=self._task_runtime
        result['task_runtime']=dict(phase=task.phase,reason_code=task.reason,recovery_anchor=task.anchor,completed_obligations=sorted(task.completed_obligations))
        emergency=self._emergency_runtime
        result['emergency_runtime']=dict(phase=emergency.phase,reason_code=emergency.reason,selected=None if emergency.selected is None else emergency.selected.candidate_action)
        junction=self._intersection_runtime
        result['intersection_runtime']=dict(phase=junction.phase,reason_code=junction.reason,
            condition_at=junction.condition_at,authorized_at=junction.authorized_at,moved_at=junction.moved_at)
        behavior=self._runtime_behavior
        result["runtime_behavior"]=dict(phase=behavior.phase,reason_code=behavior.reason,
            intent_id=behavior.session.intent_id,stage=behavior.session.stage,revision=behavior.session.revision,
            events=list(behavior.events))
        return result

    def _source_reason(self, perception):
        if getattr(perception, "target_source", "") == "ground_truth":
            return "TARGET_SOURCE:ground_truth_only"
        statuses = getattr(perception, "source_status", {})
        status = statuses.get("targets", {}) if isinstance(statuses, dict) else {}
        if not isinstance(status, dict):
            return "TARGET_SOURCE:metadata_invalid"
        if status.get("sensor_presence") == "not_configured":
            return "TARGET_SOURCE:not_configured"
        if status.get("sensor_read_ok") is False:
            return "TARGET_SOURCE:sensor_read_failed"
        quality = status.get("quality", "unknown")
        if quality in ("stale", "unsynchronized", "regressed"):
            return "TARGET_SOURCE:" + quality
        if quality == "invalid" or status.get("data_invalid") is True:
            return "TARGET_SOURCE:record_invalid"
        return "TARGET_SOURCE:unusable,presence={0},quality={1}".format(
            status.get("sensor_presence", "unknown"), quality)

    def _protect(self, output, ego_speed, reason, known_current=False,
                 force_emergency=False):
        moving = force_emergency or ego_speed > self.settings.standstill_speed
        output.mode = DecisionMode.EMERGENCY_BRAKE if moving else DecisionMode.STOP
        output.target_speed = 0.0
        output.stop_distance = 0.0 if known_current or not moving else -1.0
        output.reason = reason
        output.valid = True
        self._normal_state = ("HOLD" if known_current and not moving else
                              "YIELD" if not moving else "PROTECT")

    def _collect(self, perception, route, ego_speed, cruising, distinct):
        result = ConstraintSet(cruising)
        settings = self.settings
        signal_stop = signal_stop_requirement(perception)
        if signal_stop is not None:
            signal_ids = signal_stop['signal_ids']
            signal_id = signal_ids[0] if len(signal_ids) == 1 else -1
            if settings.front_offset_m is None:
                result.add("PROTECT", "traffic", signal_id,
                           "TRAFFIC_DISTANCE:vehicle_front_unknown",
                           valid_until=perception.valid_until)
            else:
                distance = signal_stop_bound(perception, settings.front_offset_m,
                                             settings.traffic_stop_margin)
                if distance is None:
                    result.add("PROTECT", "traffic", signal_id,
                               "TRAFFIC_DISTANCE:stop_line_unknown",
                               valid_until=perception.valid_until)
                else:
                    result.add("STOP", "traffic", signal_id,
                               "STOP_TRAFFIC:id={0},distance={1:.2f}".format(
                                   signal_id, distance), distance=distance,
                               valid_until=perception.valid_until)
        saw_target_conflict = False
        hazard_ids, confirmed_clear_ids = set(), set()
        result.obstacle_clearance_m = settings.obstacle_stop_margin
        seen_ids = set()
        for target in perception.targets if protocol.targets_usable(perception) else []:
            item = candidates.build_candidate(target, perception.ego,
                                              perception, route, settings)
            self._history.observe(perception, item)
            if not item.conflict:
                continue
            identifier = getattr(target, "id", -1)
            if self._history.discontinuous(identifier):
                self._follow_targets.discard(identifier)
                self._static_targets.pop(identifier, None)
                self._moving_confirm.pop(identifier, None)
            if type(identifier) is int and identifier >= 0:
                seen_ids.add(identifier)
            if item.coverage_stop_distance is not None:
                saw_target_conflict = True
                hazard_ids.add(identifier)
                result.add("STOP", "target", identifier,
                    "STOP_ROUTE_COVERAGE:id={0},distance={1:.2f}".format(
                        identifier, item.coverage_stop_distance),
                    distance=item.coverage_stop_distance,
                    valid_until=perception.valid_until)
                continue
            if item.gap is not None and speed_policy.is_emergency(item, ego_speed, settings):
                saw_target_conflict = True
                hazard_ids.add(identifier)
                if item.static and type(identifier) is int and identifier >= 0:
                    self._emergency_static_targets.add(identifier)
                result.add("EMERGENCY", "target", identifier,
                           "EMERGENCY_TARGET:id={0},gap={1:.2f},ttc={2:.2f}".format(
                               identifier, item.gap, item.ttc),
                           valid_until=perception.valid_until)
                continue
            if item.reason or item.gap is None:
                saw_target_conflict = True
                hazard_ids.add(identifier)
                result.add("PROTECT", "target", identifier,
                           "TARGET_UNKNOWN:id={0},{1}".format(
                               identifier, item.reason or "gap_unknown"),
                           valid_until=perception.valid_until)
                continue
            if (item.relation not in (candidates.CURRENT_LANE,
                                      candidates.FORWARD_ROUTE)
                    or not item.motion_supported):
                if (not item.motion_relevant and item.stop_distance is not None
                        and item.relation in (candidates.CURRENT_LANE, candidates.FORWARD_ROUTE)):
                    result.add("STOP", "target", identifier,
                               "TARGET_MOTION_AHEAD:id={0},distance={1:.2f}".format(
                                   identifier, item.stop_distance), distance=item.stop_distance,
                               valid_until=perception.valid_until)
                    continue
                saw_target_conflict = True
                hazard_ids.add(identifier)
                result.add("PROTECT", "target", identifier,
                           "TARGET_CONFLICT:id={0},relation={1}".format(
                               identifier, item.relation),
                           valid_until=perception.valid_until)
                continue
            static = item.static
            follow_context = (perception.scene_id in FOLLOW_SCENES
                              or ((identifier in self._follow_targets
                                   or self._history.was_followed(identifier))
                                  and perception.scene_id not in AEB_SCENES))
            if not static and perception.scene_id not in AEB_SCENES:
                self._follow_targets.add(identifier)
                follow_context = True
            gap = settings.min_gap if follow_context or not static else settings.obstacle_stop_margin
            if type(identifier) is int and identifier >= 0:
                result.obstacle_clearances_m[str(identifier)] = gap
            else:
                result.obstacle_clearance_m = max(result.obstacle_clearance_m, gap)
            if type(identifier) is int and identifier >= 0:
                if static:
                    self._static_targets[identifier] = True
                    self._moving_confirm[identifier] = 0
                elif self._static_targets.get(identifier):
                    if distinct:
                        self._moving_confirm[identifier] = (
                            self._moving_confirm.get(identifier, 0) + 1)
                    if self._moving_confirm[identifier] < settings.release_frames:
                        static = True
                    else:
                        self._static_targets.pop(identifier, None)
                        self._moving_confirm.pop(identifier, None)
                        confirmed_clear_ids.add(identifier)
            if static:
                saw_target_conflict = True
                hazard_ids.add(identifier)
                stop_distance = (max(0.0, item.gap - settings.min_gap)
                                 if follow_context else item.stop_distance)
                if (identifier in self._emergency_static_targets
                        and stop_distance is not None
                        and stop_distance <= settings.resume_margin):
                    # A near static target that already caused emergency
                    # braking must not launch another approach at standstill.
                    stop_distance = 0.0
                else:
                    self._emergency_static_targets.discard(identifier)
                if item.stop_distance is None:
                    result.add("PROTECT", "target", identifier,
                               "TARGET_STOP:distance_unknown",
                               valid_until=perception.valid_until)
                else:
                    result.add("STOP", "target", identifier,
                               "STOP_TARGET:id={0},distance={1:.2f}".format(
                                   identifier, stop_distance),
                               distance=stop_distance,
                               valid_until=perception.valid_until)
                    if not follow_context and stop_distance is not None:
                        result.blockage_candidates.append((identifier, stop_distance))
            else:
                self._emergency_static_targets.discard(identifier)
                demand = speed_policy.follow_speed(item, ego_speed, cruising, settings)
                self._history.followed(identifier, demand)
                result.add("FOLLOW", "target", identifier,
                           "FOLLOW_TARGET:id={0},gap={1:.2f},speed={2:.2f}".format(
                               identifier, item.gap, demand), speed=demand,
                           valid_until=perception.valid_until)
        for identifier in list(self._static_targets):
            if identifier not in seen_ids:
                self._static_targets.pop(identifier, None)
                self._moving_confirm.pop(identifier, None)
        self._follow_targets.intersection_update(seen_ids)
        self._emergency_static_targets.intersection_update(seen_ids)
        if saw_target_conflict:
            self._had_target_conflict = True
            self._target_clear_count = 0
            self._target_hazard_ids.update(hazard_ids)
        elif self._had_target_conflict:
            # Confirmed movement of the actual old static hazard already has
            # its own fresh-frame gate. A different safe lead is not evidence
            # that a disappeared hazard is clear.
            if self._target_hazard_ids and self._target_hazard_ids <= confirmed_clear_ids:
                self._target_clear_count = settings.release_frames
            elif distinct:
                self._target_clear_count += 1
            if self._target_clear_count < settings.release_frames:
                result.add("PROTECT", "target", -1,
                           "TARGET_CLEAR:confirmation_pending",
                           valid_until=perception.valid_until)
            else:
                self._had_target_conflict = False
                self._target_clear_count = 0
                self._static_targets.clear()
                self._moving_confirm.clear()
                self._target_hazard_ids.clear()
        memories=(self._history.retained_constraints(ego_speed,cruising)
                  +self._history.missing_danger_constraints(perception))
        for kind, identifier, demand, reason in memories:
            result.add(kind, "target_history", identifier,
                       "{0}:id={1}".format(reason, identifier), speed=demand,
                       valid_until=perception.valid_until)
        return result

    def _arbitrate(self, perception, output, distinct):
        settings = self.settings
        if not protocol.perception_usable(perception):
            self._blind_stop = True
            self._fault_state = "STOP_LATCHED"
            self._recovery_count = 0
            self._protect(output, float("inf"), "EGO_SOURCE:invalid_or_expired")
            return
        ego = perception.ego
        if not protocol.lane_usable(perception):
            self._protect(output, ego.speed, "LANE_SOURCE:unavailable")
            return
        route = candidates.RouteContext(perception, settings)
        if route.ego_s is None:
            self._protect(output, ego.speed, "ROUTE_GEOMETRY:" + route.reason)
            return
        signed_speed = (math.cos(route.ego_heading) * ego.vx
                        + math.sin(route.ego_heading) * ego.vy)
        if opposes_direction(ego.vx, ego.vy, route.ego_heading, ego.speed):
            self._protect(output, ego.speed, "EGO_MOTION:reverse_unsupported")
            return
        ego_speed = max(0.0, signed_speed)
        signal_stop = signal_stop_requirement(perception)
        if signal_stop is not None and signal_stop['distance'] < 0:
            reason = ('TRAFFIC_DISTANCE:stop_line_unknown' if signal_stop.get('state_known') is True
                      else 'TRAFFIC_SOURCE:'+signal_stop['reason'])
            self._protect(output, ego.speed, reason)
            return
        self._history.begin(perception, self._clock(), distinct)
        if self._history.reset_reason == "SOURCE_CHANGED":
            self._follow_targets.clear()
            self._static_targets.clear()
            self._moving_confirm.clear()
            self._emergency_static_targets.clear()
        targets_fresh = protocol.targets_usable(perception)
        required = requires_targets(perception.scene_id)
        if required and not targets_fresh:
            if distinct:
                self._blind_fault_count += 1
            self._recovery_count = 0
            self._static_targets.clear()
            self._moving_confirm.clear()
            self._fault_state = ("STOP_LATCHED" if self._blind_fault_count >= 2
                                 else "FAULT_PENDING")
            self._blind_stop = self._fault_state == "STOP_LATCHED"
            self._protect(output, ego.speed, self._source_reason(perception)
                          + ";state=" + self._fault_state)
            return
        self._blind_fault_count = 0
        cruising = speed_policy.desired_speed(perception, settings)
        constraints = self._collect(perception, route, ego_speed, cruising,
                                    distinct)
        self._last_constraints = constraints
        output.obstacle_clearance_m = constraints.obstacle_clearance_m
        output.obstacle_clearances_m = dict(constraints.obstacle_clearances_m)
        emergency = constraints.first("EMERGENCY")
        if emergency is not None:
            self._protect(output, ego.speed, constraints.reason(emergency), True,
                          force_emergency=True)
            return
        if self._blind_stop:
            self._fault_state = "RECOVERING"
            if ego.speed > settings.blind_speed_tolerance:
                self._recovery_count = 0
                self._protect(output, ego.speed, "TARGET_RECOVERY:vehicle_moving")
                return
            if distinct:
                self._recovery_count += 1
            if self._recovery_count < settings.recovery_frames:
                self._protect(output, ego.speed,
                              "TARGET_RECOVERY:good_frames={0}/{1}".format(
                                  self._recovery_count, settings.recovery_frames))
                return
            self._blind_stop = False
            self._recovery_count = 0
        self._fault_state = "NORMAL"
        protection = constraints.first("PROTECT")
        if protection is not None:
            self._protect(output, ego.speed, constraints.reason(protection))
            return
        if cruising <= 0.0:
            self._protect(output, ego.speed, "SPEED_LIMIT:zero")
            return
        stop = constraints.stop()
        if stop is not None:
            if (self._normal_state == "HOLD"
                    and stop.distance <= settings.resume_margin):
                self._clear_count = 0
                self._protect(output, ego.speed,
                              "HOLD_STOP:" + constraints.reason(stop), True)
                return
            if self._normal_state == "HOLD":
                if distinct:
                    self._clear_count += 1
                if self._clear_count < settings.release_frames:
                    self._protect(output, ego.speed,
                                  "HOLD_RELEASE:clear_frames={0}/{1}".format(
                                      self._clear_count, settings.release_frames), True)
                    return
            self._clear_count = 0
            if (stop.distance <= settings.hold_distance
                    and ego.speed <= settings.standstill_speed):
                self._protect(output, ego.speed, constraints.reason(stop), True)
                return
            if speed_policy.stop_is_infeasible(stop.distance, ego.speed, settings):
                self._protect(output, ego.speed,
                              "STOP_INFEASIBLE:" + constraints.reason(stop), True)
                return
        elif self._normal_state == "HOLD":
            if distinct:
                self._clear_count += 1
            if self._clear_count < settings.release_frames:
                self._protect(output, ego.speed,
                              "HOLD_RELEASE:clear_frames={0}/{1}".format(
                                  self._clear_count, settings.release_frames), True)
                return
            self._clear_count = 0
        else:
            self._clear_count = 0
        speed = constraints.speed()
        if stop is not None:
            speed = min(speed, speed_policy.approach_speed(
                stop.distance, ego.speed, cruising, settings))
            self._normal_state = "APPROACH_STOP"
        elif constraints.follows():
            self._normal_state = "FOLLOW"
        else:
            self._normal_state = "CRUISE"
        output.mode = DecisionMode.FOLLOW if constraints.follows() else DecisionMode.KEEP_LANE
        output.target_speed = float(speed)
        output.target_lane_id = perception.lane.lane_id
        output.stop_distance = -1.0 if stop is None else float(stop.distance)
        output.precision_stop = stop is not None
        primary = stop if stop is not None else constraints.first("FOLLOW")
        if primary is None:
            output.reason = "CRUISE:limit={0:.2f}".format(cruising)
        else:
            output.reason = constraints.reason(primary)
        output.reason += ";state=" + self._normal_state
        output.valid = True
