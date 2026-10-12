"""Bounded safety overrides for the captain's perception-to-control chain.

The supervisor is not a path tracker. It never commands propulsion, and a
candidate brake command is not proof that the simulator executed it.
"""

import math
import os
import time
from types import SimpleNamespace

from core.interfaces import ControlOut, DecisionMode
from core.scene_requirements import requires_targets
from core.traffic_quality import signal_stop_requirement, signal_stop_bound
from core.validation import current, number
from core.target_semantics import mapped_traffic_light
from core.route_obstacles import RouteContext, mapped_route_motion, route_footprint_relevant
from core.geometry import opposes_direction


class SafetyAssessment(object):
    def __init__(self, mode="normal", reason="", control=None):
        self.mode = mode
        self.reason = reason
        self.control = control

    @property
    def active(self):
        return self.mode != "normal"


class SafetySupervisor(object):
    """Classify missing inputs and prepare a brake-only candidate.

    A previous GPS header can only be reused for a short, explicit interval.
    SimOne may reject repeated headers; the sender must report its receipt.
    """

    def __init__(self, config, clock=None, planning_settings=None, decision_settings=None,parking_guard=None,
                 lane_change_guard=None, driving_guard=None, execution_guard=None):
        self.clock = clock or time.monotonic
        if parking_guard is not None and not callable(parking_guard):
            raise ValueError('invalid parking safety guard')
        self.parking_guard=parking_guard
        if any(v is not None and not callable(v) for v in (lane_change_guard, driving_guard, execution_guard)):
            raise ValueError('invalid maneuver safety guard')
        self.lane_change_guard, self.driving_guard = lane_change_guard, driving_guard
        self.execution_guard=execution_guard
        self.repeat_fault_ms = max(1, int(getattr(config, "pipeline_timeout_ms", 200)))
        self.last_gps_ms = max(1, int(getattr(config, "sensor_timeout_ms", 500)))
        self.recovery_frames = max(1, int(getattr(config, "safety_recovery_frames", 3)))
        self.controlled_brake = self._brake(getattr(config, "safety_controlled_brake", 0.3))
        self.emergency_brake = self._brake(getattr(config, "safety_emergency_brake", 1.0))
        self.normal_control_required = bool(getattr(config, "send_control", False))
        self.signal_stop_margin = getattr(planning_settings, 'traffic_stop_margin',
                                         getattr(decision_settings, 'traffic_stop_margin', .3))
        if not number(self.signal_stop_margin) or self.signal_stop_margin < 0:
            raise ValueError('invalid safety traffic stop margin')
        self.last_case = None
        self.last_frame = None
        self.last_frame_at = None
        self.last_good_header = None
        self.last_good_at = None
        self.recovering = False
        self.good_frames = 0
        # Runtime passes validated snapshots. Legacy callers can supply the
        # same measured dimensions through config or common environment names.
        values = {}
        for field, config_field, env, default in (
                ('front_offset_m', 'vehicle_front_offset_m', 'NEVC_VEHICLE_FRONT_OFFSET_M', None),
                ('half_width_m', 'vehicle_half_width_m', 'NEVC_VEHICLE_HALF_WIDTH_M', None),
                ('horizon', 'planning_horizon_m', 'NEVC_PLANNING_HORIZON_M', 60.0),
                ('deceleration', 'planning_deceleration_mps2', 'NEVC_PLANNING_DECELERATION_MPS2', 2.0),
                ('lateral_guard_time_s', 'planning_lateral_guard_time_s', 'NEVC_PLANNING_LATERAL_GUARD_TIME_S', 3.0),
                ('lateral_margin_m', 'planning_lateral_margin_m', 'NEVC_PLANNING_LATERAL_MARGIN_M', 0.0)):
            value = getattr(planning_settings, field, None)
            if value is None:
                value = getattr(config, config_field, None)
            if value is None:
                value = float(os.environ[env]) if env in os.environ else default
            if value is not None and (not number(value) or value < 0
                    or (field != 'lateral_margin_m' and value <= 0)):
                raise ValueError('invalid safety obstacle setting: '+field)
            values[field] = value
        self.collision_settings = SimpleNamespace(**values)
        self.route_settings = SimpleNamespace(**dict(
            (field, getattr(decision_settings, field, default)) for field, default in (
                ('projection_tolerance_m', 2.5), ('route_ambiguity_m', 2.0),
                ('emergency_ttc', 2.0), ('emergency_clearance', 2.0),
                ('standstill_speed', .1), ('static_speed_threshold', .3))))
        for field, value in vars(self.route_settings).items():
            if (not number(value) or value < 0
                    or (field not in ('standstill_speed', 'emergency_clearance') and value <= 0)):
                raise ValueError('invalid safety route setting: '+field)

    @staticmethod
    def _brake(value):
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("safety brake must be finite and in (0, 1]")
        return value

    def _signal_guard(self, perception, decision, trajectory):
        requirement = signal_stop_requirement(perception)
        if requirement is None:
            return ''
        front = self.collision_settings.front_offset_m
        # Legacy callers without body data can still verify the old rear-axle
        # bound; runtime always supplies the shared measured front offset.
        bound = signal_stop_bound(perception, front, self.signal_stop_margin if front is not None else 0.)
        if (bound is None or not current(decision, perception)
                or not current(trajectory, decision) or trajectory.stop_required is not True
                or not number(trajectory.stop_distance)
                or not 0 <= trajectory.stop_distance <= bound+1e-6):
            return ('required_traffic_unavailable' if bound is None or requirement.get('state_known') is not True
                    else 'traffic_stop_constraint_missing')
        if (requirement.get('state_known') is not True
                and (not number(decision.stop_distance) or decision.stop_distance < 0
                     or decision.stop_distance > bound+1e-6
                     or trajectory.stop_distance > decision.stop_distance+1e-6)):
            return 'required_traffic_unavailable'
        route = RouteContext(perception, self.route_settings)
        if route.ego_s is None or not trajectory.points:
            return 'traffic_trajectory_boundary_unverified'
        for point in trajectory.points:
            projection = route.project(point.x, point.y, self.route_settings)
            if projection is None or projection['s']-route.ego_s > bound+1e-6:
                return 'traffic_trajectory_crosses_boundary'
        return ''

    def reset(self, require_recovery=False):
        self.last_frame = None
        self.last_frame_at = None
        self.last_good_header = None
        self.last_good_at = None
        self.recovering = require_recovery
        self.good_frames = 0

    @staticmethod
    def _imminent_collision(perception, planning_settings=None, decision_settings=None):
        """Use only a fresh target stream for an independent brake override."""
        statuses = getattr(perception, "source_status", {})
        target_status = statuses.get("targets", {}) if isinstance(statuses, dict) else {}
        if (not getattr(perception, "targets_valid", False)
                or not getattr(perception, "target_source", "").startswith("sensor:")
                or not isinstance(target_status, dict)
                or target_status.get("usable") is not True):
            return False
        route = None
        if (planning_settings is not None and decision_settings is not None
                and number(planning_settings.front_offset_m)
                and number(planning_settings.half_width_m)
                and getattr(getattr(perception, 'lane', None), 'valid', False)):
            try:
                route = RouteContext(perception, decision_settings)
            except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
                route = None
        for target in getattr(perception, "targets", []):
            if mapped_traffic_light(target, perception):
                continue
            relevant = (getattr(target, "same_lane", False)
                        if getattr(target, "same_lane_valid", False)
                        else getattr(target, "lateral_band_match", False))
            distance = getattr(target, "longitudinal_distance", None)
            ttc = getattr(target, "ttc", None)
            if getattr(target, 'valid', False) and route is not None:
                try:
                    motion = mapped_route_motion(perception, target, route,
                        decision_settings, planning_settings.front_offset_m)
                except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
                    motion = None
                if motion is not None:
                    distance, ttc = motion['distance'], motion['ttc']
                    # A currently observed footprint ahead with signed road
                    # closing speed is still an independent collision check.
                    # This also retains real oncoming traffic on a bend.
                    if (distance > 0 and motion['gap'] <= decision_settings.emergency_clearance
                            and (perception.ego.speed > decision_settings.standstill_speed
                                 or motion['lead_speed'] < -decision_settings.static_speed_threshold)):
                        return True
                    relevant = True
                else:
                    try:
                        intersects = route_footprint_relevant(perception, target, route,
                                                             planning_settings)
                    except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
                        intersects = None
                    if intersects is False:
                        continue
                    if intersects is True:
                        relevant = True
            if (getattr(target, "valid", False) and relevant
                    and type(distance) in (int, float) and math.isfinite(distance)
                    and distance > 0 and type(ttc) in (int, float)
                    and math.isfinite(ttc) and 0 <= ttc <= (
                        decision_settings.emergency_ttc if decision_settings is not None else 2.0)):
                return True
        return False

    @staticmethod
    def _target_records_valid(perception):
        for target in getattr(perception, "targets", []):
            distance = getattr(target, "longitudinal_distance", None)
            ttc = getattr(target, "ttc", None)
            if (not getattr(target, "valid", False)
                    or type(distance) not in (int, float) or not math.isfinite(distance)
                    or type(ttc) not in (int, float) or not math.isfinite(ttc)):
                return False
        return True

    def evaluate(self, perception, decision, trajectory, control=None):
        now = self.clock()
        case = (getattr(perception, "case_id", ""), getattr(perception, "case_name", ""))
        if self.last_case is not None and case != self.last_case:
            self.reset(require_recovery=True)
        self.last_case = case
        frame = getattr(perception, "frame_id", -1)
        new_frame = frame != self.last_frame
        if new_frame:
            self.last_frame, self.last_frame_at = frame, now
        repeat_ms = ((now - self.last_frame_at) * 1000.0
                     if self.last_frame_at is not None else 0.0)
        source_status = getattr(perception, "source_status", {})
        if not isinstance(source_status, dict):
            source_status = {}
        gps_status = source_status.get("gps", {})
        gps_ok = (current(perception) and getattr(perception.ego, "valid", False)
                  and isinstance(gps_status, dict) and gps_status.get("usable") is True)
        if (gps_ok and new_frame and 0 <= perception.ego.age_ms < self.repeat_fault_ms):
            self.last_good_header = (perception.frame_id, perception.timestamp)
            self.last_good_at = now

        signal_fault = self._signal_guard(perception, decision, trajectory) if gps_ok else ''
        parking = self._parking_assessment(perception,decision,trajectory,control) if gps_ok else None
        maneuver = self._maneuver_assessment(perception,decision,trajectory,control) if gps_ok else None
        execution=None
        if gps_ok and self.execution_guard is not None:
            try:
                execution=self.execution_guard(perception,trajectory)
                if not isinstance(execution,SafetyAssessment):
                    raise ValueError('invalid execution assessment')
            except Exception as error:
                execution=SafetyAssessment('fault_stop','execution_monitor_failed:'+str(error))
        verified_motion = ((parking is not None and not parking.active)
                           or (maneuver is not None and not maneuver.active))
        mode, reason = "normal", ""
        if not gps_ok:
            mode, reason = "fault_stop", "gps_invalid_or_expired"
        elif (getattr(perception.ego, "age_ms", -1) >= self.repeat_fault_ms
              or repeat_ms >= self.repeat_fault_ms):
            mode, reason = "fault_stop", "gps_frame_stalled"
        elif parking is not None and parking.active:
            mode,reason=parking.mode,parking.reason
        elif execution is not None and execution.active:
            mode,reason=execution.mode,execution.reason
        elif maneuver is not None and maneuver.active:
            mode,reason=maneuver.mode,maneuver.reason
        elif not verified_motion and self._imminent_collision(perception, self.collision_settings, self.route_settings):
            mode, reason = "emergency_stop", "imminent_target_collision"
        elif not getattr(perception, "lane", None) or not perception.lane.valid:
            mode, reason = "controlled_stop", "lane_unavailable"
        elif (requires_targets(getattr(perception, "scene_id", 0))
              and (not getattr(perception, "targets_valid", False)
                   or not isinstance(source_status.get("targets"), dict)
                   or source_status["targets"].get("usable") is not True
                   or not getattr(perception, "target_source", "").startswith("sensor:")
                   or not self._target_records_valid(perception))):
            mode, reason = "controlled_stop", "required_targets_unavailable"
        elif signal_fault:
            mode, reason = 'controlled_stop', signal_fault
        elif (current(decision, perception)
              and decision.mode == DecisionMode.EMERGENCY_BRAKE
              and not (maneuver is not None and not maneuver.active
                       and isinstance(getattr(decision,'behavior_request',None),dict)
                       and decision.behavior_request.get('maneuver')=='EMERGENCY'
                       and control is not None and control.throttle==0 and control.brake>0)):
            mode, reason = "emergency_stop", "collision_or_emergency_request"
        elif not current(trajectory, decision):
            mode, reason = "fault_stop", "trajectory_invalid_or_expired"
        elif (trajectory.emergency_stop and not (maneuver is not None and not maneuver.active
              and isinstance(getattr(decision,'behavior_request',None),dict)
              and decision.behavior_request.get('maneuver')=='EMERGENCY'
              and control is not None and control.valid and control.throttle==0 and control.brake>0)):
            mode, reason = "emergency_stop", "trajectory_emergency_request"
        elif (self.normal_control_required and new_frame and control is not None
              and not current(control, trajectory)):
            mode, reason = "fault_stop", "controller_invalid_or_expired"
        # Independent IMU/radar/ultrasonic/environment errors are deliberately
        # absent from these gates. Only channels needed for this baseline stop it.
        if mode != "normal":
            self.recovering, self.good_frames = True, 0
        elif self.recovering:
            if new_frame:
                self.good_frames += 1
            if self.good_frames < self.recovery_frames:
                mode, reason = "controlled_stop", "recovery_waiting_for_fresh_frames"
            else:
                self.recovering, self.good_frames = False, 0
        control = self._candidate(mode, reason, now,
                                  perception.valid_until if gps_ok else None)
        request=getattr(decision,'behavior_request',None)
        if (control is not None and mode!='normal' and isinstance(request,dict)
                and request.get('maneuver')=='EMERGENCY'):
            control.hazard_signal=True
        return SafetyAssessment(mode, reason, control)

    def _maneuver_assessment(self,p,decision,trajectory,control):
        request=getattr(decision,'behavior_request',None)
        if not isinstance(request,dict) or request.get('maneuver')=='SIGNAL_STOP':
            if self.driving_guard is not None:
                try:
                    result=self.driving_guard(p,decision,trajectory,control)
                    if not isinstance(result,SafetyAssessment):raise ValueError('invalid ordinary guard')
                    return result
                except Exception as error:
                    return SafetyAssessment('fault_stop','ordinary_monitor_failed:'+str(error))
            return None
        action=request.get('maneuver')
        if action in ('LANE_CHANGE','OVERTAKE','AVOID','MERGE'):
            guard=self.lane_change_guard
        elif action in ('INTERSECTION','CROSSWALK','JUNCTION_ENTRY','EMERGENCY'):
            guard=self.driving_guard
        else:
            return None
        if guard is None:
            return SafetyAssessment('controlled_stop','maneuver_monitor_unavailable')
        try:
            result=guard(p,decision,trajectory,control)
            if not isinstance(result,SafetyAssessment) or result.mode not in (
                    'normal','controlled_stop','fault_stop','emergency_stop'):
                raise ValueError('invalid maneuver guard assessment')
            return result
        except Exception as error:
            return SafetyAssessment('fault_stop','maneuver_monitor_failed:'+str(error))

    def _parking_assessment(self,p,decision,trajectory,control):
        request=getattr(decision,'behavior_request',None)
        parking=isinstance(request,dict) and request.get('maneuver')=='PARK'
        if parking:
            if self.parking_guard is None:
                return SafetyAssessment('controlled_stop','parking_monitor_unavailable')
            try:
                result=self.parking_guard(p,decision,trajectory,control)
                if not isinstance(result,SafetyAssessment) or result.mode not in (
                        'normal','controlled_stop','fault_stop','emergency_stop'):
                    raise ValueError('invalid parking guard assessment')
                return result
            except Exception as error:
                return SafetyAssessment('fault_stop','parking_monitor_failed:'+str(error))
        if getattr(trajectory,'motion_direction',1)!=1:
            return SafetyAssessment('fault_stop','reverse_maneuver_unverified')
        if not all(number(v) for v in (p.ego.vx,p.ego.vy,p.ego.heading,p.ego.speed)):
            return SafetyAssessment('fault_stop','actual_velocity_unavailable')
        if opposes_direction(p.ego.vx,p.ego.vy,p.ego.heading,p.ego.speed):
            return SafetyAssessment('emergency_stop','actual_motion_opposes_forward_path')
        return None

    def _candidate(self, mode, reason, now, source_deadline):
        if mode == "normal" or self.last_good_header is None:
            return None
        if (self.last_good_at is None or
                (now - self.last_good_at) * 1000.0 > self.last_gps_ms):
            return None
        frame, timestamp = self.last_good_header
        if type(frame) is not int or frame < 0 or type(timestamp) is not int or timestamp < 0:
            return None
        output = ControlOut()
        output.frame_id, output.timestamp = frame, timestamp
        output.valid_until = min(
            now + min(self.repeat_fault_ms, 100) / 1000.0,
            self.last_good_at + self.last_gps_ms / 1000.0)
        if source_deadline is not None:
            output.valid_until = min(output.valid_until, source_deadline)
        if output.valid_until <= now:
            return None
        # Brake-only override: never reinterpret raw GPS gearbox positions as
        # control commands (GPS 2 would incorrectly command Reverse=2).
        output.gear = 0
        output.throttle = 0.0
        output.brake = (self.controlled_brake if mode == "controlled_stop"
                        else self.emergency_brake)
        output.steering = 0.0
        output.source = "safety:" + mode + ":" + reason
        output.valid = True
        output.clamp()
        return output
