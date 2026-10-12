"""Bidirectional Pure Pursuit + PI/profile-feedforward trajectory execution.

The engine computes spatial diagnostics before calibration, but never marks a
pedal/steering command valid until vehicle-specific mapping is supplied.
"""

import math
import time

from core.interfaces import ControlOut, Trajectory
from core.geometry import normalize_angle, opposes_direction
from core.validation import current, validate_output
from members.control.lateral import geometry_request, steering_request
from members.control.longitudinal import pedal_request, applied_integral
from members.control.parameters import ControllerSettings
from members.control.trajectory import PreparedPath
from members.control.maneuvers import DirectionInterlock, StandstillGuard


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _approach(previous, desired, amount):
    return max(previous - amount, min(previous + amount, desired))


def _stationary_stop(trajectory):
    if not trajectory.stop_required or trajectory.target_speed != 0:
        return False
    origin = trajectory.points[0]
    return all(point.speed == 0 and
               math.hypot(point.x - origin.x, point.y - origin.y) <= 1e-6
               for point in trajectory.points)


class ControlEngine(object):
    def __init__(self, calibration=None, settings=None, clock=None):
        self.calibration = calibration
        self.settings = (settings or ControllerSettings()).validate()
        self.clock = clock or time.monotonic
        self.reset()

    def reset(self):
        self._execution_progress = {}
        self.case = None
        self.last_frame = None
        self.last_time = None
        self.last_output = None
        self.integral = 0.0
        self.state = "INIT"
        self.release_frames = 0
        self.direction = DirectionInterlock()
        self.standstill = StandstillGuard()
        self._forget_projection()

    def _forget_projection(self):
        self.path_signature = self.path_progress = None
        self.path_speed = 0.0

    def _invalid(self, output, reason):
        self.integral = 0.0
        self.release_frames = 0
        self.last_output = None
        self.state = "FAULT_STOP"
        self.direction.disarm()
        self.standstill.interrupt()
        self._forget_projection()
        output.throttle = output.brake = output.steering = 0.0
        output.source = "control:FAULT_STOP"
        output.errors.append(reason)
        output.diagnostics["state"] = self.state
        return output

    def _braking_output(self, output, ego, state, gear=0, handbrake=False):
        if self.calibration is None:
            return self._invalid(output, "vehicle calibration required")
        self.integral = 0.0
        if state != "HOLD":
            self.release_frames = 0
        self.state = state
        if state in ("GEAR_STOP", "GEAR_ENGAGE"):
            self._forget_projection()
        output.gear = gear
        output.throttle = output.steering = 0.0
        output.brake = (self.calibration.hold_brake
                        if ego.speed <= self.settings.gear_standstill_speed_mps
                        else self.calibration.max_brake)
        output.handbrake = handbrake and ego.speed <= self.settings.gear_standstill_speed_mps
        output.diagnostics.update({"state": state, "reference_speed_mps": 0.0,
                                   "speed_error_mps": -ego.speed,
                                   "hold_remaining_s": max(0.0, self.standstill.duration -
                                                            self.standstill.elapsed)})
        output.source, output.valid = "control:" + state, True
        output.clamp()
        self.last_output = output
        return output

    def _hold_output(self, output, ego, trajectory, arrived):
        output.diagnostics["stop_pose_arrived"] = arrived
        if arrived and ego.speed <= self.settings.gear_standstill_speed_mps:
            self.standstill.begin(trajectory, ego)
        if self.standstill.active:
            self.direction.disarm()
            return self._braking_output(output, ego, "DWELL", 0,
                                        self.standstill.parking_brake)
        gear = (1 if self.direction.active_direction == 1 else
                2 if self.direction.active_direction == -1 else 0)
        return self._braking_output(output, ego, "HOLD", gear,
                                    getattr(trajectory, "parking_brake_at_stop", False))

    def execution_feedback(self, perception, trajectory, output):
        from core.behavior_protocol import RuntimeRequest, ExecutionFeedback, GoalPose, behavior_frame
        request = trajectory.behavior_request
        if not isinstance(request, RuntimeRequest) or not output.valid:
            return None
        if (request.authorization == "EXECUTE_SELECTED" and
                trajectory.execution_path_id != request.candidate_reference.candidate_id):
            return None
        observed = self.clock()
        frame = behavior_frame(perception, observed)
        ego = perception.ego
        pose = GoalPose(ego.x, ego.y, ego.heading)
        goal = trajectory.goal_pose or request.goal.goal_pose
        arrived = (goal is not None and math.hypot(ego.x-goal.x, ego.y-goal.y)
                   <= self.settings.precision_arrival_tolerance_m
                   and abs(normalize_angle(ego.heading-goal.body_heading_rad))
                   <= self.settings.precision_heading_tolerance_rad)
        stopped = ego.speed <= self.settings.gear_standstill_speed_mps
        hold_complete = (arrived and stopped and self.standstill.anchor is not None
                         and self.standstill.elapsed >= request.goal.minimum_standstill_duration_s)
        if not hasattr(self, "_execution_progress"):
            self._execution_progress = {}
        identity = (frame.context.key(), request.intent_id, request.stage, request.revision)
        previous = self._execution_progress.get(identity)
        distance = 0.0 if previous is None else previous[2]
        if previous is not None and frame.frame_id > previous[0]:
            distance += math.hypot(pose.x-previous[1].x, pose.y-previous[1].y)
        if previous is None or frame.frame_id > previous[0]:
            self._execution_progress[identity] = (frame.frame_id, pose, distance)
        while len(self._execution_progress) > 16:
            self._execution_progress.pop(next(iter(self._execution_progress)))
        return ExecutionFeedback(frame.context, request.intent_id, request.stage, request.revision,
            "control", request.source_frame_id, frame.frame_id, frame.observed_at_s, frame.valid_until_s,
            "COMPLETED" if hold_complete else "ARRIVED" if arrived and stopped else "EXECUTING",
            usable=True, actual_pose=pose, actual_lane_id=perception.lane.lane_id,
            actual_standstill_confirmed=stopped, goal_pose_arrived=arrived,
            standstill_duration_s=self.standstill.elapsed if arrived and stopped else 0.0,
            hold_completed=hold_complete, motion_direction=trajectory.motion_direction,
            candidate_id=trajectory.execution_path_id,
            progress=1.0 if arrived else 0.0, actual_progress_m=distance,actual_speed_mps=ego.speed,
            control_limited=(self.calibration is not None and (
                output.throttle>=self.calibration.max_throttle-1e-6
                or output.brake>=self.calibration.max_brake-1e-6 or abs(output.steering)>=1-1e-6)),
            evidence_level="observed_pose_and_speed")

    def compute(self, perception, trajectory):
        output = ControlOut()
        output.source = "control:INIT"
        output.diagnostics = {}
        now = self.clock()
        try:
            output.bind(trajectory)
            case = (perception.case_id, perception.case_name,
                    getattr(perception, "task_id", ""), perception.scene_id)
            case_changed = self.case != case
            if type(perception.frame_id) is not int or perception.frame_id < 0:
                return self._invalid(output, "non-increasing perception frame")
            frame_rollback = (not case_changed and self.last_frame is not None
                              and perception.frame_id < self.last_frame)
            dt = 0.0 if case_changed or self.last_time is None else now - self.last_time
            if not case_changed and self.last_time is not None and (dt < 0 or dt > self.settings.max_dt_s or
                    (dt == 0 and perception.frame_id != self.last_frame)):
                self.last_frame, self.last_time = perception.frame_id, now
                return self._invalid(output, "control loop dt outside bound")
            if not current(perception) or not perception.ego.valid:
                raise ValueError("ego perception invalid or expired")
            gps_status = perception.source_status.get("gps", {})
            if not isinstance(gps_status, dict) or gps_status.get("usable") is not True:
                raise ValueError("GPS source unusable")
            ego = perception.ego
            if (type(ego.frame_id) is not int or ego.frame_id != perception.frame_id or
                    type(ego.age_ms) is not int or ego.age_ms < 0 or
                    not all(_finite(getattr(ego, name)) for name in
                            ("x", "y", "heading", "speed")) or ego.speed < 0):
                raise ValueError("ego pose or speed invalid")
            if type(ego.gear) is not int:
                raise ValueError("invalid GPS gear position")
            validate_output(trajectory, Trajectory, perception)
            if case_changed:
                # Invalid packets with a changed task/case label cannot erase
                # an already accepted minimum standstill obligation.
                self.reset()
                self.case = case
            elif frame_rollback:
                # Reject a trusted discontinuity and re-arm from subsequent
                # fresh frames, without waiting for the old high-water mark.
                # An unfinished dwell survives a same-context discontinuity;
                # only an explicit run/context reset can erase that obligation.
                standstill = self.standstill
                self.reset()
                self.standstill = standstill
                self.case = case
                self.last_frame, self.last_time = perception.frame_id, now
                return self._invalid(output, "non-increasing perception frame")
            if perception.frame_id == self.last_frame:
                # Runtime must not resend a normal command on a repeated GPS
                # frame. Keep PI/slew/HOLD memory until a genuinely new frame;
                # expiry and stalling remain checked above and by supervisor.
                output.source = "control:DUPLICATE"
                output.errors.append("duplicate perception frame; no new command")
                output.diagnostics["state"] = self.state
                return output
            self.last_frame, self.last_time = perception.frame_id, now
            # A raw gearbox position cannot be used as an N/D/R/P command.
            # Brake-only output uses Neutral; forward tracking requests Drive.
            output.gear = 0
            output.diagnostics["dt_s"] = dt
            direction = getattr(trajectory, "motion_direction", 1)
            precision = getattr(trajectory, "precision_stop", False)
            for name in ("left_signal", "right_signal", "hazard_signal"):
                setattr(output, name, getattr(trajectory, name, False))
            output.diagnostics["motion_direction"] = direction
            if trajectory.emergency_stop:
                self.integral = 0.0
                self.release_frames = 0
                self.state = "EMERGENCY"
                self.direction.disarm()
                self.standstill.interrupt()
                self._forget_projection()
                output.diagnostics["state"] = self.state
                if self.calibration is None:
                    return self._invalid(output, "vehicle calibration required")
                output.brake = self.calibration.emergency_brake
                output.source = "control:EMERGENCY"
                output.valid = True
                output.clamp()
                self.last_output = output
                return output

            if not all(_finite(getattr(ego, name)) for name in ("vx", "vy")):
                raise ValueError("reverse or invalid velocity")
            signed_speed = ego.vx * math.cos(ego.heading) + ego.vy * math.sin(ego.heading)
            vector_speed = math.hypot(ego.vx, ego.vy)
            if (vector_speed > ego.speed + self.settings.gear_standstill_speed_mps or
                    ((direction == -1 or self.direction.pending_direction is not None) and
                     abs(vector_speed - ego.speed) > self.settings.gear_standstill_speed_mps)):
                raise ValueError("reverse or invalid velocity magnitude disagrees with ego speed")
            if self.standstill.update(ego, dt, self.settings):
                return self._braking_output(output, ego, "DWELL", 0,
                                            self.standstill.parking_brake)
            if (direction == self.direction.last_direction and
                    self.direction.pending_direction is None and
                    opposes_direction(ego.vx, ego.vy, ego.heading, ego.speed, direction,
                                      self.settings.gear_standstill_speed_mps)):
                raise ValueError("reverse or invalid velocity opposes requested motion")
            output.gear = 1 if direction == 1 else 2
            output.diagnostics["observed_gear"] = ego.gear
            output.diagnostics["requested_gear"] = output.gear
            output.diagnostics["signed_body_speed_mps"] = signed_speed
            if _stationary_stop(trajectory):
                self._forget_projection()
                origin = trajectory.points[0]
                if math.hypot(ego.x - origin.x, ego.y - origin.y) > self.settings.max_projection_error_m:
                    raise ValueError("stationary stop is too far from vehicle")
                if self.calibration is None:
                    return self._invalid(output, "vehicle calibration required")
                self.release_frames = 0
                if ego.speed <= self.settings.hold_speed_mps:
                    aligned = (not precision or abs(normalize_angle(
                        ego.heading - trajectory.points[-1].heading)) <=
                        self.settings.precision_heading_tolerance_rad)
                    return self._hold_output(output, ego, trajectory,
                        math.hypot(ego.x - origin.x, ego.y - origin.y) <=
                        self.settings.precision_arrival_tolerance_m and aligned)
                return self._braking_output(output, ego, "STOPPING")

            gear_stage = self.direction.update(direction, ego.speed, now, self.settings)
            if gear_stage is not None:
                return self._braking_output(output, ego, gear_stage[0], gear_stage[1])
            if opposes_direction(ego.vx, ego.vy, ego.heading, ego.speed, direction,
                                  self.settings.gear_standstill_speed_mps):
                raise ValueError("velocity opposes engaged trajectory direction")

            path = PreparedPath(trajectory.points, self.settings.max_segment_m,
                                self.settings.curve_window_m)
            signature = (direction, tuple(point[:2] for point in path.points))
            hint = self.path_progress if signature == self.path_signature else None
            delta = (max(ego.speed, self.path_speed)*dt +
                     self.settings.projection_progress_slack_m)
            progress, cross_track, heading_error = path.project(
                ego.x, ego.y, ego.heading + (math.pi if direction == -1 else 0.0),
                self.settings,
                self.settings.precision_arrival_tolerance_m if precision else 0.0,
                hint, delta)
            self.path_signature, self.path_progress, self.path_speed = signature, progress, ego.speed
            reference, remaining = path.speed_reference(progress, trajectory,
                                                        self.settings)
            curve_limit = path.curve_speed_limit(progress, self.settings)
            reference = min(reference, curve_limit)
            if trajectory.stop_required and trajectory.target_speed == 0:
                reference = min(reference, ego.speed)
                self.integral = min(0.0, self.integral)
            output.diagnostics.update({
                "path_progress_m": progress, "path_remaining_m": remaining,
                "cross_track_abs_m": cross_track, "heading_error_rad": heading_error,
                "projection_continuity_used": hint is not None,
                "curve_speed_limit_mps": curve_limit,
                "reference_speed_mps": reference,
                "speed_error_mps": reference - ego.speed})
            if precision:
                stop_heading = path.heading_at(min(path.length, progress + trajectory.stop_distance))
                stop_heading_error = normalize_angle(stop_heading - ego.heading)
                output.diagnostics["stop_heading_error_rad"] = stop_heading_error
            crawl_intent = (trajectory.target_speed > 0 and
                            0 < reference <= self.settings.hold_release_speed_mps)
            hold_speed = (self.settings.precision_speed_mps if precision or direction == -1 or crawl_intent
                          else self.settings.hold_speed_mps)
            hold_request = reference <= hold_speed and ego.speed <= hold_speed
            # At a finite path end there may be no valid forward carrot left,
            # but braking must remain executable while the vehicle is moving.
            if reference <= hold_speed and not hold_request:
                if self.calibration is None:
                    return self._invalid(output, "vehicle calibration required")
                self.integral, self.release_frames = 0.0, 0
                self.state = "STOPPING"
                output.brake = self.calibration.max_brake
                output.steering = (self.last_output.steering if self.last_output else 0.0)
                output.diagnostics["state"] = self.state
                output.source, output.valid = "control:STOPPING", True
                output.clamp()
                self.last_output = output
                return output
            if hold_request or self.state in ("HOLD", "DWELL"):
                release_speed = (self.settings.precision_speed_mps
                                 if precision or direction == -1 or crawl_intent
                                 else self.settings.hold_release_speed_mps)
                if (not hold_request and trajectory.target_speed > 0 and
                        reference >= release_speed):
                    self.release_frames += 1
                else:
                    self.release_frames = 0
                if hold_request or self.release_frames < self.settings.hold_release_frames:
                    arrived = (trajectory.stop_required and
                               0 <= trajectory.stop_distance <=
                               self.settings.precision_arrival_tolerance_m)
                    if precision:
                        arrived = (arrived and cross_track <=
                                   self.settings.precision_arrival_tolerance_m and
                                   abs(heading_error) <=
                                   self.settings.precision_heading_tolerance_rad and
                                   abs(stop_heading_error) <=
                                   self.settings.precision_heading_tolerance_rad)
                    output.diagnostics["stop_pose_arrived"] = arrived
                    return self._hold_output(output, ego, trajectory, arrived)
                self.release_frames = 0

            curvature, geometry = geometry_request(path, progress, ego,
                                                    self.settings, direction, precision)
            output.diagnostics.update(geometry)
            if self.calibration is None:
                return self._invalid(output, "vehicle calibration required")
            steering, front_angle = steering_request(curvature, self.calibration)
            output.diagnostics["front_angle_rad"] = front_angle
            acceleration = path.acceleration_reference(
                progress, trajectory, self.settings, reference)
            if trajectory.stop_required and trajectory.target_speed == 0:
                acceleration = min(0.0, acceleration)
                # A rolling STOP profile pins its first point to GPS speed
                # and forbids propulsion. Once its current brake demand is
                # neutral, a negative integral has no positive-error path to
                # unwind and would keep braking before the requested endpoint.
                # Retain it whenever the profile still asks for deceleration
                # or the measured vehicle is above the reference.
                if acceleration == 0.0 and reference >= ego.speed:
                    self.integral = 0.0
            throttle, brake, next_integral, speed_error = pedal_request(
                ego.speed, reference, self.integral, dt,
                self.settings, self.calibration, acceleration)
            if trajectory.stop_required and trajectory.target_speed == 0:
                throttle = 0.0
            requested_throttle, requested_brake = throttle, brake
            if self.last_output is not None and dt > 0:
                steering = _approach(self.last_output.steering, steering,
                                     self.settings.steering_slew_per_s * dt)
                if throttle > self.last_output.throttle:
                    throttle = min(throttle, self.last_output.throttle +
                                   self.settings.throttle_slew_per_s * dt)
                if brake < self.last_output.brake:
                    brake = max(brake, self.last_output.brake -
                                self.settings.brake_slew_per_s * dt)
            if brake > 0:
                throttle = 0.0
            if throttle > 0:
                brake = 0.0
            next_integral = applied_integral(
                self.integral, next_integral, speed_error,
                requested_throttle, requested_brake, throttle, brake,
                self.calibration)
            output.throttle, output.brake, output.steering = throttle, brake, steering
            output.diagnostics["speed_error_mps"] = speed_error
            output.diagnostics["integral_m"] = next_integral
            output.diagnostics["reference_acceleration_mps2"] = acceleration
            self.integral = next_integral
            self.state = ("STOPPING" if trajectory.stop_required and
                          trajectory.target_speed == 0 else "TRACK")
            output.diagnostics["state"] = self.state
            output.source = "control:" + self.state
            output.valid = True
            output.clamp()
            self.last_output = output
            return output
        except (AttributeError, TypeError, ValueError, OverflowError, IndexError) as exc:
            return self._invalid(output, str(exc))
