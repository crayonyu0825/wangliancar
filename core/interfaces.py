"""Four-member data contracts.

Keep this module independent from SimOne SDK so every member can import and test it.
Python 3.6 compatible by design.
"""

import math


class FrameOutput(object):
    """Monotonic deadline is process-local; timestamp retains SDK units."""

    def __init__(self):
        self.frame_id = -1
        self.timestamp = 0
        self.valid_until = 0.0
        self.errors = []

    def bind(self, source):
        self.frame_id = source.frame_id
        self.timestamp = source.timestamp
        self.valid_until = source.valid_until
        return self


class DecisionMode(object):
    KEEP_LANE = "KEEP_LANE"
    FOLLOW = "FOLLOW"
    STOP = "STOP"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


class EgoState(object):
    def __init__(self):
        self.frame_id = 0
        self.timestamp = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.heading = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.speed = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        # Signed acceleration along the vehicle's forward axis (m/s^2).
        self.acceleration = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.wheel_speeds = []
        self.odometer = -1.0
        # Raw GPS gearbox position (int), not the control N/D/R/P enum.
        # Keep it unchanged; do not copy it into ControlOut.gear.
        self.gear = 0
        self.valid = False
        # Local age since this GPS frame was first received, or -1 if unknown.
        self.age_ms = -1
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.engine_rpm = 0.0
        self.extra_states = []
        self.sdk_data = {}


class Target(object):
    def __init__(self):
        self.id = -1
        self.type = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.heading = None  # radians; absent orientation uses a bounding circle
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.probability = 1.0
        self.source = ""
        self.sensor_range = -1.0
        self.relative_x = None
        self.relative_y = None
        self.relative_z = None
        self.relative_vx = None
        self.relative_vy = None
        self.relative_vz = None
        self.bbox2d = None
        self.length = 0.0
        self.width = 0.0
        self.height = 0.0
        self.distance = -1.0
        self.longitudinal_distance = 0.0
        self.lateral_distance = 0.0
        self.relative_speed = 0.0
        self.same_lane = False
        self.same_lane_valid = False
        self.lane_source = "unknown"
        self.lateral_band_match = False
        self.lane_id = ""
        # HD-map width measured at this target's own position, in metres.
        # None means unavailable; never borrow the ego lane's width for a successor.
        self.lane_width_m = None
        self.roll = 0.0
        self.pitch = 0.0
        self.relative_roll = None
        self.relative_pitch = None
        self.relative_heading = None
        self.sdk_data = {}
        self.ttc = -1.0
        self.valid = False


class LaneContext(object):
    def __init__(self):
        self.lane_id = ""
        self.center_line = []
        self.left_boundary = []
        self.right_boundary = []
        self.left_lane_id = ""
        self.right_lane_id = ""
        self.left_mark_type = "UNKNOWN"
        self.right_mark_type = "UNKNOWN"
        self.lane_width = 3.5
        self.lane_width_valid = False
        self.speed_limit = -1.0
        self.speed_limit_source = "unavailable"
        self.predecessor_lane_ids = []
        self.successor_lane_ids = []
        # HD-map reference in the travel direction, including verified unique
        # successors. center_line/boundaries still describe only lane_id.
        # The full current center_line is the unchanged prefix. Coordinates
        # are world metres; IDs are plain strings, never SDK objects.
        self.forward_reference = []
        self.forward_lane_ids = []
        # Inclusive point indices in forward_reference, in the same order as
        # forward_lane_ids. Adjacent spans share their verified join point.
        # An empty list denotes legacy/unknown segment boundaries.
        self.forward_lane_spans = []
        self.forward_reference_valid = False
        # map_end: confirmed terminal; lookahead_limit: more links exist.
        # Other values describe why expansion stopped at a known boundary.
        self.forward_reference_status = "unavailable"
        self.source = "none"
        self.heading_error = 0.0
        self.lateral_offset = 0.0
        self.valid = False


class TrafficControl(object):
    def __init__(self):
        self.signal_state = "UNKNOWN"
        self.signal_id = -1
        self.count_down = -1
        self.candidates = []
        # Independent, route-scoped signal groups. Distances are rear-axle
        # route metres; scope/coverage boundaries are explicitly distinguished
        # from mapped stop lines. False retains legacy single-signal semantics.
        self.signal_groups = []
        self.signal_groups_valid = False
        self.diagnostics = []
        self.ambiguous = False
        self.reason = "unavailable"
        self.observed = False
        # Static lane association and dynamic read are independent facts.
        self.association_valid = False
        self.signal_presence = "unknown"  # unknown / absent / present
        self.required = False  # applicable signal ahead, even if its read failed
        self.signal_distance = -1.0
        self.stop_line_distance = -1.0
        self.speed_limit = -1.0
        self.valid = False


class Perception(object):
    """Captain -> decision member."""

    def __init__(self):
        self.valid_until = 0.0
        # Runtime-owned identity and previous-cycle formal feedback. Empty keeps
        # the legacy forward chain; model inputs require an explicit test mode.
        self.session_id = ""
        self.observed_at_s = None
        self.observation_mode = "production"
        self.behavior_environment = None
        self.behavior_capabilities = None
        self.behavior_feedback = []
        self.ego = EgoState()
        self.targets = []
        self.lane = LaneContext()
        self.traffic = TrafficControl()
        self.traffic_signs = []
        self.traffic_signs_valid = False
        # Detached static HDMap observations in world metres. These are not
        # synchronized Sensor API frames, occupancy or planned trajectories.
        self.parking_spaces = []
        self.parking_spaces_valid = False
        self.map_stop_lines = []
        self.map_stop_lines_valid = False
        self.map_crosswalks = []
        self.map_crosswalks_valid = False
        self.map_observation_status = {}
        self.speed_limit_observations = []
        # Scenario waypoints are route hints, not a planned trajectory.
        self.route_points = []
        self.route_waypoints = []
        self.route_valid = False
        self.sensor_configurations = []
        self.sensor_configurations_valid = False
        self.environment = {}
        self.environment_valid = False
        self.imu = {}
        self.imu_valid = False
        self.radar_detections = []
        self.radar_status = {}
        self.ultrasonic_detections = []
        self.ultrasonic_valid = False
        self.sensor_lane_observations = []
        self.sensor_lane_status = {}
        self.sensor_errors = []
        self.source_status = {}
        self.target_source = "none"
        self.targets_valid = False
        self.targets_frame_id = -1
        self.targets_timestamp = 0
        self.targets_age_ms = -1
        self.traffic_source = "none"
        self.scene_id = 0
        self.case_name = ""
        self.case_id = ""
        self.task_id = ""
        self.frame_id = 0
        self.timestamp = 0
        self.valid = False
        self.errors = []


class DecisionTarget(FrameOutput):
    """Decision member -> planning member."""

    def __init__(self):
        super(DecisionTarget, self).__init__()
        self.mode = DecisionMode.KEEP_LANE
        self.behavior_request = None
        self.target_speed = 0.0
        self.target_lane_id = ""
        self.stop_distance = -1.0
        # Business net clearance; -1 lets a legacy producer use planner fallback.
        self.obstacle_clearance_m = -1.0
        # Canonical decimal ID strings keep JSON round trips lossless.
        self.obstacle_clearances_m = {}
        self.precision_stop = False
        self.reason = ""
        self.valid = False


class TrajectoryPoint(object):
    def __init__(self, x=0.0, y=0.0, speed=0.0, heading=0.0, relative_time=0.0):
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)
        self.heading = float(heading)
        self.relative_time = float(relative_time)


class Trajectory(FrameOutput):
    """Planning member -> control member."""

    def __init__(self):
        super(Trajectory, self).__init__()
        self.points = []
        self.behavior_request = None
        self.planning_feedback = []
        self.execution_path_id = None
        self.stop_obligation_id = None
        self.goal_pose = None
        self.target_speed = 0.0
        self.emergency_stop = False
        self.emergency_maneuver_authorized = False
        self.stop_required = False
        self.stop_distance = -1.0
        self.target_lane_id = ""
        # Points follow travel order; speed is a nonnegative magnitude. Ego
        # heading remains body yaw. A segment has one explicit direction.
        self.motion_direction = 1  # +1 forward, -1 reverse; never raw GPS gear
        self.precision_stop = False
        # Minimum continuous standstill at the requested stop, in seconds.
        self.hold_duration_s = 0.0
        self.parking_brake_at_stop = False
        # Explicit upstream intentions; steering does not imply a turn signal.
        self.left_signal = False
        self.right_signal = False
        self.hazard_signal = False
        self.reason = ""
        self.valid = False


class ControlOut(FrameOutput):
    """Control member -> captain/runtime -> SimOne API."""

    def __init__(self):
        super(ControlOut, self).__init__()
        self.execution_feedback = None
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        # Command enum: Neutral=0, Drive=1, Reverse=2, Parking=3.
        self.gear = 1
        self.handbrake = False
        self.left_signal = False
        self.right_signal = False
        self.hazard_signal = False
        self.valid = False
        self.source = ""
        self.diagnostics = {}

    def clamp(self):
        values = (self.throttle, self.brake, self.steering)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in values):
            self.valid = False
            raise ValueError("nonfinite or nonnumeric actuator command")
        if type(self.gear) is not int or self.gear not in (0, 1, 2, 3):
            self.valid = False
            raise ValueError("automatic gear must be N/D/R/P (0..3)")
        if not all(type(getattr(self, name)) is bool for name in
                   ("handbrake", "left_signal", "right_signal", "hazard_signal")):
            self.valid = False
            raise ValueError("signal and handbrake fields must be bool")
        self.throttle = max(0.0, min(1.0, self.throttle))
        self.brake = max(0.0, min(1.0, self.brake))
        self.steering = max(-1.0, min(1.0, self.steering))
        # Brake/handbrake takes precedence over propulsion.
        if self.brake > 0.0 or self.handbrake or self.gear in (0, 3):
            self.throttle = 0.0
        return self
