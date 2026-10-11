"""Bounded, provisional decision parameters (Python 3.6, standard library)."""

import math
import os


ENV_PREFIX = "NEVC_DECISION_"
FIELDS = (
    "cruise_speed", "min_gap", "time_headway", "gap_gain", "resume_margin",
    "hold_distance",
    "follow_deceleration", "approach_deceleration_ratio", "reaction_time", "front_offset_m", "half_width_m",
    "emergency_clearance", "emergency_ttc", "static_speed_threshold",
    "standstill_speed", "blind_speed_tolerance", "traffic_stop_margin",
    "obstacle_stop_margin", "projection_tolerance_m", "route_ambiguity_m",
    "conflict_horizon_s", "recovery_frames", "release_frames",
    "motion_horizon_m", "motion_deceleration", "motion_guard_time_s", "motion_lateral_margin_m",
    "motion_tolerance_mps",
    "history_max_tracks", "history_samples", "history_age_s", "history_handoff_s",
    "history_max_gap_s", "history_position_jump_m", "history_braking_uncertainty_mps2",
    "behavior_feedback_max_age_s", "behavior_ack_timeout_s", "behavior_progress_timeout_s",
    "behavior_retry_delay_s", "behavior_max_retries", "behavior_observation_gap_s",
    "signal_dwell_duration_s", "slow_lead_confirmation_s", "slow_lead_min_advantage_mps",
    "indicator_lead_s", "maneuver_settle_frames", "trajectory_progress_timeout_s",)
INT_FIELDS = ("recovery_frames", "release_frames", "history_max_tracks", "history_samples", "behavior_max_retries", "maneuver_settle_frames")
DEPRECATED = ("launch_ttc_cap", "stop_margin")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _parse(name, raw, integer=False):
    value = str(raw).strip()
    if integer:
        if not value.isdigit():
            raise ValueError("decision override {0} must be a positive integer".format(name))
        return int(value)
    try:
        result = float(value)
    except ValueError:
        raise ValueError("decision override {0} is not numeric".format(name))
    if not math.isfinite(result):
        raise ValueError("decision override {0} is not finite".format(name))
    return result


def overrides_from_environment(environ=None):
    environ = os.environ if environ is None else environ
    for name in DEPRECATED:
        key = ENV_PREFIX + name.upper()
        if key in environ:
            raise ValueError("decision override {0} is retired".format(key))
    values = {}
    if "NEVC_VEHICLE_FRONT_OFFSET_M" in environ:
        values["front_offset_m"] = _parse("NEVC_VEHICLE_FRONT_OFFSET_M",
                                          environ["NEVC_VEHICLE_FRONT_OFFSET_M"])
    for key, field in (("NEVC_VEHICLE_HALF_WIDTH_M", "half_width_m"),
                       ("NEVC_PLANNING_HORIZON_M", "motion_horizon_m"),
                       ("NEVC_PLANNING_DECELERATION_MPS2", "motion_deceleration"),
                       ("NEVC_PLANNING_DECELERATION_MPS2", "follow_deceleration"),
                       ("NEVC_PLANNING_MOTION_TOLERANCE_MPS", "motion_tolerance_mps"),
                       ("NEVC_PLANNING_LATERAL_GUARD_TIME_S", "motion_guard_time_s"),
                       ("NEVC_PLANNING_LATERAL_MARGIN_M", "motion_lateral_margin_m")):
        if key in environ:
            values[field] = _parse(key, environ[key])
    for name in FIELDS:
        key = ENV_PREFIX + name.upper()
        if key in environ:
            values[name] = _parse(key, environ[key], name in INT_FIELDS)
    return values


class DecisionSettings(object):
    def __init__(self, cruise_speed=30.0 / 3.6, min_gap=10.5,
                 time_headway=1.2, gap_gain=0.5, resume_margin=2.0,
                 hold_distance=0.05,
                 follow_deceleration=2.0, reaction_time=0.3,
                 front_offset_m=None, emergency_clearance=2.0,
                 emergency_ttc=2.0, static_speed_threshold=0.3,
                 standstill_speed=0.1, blind_speed_tolerance=0.5,
                 traffic_stop_margin=0.3, obstacle_stop_margin=0.5,
                 projection_tolerance_m=2.5, route_ambiguity_m=2.0,
                 conflict_horizon_s=3.0, recovery_frames=3,
                 release_frames=2, approach_deceleration_ratio=0.5,
                 half_width_m=None, motion_horizon_m=60.0, motion_deceleration=2.0,
                 motion_guard_time_s=3.0, motion_lateral_margin_m=0.0,
                 motion_tolerance_mps=0.0, history_max_tracks=128, history_samples=8,
                 history_age_s=1.0, history_handoff_s=0.4, history_max_gap_s=0.5,
                 history_position_jump_m=3.0, history_braking_uncertainty_mps2=2.0,
                 behavior_feedback_max_age_s=0.5, behavior_ack_timeout_s=2.0,
                 behavior_progress_timeout_s=5.0, behavior_retry_delay_s=0.5,
                 behavior_max_retries=2, behavior_observation_gap_s=0.5,
                 signal_dwell_duration_s=5.1, slow_lead_confirmation_s=2.0,
                 slow_lead_min_advantage_mps=1.5, indicator_lead_s=1.0,
                 maneuver_settle_frames=3, trajectory_progress_timeout_s=20.0):
        for name in FIELDS:
            setattr(self, name, locals()[name])

    def validate(self):
        for name in FIELDS:
            value = getattr(self, name)
            if name in INT_FIELDS:
                minimum = 0 if name == "behavior_max_retries" else 1
                if type(value) is not int or value < minimum:
                    raise ValueError("decision setting {0} must be a positive integer".format(name))
                continue
            if name in ("front_offset_m", "half_width_m") and value is None:
                continue
            if not _finite(value) or value < 0.0:
                raise ValueError("decision setting {0} must be finite and nonnegative".format(name))
        for name in ("cruise_speed", "time_headway", "gap_gain", "resume_margin", "follow_deceleration",
                     "emergency_ttc", "static_speed_threshold", "blind_speed_tolerance",
                     "projection_tolerance_m", "route_ambiguity_m", "conflict_horizon_s",
                     "motion_horizon_m", "motion_deceleration", "motion_guard_time_s",
                     "history_age_s", "history_handoff_s", "history_max_gap_s",
                     "history_position_jump_m", "history_braking_uncertainty_mps2",
                     "behavior_feedback_max_age_s", "behavior_ack_timeout_s",
                     "behavior_progress_timeout_s", "behavior_retry_delay_s",
                     "behavior_observation_gap_s", "slow_lead_confirmation_s",
                     "slow_lead_min_advantage_mps", "indicator_lead_s", "trajectory_progress_timeout_s") :
            if getattr(self, name) <= 0.0:
                raise ValueError("decision setting {0} must be positive".format(name))
        if self.front_offset_m is not None and self.front_offset_m <= 0.0:
            raise ValueError("measured front offset must be positive")
        if self.half_width_m is not None and self.half_width_m <= 0.0:
            raise ValueError("measured half width must be positive")
        if self.standstill_speed >= self.blind_speed_tolerance:
            raise ValueError("standstill_speed must be below blind_speed_tolerance")
        if self.hold_distance >= self.resume_margin:
            raise ValueError("hold_distance must be below resume_margin")
        if not 0.0 < self.approach_deceleration_ratio <= 1.0:
            raise ValueError("approach_deceleration_ratio must be in (0, 1]")
        if self.history_samples < 2 or self.history_max_tracks > 4096 or self.history_samples > 128:
            raise ValueError("target history budget must be bounded and contain at least two samples")
        if self.history_handoff_s > self.history_age_s or self.history_max_gap_s > self.history_age_s:
            raise ValueError("target history handoff/gap cannot exceed history age")
        if self.signal_dwell_duration_s <= 5.0:
            raise ValueError("signal dwell policy must exceed five seconds")
        return self

    def replace(self, **overrides):
        values = dict((name, getattr(self, name)) for name in FIELDS)
        for name, value in overrides.items():
            if name not in values:
                raise ValueError("unknown decision setting: " + str(name))
            values[name] = value
        return DecisionSettings(**values).validate()

    @classmethod
    def from_environment(cls, environ=None, **overrides):
        values = overrides_from_environment(environ)
        values.update(overrides)
        return cls().replace(**values)
