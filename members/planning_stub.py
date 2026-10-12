"""Fixed planning entry for the lane reference and guarded obstacles."""

import math

from members.planning.lane_planner import build_trajectory, PlannerSettings
from core.interfaces import DecisionTarget, Trajectory
from members.planning.behavior_planner import BehaviorPlanner, capabilities
from members.planning.parking_planner import ParkingPlanner
from members.planning.intersection_planner import IntersectionPlanner
from members.planning.emergency_planner import EmergencyPlanner


_CONTROL_GEOMETRY = {}
_BEHAVIOR_PLANNER = BehaviorPlanner()
_PARKING_PLANNER = ParkingPlanner()
_INTERSECTION_PLANNER = IntersectionPlanner()
_EMERGENCY_PLANNER = EmergencyPlanner()


def _settings_with_control_geometry():
    settings = PlannerSettings.from_environment()
    for name, value in _CONTROL_GEOMETRY.items():
        explicit = getattr(settings, name)
        if explicit is not None and not math.isclose(explicit, value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("planning/control vehicle geometry differs: " + name)
        setattr(settings, name, value)
    return settings


def configure_planning(config=None):
    """Captain supplies the same configured steering geometry as control.

    This is an explicit trial capability, not a claim of live calibration.
    Body extents continue to come from the shared NEVC_VEHICLE_* overrides.
    """
    global _CONTROL_GEOMETRY
    _BEHAVIOR_PLANNER.reset()
    _PARKING_PLANNER.reset()
    _INTERSECTION_PLANNER.reset()
    _EMERGENCY_PLANNER.reset()
    geometry = {}
    if config is not None and getattr(config, "control_calibrated", False):
        for name in ("wheelbase_m", "front_steer_max_rad"):
            value = getattr(config, "control_" + name, None)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("configured control vehicle geometry is unavailable: " + name)
            geometry[name] = float(value)
    _CONTROL_GEOMETRY = geometry
    settings = _settings_with_control_geometry()
    settings.validate()
    return settings


def plan(perception, decision):
    try:
        settings = _settings_with_control_geometry()
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        output = Trajectory()
        if isinstance(decision, DecisionTarget):
            output.bind(decision)
        output.reason = str(exc)
        output.errors.append(str(exc))
        return output
    fallback = build_trajectory(perception, decision, settings)
    if isinstance(decision, DecisionTarget) and decision.behavior_request is not None:
        if decision.mode=='EMERGENCY_BRAKE' and decision.behavior_request.maneuver!='EMERGENCY':
            return fallback
        if decision.behavior_request.maneuver=='EMERGENCY':
            return _EMERGENCY_PLANNER.plan(perception,decision,settings,fallback)
        if decision.behavior_request.maneuver=='BLOCKED_ROAD':
            fallback.behavior_request=decision.behavior_request
            fallback.goal_pose=decision.behavior_request.goal.goal_pose
            fallback.hazard_signal=True
            return fallback
        if decision.behavior_request.maneuver=='CONTINUOUS_TASK':
            fallback.behavior_request=decision.behavior_request
            fallback.goal_pose=decision.behavior_request.goal.goal_pose
            return fallback
        if decision.behavior_request.maneuver=='SIGNAL_STOP':
            req=decision.behavior_request
            fallback.behavior_request=req;fallback.hold_duration_s=req.goal.minimum_standstill_duration_s
            fallback.stop_obligation_id=req.goal.stop_obligation_id;fallback.goal_pose=req.goal.goal_pose
            fallback.precision_stop=True
            return fallback
        if decision.behavior_request.maneuver in ('INTERSECTION','CROSSWALK','JUNCTION_ENTRY'):
            return _INTERSECTION_PLANNER.plan(perception,decision,settings,fallback)
        if decision.behavior_request.maneuver=='PARK':
            if not fallback.valid or decision.behavior_request.authorization=='EVALUATE_ONLY':
                from core.interfaces import TrajectoryPoint
                fallback=Trajectory().bind(decision);fallback.valid=True;fallback.target_speed=0.0
                fallback.stop_required=True;fallback.stop_distance=0.0
                fallback.points=[TrajectoryPoint(perception.ego.x,perception.ego.y,0,perception.ego.heading,0),
                                 TrajectoryPoint(perception.ego.x,perception.ego.y,0,perception.ego.heading,0.1)]
            return _PARKING_PLANNER.plan(perception,decision,settings,fallback)
        return _BEHAVIOR_PLANNER.plan(perception, decision, settings, fallback)
    return fallback


def planning_capabilities(perception):
    """Actual registered generators with the shared configured vehicle body."""
    settings = _settings_with_control_geometry()
    result = capabilities(perception, settings)
    if result is not None:
        # Explicit geometry is carried by a shared subtype, not dynamic fields.
        from core.behavior_protocol import RuntimeCapabilities
        return RuntimeCapabilities(result, dict(front_offset_m=settings.front_offset_m,
            rear_offset_m=settings.rear_offset_m, half_width_m=settings.half_width_m,
            wheelbase_m=settings.wheelbase_m, front_steer_max_rad=settings.front_steer_max_rad))
    return None
