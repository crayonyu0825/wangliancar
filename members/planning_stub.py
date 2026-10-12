"""Fixed planning entry for the lane reference and guarded obstacles."""

import math

from members.planning.lane_planner import build_trajectory, PlannerSettings
from core.interfaces import DecisionTarget, Trajectory
from members.planning.stop_behavior import StopBehaviorPlanner
from members.planning.lane_change_behavior import LaneChangeBehaviorPlanner
from members.planning.behavior_contract import LANE_CHANGE_ACTIONS
from members.planning.parking_behavior import ParkingBehaviorPlanner
from members.planning.driving_behavior import DrivingBehaviorPlanner
from members.planning.behavior_contract import DRIVING_ACTIONS


_CONTROL_GEOMETRY = {}
_BEHAVIOR_PLANNER = StopBehaviorPlanner()
_LANE_CHANGE_PLANNER = LaneChangeBehaviorPlanner()
_PARKING_PLANNER = ParkingBehaviorPlanner()
_DRIVING_PLANNER = DrivingBehaviorPlanner()


def _settings_with_control_geometry():
    settings = PlannerSettings.from_environment()
    for name, value in _CONTROL_GEOMETRY.items():
        explicit = getattr(settings, name)
        if explicit is not None and not math.isclose(explicit, value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("planning/control vehicle geometry differs: " + name)
        setattr(settings, name, value)
    return settings


def configure_planning(config=None,lane_change_inputs_provider=None,parking_inputs_provider=None,driving_inputs_provider=None,
                       vehicle_geometry=None):
    """Captain supplies the same configured steering geometry as control.

    This is an explicit trial capability, not a claim of live calibration.
    Body extents continue to come from the shared NEVC_VEHICLE_* overrides.
    lane_change_inputs_provider is an optional captain-owned source/model
    adapter. None keeps P03 unavailable; it never advertises capabilities.
    parking_inputs_provider similarly supplies P04's original task and model.
    Neither provider is installed or advertised by default.
    """
    global _CONTROL_GEOMETRY,_BEHAVIOR_PLANNER,_LANE_CHANGE_PLANNER,_PARKING_PLANNER,_DRIVING_PLANNER
    _BEHAVIOR_PLANNER = StopBehaviorPlanner()
    _LANE_CHANGE_PLANNER = LaneChangeBehaviorPlanner(lane_change_inputs_provider)
    _PARKING_PLANNER = ParkingBehaviorPlanner(parking_inputs_provider)
    _DRIVING_PLANNER = DrivingBehaviorPlanner(driving_inputs_provider)
    geometry = {}
    if vehicle_geometry is not None:
        for name in ('front_offset_m','rear_offset_m','half_width_m'):
            value=getattr(vehicle_geometry,name,None)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('configured body geometry unavailable: '+name)
            geometry[name]=float(value)
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
    if getattr(decision,"behavior_request",None) is not None:
        request=decision.behavior_request
        if isinstance(request,dict) and request.get('maneuver') in LANE_CHANGE_ACTIONS:
            return _LANE_CHANGE_PLANNER.plan(perception,decision,settings)
        if isinstance(request,dict) and request.get('maneuver')=='PARK':
            return _PARKING_PLANNER.plan(perception,decision,settings)
        if isinstance(request,dict) and request.get('maneuver') in DRIVING_ACTIONS:
            return _DRIVING_PLANNER.plan(perception,decision,settings)
        return _DRIVING_PLANNER.ordinary(_BEHAVIOR_PLANNER.plan(perception,decision,settings),
                                        perception,decision,settings)
    return _DRIVING_PLANNER.ordinary(build_trajectory(perception, decision, settings),
                                    perception,decision,settings)
