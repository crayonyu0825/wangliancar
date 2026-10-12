"""Fixed team entry for forward/reverse trajectory execution."""

from members.control.controller import ControlEngine
from members.control.parameters import VehicleCalibration
import math
from core.geometry import normalize_angle


_engine = ControlEngine()


def configure_control(config):
    """Captain injects vehicle-specific calibration; never read SDK or INI here."""
    global _engine
    _engine = ControlEngine(VehicleCalibration.from_app_config(config))


def compute_control(perception, trajectory):
    output = _engine.compute(perception, trajectory)
    identity = trajectory.behavior_identity
    if output.valid and isinstance(identity,dict):
        guard,ego = _engine.standstill,perception.ego
        same = bool(trajectory.stop_obligation_id and trajectory.stop_obligation_id==guard.obligation_id)
        arrived = bool(same and guard.anchor is not None and ego.speed<=_engine.settings.gear_standstill_speed_mps
            and math.hypot(ego.x-guard.anchor[0],ego.y-guard.anchor[1])<=_engine.settings.precision_arrival_tolerance_m
            and abs(normalize_angle(ego.heading-guard.anchor[2]))<=_engine.settings.precision_heading_tolerance_rad)
        if (not trajectory.stop_obligation_id and not guard.active
                and (identity.get('stage') in ('APPROACH','POSITION','ALIGN','EXIT_PREPARE','EXIT')
                     or identity.get('stage') in ('EXECUTE_BRAKE','EXECUTE_LEFT','EXECUTE_RIGHT',
                                                'EXECUTE_BRAKE_LEFT','EXECUTE_BRAKE_RIGHT'))
                and trajectory.precision_stop and trajectory.stop_required and trajectory.points):
            end=trajectory.points[-1]
            arrived=bool(output.diagnostics.get('stop_pose_arrived') is True
                and ego.speed<=_engine.settings.gear_standstill_speed_mps
                and math.hypot(ego.x-end.x,ego.y-end.y)<=_engine.settings.precision_arrival_tolerance_m
                and abs(normalize_angle(ego.heading-end.heading))<=_engine.settings.precision_heading_tolerance_rad)
        brake_hold=trajectory.planning_metrics.get('emergency_outcome')=='verified_safe_brake_stop'
        if brake_hold:
            arrived=False
        output.execution_observation = dict(identity=dict(identity),goal_pose_arrived=arrived,
            candidate_id=getattr(trajectory,'candidate_id',None),
            actual_brake_stop_confirmed=bool(brake_hold and ego.speed<=1e-9
                and output.throttle==0 and output.brake>0 and output.hazard_signal),
            standstill_duration_s=guard.elapsed if arrived else 0.,
            hold_completed=bool(arrived and guard.elapsed>=guard.duration>=trajectory.hold_duration_s>0))
    return output
