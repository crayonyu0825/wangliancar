"""SDK-free check of measured tracking against a previously sent candidate.

Only a successful attempted send receipt arms a copied timeline. The report is
a request for the captain's safety brake; it is not an actuator command, proof
of arrival, or evidence that caller-supplied model bounds were calibrated.
"""
import math
from collections import deque

from core.geometry import normalize_angle
from core.interfaces import Perception, Trajectory
from core.validation import number


EPS = 1e-9


class _Fault(Exception):
    def __init__(self, reason, details=None):
        self.reason, self.details = reason, details or {}


def _require(condition, reason, details=None):
    if not condition:
        raise _Fault(reason, details)


def _bounds(envelope):
    names = ('position_error_m', 'heading_error_rad', 'early_time_s', 'late_time_s')
    values = tuple(getattr(envelope, name, None) for name in names)
    _require(all(number(v) and v >= 0 for v in values) and values[1] <= math.pi,
             'UNVERIFIED_EXECUTION_ENVELOPE')
    return values


def _fraction(segment, local_time):
    dt, v0, v1, distance = segment[4:8]
    expected = .5*(v0+v1)*dt
    if distance <= EPS:
        return 0.
    acceleration = (v1-v0)/dt
    return max(0., min(1., (v0*local_time+.5*acceleration*local_time**2)/expected))


def _intersect(first, second):
    lo, hi = max(first[0], second[0]), min(first[1], second[1])
    return (lo, hi) if lo <= hi+EPS else None


def _matches(entry, ego, elapsed):
    """Analytic intersection: position, yaw and speed share one nominal time.

    Checking each quantity at a different nearest point can falsely accept a
    vehicle that is in none of the allowed timed poses. Constant acceleration
    maps nominal time monotonically to segment progress, so interval overlap
    provides a joint witness without an approximate sampling grid.
    """
    position_error, yaw_error, early, late = entry['bounds']
    earliest, latest = max(0., elapsed-late), elapsed+early
    _require(latest >= 0 and earliest <= entry['horizon']+EPS,
             'EXECUTION_TIMELINE_EXHAUSTED', {'elapsed_s': elapsed,
                                             'horizon_s': entry['horizon']})
    found_position, found_heading = False, False
    checks = 0
    for segment in entry['segments']:
        checks += 1
        x, y, dx, dy, dt, v0, v1, distance, yaw, turn, start = segment
        time_range = _intersect((max(0., earliest-start), min(dt, latest-start)), (0., dt))
        if time_range is None:
            continue
        acceleration = (v1-v0)/dt
        scale = distance/(.5*(v0+v1)*dt) if distance > EPS else 1.
        if distance <= EPS:
            if math.hypot(ego.x-x, ego.y-y) > position_error+EPS:
                continue
            found_position = True
            if abs(normalize_angle(ego.heading-yaw)) > yaw_error+EPS:
                continue
            found_heading = True
            if ego.speed <= entry['speed_error_mps']+EPS:
                return dict(checks=checks, nominal_time_interval_s=[start+time_range[0], start+time_range[1]],
                            position_error_m=math.hypot(ego.x-x, ego.y-y),
                            heading_error_rad=abs(normalize_angle(ego.heading-yaw)),
                            speed_error_mps=ego.speed)
            continue
        projection = ((ego.x-x)*dx+(ego.y-y)*dy)/(distance*distance)
        cross = abs((ego.x-x)*dy-(ego.y-y)*dx)/distance
        if cross > position_error+EPS:
            continue
        half_span = math.sqrt(max(0., position_error**2-cross**2))/distance
        progress = _intersect((projection-half_span, projection+half_span),
                              (_fraction(segment,time_range[0]), _fraction(segment,time_range[1])))
        if progress is None:
            continue
        found_position = True
        yaw_ranges = []
        if abs(turn) <= EPS:
            if abs(normalize_angle(ego.heading-yaw)) <= yaw_error+EPS:
                yaw_ranges.append(progress)
        else:
            base = yaw+normalize_angle(ego.heading-yaw)
            for revolution in (-1, 0, 1):
                a = (base+revolution*2*math.pi-yaw-yaw_error)/turn
                b = (base+revolution*2*math.pi-yaw+yaw_error)/turn
                overlap = _intersect(progress, (min(a,b),max(a,b)))
                if overlap is not None:
                    yaw_ranges.append(overlap)
        if not yaw_ranges:
            continue
        found_heading = True
        allowed_speed = (max(0.,ego.speed-entry['speed_error_mps']),
                         ego.speed+entry['speed_error_mps'])
        if abs(acceleration*scale) <= EPS:
            if not allowed_speed[0]-EPS <= v0*scale <= allowed_speed[1]+EPS:
                continue
            speed_time = time_range
        else:
            a = (allowed_speed[0]/scale-v0)/acceleration
            b = (allowed_speed[1]/scale-v0)/acceleration
            speed_time = _intersect(time_range, (min(a,b),max(a,b)))
            if speed_time is None:
                continue
        speed_progress = (_fraction(segment,speed_time[0]), _fraction(segment,speed_time[1]))
        for yaw_range in yaw_ranges:
            overlap = _intersect(yaw_range,speed_progress)
            if overlap is not None:
                u = .5*(overlap[0]+overlap[1])
                ds = distance*u
                if abs(acceleration) <= EPS:
                    local = ds/(v0*scale)
                else:
                    terminal = math.sqrt(max(0.,v0*v0+2*acceleration*ds/scale))
                    local = 2*(ds/scale)/(v0+terminal)
                return dict(checks=checks, nominal_time_s=start+local,
                            progress_fraction=u,
                            position_error_m=math.hypot(ego.x-x-u*dx,ego.y-y-u*dy),
                            heading_error_rad=abs(normalize_angle(ego.heading-yaw-u*turn)),
                            speed_error_mps=abs(ego.speed-scale*(v0+acceleration*local)))
    reason = ('EXECUTION_POSITION_OR_TIME_ERROR' if not found_position else
              'EXECUTION_HEADING_ERROR' if not found_heading else 'EXECUTION_SPEED_ERROR')
    raise _Fault(reason, dict(checks=checks, elapsed_s=elapsed,
                             allowed_nominal_time_interval_s=[earliest,latest]))


class ExecutionMonitor(object):
    """Finite sent history; explicit reset is reserved for a trusted task reset.

    max_speed_mps and speed_error_mps are explicit assertions from the same
    motion model as the candidate envelope. Missing speed error fails closed;
    position/timing bounds do not imply an instantaneous velocity error bound.
    """
    def __init__(self, max_speed_mps=None, speed_error_mps=None, max_gap_s=.5,
                 max_points=2000, max_entries=8):
        self.max_speed_mps, self.speed_error_mps = max_speed_mps, speed_error_mps
        _require(number(max_gap_s) and max_gap_s>0 and type(max_points) is int
                 and 2<=max_points<=10000 and type(max_entries) is int
                 and 1<=max_entries<=128,'EXECUTION_MONITOR_CONFIG_INVALID')
        self.max_gap_s, self.max_points = max_gap_s, max_points
        self.history = deque(maxlen=max_entries)
        self.reset()

    def reset(self):
        self.history.clear()
        self.last_frame, self.last_timestamp, self.last_now = None, None, None
        self.context = None

    def _config(self):
        _require(number(self.max_speed_mps) and self.max_speed_mps>0,'UNVERIFIED_SPEED_LIMIT')
        _require(number(self.speed_error_mps) and self.speed_error_mps>=0,'UNVERIFIED_SPEED_ERROR')

    def _actual(self,p,now):
        self._config()
        _require(isinstance(p,Perception) and p.valid is True and p.ego.valid is True
                 and type(p.frame_id) is int and p.frame_id>=0
                 and type(p.timestamp) is int and p.timestamp>=0
                 and p.ego.frame_id==p.frame_id and p.ego.timestamp==p.timestamp
                 and number(now) and now>=0 and number(p.valid_until) and now<p.valid_until
                 and all(number(getattr(p.ego,name,None)) for name in ('x','y','heading','speed'))
                 and p.ego.speed>=0,'EXECUTION_GPS_INVALID_OR_EXPIRED')
        gps=p.source_status.get('gps',{}) if isinstance(p.source_status,dict) else {}
        _require(isinstance(gps,dict) and gps.get('usable') is True
                 and gps.get('quality') not in ('stale','unsynchronized','regressed','invalid','timing_unknown')
                 and ('frame_id' not in gps or gps['frame_id']==p.frame_id), 'EXECUTION_GPS_UNAVAILABLE')
        env=getattr(p,'maneuver_environment',None)
        observed=getattr(env,'observed_at_s',None) if getattr(env,'valid',False) is True else gps.get('observed_at_s')
        if observed is not None:
            _require(number(observed) and 0<=observed<=now,'EXECUTION_FUTURE_OBSERVATION')
            _require(now-observed<=self.max_gap_s,'EXECUTION_OBSERVATION_GAP')
        _require(p.ego.speed<=self.max_speed_mps+EPS,'EXECUTION_HARD_SPEED_LIMIT',
                 {'actual_speed_mps':p.ego.speed,'max_speed_mps':self.max_speed_mps})
        context=(p.case_id,p.task_id,p.scene_id)
        _require(self.context is None or context==self.context,'EXECUTION_CONTEXT_CHANGED_RESET_REQUIRED')
        return context, now if observed is None else observed

    def _copy_candidate(self,p,trajectory,envelope,now):
        _require(isinstance(trajectory,Trajectory) and trajectory.valid is True
                 and type(trajectory.frame_id) is int and trajectory.frame_id==p.frame_id
                 and trajectory.timestamp==p.timestamp and number(trajectory.valid_until)
                 and now<trajectory.valid_until<=p.valid_until,'EXECUTION_CANDIDATE_SOURCE_INVALID_OR_EXPIRED')
        bounds=_bounds(envelope)
        _require(type(trajectory.motion_direction) is int and trajectory.motion_direction in (-1,1)
                 and isinstance(trajectory.points,(list,tuple)) and 2<=len(trajectory.points)<=self.max_points,
                 'EXECUTION_CANDIDATE_INVALID_OR_TOO_LARGE')
        points=[]
        for point in trajectory.points:
            values=tuple(getattr(point,name,None) for name in ('x','y','speed','heading','relative_time'))
            _require(all(number(v) for v in values) and values[2]>=0 and values[4]>=0,
                     'EXECUTION_CANDIDATE_INVALID')
            points.append(values)
        _require(points[0][4]==0,'EXECUTION_CANDIDATE_TIME_ORIGIN_INVALID')
        first=points[0]
        _require(math.hypot(first[0]-p.ego.x,first[1]-p.ego.y)<=bounds[0]+EPS
                 and abs(normalize_angle(first[3]-p.ego.heading))<=bounds[1]+EPS
                 and abs(first[2]-p.ego.speed)<=self.speed_error_mps+EPS,
                 'EXECUTION_CANDIDATE_START_MISMATCH')
        segments=[]
        for a,b in zip(points,points[1:]):
            dt=b[4]-a[4]; dx,dy=b[0]-a[0],b[1]-a[1]
            distance=math.hypot(dx,dy); turn=normalize_angle(b[3]-a[3])
            _require(dt>0 and (distance>EPS and a[2]+b[2]>EPS or
                               distance<=EPS and a[2]<=EPS and b[2]<=EPS and abs(turn)<=EPS),
                     'EXECUTION_CANDIDATE_KINEMATICS_INVALID')
            segments.append((a[0],a[1],dx,dy,dt,a[2],b[2],distance,a[3],turn,a[4]))
        identity=getattr(trajectory,'candidate_id',None)
        _require(identity is None or isinstance(identity,str) and 0<len(identity)<=256,
                 'EXECUTION_CANDIDATE_ID_INVALID')
        return dict(segments=tuple(segments),bounds=tuple(bounds),horizon=points[-1][4],
                    candidate_id=identity,source_frame_id=p.frame_id,sent_at_s=now,
                    valid_until_s=trajectory.valid_until,max_speed_mps=self.max_speed_mps,
                    speed_error_mps=self.speed_error_mps)

    def _brake_command(self,p,trajectory,now):
        """An emergency zero-speed command does not promise instant standstill."""
        if getattr(trajectory,'emergency_stop',False) is not True:
            return False
        _require(isinstance(trajectory,Trajectory) and trajectory.valid is True
                 and trajectory.frame_id==p.frame_id and trajectory.timestamp==p.timestamp
                 and number(trajectory.valid_until) and now<trajectory.valid_until<=p.valid_until
                 and trajectory.target_speed==0 and trajectory.stop_required is True
                 and isinstance(trajectory.points,(list,tuple)) and 2<=len(trajectory.points)<=self.max_points
                 and all(all(number(getattr(v,k,None)) for k in ('x','y','speed','heading','relative_time'))
                         and v.speed==0 and math.hypot(v.x-p.ego.x,v.y-p.ego.y)<=EPS
                         and abs(normalize_angle(v.heading-p.ego.heading))<=EPS for v in trajectory.points)
                 and trajectory.points[0].relative_time==0
                 and all(a.relative_time<b.relative_time for a,b in zip(trajectory.points,trajectory.points[1:])),
                 'EXECUTION_EMERGENCY_COMMAND_INVALID')
        return True

    def _result(self,active,status,reason,details=None):
        entry=self.history[-1] if self.history else None
        return dict(active=active,brake_required=active,status=status,reason_code=reason,
                    details=details or {},history_size=len(self.history),
                    candidate_id=entry['candidate_id'] if entry else None,
                    sent_source_frame_id=entry['source_frame_id'] if entry else None,
                    authority='tracking_bounds_only_not_arrival_or_actuator_receipt')

    def observe(self,actual_perception,current_trajectory,execution_envelope,now_s):
        """Inspect last actually sent timeline before accepting a new candidate."""
        try:
            p=actual_perception; context,observed_at=self._actual(p,now_s)
            _bounds(execution_envelope)
            if self.last_frame is not None:
                _require(p.frame_id>self.last_frame,'EXECUTION_REPEATED_FRAME' if p.frame_id==self.last_frame
                         else 'EXECUTION_OUT_OF_ORDER_FRAME')
                _require(p.timestamp>self.last_timestamp,'EXECUTION_OUT_OF_ORDER_TIMESTAMP')
                _require(now_s>self.last_now,'EXECUTION_OUT_OF_ORDER_TIME')
                _require(now_s-self.last_now<=self.max_gap_s,'EXECUTION_OBSERVATION_GAP')
            if self.history:
                entry=self.history[-1]
                _require(p.frame_id>entry['source_frame_id'],'EXECUTION_REPEATED_SENT_FRAME')
                _require(now_s>=entry['sent_at_s'],'EXECUTION_FUTURE_SENT_TIMELINE')
                _require(now_s-entry['sent_at_s']<=self.max_gap_s,'EXECUTION_OBSERVATION_GAP')
                _require(now_s<entry['valid_until_s'],'EXECUTION_SENT_CANDIDATE_EXPIRED')
                _require(p.ego.speed<=entry['max_speed_mps']+EPS,'EXECUTION_HARD_SPEED_LIMIT')
                detail=_matches(entry,p.ego,observed_at-entry['sent_at_s'])
            else:
                detail={}
            # New candidate cannot overwrite history to hide a tracking error.
            if self._brake_command(p,current_trajectory,now_s):
                detail['current_candidate_authority']='brake_command_only_no_tracking_promise'
            else:
                self._copy_candidate(p,current_trajectory,execution_envelope,now_s)
            self.last_frame,self.last_timestamp,self.last_now=p.frame_id,p.timestamp,now_s
            self.context=context
            return self._result(False,'CLEAR' if self.history else 'UNARMED',
                                'EXECUTION_WITHIN_BOUNDS' if self.history else 'EXECUTION_NO_SENT_HISTORY',detail)
        except _Fault as error:
            return self._result(True,'FAULT',error.reason,error.details)
        except (AttributeError,TypeError,ValueError,OverflowError,ZeroDivisionError):
            return self._result(True,'FAULT','EXECUTION_INPUT_INVALID')

    def record_sent(self,actual_perception,trajectory,execution_envelope,now_s,receipt):
        """Arm only after transport confirms an attempted successful send."""
        if not isinstance(receipt,dict) or receipt.get('attempted') is not True or receipt.get('ok') is not True:
            return self._result(False,'UNARMED' if not self.history else 'UNCHANGED','EXECUTION_SEND_NOT_CONFIRMED')
        try:
            context,observed_at=self._actual(actual_perception,now_s)
            if self.history:
                old=self.history[-1]
                _require(actual_perception.frame_id>old['source_frame_id'] and now_s>old['sent_at_s'],
                         'EXECUTION_DUPLICATE_SEND_CANNOT_RENEW')
                _require(now_s-old['sent_at_s']<=self.max_gap_s,'EXECUTION_OBSERVATION_GAP')
                _require(now_s<old['valid_until_s'],'EXECUTION_SENT_CANDIDATE_EXPIRED')
                _matches(old,actual_perception.ego,observed_at-old['sent_at_s'])
            if self._brake_command(actual_perception,trajectory,now_s):
                return self._result(False,'UNCHANGED' if self.history else 'UNARMED',
                                    'EXECUTION_BRAKE_COMMAND_NOT_TIMED_PROMISE')
            entry=self._copy_candidate(actual_perception,trajectory,execution_envelope,now_s)
            self.history.append(entry); self.context=context
            return self._result(False,'ARMED','EXECUTION_SENT_TIMELINE_RECORDED')
        except _Fault as error:
            return self._result(True,'FAULT',error.reason,error.details)
        except (AttributeError,TypeError,ValueError,OverflowError,ZeroDivisionError):
            return self._result(True,'FAULT','EXECUTION_INPUT_INVALID')
