"""Observe sent-plan errors for calibration; never grant maneuver authority."""
import math
from collections import deque

from core.geometry import normalize_angle
from core.validation import number


class TrackingMeasurements(object):
    def __init__(self):
        self.scope, self.previous, self.high_header = None, None, (-1,-1)
        self.samples = deque(maxlen=16)
        self.count = 0
        self.maxima = dict(position_error_m=0.,heading_error_rad=0.,speed_error_mps=0.)

    def record(self,p,trajectory,receipt,safety_active,now):
        scope = (p.case_id,p.task_id,p.scene_id)
        if scope != self.scope:
            self.__init__()
            self.scope = scope
        gps = p.source_status.get('gps',{})
        if (not all(scope[:2]) or p.valid is not True or p.ego.valid is not True
                or now>=p.valid_until or gps.get('usable') is not True or gps.get('clock')!='sdk_frame'
                or gps.get('frame_id')!=p.frame_id or gps.get('timestamp')!=p.timestamp
                or p.ego.frame_id!=p.frame_id or p.ego.timestamp!=p.timestamp
                or type(p.frame_id) is not int or type(p.timestamp) is not int
                or not all(number(getattr(p.ego,k)) for k in ('x','y','heading','speed','steering'))):
            self.previous = None
            return
        if p.frame_id<=self.high_header[0] or p.timestamp<=self.high_header[1]:
            if (p.frame_id,p.timestamp)!=self.high_header or safety_active or receipt.get('ok') is not True:
                self.previous = None
            return
        self.high_header = (p.frame_id,p.timestamp)
        prior = self.previous
        if prior is not None and now<prior['valid_until']:
            # SDK 3.0.0001 SimOne_Data.timestamp is milliseconds. This measures
            # source-time nominal error, not an inferred actuator delay bound.
            elapsed = (p.timestamp-prior['timestamp'])/1000.
            if 0<elapsed<=.5:
                for a,b in zip(prior['points'],prior['points'][1:]):
                    if not a[4]<=elapsed<=b[4]:continue
                    dt,local = b[4]-a[4],elapsed-a[4]
                    fraction = local/dt
                    if a[2]+b[2]>1e-9:
                        fraction = (a[2]*local+.5*(b[2]-a[2])/dt*local*local)/(.5*(a[2]+b[2])*dt)
                    x,y = a[0]+fraction*(b[0]-a[0]),a[1]+fraction*(b[1]-a[1])
                    yaw = a[3]+fraction*normalize_angle(b[3]-a[3])
                    speed = a[2]+(b[2]-a[2])*local/dt
                    sample = dict(source_frame_id=prior['frame'],actual_frame_id=p.frame_id,
                        source_elapsed_s=elapsed,position_error_m=math.hypot(p.ego.x-x,p.ego.y-y),
                        heading_error_rad=abs(normalize_angle(p.ego.heading-yaw)),speed_error_mps=abs(p.ego.speed-speed),
                        actual_speed_mps=p.ego.speed,actual_steering_raw=p.ego.steering)
                    self.samples.append(sample)
                    self.count += 1
                    for k in self.maxima:self.maxima[k]=max(self.maxima[k],sample[k])
                    break
        self.previous = None
        if (safety_active or receipt.get('attempted') is not True or receipt.get('ok') is not True
                or receipt.get('reason')!='sent' or trajectory.valid is not True or trajectory.emergency_stop
                or trajectory.frame_id!=p.frame_id or trajectory.timestamp!=p.timestamp
                or now>=trajectory.valid_until or not 2<=len(trajectory.points)<=1000):return
        points = [tuple(getattr(v,k) for k in ('x','y','speed','heading','relative_time')) for v in trajectory.points]
        if (not all(all(number(v) for v in point) and point[2]>=0 for point in points)
                or points[0][4]!=0 or not all(a[4]<b[4] for a,b in zip(points,points[1:]))):return
        self.previous = dict(points=points,frame=p.frame_id,timestamp=p.timestamp,valid_until=trajectory.valid_until)

    def summary(self):
        return dict(status='OBSERVED_ONLY_NOT_CALIBRATED',scope=self.scope,sample_count=self.count,
            observed_maxima=dict(self.maxima),recent_samples=list(self.samples),
            time_basis='SDK milliseconds relative to sent trajectory source frame',
            missing_bounds=['early_time_s','late_time_s','unobserved operating conditions'])
