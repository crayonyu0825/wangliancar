"""Bounded exact-frame GPS poses for old Sensor packets; no interpolation."""
import copy
from collections import OrderedDict
from types import SimpleNamespace

from core.validation import number


INSTALL_FIELDS=("type","x","y","z","roll","pitch","yaw","hz")
POSE_FIELDS=("x","y","z","heading","roll","pitch")
MAX_POSES=128


def _installations(p):
    if (p.sensor_configurations_valid is not True or not isinstance(p.sensor_configurations,list)
            or len(p.sensor_configurations)>128): return {}
    records={}
    for item in p.sensor_configurations:
        if not isinstance(item,dict) or not isinstance(item.get("id"),str): return {}
        identity=item["id"]
        if not identity or identity in records: return {}
        if not all(key in item for key in INSTALL_FIELDS): return {}
        records[identity]={key:copy.deepcopy(item[key]) for key in INSTALL_FIELDS}
    return records


class SensorPoseHistory(object):
    def __init__(self,vehicle_id,timeout_ms):
        self.vehicle_id,self.timeout_s=vehicle_id,timeout_ms/1000.
        self.scope,self.high_frame,self.high_timestamp,self.last_now=None,-1,-1,None
        self.poses=OrderedDict()

    def observe(self,p,now):
        if not number(now):
            self.poses.clear(); self.scope=None
            return
        scope=(p.case_id,p.task_id,p.scene_id,str(self.vehicle_id))
        if not all(isinstance(v,str) and v for v in scope[:2]):
            self.poses.clear(); self.scope=None
            return
        if scope!=self.scope or (self.last_now is not None and now<self.last_now):
            self.poses.clear(); self.high_frame,self.high_timestamp=-1,-1
            self.scope=scope
        self.last_now=now
        for frame,item in list(self.poses.items()):
            if now>=item["valid_until_s"]: del self.poses[frame]
        gps=p.source_status.get("gps",{}) if isinstance(p.source_status,dict) else {}
        age=gps.get("age_ms",-1) if isinstance(gps,dict) else -1
        if (p.valid is not True or p.ego.valid is not True or not isinstance(gps,dict)
                or gps.get("usable") is not True or gps.get("clock")!="sdk_frame"
                or type(p.frame_id) is not int or p.frame_id<0 or p.ego.frame_id!=p.frame_id
                or type(p.timestamp) is not int or p.timestamp<=0 or p.ego.timestamp!=p.timestamp
                or gps.get("frame_id")!=p.frame_id or gps.get("timestamp")!=p.timestamp
                or not number(age) or not 0<=age<self.timeout_s*1000
                or not all(number(getattr(p.ego,key)) for key in POSE_FIELDS)
                or now>=p.valid_until): return
        if p.frame_id<self.high_frame or p.timestamp<self.high_timestamp:
            # Retain the high-water mark, so a regressed stream cannot seed a
            # new apparent history for another packet from the old sequence.
            self.poses.clear()
            return
        if p.frame_id==self.high_frame and p.frame_id not in self.poses:
            return  # An expired repeated frame cannot become a new observation.
        if p.frame_id>self.high_frame and p.timestamp<=self.high_timestamp:
            self.poses.clear()
            return
        self.high_frame,self.high_timestamp=p.frame_id,p.timestamp
        pose={key:getattr(p.ego,key) for key in POSE_FIELDS}
        installations=_installations(p)
        observed=now-age/1000.
        record=dict(frame_id=p.frame_id,timestamp=p.timestamp,pose=pose,
                    installations=installations,observed_at_s=observed,
                    valid_until_s=observed+self.timeout_s,consistent=True)
        previous=self.poses.get(p.frame_id)
        if previous is not None:
            if any(previous[key]!=record[key] for key in ("timestamp","pose","installations")):
                previous["consistent"]=False
            # A repeated packet never refreshes the first observation or TTL.
            previous["observed_at_s"]=min(previous["observed_at_s"],observed)
            previous["valid_until_s"]=min(previous["valid_until_s"],record["valid_until_s"])
        else: self.poses[p.frame_id]=record
        while len(self.poses)>MAX_POSES: self.poses.popitem(last=False)

    def select(self,p,now):
        if ((p.case_id,p.task_id,p.scene_id,str(self.vehicle_id))!=self.scope
                or type(p.targets_frame_id) is not int or p.targets_frame_id>=p.frame_id):
            return None,"sensor_pose_history_scope_mismatch"
        record=self.poses.get(p.targets_frame_id)
        if record is None: return None,"sensor_pose_history_unavailable"
        if not record["consistent"]: return None,"sensor_pose_history_conflicting"
        if now>=record["valid_until_s"]: return None,"sensor_pose_history_expired"
        meta=p.source_status.get("targets",{})
        if (not isinstance(meta,dict) or meta.get("usable") is not True
                or meta.get("clock")!="sdk_frame" or meta.get("frame_id")!=p.targets_frame_id
                or type(p.targets_timestamp) is not int or p.targets_timestamp<=0
                or p.targets_timestamp!=record["timestamp"] or meta.get("timestamp")!=p.targets_timestamp):
            return None,"sensor_pose_history_time_mismatch"
        sensor=p.target_source[7:] if isinstance(p.target_source,str) and p.target_source.startswith("sensor:") else ""
        installation=record["installations"].get(sensor)
        if installation is None or installation!=_installations(p).get(sensor):
            return None,"sensor_pose_history_installation_mismatch"
        if abs(record["pose"]["z"]-p.ego.z)>.1:
            return None,"sensor_pose_history_plane_mismatch"
        result=copy.deepcopy(record)
        result["ego"]=SimpleNamespace(**result["pose"])
        return result,"sensor_pose_history_exact_frame"
