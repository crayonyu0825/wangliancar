"""Explicit Sensor-model visibility, separate from a readable empty packet.

The current SDK lacks range/FOV/occlusion fields. Never infer them from object
range, sensor name, ground truth or a successful read. Profiles are opt-in
calibration data and must match task, vehicle and live sensor installation.
"""
import json
import math
import os

from core.region_geometry import convex_polygon


class SensorVisibility(object):
    def __init__(self, filename=""):
        self.filename = filename
        self.signature = None
        self.profiles = []
        self.reason = "visibility_model_not_configured"

    def regions(self, perception, vehicle_id, pose=None):
        if not self.filename:
            return [],self.reason
        try:
            stat = os.stat(self.filename)
            signature = (stat.st_mtime_ns,stat.st_size)
            if stat.st_size > 1048576:
                raise ValueError("visibility model too large")
            if signature != self.signature:
                with open(self.filename,encoding="utf-8") as stream:
                    data = json.load(stream)
                if (not isinstance(data,dict) or data.get("version")!="sensor-visibility-v1"
                        or not isinstance(data.get("profiles"),list) or len(data["profiles"])>32):
                    raise ValueError("invalid visibility model")
                self.profiles,self.signature = data["profiles"],signature
            records = []
            source = perception.target_source
            if not isinstance(source,str) or not source.startswith("sensor:"):
                return [],"formal_sensor_source_unavailable"
            sensor_id = source[7:]
            configs = [s for s in perception.sensor_configurations
                       if isinstance(s,dict) and s.get("id")==sensor_id]
            if not perception.sensor_configurations_valid or len(configs)!=1:
                return [],"sensor_installation_unverified"
            for profile in self.profiles:
                if not isinstance(profile,dict):
                    raise ValueError("invalid visibility profile")
                if (profile.get("task_id")!=perception.task_id or not perception.task_id
                        or profile.get("case_id")!=perception.case_id or not perception.case_id
                        or str(profile.get("vehicle_id"))!=str(vehicle_id)
                        or profile.get("sensor_id")!=sensor_id):
                    continue
                if (profile.get("complete_detections") is not True
                        or not isinstance(profile.get("verification_reference"),str)
                        or not profile["verification_reference"].strip()):
                    continue
                installation = profile.get("installation")
                fields = ("type","x","y","z","roll","pitch","yaw","hz")
                if not isinstance(installation,dict) or set(installation)!=set(fields):
                    raise ValueError("full SDK-native installation required")
                for key in fields:
                    actual,expected = configs[0].get(key),installation[key]
                    if key=="type":
                        if str(actual)!=str(expected):
                            break
                    elif (type(actual) not in (int,float) or type(expected) not in (int,float)
                          or not math.isfinite(actual) or not math.isfinite(expected)
                          or not math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-6)):
                        break
                else:
                    local = convex_polygon(profile.get("body_region_m"))
                    if any(math.hypot(x,y)>250 for x,y in local):
                        raise ValueError("visibility region outside supported range")
                    ego = perception.ego if pose is None else pose
                    c,s = math.cos(ego.heading),math.sin(ego.heading)
                    world = [(ego.x+x*c-y*s,ego.y+x*s+y*c) for x,y in local]
                    records.append(dict(sensor_id=sensor_id,source=source,source_kind="sensor",
                        polygon=world,reference_z_m=ego.z,complete_detections=True,
                        verification_reference=profile["verification_reference"]))
            return records,"verified_model_matched" if records else "visibility_model_scope_or_installation_mismatch"
        except (OSError,ValueError,TypeError,KeyError,OverflowError) as exc:
            self.signature,self.profiles = None,[]
            return [],"visibility_model_invalid:"+type(exc).__name__
