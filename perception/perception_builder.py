"""Convert raw official API data into the captain's stable team interface."""

import math
import copy
import time

from core.geometry import calculate_ttc, speed_2d, world_to_ego, project_polyline
from core.interfaces import EgoState, Perception, Target
from core.scene_requirements import requires_targets
from simone_platform.case_resolver import resolve_scene_id
from perception.map_semantics import speed_limit_observations
from perception.traffic_signals import build_traffic
from perception.maneuver_environment import ManeuverEnvironmentBuilder


def _finite_data(value):
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_data(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_data(item) for item in value)
    return True


class PerceptionBuilder(object):
    def __init__(self, adapter, route_manager, scene_override=0):
        self.adapter = adapter
        self.route_manager = route_manager
        self.scene_override = scene_override
        self.case_name = ""
        self.case_id = ""
        self.task_id = ""
        self.scene_id = int(scene_override)
        self.maneuver_builder = ManeuverEnvironmentBuilder(route_manager,getattr(adapter,"config",None))

    def update_case_info(self):
        info = self.adapter.get_case_info()
        self.case_name = info.get("case_name", "")
        self.case_id = info.get("case_id", "")
        self.task_id = info.get("task_id", "")
        self.scene_id = resolve_scene_id(self.case_name, self.scene_override)
        return info

    def build(self):
        return self.build_from_raw(self.adapter.read_raw_snapshot())

    def build_from_raw(self, raw):
        result = Perception()
        result.case_name, result.case_id = self.case_name, self.case_id
        result.task_id, result.scene_id = self.task_id, self.scene_id
        config = getattr(self.adapter, "config", None)
        timeout = getattr(config, "sensor_timeout_ms", 500)
        max_gap = getattr(config, "max_sensor_frame_gap", 10)
        ttl = getattr(config, "pipeline_timeout_ms", 200)
        result.valid_until = time.monotonic() + ttl / 1000.0
        if not isinstance(raw, dict):
            result.errors.append("SNAPSHOT_UNAVAILABLE")
            return result
        raw = copy.deepcopy(raw)
        if not isinstance(raw.get("source_status", {}), dict):
            result.errors.append("SOURCE_STATUS_INVALID")
            raw["source_status"] = {}
        for name in ("errors", "reference_errors"):
            value = raw.get(name, [])
            if isinstance(value, list):
                result.errors.extend(str(item) for item in value)
            else:
                result.errors.append(name.upper() + "_INVALID")
        sensor_errors = raw.get("sensor_errors", [])
        result.sensor_errors = list(sensor_errors) if isinstance(sensor_errors, list) else ["SENSOR_ERRORS_INVALID"]
        result.source_status = raw.get("source_status", {})
        gps = raw.get("gps")
        result.ego = self._build_ego(gps, timeout)
        result.frame_id, result.timestamp = result.ego.frame_id, result.ego.timestamp
        result.source_status.setdefault("gps", {
            "read_ok": isinstance(gps, dict), "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "age_ms": result.ego.age_ms})
        if not isinstance(result.source_status["gps"], dict):
            result.source_status["gps"] = {"read_ok": False, "data_invalid": True}
        # Keep absent raw motion fields distinct from EgoState's compatibility
        # defaults. Configured maneuver models must not treat a default zero as
        # a current steering/yaw observation.
        result.source_status["gps"]["motion_fields"] = [
            name for name in ("steering", "yaw_rate", "vx", "vy")
            if isinstance(gps, dict) and name in gps]
        if not result.ego.valid:
            result.source_status["gps"]["data_invalid"] = True
        if not result.ego.valid:
            result.errors.append("GPS_UNAVAILABLE" if not gps else "GPS_INVALID_OR_STALE")
        # Keep auxiliary observations available even when GPS is unavailable.
        for name, empty in (("route_points", []), ("route_waypoints", []),
                            ("traffic_signs", []),
                            ("parking_spaces", []),
                            ("sensor_configurations", []), ("environment", {}), ("imu", {})):
            value = raw.get(name, copy.deepcopy(empty))
            flag = "route_valid" if name.startswith("route_") else name + "_valid"
            ok = isinstance(value, type(empty)) and _finite_data(value)
            setattr(result, name, value if ok else empty)
            setattr(result, flag, bool(raw.get(flag, False)) and ok)
            if not ok:
                result.errors.append(name.upper() + "_INVALID")
        # route_points and route_waypoints share one status; a bad one invalidates both.
        result.route_valid = bool(raw.get("route_valid", False)) and all(
            _finite_data(raw.get(n, [])) and isinstance(raw.get(n, []), list)
            for n in ("route_points", "route_waypoints"))
        result.sensor_configurations_valid = bool(raw.get("sensor_configurations_valid", False)) and bool(
            isinstance(raw.get("sensor_configurations", []), list) and
            _finite_data(raw.get("sensor_configurations", [])))
        result.environment_valid = bool(raw.get("environment_valid", False)) and isinstance(
            raw.get("environment", {}), dict) and _finite_data(raw.get("environment", {}))
        result.imu_valid = bool(raw.get("imu_valid", False)) and isinstance(
            raw.get("imu", {}), dict) and _finite_data(raw.get("imu", {}))
        for data_name, status_name, prefix in (
                ("radar_detections", "radar_status", "radar:"),
                ("sensor_lane_observations", "sensor_lane_status", "lane:"),
                ("ultrasonic_detections", None, "ultrasonic:")):
            data = raw.get(data_name, [])
            if not isinstance(data, list):
                result.sensor_errors.append(data_name.upper() + "_INVALID")
                data = []
            statuses = dict(raw.get(status_name, {})) if status_name and isinstance(raw.get(status_name, {}), dict) else {}
            clean = []
            for item in data:
                if isinstance(item, dict) and _finite_data(item):
                    clean.append(item)
                else:
                    sensor = item.get("sensor_id", "") if isinstance(item, dict) else ""
                    statuses[sensor] = False
                    result.sensor_errors.append(data_name.upper() + "_INVALID")
                    result.source_status.setdefault(prefix + sensor, {})["data_invalid"] = True
            setattr(result, data_name, clean)
            if status_name:
                setattr(result, status_name, statuses)
            else:
                result.ultrasonic_valid = bool(raw.get("ultrasonic_valid", False)) and len(clean) == len(data)
        result.target_source = raw.get("target_source", "none")
        result.targets_valid = bool(raw.get("targets_valid", False))
        try:
            result.targets_frame_id = int(raw.get("targets_frame", -1))
            result.targets_timestamp = int(raw.get("targets_timestamp", 0))
            result.targets_age_ms = int(raw.get("targets_age_ms", -1))
        except (ValueError, TypeError, OverflowError):
            result.targets_valid = False
        target_meta = result.source_status.setdefault("targets", {
            "read_ok": result.targets_valid, "frame_id": result.targets_frame_id,
            "timestamp": result.targets_timestamp, "age_ms": result.targets_age_ms})
        if not isinstance(target_meta, dict):
            result.source_status["targets"] = target_meta = {"read_ok": False,
                                                                "data_invalid": True}
        if result.ego.valid:
            try:
                if hasattr(self.route_manager,'set_route_hint'):
                    self.route_manager.set_route_hint(result.route_points,result.route_valid,
                                                       (result.case_id,result.task_id))
                result.lane = self.route_manager.update(result.ego)
            except Exception as exc:
                result.errors.append("LANE_UNAVAILABLE:" + type(exc).__name__)
        if not result.lane.valid:
            result.errors.append("LANE_UNAVAILABLE")
        self._build_map_observations(result, raw)
        lane_half_width = max(1.4, result.lane.lane_width * 0.5)
        target_data = raw.get("targets", [])
        if not isinstance(target_data, list):
            target_data = []
            result.targets_valid = False
            target_meta["data_invalid"] = True
        # Legacy Target defaults remain compatible, but a default zero is not
        # an observed velocity for prediction or maneuver admission.
        target_meta['motion_fields_complete']=all(isinstance(item,dict)
            and all(name in item and isinstance(item[name],(int,float))
                    and not isinstance(item[name],bool) and math.isfinite(item[name])
                    for name in ('vx','vy')) for item in target_data)
        if result.ego.valid:
            for item in target_data:
                try:
                    target = self._build_target(item, result.ego, lane_half_width)
                    target.source = result.target_source
                    if target.valid:
                        if hasattr(self.route_manager, "locate_target"):
                            target.lane_id, target.same_lane_valid = self.route_manager.locate_target(target)
                            target.same_lane = target.same_lane_valid and target.lane_id == result.lane.lane_id
                            target.lane_source = "hdmap" if target.same_lane_valid else "unknown"
                        result.targets.append(target)
                    elif not (math.isfinite(target.probability) and 0 <= target.probability < 0.05
                              and _finite_data(item)):
                        raise ValueError("invalid target")
                except (AttributeError, TypeError, ValueError, OverflowError):
                    result.errors.append("TARGET_INVALID")
                    result.targets_valid = False
                    target_meta["data_invalid"] = True
        else:
            result.targets_valid = False
        result.targets.sort(key=lambda item: item.distance)
        for source, meta in result.source_status.items():
            if not isinstance(meta, dict):
                result.source_status[source] = meta = {"data_invalid": True}
            try:
                age, frame = int(meta.get("age_ms", -1)), int(meta.get("frame_id", -1))
                quality = "ok"
                if not meta.get("read_ok", False):
                    quality = ("not_configured" if source == "targets" and
                               meta.get("sensor_presence") == "not_configured"
                               else "unavailable")
                elif meta.get("data_invalid", False):
                    quality = "invalid"
                elif age == -2:
                    quality = "regressed"
                elif age >= timeout:
                    quality = "stale"
                elif frame >= 0 and result.ego.valid and abs(frame - result.frame_id) > max_gap:
                    quality = "unsynchronized"
                elif age < 0 or frame < 0:
                    quality = "timing_unknown"
                meta["quality"] = quality
                meta["usable"] = quality == "ok"
            except (ValueError, TypeError, OverflowError):
                meta["quality"], meta["usable"] = "invalid", False
            if not meta["usable"]:
                if not (source == "targets" and meta["quality"] == "not_configured"
                        and not requires_targets(result.scene_id)):
                    result.sensor_errors.append(source + ":" + meta["quality"])
                if source == "targets":
                    result.targets_valid = False
                elif source.startswith("radar:"):
                    result.radar_status[source.split(":", 1)[1]] = False
                elif source.startswith("lane:"):
                    result.sensor_lane_status[source.split(":", 1)[1]] = False
                elif source.startswith("ultrasonic"):
                    result.ultrasonic_valid = False
                elif source == "imu" and meta["quality"] != "timing_unknown":
                    # This SDK's IMU struct has no frame header. Keep readable
                    # measurements, but do not claim synchronized usability.
                    result.imu_valid = False
        if not result.targets_valid and requires_targets(result.scene_id):
            result.errors.append("TARGETS_UNAVAILABLE")
        result.traffic = self._build_traffic(result.ego, result.lane, result.errors)
        result.source_status["traffic"] = {
            "clock": "host_query",  # this SDK structure has no frame header
            "association_valid": result.traffic.association_valid,
            "signal_presence": result.traffic.signal_presence,
            "required": result.traffic.required,
            "usable": result.traffic.valid,
            "quality": "ok" if result.traffic.valid else result.traffic.reason,
        }
        if result.traffic.observed:
            result.traffic_source = "hdmap+simone"
        result.valid = result.ego.valid
        if result.ego.valid and result.ego.age_ms >= 0:
            result.valid_until = min(result.valid_until, time.monotonic() +
                                     max(0, timeout - result.ego.age_ms) / 1000.0)
        try:
            if raw.get('sensor_pose_sample') is not None:
                result.maneuver_environment = self.maneuver_builder.build(result,raw['sensor_pose_sample'])
            else:
                result.maneuver_environment = self.maneuver_builder.build(result)
        except Exception as exc:
            # Optional new environment must not terminate the established
            # GPS/target/signal chain, nor masquerade as valid on failure.
            result.maneuver_environment.bind(result)
            result.maneuver_environment.errors.append("MANEUVER_ENVIRONMENT:"+type(exc).__name__)
        return result

    def _build_map_observations(self, result, raw):
        statuses = raw.get("map_observation_status", {})
        result.map_observation_status = copy.deepcopy(statuses) if isinstance(statuses, dict) else {}
        if not all(isinstance(v, dict) and _finite_data(v)
                   for v in result.map_observation_status.values()):
            result.map_observation_status = {}
            result.errors.append("MAP_OBSERVATION_STATUS_INVALID")
        if not all(isinstance(v, dict) and _finite_data(v)
                   for v in result.parking_spaces):
            result.parking_spaces, result.parking_spaces_valid = [], False
            result.errors.append("PARKING_SPACES_INVALID")
        context = {}
        if hasattr(self.adapter, "read_map_observations"):
            try:
                lane_ids = ([result.lane.lane_id] if result.lane.valid else [])
                context = self.adapter.read_map_observations(lane_ids)
            except Exception as exc:
                result.errors.append("MAP_OBSERVATIONS_UNAVAILABLE:" + type(exc).__name__)
        if not isinstance(context, dict):
            context = {}
        extra_statuses = context.get("map_observation_status", {})
        if (isinstance(extra_statuses, dict) and all(isinstance(v, dict) and _finite_data(v)
                                                   for v in extra_statuses.values())):
            result.map_observation_status.update(copy.deepcopy(extra_statuses))
        for name in ("map_stop_lines", "map_crosswalks"):
            values = context.get(name, [])
            good = isinstance(values, list) and all(isinstance(v, dict) and _finite_data(v)
                                                     for v in values)
            setattr(result, name, copy.deepcopy(values) if good else [])
            setattr(result, name + "_valid", good and context.get(name + "_valid") is True)
            if not good:
                result.errors.append(name.upper() + "_INVALID")
        observations, limit, source = speed_limit_observations(
            result.traffic_signs, result.traffic_signs_valid, result.ego, result.lane)
        result.speed_limit_observations = observations
        if limit > 0:
            existing = result.lane.speed_limit
            if existing <= 0 or limit < existing:
                result.lane.speed_limit, result.lane.speed_limit_source = limit, source

    def _build_traffic(self, ego, lane, errors):
        return build_traffic(self.adapter, ego, lane, errors)

    @staticmethod
    def _path_position(points, x, y):
        projection = project_polyline(points, x, y)
        return None if projection is None else projection["s"]

    @staticmethod
    def _build_ego(gps, timeout_ms=500):
        try:
            if (not isinstance(gps, dict) or not _finite_data(gps)
                    or not all(name in gps for name in ("frame", "x", "y", "heading", "vx", "vy"))):
                return EgoState()
            ego = PerceptionBuilder._parse_ego(gps)
            if ego.age_ms < -1 or ego.age_ms >= timeout_ms:
                ego.valid = False
            return ego
        except (AttributeError, TypeError, ValueError, OverflowError):
            return EgoState()

    @staticmethod
    def _parse_ego(gps):
        ego = EgoState()
        ego.frame_id = int(gps.get("frame", 0))
        ego.timestamp = int(gps.get("timestamp", 0))
        ego.x = float(gps.get("x", 0.0))
        ego.y = float(gps.get("y", 0.0))
        ego.z = float(gps.get("z", 0.0))
        ego.heading = float(gps.get("heading", 0.0))
        ego.roll = float(gps.get("roll", 0.0))
        ego.pitch = float(gps.get("pitch", 0.0))
        ego.vx = float(gps.get("vx", 0.0))
        ego.vy = float(gps.get("vy", 0.0))
        ego.vz = float(gps.get("vz", 0.0))
        ego.ax = float(gps.get("ax", 0.0))
        ego.ay = float(gps.get("ay", 0.0))
        ego.az = float(gps.get("az", 0.0))
        ego.yaw_rate = float(gps.get("yaw_rate", 0.0))
        for name in ("roll_rate", "pitch_rate", "engine_rpm"):
            setattr(ego, name, float(gps.get(name, 0.0)))
        ego.extra_states = list(gps.get("extra_states", []))
        ego.sdk_data = dict(gps.get("sdk_data", {}))
        if not math.isfinite(ego.heading):
            return EgoState()
        ego.speed = speed_2d(ego.vx, ego.vy)
        ego.acceleration = math.cos(ego.heading) * ego.ax + math.sin(ego.heading) * ego.ay
        ego.throttle = float(gps.get("throttle", 0.0))
        ego.brake = float(gps.get("brake", 0.0))
        ego.steering = float(gps.get("steering", 0.0))
        ego.gear = int(gps.get("gear", 0))
        ego.wheel_speeds = list(gps.get("wheel_speeds", []))
        ego.odometer = float(gps.get("odometer", -1.0))
        ego.age_ms = int(gps.get("age_ms", -1))
        ego.valid = (ego.frame_id >= 0 and ego.timestamp >= 0 and ego.age_ms >= -1
                     and not bool(gps.get("is_lost", False))) and all(
            math.isfinite(value) for value in
            (ego.x, ego.y, ego.z, ego.heading, ego.roll, ego.pitch,
             ego.vx, ego.vy, ego.vz,
             ego.ax, ego.ay, ego.az, ego.speed, ego.acceleration,
             ego.yaw_rate, ego.throttle, ego.brake, ego.steering,
             ego.roll_rate, ego.pitch_rate, ego.engine_rpm, ego.odometer)
        ) and _finite_data(ego.wheel_speeds) and _finite_data(ego.extra_states)
        return ego

    @staticmethod
    def _build_target(item, ego, lane_half_width):
        target = Target()
        target.id = int(item.get("id", -1))
        target.type = int(item.get("type", 0))
        target.x = float(item.get("x", 0.0))
        target.y = float(item.get("y", 0.0))
        target.z = float(item.get("z", 0.0))
        target.vx = float(item.get("vx", 0.0))
        target.vy = float(item.get("vy", 0.0))
        target.vz = float(item.get("vz", 0.0))
        target.heading = float(item["heading"]) if item.get("heading") is not None else None
        target.roll = float(item.get("roll", 0.0))
        target.pitch = float(item.get("pitch", 0.0))
        target.sdk_data = dict(item.get("sdk_data", {}))
        target.ax = float(item.get("ax", 0.0))
        target.ay = float(item.get("ay", 0.0))
        target.az = float(item.get("az", 0.0))
        target.probability = float(item.get("probability", 1.0))
        target.sensor_range = float(item.get("sensor_range", -1.0))
        for name in ("relative_x", "relative_y", "relative_z",
                     "relative_vx", "relative_vy", "relative_vz",
                     "relative_roll", "relative_pitch", "relative_heading"):
            if item.get(name) is not None:
                setattr(target, name, float(item[name]))
        if item.get("bbox2d") is not None:
            target.bbox2d = list(item["bbox2d"])
        target.length = float(item.get("length", 0.0))
        target.width = float(item.get("width", 0.0))
        target.height = float(item.get("height", 0.0))
        target.longitudinal_distance, target.lateral_distance = world_to_ego(
            ego.x, ego.y, ego.heading, target.x, target.y
        )
        target.distance = math.sqrt(
            target.longitudinal_distance * target.longitudinal_distance
            + target.lateral_distance * target.lateral_distance
        )
        target.ttc, target.relative_speed = calculate_ttc(
            target.longitudinal_distance, ego.speed, target.vx, target.vy, ego.heading
        )
        target.lateral_band_match = (
            target.longitudinal_distance > -target.length
            and abs(target.lateral_distance) <= lane_half_width
        )
        target.valid = (target.id >= 0 and 0.05 <= target.probability <= 1.0 and
                        all(math.isfinite(value) for value in
                            (target.x, target.y, target.z, target.vx, target.vy, target.vz,
                             target.ax, target.ay, target.az,
                             target.length, target.width, target.height,
                             target.probability, target.distance, target.sensor_range))
                        and (target.heading is None or math.isfinite(target.heading))
                        and _finite_data(item)
                        and min(target.length, target.width, target.height) >= 0
                        and _finite_data(target.bbox2d)
                        and _finite_data([target.relative_x, target.relative_y,
                                          target.relative_z, target.relative_vx,
                                          target.relative_vy, target.relative_vz]))
        return target
