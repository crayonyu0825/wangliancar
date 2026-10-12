"""Captain-owned gateway to official SimOne APIs and ABI-isolated helpers."""

import importlib
import copy
import ctypes
import math
import os
import sys
import threading
import time

from core.interfaces import ControlOut
from core.map_scope import matches_lane
from simone_platform.evaluation import LocalEvaluation
from simone_platform.sdk_compat import polling_structs
from simone_platform.sensor_catalog import (TARGET_INGESTION_VERSION, sensor_kind,
                                           target_sensor_ids)
from simone_platform.map_observations import MapObservationReader, vector_items
from simone_platform.map_document import MapDocument


def _decode_sdk_text(value):
    if isinstance(value, bytes):
        raw = value.split(b"\0", 1)[0]
        for encoding in ("utf-8", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                pass
        return raw.decode("utf-8", errors="replace")
    return str(value)


def _sdk_int(value):
    """ctypes enum fields need .value on the bundled Python 3.6 SDK."""
    return int(getattr(value, "value", value))


def _map_string(value):
    return value.GetString() if hasattr(value, "GetString") else _decode_sdk_text(value)


def _native_dict(value):
    """Detach small SDK structs, including inherited headers and enum values.

    Fixed lane arrays have no count in this SDK: retain them without inventing
    a valid-point count. Never use this helper for image/point-cloud buffers.
    """
    if isinstance(value, bytes):
        return _decode_sdk_text(value)
    if isinstance(value, ctypes.Array) or isinstance(value, (tuple, list)):
        return [_native_dict(item) for item in value]
    if isinstance(value, ctypes.Structure):
        fields = []
        for cls in reversed(type(value).__mro__):
            fields.extend(cls.__dict__.get("_fields_", []))
        return {field[0]: _native_dict(getattr(value, field[0])) for field in fields}
    if hasattr(value, "value"):
        return _native_dict(value.value)
    if hasattr(value, "__dict__"):
        return {key: _native_dict(item) for key, item in vars(value).items()}
    return value


def _finite_sensor_data(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_sensor_data(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_sensor_data(item) for item in value)
    return True


class SimOneAdapter(object):
    CASE_STOP = 1
    CASE_RUNNING = 2
    CASE_PAUSE = 3

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.structs = None
        self.sensor_api = None
        self.pnc_api = None
        self.service_api = None
        self.hdmap = None
        self.connected = False
        self.map_loaded = False
        self.last_send_result = {}
        self._source_status = {}
        self._reference_snapshot = {"route_points": [], "route_valid": False,
                                    "route_waypoints": [],
                                    "sensor_configurations": [],
                                    "sensor_configurations_valid": False,
                                    "environment": {}, "environment_valid": False}
        self._reference_update = 0.0
        self._map_observations = None
        self._map_document = MapDocument()
        self._traffic_candidates = {}
        self._last_frame_times = {}
        self.sdk_version = ""
        self._target_callback = None
        self._target_callback_status = "not_registered"
        self._target_streams = {}
        self._target_lock = threading.Lock()
        self._target_accepting = True
        self._target_probe_after = {}
        self._target_log_signature = None
        self._target_log_time = 0.0
        self._evaluation = None

    def bootstrap(self):
        sdk_dir = os.path.abspath(self.config.sdk_dir)
        required = ("SimOneIOStruct.py", "SimOneServiceAPI.py", "HDMapAPI.pyd")
        missing = [name for name in required if not os.path.exists(os.path.join(sdk_dir, name))]
        if missing:
            raise RuntimeError("SimOne SDK 文件缺失: {0}".format(", ".join(missing)))
        if sdk_dir not in sys.path:
            sys.path.insert(0, sdk_dir)
        # The SDK's native DLL loader expects this directory to be current.
        os.chdir(sdk_dir)
        self.structs = importlib.import_module("SimOneIOStruct")
        self.sensor_api = importlib.import_module("SimOneSensorAPI")
        self.pnc_api = importlib.import_module("SimOnePNCAPI")
        self.service_api = importlib.import_module("SimOneServiceAPI")
        self.hdmap = importlib.import_module("HDMapAPI")
        version = self.service_api.SoAPIGetVersion()
        self.sdk_version = _decode_sdk_text(version)
        self.structs, repairs = polling_structs(self.structs, version)
        if getattr(self.config, "evaluation_enabled", True):
            self._evaluation = LocalEvaluation(
                self.service_api.SimoneAPI, version, self.logger,
                getattr(self.config, "evaluation_flush_interval_sec", 1.0))
        if repairs:
            self.logger.warning("已修正 SDK %s 轮询结构布局: %s",
                                _decode_sdk_text(version), ", ".join(repairs))
        self.logger.info("SimOne SDK 已加载: %s", sdk_dir)
        return True

    def initialize(self):
        if self.service_api is None:
            self.bootstrap()
        deadline = time.time() + self.config.connect_timeout_sec
        while time.time() < deadline:
            result = self.service_api.SoInitSimOneAPI(
                self.config.vehicle_id, 0, self.config.server_ip
            )
            if result:
                self.connected = True
                self._target_accepting = True
                if self._evaluation is not None:
                    self._evaluation.initialize(self.config.vehicle_id)
                self.pnc_api.SoSetDriverName(self.config.vehicle_id, "Captain")
                self._register_target_callback()
                self.logger.info(
                    "已连接 SimOne: vehicle=%s server=%s",
                    self.config.vehicle_id,
                    self.config.server_ip,
                )
                return True
            time.sleep(1.0)
        raise RuntimeError("连接 SimOne 超时，请确认案例和 BridgeIO 已启动")

    def load_hdmap(self):
        if self.hdmap is None:
            raise RuntimeError("SDK 尚未加载")
        selected = MapDocument.selected_identity(self.service_api,self.structs,self.sdk_version)
        self._map_document.clear("MAP_NOT_LOADED")
        self.map_loaded = bool(self.hdmap.loadHDMap(int(self.config.map_timeout_sec)))
        self._traffic_candidates = {}
        self._map_observations = None
        self._reference_update = 0.0
        if not self.map_loaded:
            self.logger.warning("高精地图加载失败；GPS/目标数据仍可继续输出")
        else:
            self.logger.info("高精地图加载成功")
            if selected is not None:
                metadata = self._map_document.load(self.service_api,self.structs,self.sdk_version,
                    self.config.sdk_dir,getattr(self.config,"map_document_file",""),
                    loaded_identity=selected)
                self.logger.info("地图语义文档 verified=%s source=%s reason=%s",
                    metadata["verified"],metadata["source"],metadata["reason"])
            else:
                self._map_document.clear("MAP_IDENTITY_API_OR_ABI_UNVERIFIED")
        return self.map_loaded

    def read_map_document(self):
        """Private captain input: exact document bytes plus current identity.

        This performs no download or HDMap reload in the driving loop. An
        identity change requires load_hdmap(), so old SDK geometry cannot be
        paired with the newly selected XML. No document/URL enters telemetry.
        """
        if not self.map_loaded:
            self._map_document.clear("MAP_NOT_LOADED")
        elif self._map_document.data is not None:
            self._map_document.verified(self.service_api,self.structs,self.sdk_version)
        metadata = self._map_document.metadata()
        return (self._map_document.data if metadata["verified"] else None),metadata

    def get_case_status(self):
        return int(self.service_api.SoGetCaseRunStatus())

    def flush_evaluation(self, force=False):
        if self._evaluation is None:
            return True
        return self._evaluation.flush(force)

    def evaluation_status(self):
        if self._evaluation is not None:
            return self._evaluation.status()
        return {"enabled": getattr(self.config, "evaluation_enabled", True),
                "initialized": False, "mode": "local", "save_count": 0,
                "last_save_ok": None}

    def get_case_info(self):
        result = {"case_name": "", "case_id": "", "task_id": ""}
        data = self.structs.SimOne_Data_CaseInfo()
        if self.service_api.SoAPIGetCaseInfo(data):
            result["case_name"] = _decode_sdk_text(data.caseName)
            result["case_id"] = _decode_sdk_text(data.caseId)
            result["task_id"] = _decode_sdk_text(data.taskId)
        return result

    def read_raw_snapshot(self):
        self._source_status = {}
        try:
            result = self._read_raw_snapshot()
        except Exception as exc:
            result = {"gps": None, "targets": [], "targets_valid": False,
                      "errors": ["SNAPSHOT_UNAVAILABLE:" + type(exc).__name__]}
        if result.get("gps") is None:
            result.update(self._read_reference_data())
            result.update(self._read_auxiliary_data(result.get("sensor_configurations", [])))
        result["source_status"] = copy.deepcopy(self._source_status)
        return result

    def _read_raw_snapshot(self):
        gps = self.structs.SimOne_Data_Gps()
        if not self.sensor_api.SoGetGps(self.config.vehicle_id, gps):
            return {"gps": None, "targets": [], "targets_valid": False,
                    "target_source": "none", "errors": ["GPS_UNAVAILABLE"]}
        gps_data = {
            "frame": int(gps.frame),
            "timestamp": int(gps.timestamp),
            "x": float(gps.posX),
            "y": float(gps.posY),
            "z": float(gps.posZ),
            "heading": float(gps.oriZ),
            "roll": float(gps.oriX),
            "pitch": float(gps.oriY),
            "vx": float(gps.velX),
            "vy": float(gps.velY),
            "vz": float(gps.velZ),
            "ax": float(gps.accelX),
            "ay": float(gps.accelY),
            "az": float(gps.accelZ),
            "yaw_rate": float(gps.angVelZ),
            "wheel_speeds": [float(gps.wheelSpeedFL), float(gps.wheelSpeedFR),
                             float(gps.wheelSpeedRL), float(gps.wheelSpeedRR)],
            "odometer": float(gps.odometer),
            "throttle": float(gps.throttle),
            "brake": float(gps.brake),
            "steering": float(gps.steering),
            "gear": int(gps.gear),
            "is_lost": bool(gps.isGPSLost),
        }
        gps_data["age_ms"] = self._frame_age_ms("gps", gps_data["frame"])
        gps_data["sdk_data"] = _native_dict(gps)
        count = int(getattr(gps, "extraStateSize", 0))
        extra = getattr(gps, "extraStates", [])
        if count < 0 or count > len(extra):
            raise ValueError("GPS extra state count out of range")
        gps_data["extra_states"] = [float(v) for v in extra[:count]]
        if "extraStates" in gps_data["sdk_data"]:
            gps_data["sdk_data"]["extraStates"] = gps_data["extra_states"]
        for name, sdk_name in (("roll_rate", "angVelX"), ("pitch_rate", "angVelY"),
                               ("engine_rpm", "engineRpm")):
            gps_data[name] = float(getattr(gps, sdk_name, 0.0))
        self._metadata("gps", gps, True)
        errors = []
        reference = self._read_reference_data()
        targets, source, target_frame, target_timestamp, target_status, target_errors = (
            self._read_target_sources(reference, int(gps.frame)))
        errors.extend(target_errors)
        if targets is None:
            try:
                targets, target_frame, target_timestamp = self._read_ground_truth_targets()
            except Exception as exc:
                errors.append("GROUND_TRUTH_INVALID:{0}".format(type(exc).__name__))
                targets = None
            source = "ground_truth" if targets is not None else "none"
        result = {"gps": gps_data, "targets": targets or [],
                  "targets_valid": targets is not None, "target_source": source,
                  "targets_frame": target_frame if targets is not None else -1,
                  "targets_timestamp": target_timestamp if targets is not None else 0,
                  "targets_age_ms": self._frame_age_ms("targets:" + source, target_frame)
                  if targets is not None else -1,
                  "errors": errors}
        result.update(reference)
        result.update(self._read_auxiliary_data(result["sensor_configurations"]))
        target_status.update({
            "read_ok": targets is not None, "source": source,
            "sensor_read_ok": source.startswith("sensor:"),
            "frame_id": result["targets_frame"], "timestamp": result["targets_timestamp"],
            "age_ms": result["targets_age_ms"], "clock": "sdk_frame"})
        self._source_status["targets"] = target_status
        self._log_target_status(target_status)
        return result

    def _register_target_callback(self):
        """Discover actual source IDs with the verified detection ABI.

        The vendor Python callback typedef uses the old 128-byte entry stride.
        Register our own typed callback against the native void-returning API,
        retaining it until termination. No SDK installation files are changed.
        """
        native = getattr(self.sensor_api, "SimoneAPI", None)
        if native is None or self.sdk_version != "3.0.0001":
            self._target_callback_status = "unsupported"
            return
        try:
            kind = self.structs.SimOne_Data_SensorDetections
            if ctypes.sizeof(kind) != 58388:
                raise ValueError("unverified callback detection ABI")
            callback_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p,
                                             ctypes.POINTER(kind))
            self._target_callback = callback_type(self._receive_target_detection)
            register = native.SetSensorDetectionsUpdateCB
            register.restype = ctypes.c_bool
            register.argtypes = [callback_type]
            ok = register(self._target_callback)
            self._target_callback_status = "registered" if ok else "registration_failed"
        except Exception as exc:
            self._target_callback_status = "unavailable:" + type(exc).__name__
        log = getattr(self.logger, "info", None)
        if callable(log):
            log("目标接入 version=%s callback=%s", TARGET_INGESTION_VERSION,
                self._target_callback_status)

    def _receive_target_detection(self, vehicle_id, sensor_id, pointer):
        """Detach native memory on the SDK thread; never run member algorithms here."""
        if _decode_sdk_text(vehicle_id) != self.config.vehicle_id or not pointer:
            return
        identifier = _decode_sdk_text(sensor_id)
        if not identifier or len(identifier.encode("utf-8")) >= 64:
            return
        now = time.monotonic()
        try:
            # Reinterpret vendor-typed pointers only with our already-verified layout.
            native = ctypes.cast(pointer, ctypes.POINTER(
                self.structs.SimOne_Data_SensorDetections)).contents
            targets, frame, timestamp = self._parse_sensor_targets(native)
            record = {"targets": targets, "frame_id": frame, "timestamp": timestamp,
                      "received_at": now, "reason": "ok"}
        except Exception as exc:
            record = {"targets": None, "frame_id": -1, "timestamp": 0,
                      "received_at": now, "reason": "invalid:" + type(exc).__name__}
        with self._target_lock:
            if self._target_accepting and (identifier in self._target_streams
                                           or len(self._target_streams) < 100):
                self._target_streams[identifier] = record

    def _read_target_sources(self, reference, gps_frame):
        """Select a fresh official sensor stream, with bounded fallback probing."""
        now = time.monotonic()
        config_ok = reference["sensor_configurations_valid"]
        configured = target_sensor_ids(reference["sensor_configurations"], self.config.sensor_id)
        with self._target_lock:
            streams = dict(self._target_streams)  # callback records are replaced, never mutated
        timeout = getattr(self.config, "sensor_timeout_ms", 500)
        # A previously delivered ID remains a real discovery even when that
        # packet expires. Poll that ID for recovery, never reuse its old objects.
        observed = sorted(streams)
        candidates = configured + [identifier for identifier in observed if identifier not in configured]
        preferred = self.config.sensor_id
        # An unknown catalog is not proof of absent sensors. Probe only the user
        # preference; other IDs must come from configurations or actual callbacks.
        if not config_ok and preferred and not candidates:
            candidates.append(preferred)
        if preferred in candidates:
            candidates.remove(preferred)
            candidates.insert(0, preferred)
        presence = "configured" if configured else ("unknown" if observed or not config_ok
                                                    else "not_configured")
        meta = {"version": TARGET_INGESTION_VERSION, "sensor_presence": presence,
                "sensor_id": candidates[0] if candidates else "", "sensor_candidates": candidates,
                "config_read_ok": config_ok,
                "config_reason": reference.get("sensor_configurations_reason", "unknown"),
                "callback_status": self._target_callback_status,
                "callback_sensor_ids": observed, "attempts": [], "transport": "none",
                "callback_errors": {identifier: record["reason"]
                                    for identifier, record in streams.items()
                                    if record["targets"] is None},
                "id_verified": False,
                "reason": "not_configured" if presence == "not_configured" else "sensor_read_failed"}
        errors = []
        for identifier in candidates:
            record = streams.get(identifier)
            transport = "callback"
            if (record is None or record["targets"] is None
                    or (now - record["received_at"]) * 1000 >= timeout):
                if now < self._target_probe_after.get(identifier, 0.0):
                    meta["attempts"].append({"sensor_id": identifier,
                                             "read_ok": False, "reason": "retry_backoff"})
                    continue
                transport = "poll"
                try:
                    targets, unused_source, frame, timestamp = self._read_sensor_targets(identifier)
                    record = {"targets": targets, "frame_id": frame, "timestamp": timestamp}
                except Exception as exc:
                    record = {"targets": None, "frame_id": -1, "timestamp": 0}
                    errors.append("SENSOR_TARGETS_INVALID:{0}:{1}".format(identifier, type(exc).__name__))
            reason = "read_failed"
            age = -1
            if record["targets"] is not None:
                age = self._frame_age_ms("targets:sensor:" + identifier, record["frame_id"])
                reason = ("regressed" if age == -2 else "stale" if age >= timeout else
                          "unsynchronized" if abs(record["frame_id"] - gps_frame) >
                          getattr(self.config, "max_sensor_frame_gap", 10) else "ok")
            attempt = {"sensor_id": identifier, "read_ok": record["targets"] is not None,
                       "reason": reason, "transport": transport,
                       "frame_id": record["frame_id"], "age_ms": age}
            meta["attempts"].append(attempt)
            if reason != "ok":
                self._target_probe_after[identifier] = now + 1.0
                continue
            meta.update(sensor_id=identifier, reason="empty" if not record["targets"] else "ok",
                        sensor_presence="configured" if identifier in configured else "unknown",
                        transport=transport, id_verified=identifier in configured or identifier in observed)
            self._target_probe_after.pop(identifier, None)
            return (copy.deepcopy(record["targets"]), "sensor:" + identifier,
                    record["frame_id"], record["timestamp"], meta, errors)
        if candidates:
            errors.append("SENSOR_TARGETS_UNAVAILABLE:" + ",".join(candidates))
        return None, "none", -1, 0, meta, errors

    def _log_target_status(self, meta):
        # Log state transitions and a ten-second reminder, never every failed poll.
        signature = tuple(meta.get(key) for key in
                          ("config_read_ok", "config_reason", "sensor_presence", "sensor_id",
                           "source", "sensor_read_ok", "reason", "transport"))
        now = time.monotonic()
        if signature == self._target_log_signature and now - self._target_log_time < 10.0:
            return
        self._target_log_signature, self._target_log_time = signature, now
        log = getattr(self.logger, "info", None)
        if callable(log):
            log("目标接入 config=%s config_reason=%s presence=%s candidates=%s selected=%s "
                "source=%s sensor_ok=%s transport=%s reason=%s attempts=%s callback_errors=%s",
                meta["config_read_ok"], meta["config_reason"], meta["sensor_presence"],
                ",".join(meta["sensor_candidates"]), meta["sensor_id"], meta["source"],
                meta["sensor_read_ok"], meta["transport"], meta["reason"], meta["attempts"],
                meta["callback_errors"])

    def _frame_age_ms(self, source, frame):
        if frame < 0:
            return -1
        now = time.monotonic()
        previous = self._last_frame_times.get(source)
        if previous is not None and frame < previous[0]:
            return -2  # Regressed stream; retain high water mark until recovery.
        if previous is None or previous[0] != frame:
            self._last_frame_times[source] = (frame, now)
            return 0
        return max(0, int((now - previous[1]) * 1000))

    def _metadata(self, source, native, ok):
        frame = int(getattr(native, "frame", -1))
        self._source_status[source] = {
            "read_ok": bool(ok), "frame_id": frame,
            "timestamp": int(getattr(native, "timestamp", 0)),
            "age_ms": self._frame_age_ms(source, frame) if ok else -1,
            "clock": "sdk_frame" if frame >= 0 else "unavailable"}

    def _read_sensor_targets(self, sensor_id):
        data = self.structs.SimOne_Data_SensorDetections()
        ok = self.sensor_api.SoGetSensorDetections(
            self.config.vehicle_id, sensor_id, data
        )
        if not ok:
            return None, "none", -1, 0
        targets, frame, timestamp = self._parse_sensor_targets(data)
        return targets, "sensor:{0}".format(sensor_id), frame, timestamp

    def _parse_sensor_targets(self, data):
        targets = []
        if int(data.objectSize) < 0 or int(data.objectSize) > len(data.objects):
            raise ValueError("sensor object count out of range")
        for index in range(int(data.objectSize)):
            item = data.objects[index]
            target = self._target_dict(item, float(item.probability))
            if not _finite_sensor_data(target):
                raise ValueError("nonfinite sensor target")
            if not 0.0 <= target["probability"] <= 1.0:
                raise ValueError("invalid sensor target probability")
            targets.append(target)
        frame, timestamp = int(data.frame), int(data.timestamp)
        if frame < 0 or timestamp < 0:
            raise ValueError("invalid sensor frame header")
        return targets, frame, timestamp

    def _read_ground_truth_targets(self):
        data = self.structs.SimOne_Data_Obstacle()
        if not self.sensor_api.SoGetGroundTruth(self.config.vehicle_id, data):
            return None, -1, 0
        targets = []
        if int(data.obstacleSize) < 0 or int(data.obstacleSize) > len(data.obstacle):
            raise ValueError("ground truth object count out of range")
        for index in range(int(data.obstacleSize)):
            targets.append(self._target_dict(data.obstacle[index], 1.0))
        return targets, int(data.frame), int(data.timestamp)

    @staticmethod
    def _target_dict(item, probability):
        result = {
            "sdk_data": _native_dict(item),
            "id": int(item.id),
            "type": _sdk_int(item.type),
            "x": float(item.posX),
            "y": float(item.posY),
            "z": float(item.posZ),
            "vx": float(item.velX),
            "vy": float(item.velY),
            "vz": float(item.velZ),
            "heading": float(item.oriZ) if getattr(item, "oriZ", None) is not None else None,
            "roll": float(getattr(item, "oriX", 0.0)),
            "pitch": float(getattr(item, "oriY", 0.0)),
            "ax": float(getattr(item, "accelX", 0.0)),
            "ay": float(getattr(item, "accelY", 0.0)),
            "az": float(getattr(item, "accelZ", 0.0)),
            "length": float(item.length),
            "width": float(item.width),
            "height": float(item.height),
            "probability": float(probability),
            "sensor_range": float(getattr(item, "range", -1.0)),
        }
        for name, sdk_name in (("relative_x", "relativePosX"),
                               ("relative_y", "relativePosY"),
                               ("relative_z", "relativePosZ"),
                               ("relative_vx", "relativeVelX"),
                               ("relative_vy", "relativeVelY"),
                               ("relative_vz", "relativeVelZ"),
                               ("relative_roll", "relativeRotX"),
                               ("relative_pitch", "relativeRotY"),
                               ("relative_heading", "relativeRotZ")):
            if hasattr(item, sdk_name):
                result[name] = float(getattr(item, sdk_name))
        if hasattr(item, "bbox2dMinX"):
            result["bbox2d"] = [float(item.bbox2dMinX), float(item.bbox2dMinY),
                                float(item.bbox2dMaxX), float(item.bbox2dMaxY)]
        return result

    def _read_reference_data(self):
        """Scenario metadata changes rarely; avoid polling it at 20 Hz."""
        now = time.monotonic()
        ready = (self._reference_snapshot["sensor_configurations_valid"]
                 and bool(self._reference_snapshot["sensor_configurations"]))
        if self._reference_update and now - self._reference_update < (5.0 if ready else 0.5):
            return copy.deepcopy(self._reference_snapshot)
        snapshot = {"route_points": [], "route_waypoints": [], "route_valid": False,
                    "sensor_configurations": [], "sensor_configurations_valid": False,
                    "sensor_configurations_reason": "api_missing",
                    "environment": {},
                    "environment_valid": False, "traffic_signs": [],
                    "traffic_signs_valid": False, "reference_errors": []}
        try:
            if hasattr(self.pnc_api, "SoGetWayPoints"):
                route = self.structs.SimOne_Data_WayPoints()
                if self.pnc_api.SoGetWayPoints(self.config.vehicle_id, route):
                    count = int(route.wayPointsSize)
                    if count < 0 or count > len(route.wayPoints):
                        raise ValueError("waypoint count out of range")
                    route_points = [
                        (float(route.wayPoints[i].posX), float(route.wayPoints[i].posY))
                        for i in range(count)
                    ]
                    route_waypoints = [
                        {"index": int(getattr(route.wayPoints[i], "index", i)),
                         "x": float(route.wayPoints[i].posX),
                         "y": float(route.wayPoints[i].posY),
                         "heading_quaternion": [
                             float(getattr(route.wayPoints[i], field, 0.0))
                             for field in ("heading_x", "heading_y", "heading_z", "heading_w")
                         ]}
                        for i in range(count)
                    ]
                    snapshot["route_points"] = route_points
                    snapshot["route_waypoints"] = route_waypoints
                    snapshot["route_valid"] = bool(count)
        except Exception as exc:
            snapshot["reference_errors"].append("ROUTE_UNAVAILABLE:{0}".format(type(exc).__name__))
        try:
            if hasattr(self.sensor_api, "SoGetSensorConfigurations"):
                snapshot["sensor_configurations_reason"] = "read_failed"
                configurations = self.structs.SimOne_Data_SensorConfigurations()
                if self.sensor_api.SoGetSensorConfigurations(self.config.vehicle_id, configurations):
                    count = int(configurations.dataSize)
                    if count < 0 or count > len(configurations.data):
                        raise ValueError("sensor configuration count out of range")
                    values = [
                        {"id": _decode_sdk_text(configurations.data[i].sensorId),
                         "type": _decode_sdk_text(configurations.data[i].sensorType),
                         "hz": int(configurations.data[i].hz),
                         "x": float(configurations.data[i].x),
                         "y": float(configurations.data[i].y),
                         "z": float(configurations.data[i].z),
                         "roll": float(getattr(configurations.data[i], "roll", 0.0)),
                         "pitch": float(getattr(configurations.data[i], "pitch", 0.0)),
                         "sdk_data": _native_dict(configurations.data[i]),
                         "yaw": float(configurations.data[i].yaw)}
                        for i in range(count)
                    ]
                    snapshot["sensor_configurations"] = values
                    snapshot["sensor_configurations_valid"] = True
                    snapshot["sensor_configurations_reason"] = "ok" if values else "empty"
        except Exception as exc:
            snapshot["reference_errors"].append("SENSOR_CONFIG_UNAVAILABLE:{0}".format(type(exc).__name__))
            snapshot["sensor_configurations_reason"] = "invalid:" + type(exc).__name__
        try:
            if hasattr(self.sensor_api, "SoGetEnvironment"):
                environment = self.structs.SimOne_Data_Environment()
                if self.sensor_api.SoGetEnvironment(environment):
                    snapshot["environment"] = _native_dict(environment)
                    snapshot["environment_valid"] = True
        except Exception as exc:
            snapshot["reference_errors"].append("ENVIRONMENT_UNAVAILABLE:{0}".format(type(exc).__name__))
        if self.map_loaded and self.hdmap is not None:
            snapshot.update(self._map_reader().catalog())
            for name, meta in snapshot["map_observation_status"].items():
                if meta["reason"] in ("read_failed", "invalid_records"):
                    snapshot["reference_errors"].append(name.upper() + "_UNAVAILABLE:" + meta["reason"])
        self._reference_snapshot = snapshot
        self._reference_update = now
        return copy.deepcopy(snapshot)

    def _map_reader(self):
        if self._map_observations is None or self._map_observations.hdmap is not self.hdmap:
            self._map_observations = MapObservationReader(self.hdmap)
        return self._map_observations

    def read_map_observations(self, lane_ids):
        """Static lane-associated objects, independent of dynamic light colour."""
        if not self.map_loaded or self.hdmap is None:
            return {}
        return self._map_reader().lane_objects(lane_ids)

    def _read_auxiliary_data(self, configurations):
        result = {"imu": {}, "imu_valid": False, "radar_detections": [],
                  "radar_status": {}, "ultrasonic_detections": [],
                  "ultrasonic_valid": False, "sensor_lane_observations": [],
                  "sensor_lane_status": {},
                  "sensor_errors": []}
        if hasattr(self.sensor_api, "SoGetImu"):
            try:
                imu = self.structs.SimOne_Data_IMU()
                if self.sensor_api.SoGetImu(self.config.vehicle_id, imu):
                    result["imu"] = {
                        "sdk_data": _native_dict(imu),
                        "vx": float(getattr(imu, "velX", 0.0)),
                        "vy": float(getattr(imu, "velY", 0.0)),
                        "vz": float(getattr(imu, "velZ", 0.0)),
                        "roll_rate": float(getattr(imu, "angVelX", 0.0)),
                        "pitch_rate": float(getattr(imu, "angVelY", 0.0)),
                        "ax": float(imu.accelX), "ay": float(imu.accelY),
                        "az": float(imu.accelZ), "yaw_rate": float(imu.angVelZ),
                        "roll": float(imu.rotX), "pitch": float(imu.rotY),
                        "yaw": float(imu.rotZ)
                    }
                    result["imu_valid"] = True
                self._metadata("imu", imu, result["imu_valid"])
            except Exception as exc:
                self._source_status["imu"] = {"read_ok": False, "frame_id": -1,
                                               "timestamp": 0, "age_ms": -1}
                result["sensor_errors"].append("IMU_UNAVAILABLE:{0}".format(type(exc).__name__))
        for sensor in configurations:
            sensor_id = sensor.get("id", "")
            kind = sensor_kind(sensor.get("type", ""))
            if not sensor_id:
                continue
            if kind == "radar" and hasattr(self.sensor_api, "SoGetRadarDetections"):
                try:
                    radar = self.structs.SimOne_Data_RadarDetection()
                    radar_ok = self.sensor_api.SoGetRadarDetections(self.config.vehicle_id, sensor_id, radar)
                    self._metadata("radar:" + sensor_id, radar, radar_ok)
                    result["radar_status"][sensor_id] = bool(radar_ok)
                    if radar_ok:
                        count = int(radar.detectNum)
                        if count < 0 or count > len(radar.detections):
                            raise ValueError("radar count out of range")
                        for i in range(count):
                            hit = radar.detections[i]
                            result["radar_detections"].append({
                                "sdk_data": _native_dict(hit),
                                "sensor_id": sensor_id, "frame_id": int(radar.frame),
                                "timestamp": int(radar.timestamp), "id": int(hit.id),
                                "type": _sdk_int(hit.type), "x": float(hit.posX),
                                "y": float(hit.posY), "vx": float(hit.velX),
                                "vy": float(hit.velY), "range": float(hit.range),
                                "range_rate": float(hit.rangeRate),
                                "probability": float(hit.probability)
                            })
                            for name, field in (("sub_id", "subId"), ("z", "posZ"),
                                                ("vz", "velZ"), ("ax", "accelX"),
                                                ("ay", "accelY"), ("az", "accelZ"),
                                                ("roll", "oriX"), ("pitch", "oriY"),
                                                ("heading", "oriZ"), ("length", "length"),
                                                ("width", "width"), ("height", "height"),
                                                ("azimuth", "azimuth"), ("vertical", "vertical"),
                                                ("snr_db", "snrdb"), ("rcs_db", "rcsdb")):
                                if hasattr(hit, field):
                                    result["radar_detections"][-1][name] = float(getattr(hit, field))
                except Exception as exc:
                    self._source_status["radar:" + sensor_id] = {
                        "read_ok": False, "frame_id": -1, "timestamp": 0, "age_ms": -1}
                    result["radar_status"][sensor_id] = False
                    result["radar_detections"] = [
                        hit for hit in result["radar_detections"]
                        if hit["sensor_id"] != sensor_id
                    ]
                    result["sensor_errors"].append("RADAR_UNAVAILABLE:{0}:{1}".format(sensor_id, type(exc).__name__))
            if kind in ("camera", "fusion") and hasattr(self.sensor_api, "SoGetSensorLaneInfo"):
                try:
                    lane = self.structs.SimOne_Data_LaneInfo()
                    lane_ok = self.sensor_api.SoGetSensorLaneInfo(self.config.vehicle_id, sensor_id, lane)
                    self._metadata("lane:" + sensor_id, lane, lane_ok)
                    result["sensor_lane_status"][sensor_id] = bool(lane_ok)
                    if lane_ok:
                        result["sensor_lane_observations"].append({
                            "sdk_data": _native_dict(lane),
                            "lines": {name: _native_dict(getattr(lane, name))
                                      for name in ("ll_Line", "l_Line", "c_Line", "r_Line", "rr_Line")},
                            "left_lane_id": int(lane.laneLeftID),
                            "right_lane_id": int(lane.laneRightID),
                            "predecessor_ids_raw": _native_dict(lane.lanePredecessorID),
                            "successor_ids_raw": _native_dict(lane.laneSuccessorID),
                            "id_namespace": "sensor_local_not_hdmap",
                            "geometry_frame": "sdk_native_unverified",
                            "fixed_arrays_have_count": False,
                            "sensor_id": sensor_id, "frame_id": int(lane.frame),
                            "timestamp": int(lane.timestamp), "lane_id": int(lane.id),
                            "lane_type": _sdk_int(lane.laneType),
                            "left_line_type": _sdk_int(lane.l_Line.lineType),
                            "right_line_type": _sdk_int(lane.r_Line.lineType),
                            "left_line_color": _sdk_int(lane.l_Line.lineColor),
                            "right_line_color": _sdk_int(lane.r_Line.lineColor)
                        })
                except Exception as exc:
                    self._source_status["lane:" + sensor_id] = {
                        "read_ok": False, "frame_id": -1, "timestamp": 0, "age_ms": -1}
                    result["sensor_lane_status"][sensor_id] = False
                    result["sensor_errors"].append("SENSOR_LANE_UNAVAILABLE:{0}:{1}".format(sensor_id, type(exc).__name__))
        if hasattr(self.sensor_api, "SoGetUltrasonicRadars"):
            try:
                ultrasound = self.structs.SimOne_Data_UltrasonicRadars()
                if self.sensor_api.SoGetUltrasonicRadars(self.config.vehicle_id, ultrasound):
                    result["ultrasonic_valid"] = True
                    count = int(ultrasound.ultrasonicRadarNum)
                    if count < 0 or count > len(ultrasound.ultrasonicRadars):
                        raise ValueError("ultrasonic sensor count out of range")
                    for i in range(count):
                        sensor = ultrasound.ultrasonicRadars[i]
                        self._metadata("ultrasonic:" + _decode_sdk_text(sensor.sensorId), sensor, True)
                        hits = int(sensor.obstacleNum)
                        if hits < 0 or hits > len(sensor.obstacleDetections):
                            raise ValueError("ultrasonic target count out of range")
                        for j in range(hits):
                            hit = sensor.obstacleDetections[j]
                            result["ultrasonic_detections"].append({
                                "sensor_id": _decode_sdk_text(sensor.sensorId),
                                "frame_id": int(getattr(sensor, "frame", -1)),
                                "timestamp": int(getattr(sensor, "timestamp", 0)),
                                "range": float(hit.obstacleRanges),
                                "x": float(hit.x), "y": float(hit.y), "z": float(hit.z)
                            })
                    if not count:
                        self._metadata("ultrasonic:no_sensor", ultrasound, True)
            except Exception as exc:
                result["ultrasonic_valid"] = False
                result["ultrasonic_detections"] = []
                self._source_status["ultrasonic:all"] = {
                    "read_ok": False, "frame_id": -1, "timestamp": 0, "age_ms": -1}
                result["sensor_errors"].append("ULTRASONIC_UNAVAILABLE:{0}".format(type(exc).__name__))
        return result

    def read_traffic(self, lane_id=""):
        """Return traffic-light info for the main vehicle.

        Uses the dynamic per-frame signal truth via SoGetTrafficLights, keyed by
        the static light ids discovered from the HD map (getTrafficLightList).
        Returns a list of dicts, one per currently relevant traffic light:
          {opendrive_id, status, count_down, x, y, stop_line_x, stop_line_y,
           stop_line_boundary (when exposed), association_lane_id}
        A failed dynamic read retains its map candidate with read_ok=False.
        last_traffic_query distinguishes confirmed association from an absent
        map API; [] alone is not proof that a road has no applicable signal.
        """
        lights = []
        # Reset every query: never reuse a previous green/read receipt.
        self.last_traffic_query = {"association_valid": False, "read_ok": False}
        if (not lane_id or not self.map_loaded or self.hdmap is None or
                not hasattr(self.hdmap, "getTrafficLightList") or
                not hasattr(self.hdmap, "getStoplineList")):
            return lights
        if lane_id not in self._traffic_candidates:
            candidates = []
            traffic_light_list = self.hdmap.getTrafficLightList()
            if traffic_light_list is None:
                return lights
            count = int(traffic_light_list.Size()) if traffic_light_list else 0
            for index in range(count):
                light = traffic_light_list.GetElement(index)
                scopes = None
                scope_confirmed = False
                if hasattr(light, 'validities'):
                    # This binding's stop-line lookup can return a road's line
                    # even for a light whose explicit validity excludes the lane.
                    try:
                        scopes = [dict(road_id=int(scope.roadId), section_index=int(scope.sectionIndex),
                                       from_lane_id=int(scope.fromLaneId), to_lane_id=int(scope.toLaneId))
                                  for scope in vector_items(light.validities)]
                        if not matches_lane(scopes, lane_id):
                            continue
                        scope_confirmed = True
                    except (AttributeError, TypeError, ValueError, OverflowError):
                        # Keep an actually returned line as an unresolved
                        # applicability fact; do not fail the whole catalog.
                        pass
                stoplines = self.hdmap.getStoplineList(light, self.hdmap.pySimString(lane_id))
                if stoplines is None:
                    return lights  # unknown association, not confirmed absence
                if not stoplines:
                    continue
                stop_points = []
                for stop_index in range(stoplines.Size()):
                    stopline = stoplines.GetElement(stop_index)
                    knots = getattr(stopline, "boundaryKnots", None)
                    boundary = None
                    if knots and knots.Size():
                        boundary = [[float(knots.GetElement(j).x),
                                     float(knots.GetElement(j).y),
                                     float(getattr(knots.GetElement(j), 'z', 0.0))]
                                    for j in range(knots.Size())]
                        sx = sum(point[0] for point in boundary) / len(boundary)
                        sy = sum(point[1] for point in boundary) / len(boundary)
                    else:
                        sx = float(stopline.pt.x)
                        sy = float(stopline.pt.y)
                    point = {'stop_line_x': sx, 'stop_line_y': sy}
                    point['signal_scope_valid'] = scope_confirmed
                    if scopes is not None:
                        point['signal_validities'] = copy.deepcopy(scopes)
                    front = getattr(light, 'heading', None)
                    if front is not None and hasattr(front, 'x') and hasattr(front, 'y'):
                        hx, hy = float(front.x), float(front.y)
                        if not all(math.isfinite(v) for v in (hx, hy)):
                            raise ValueError('invalid signal heading')
                        if math.hypot(hx, hy) > 1e-9:
                            point['signal_heading'] = math.atan2(hy, hx)
                    if boundary is not None:
                        point['stop_line_boundary'] = boundary
                    stop_points.append(point)
                if stop_points:
                    pt = getattr(light, "pt", None)
                    candidates.append((int(light.id), getattr(pt, "x", None),
                                       getattr(pt, "y", None), stop_points))
            self._traffic_candidates[lane_id] = candidates
        self.last_traffic_query = {"association_valid": True, "read_ok": True}
        for opendrive_id, x, y, stop_points in self._traffic_candidates[lane_id]:
            read_ok = False
            status, count_down = 0, -1
            try:
                native = self.structs.SimOne_Data_TrafficLight()
                read_ok = bool(self.sensor_api.SoGetTrafficLights(
                    self.config.vehicle_id, opendrive_id, native))
                if read_ok:
                    status, count_down = _sdk_int(native.status), int(native.countDown)
            except Exception:
                read_ok, status, count_down = False, 0, -1
            if not read_ok:
                self.last_traffic_query["read_ok"] = False
            for stop_point in stop_points:
                # Accept older in-memory cache fixtures as point-only records.
                point = (copy.deepcopy(stop_point) if isinstance(stop_point, dict) else
                         dict(stop_line_x=stop_point[0], stop_line_y=stop_point[1]))
                point.update({
                    "opendrive_id": opendrive_id,
                    "status": status,
                    "count_down": count_down,
                    "read_ok": read_ok,
                    "association_lane_id": lane_id,
                    "x": float(x) if isinstance(x, (int, float)) else None,
                    "y": float(y) if isinstance(y, (int, float)) else None,
                })
                lights.append(point)
        return lights

    def get_driver_control(self):
        native = self.structs.SimOne_Data_Control()
        if not self.pnc_api.SoGetDriverControl(self.config.vehicle_id, native):
            return None
        if any(_sdk_int(getattr(native, field)) != 0 for field in
               ("EThrottleMode", "EBrakeMode", "ESteeringMode")) or native.isManualGear:
            return None  # Do not reinterpret angles/torques as percentages.
        result = ControlOut()
        result.throttle = float(native.throttle)
        result.brake = float(native.brake)
        result.steering = float(native.steering)
        result.gear = _sdk_int(native.gear)
        result.handbrake = bool(native.handbrake)
        result.valid = True
        result.source = "SimOneDriver"
        result.frame_id = int(native.frame)
        result.timestamp = int(native.timestamp)
        # A driver snapshot is observational, not a fresh command authorization.
        result.valid_until = 0.0
        try:
            result.clamp()
        except ValueError:
            return None
        return result

    def send_control(self, control):
        receipt = {"mode_ok": False, "drive_ok": False, "signals_ok": False,
                   "ok": False, "error": ""}
        self.last_send_result = receipt
        try:
            if not isinstance(control, ControlOut) or not control.valid:
                raise ValueError("control invalid")
            control.clamp()
            if (type(control.frame_id) is not int or not 0 <= control.frame_id <= 2147483647
                    or type(control.timestamp) is not int or not 0 <= control.timestamp < 2 ** 63
                    or not math.isfinite(control.valid_until)
                    or time.monotonic() >= control.valid_until):
                raise ValueError("control provenance missing or expired")
            native = self.structs.SimOne_Data_Control()
            native.frame, native.timestamp = control.frame_id, control.timestamp
            native.EThrottleMode = native.EBrakeMode = native.ESteeringMode = 0
            native.throttle, native.brake, native.steering = control.throttle, control.brake, control.steering
            native.handbrake = control.handbrake
            native.isManualGear = False
            native.gear = control.gear
            receipt["mode_ok"] = bool(self.pnc_api.SoSetDriveMode(self.config.vehicle_id, 0))
            if not receipt["mode_ok"]:
                raise RuntimeError("drive mode rejected")
            receipt["drive_ok"] = bool(self.pnc_api.SoSetDrive(self.config.vehicle_id, native))
            receipt["signals_ok"] = bool(self._send_signals(control))
            receipt["ok"] = receipt["drive_ok"] and receipt["signals_ok"]
        except Exception as exc:
            receipt["error"] = "{0}:{1}".format(type(exc).__name__, exc)
        return receipt["ok"]

    def _send_signals(self, control):
        lights = self.structs.SimOne_Data_Signal_Lights()
        lights.frame, lights.timestamp = control.frame_id, control.timestamp
        if control.hazard_signal:
            lights.signalLights = 4
        else:
            lights.signalLights = (2 if control.left_signal else 0) | (
                1 if control.right_signal else 0
            )
        return self.pnc_api.SoSetSignalLights(self.config.vehicle_id, lights)

    def shutdown(self):
        with self._target_lock:
            self._target_accepting = False
        if self.connected and self.service_api is not None:
            saved = True
            try:
                # SDK recording must still be alive when it writes its JSON.
                saved = self.flush_evaluation(force=True)
            finally:
                try:
                    self.service_api.SoTerminateSimOneAPI()
                finally:
                    self.connected = False
                    if self._evaluation is not None:
                        self._evaluation.active = False
                    with self._target_lock:
                        self._target_streams.clear()
                    self._target_probe_after.clear()
                    self._last_frame_times.clear()
                    self._reference_update = 0.0
            if not saved:
                raise RuntimeError("评价记录最终保存失败，请检查本次 captain.log")

    @staticmethod
    def gps_speed(gps):
        return math.sqrt(gps["vx"] * gps["vx"] + gps["vy"] * gps["vy"])
