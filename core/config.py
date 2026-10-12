"""INI configuration with project-relative defaults."""

import configparser
import math
import os


class AppConfig(object):
    def __init__(self, project_dir, values):
        self.project_dir = project_dir
        self.sdk_dir = values.get("sdk_dir", r"E:\Sim-One\SimOneAPI\lib\Win64")
        self.python_exe = values.get("python_exe", r"E:\Sim-One\Tools\python36\python.exe")
        self.server_ip = values.get("server_ip", "127.0.0.1")
        self.vehicle_id = values.get("vehicle_id", "0")
        self.sensor_id = values.get("sensor_id", "perfectPerception1")
        coverage = values.get("sensor_visibility_file", "").strip()
        self.sensor_visibility_file = (os.path.abspath(os.path.join(project_dir, coverage))
                                       if coverage else "")
        behavior_model = values.get("behavior_model_file", "").strip()
        self.behavior_model_file = (os.path.abspath(os.path.join(project_dir, behavior_model))
                                    if behavior_model else "")
        map_file = values.get("map_document_file", "").strip()
        self.map_document_file = (os.path.abspath(os.path.join(project_dir,map_file)) if map_file else "")
        self.loop_hz = max(1.0, float(values.get("loop_hz", "20")))
        self.sensor_timeout_ms = max(1, int(values.get("sensor_timeout_ms", "500")))
        self.max_sensor_frame_gap = max(0, int(values.get("max_sensor_frame_gap", "10")))
        self.pipeline_timeout_ms = max(1, int(values.get("pipeline_timeout_ms", "200")))
        self.map_timeout_sec = max(1.0, float(values.get("map_timeout_sec", "100")))
        self.connect_timeout_sec = max(1.0, float(values.get("connect_timeout_sec", "30")))
        self.scene_id_override = int(values.get("scene_id_override", "0"))
        self.send_control = _as_bool(values.get("send_control", "false"))
        self.evaluation_enabled = _as_bool(values.get("evaluation_enabled", "true"))
        self.evaluation_flush_interval_sec = float(
            values.get("evaluation_flush_interval_sec", "1.0"))
        if (not math.isfinite(self.evaluation_flush_interval_sec) or
                self.evaluation_flush_interval_sec < 0.1):
            raise ValueError("evaluation_flush_interval_sec must be finite and >= 0.1")
        self.control_calibrated = _as_bool(values.get("control_calibrated", "false"))
        for name in ("wheelbase_m", "front_steer_max_rad", "throttle_per_mps",
                     "brake_per_mps", "hold_brake", "emergency_brake",
                     "max_throttle", "max_brake"):
            setattr(self, "control_" + name,
                    _optional_float(values.get("control_" + name, "")))
        sign = values.get("control_steering_sign", "").strip()
        self.control_steering_sign = int(sign) if sign else None
        # Normal sending and brake-only overrides both require this safety exit.
        self.safety_brake_enabled = _as_bool(values.get("safety_brake_enabled", "false"))
        self.safety_controlled_brake = float(values.get("safety_controlled_brake", "0.3"))
        self.safety_emergency_brake = float(values.get("safety_emergency_brake", "1.0"))
        self.safety_recovery_frames = max(1, int(values.get("safety_recovery_frames", "3")))
        self.publish_json = _as_bool(values.get("publish_json", "true"))
        self.log_level = values.get("log_level", "INFO").upper()
        self.runtime_dir = os.path.join(project_dir, "runtime_data")


def _as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _optional_float(value):
    text = str(value).strip()
    return float(text) if text else None


def load_config(project_dir, custom_path=None):
    parser = configparser.ConfigParser()
    default_path = os.path.join(project_dir, "config", "default.ini")
    loaded = parser.read(default_path, encoding="utf-8")
    if not loaded:
        raise RuntimeError("配置文件不存在: {0}".format(default_path))
    if custom_path:
        custom_path = os.path.abspath(custom_path)
        if not os.path.exists(custom_path):
            raise RuntimeError("指定配置文件不存在: {0}".format(custom_path))
        parser.read(custom_path, encoding="utf-8")
    if not parser.has_section("app"):
        raise RuntimeError("配置缺少 [app] 段")
    return AppConfig(project_dir, dict(parser.items("app")))
