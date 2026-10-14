"""Captain-owned process lifecycle and four-member integration pipeline."""

import json
import math
import os
import signal
import time

from core.serialization import perception_to_dict, to_dict
from core.diagnostic_writer import DiagnosticWriter
from core.tracking_measurements import TrackingMeasurements
from core.validation import validate_output
from core.interfaces import DecisionTarget, Trajectory, ControlOut
from core.safety_supervisor import SafetySupervisor,SafetyAssessment
from core.scene_requirements import requires_targets
from members.control_stub import compute_control, configure_control
from members.decision_stub import decide, decision_info, decision_settings, reset_decision
from members.planning_stub import plan, configure_planning
from core.behavior_channel import BehaviorTransport
from parking_integration import ParkingInputBridge,PARKING_ACTIONS
from lane_change_integration import LaneChangeInputBridge,LANE_CHANGE_CAPABILITIES
from driving_integration import DrivingInputBridge,DRIVING_ACTIONS
from behavior_inputs import load_behavior_inputs
from members.decision.settings import DecisionSettings
from perception.perception_builder import PerceptionBuilder
from perception.route_manager import RouteManager, ROUTE_REFERENCE_VERSION
from simone_platform.simone_adapter import SimOneAdapter
from simone_platform.sensor_catalog import TARGET_INGESTION_VERSION
from simone_platform.task_configuration import installed_task_parameters


def _member_output(value, expected, source):
    # An explicit failure is already unusable. Preserve its diagnostic instead
    # of replacing it with the generic validator's invalid-frame exception.
    if isinstance(value, expected) and value.valid is False:
        return value.bind(source)
    return validate_output(value, expected, source)


def _validate_motion_contract(planning, decision):
    """One set of body dimensions and motion-check bounds across members."""
    pairs = (("front_offset_m", "front_offset_m"),
             ("half_width_m", "half_width_m"),
             ("horizon", "motion_horizon_m"),
             ("deceleration", "motion_deceleration"),
             ("deceleration", "follow_deceleration"),
             ("motion_tolerance_mps", "motion_tolerance_mps"),
             ("projection_tolerance_m", "projection_tolerance_m"),
             ("route_ambiguity_m", "route_ambiguity_m"),
             ("lateral_guard_time_s", "motion_guard_time_s"),
             ("lateral_margin_m", "motion_lateral_margin_m"),
             ("traffic_stop_margin", "traffic_stop_margin"))
    for planning_name, decision_name in pairs:
        left, right = getattr(planning, planning_name), getattr(decision, decision_name)
        if left is None and right is None:
            continue
        if (left is None or right is None or
                not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("decision/planning motion settings differ: " + planning_name)


def target_input_summary(perception):
    """Report formal target state separately from diagnostic object count."""
    statuses = getattr(perception, "source_status", {})
    meta = statuses.get("targets", {}) if isinstance(statuses, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    source = getattr(perception, "target_source", "none")
    usable = (perception.targets_valid is True and isinstance(source, str)
              and source.startswith("sensor:") and meta.get("usable") is True
              and meta.get("sensor_read_ok", True) is not False)
    count = len(perception.targets)
    return {"required": requires_targets(getattr(perception, "scene_id", 0)),
            "state": ("valid_objects" if count else "valid_empty") if usable else "unavailable",
            "usable": usable, "source": source, "object_count": count,
            "diagnostic_only": source == "ground_truth",
            "frame_id": perception.targets_frame_id,
            "sensor_presence": meta.get("sensor_presence", "unknown"),
            "sensor_id": meta.get("sensor_id", ""),
            "sensor_read_ok": meta.get("sensor_read_ok", False),
            "quality": meta.get("quality", "unknown"),
            "reason": meta.get("reason", "unknown"),
            "transport": meta.get("transport", "none"),
            "id_verified": meta.get("id_verified", False),
            "candidates": list(meta.get("sensor_candidates", []))}


class CaptainRuntime(object):
    def __init__(self, config, logger,parking_inputs_provider=None,parking_actions=(),
                 lane_change_inputs_provider=None,lane_change_actions=(),
                 driving_inputs_provider=None,driving_actions=(),task_inputs_provider=None):
        self.config = config
        self.logger = logger
        self.configured_behaviors=load_behavior_inputs(config)
        if task_inputs_provider is None:
            task_inputs_provider = self.configured_behaviors.task_provider
        if parking_inputs_provider is None and not parking_actions:
            parking_inputs_provider=self.configured_behaviors.parking_provider
            parking_actions=self.configured_behaviors.parking_actions
        if lane_change_inputs_provider is None and not lane_change_actions:
            lane_change_inputs_provider=self.configured_behaviors.lane_change_provider
            lane_change_actions=self.configured_behaviors.lane_change_actions
        if driving_inputs_provider is None and not driving_actions:
            driving_inputs_provider=self.configured_behaviors.driving_provider
            driving_actions=self.configured_behaviors.driving_actions
        self.adapter = SimOneAdapter(config, logger)
        self.stop_requested = False
        self.pid_path = os.path.join(config.runtime_dir, "captain.pid")
        self.lock_path = os.path.join(config.runtime_dir, "captain.lock")
        self.stop_path = os.path.join(config.runtime_dir, "captain.stop")
        self._lock_stream = None
        self.snapshot_path = os.path.join(config.runtime_dir, "latest_perception.json")
        self.pipeline_path = os.path.join(config.runtime_dir, "latest_pipeline.json")
        self._warned_no_control = False
        self._snapshot_failures = set()
        self._diagnostics = DiagnosticWriter(lambda path, data:self._publish_json(path,data))
        self._diagnostic_next, self._diagnostic_event_next = 0.0, 0.0
        self._diagnostic_cycle = None
        self._loop_timing = {}
        self._input_task, self._input_readiness = None, {}
        self._input_retry_at = 0.
        self._tracking_measurements = TrackingMeasurements()
        self._timing_summary = dict(cycles=0,over_budget_cycles=0,max_ms={})
        self._braking_signature = None
        self._braking_next_sample = 0.0
        self._braking_event_count = 0
        self._braking_history = []
        self._braking_io_warned = False
        if (not isinstance(parking_actions,(tuple,list)) or
                (parking_actions and (set(parking_actions)!=set(PARKING_ACTIONS)
                 or len(parking_actions)!=len(PARKING_ACTIONS)
                 or parking_inputs_provider is None or getattr(config,'control_calibrated',False) is not True))):
            raise ValueError('parking capabilities require explicit paired inputs and configured control')
        self.parking=(ParkingInputBridge(parking_inputs_provider)
                      if parking_inputs_provider is not None else None)
        if (not isinstance(lane_change_actions,(tuple,list)) or
                (lane_change_actions and (set(lane_change_actions)!=set(LANE_CHANGE_CAPABILITIES)
                 or len(lane_change_actions)!=len(LANE_CHANGE_CAPABILITIES)
                 or lane_change_inputs_provider is None or getattr(config,'control_calibrated',False) is not True))):
            raise ValueError('lane-change capabilities require explicit paired inputs and configured control')
        # Both task assertions and error bounds come from the captain's provider.
        # A normal runtime never creates synthetic model or visibility evidence.
        self.lane_changes=(LaneChangeInputBridge(lane_change_inputs_provider)
                           if lane_change_inputs_provider is not None else None)
        if (not isinstance(driving_actions,(tuple,list)) or
                (driving_actions and (set(driving_actions)!=set(DRIVING_ACTIONS)
                 or len(driving_actions)!=len(DRIVING_ACTIONS)
                 or driving_inputs_provider is None or getattr(config,'control_calibrated',False) is not True))):
            raise ValueError('driving capabilities require explicit paired inputs and configured control')
        self.driving=(DrivingInputBridge(driving_inputs_provider)
                      if driving_inputs_provider is not None else None)
        planning_settings = configure_planning(config,parking_inputs_provider=(
            self.parking.planning_inputs if self.parking else None),lane_change_inputs_provider=(
            self.lane_changes.planning_inputs if self.lane_changes else None),driving_inputs_provider=(
            self.driving.planning_inputs if self.driving else None),
            vehicle_geometry=self.configured_behaviors.vehicle_geometry)
        if self.parking: self.parking.settings=planning_settings
        if self.lane_changes: self.lane_changes.settings=planning_settings
        if self.driving: self.driving.settings=planning_settings
        decision_model_settings=DecisionSettings.from_environment()
        geometry=self.configured_behaviors.vehicle_geometry
        if geometry is not None:
            for field in ('front_offset_m','half_width_m'):
                value=getattr(geometry,field); existing=getattr(decision_model_settings,field)
                if existing is not None and not math.isclose(existing,value,rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('decision/configured model body geometry differs: '+field)
                setattr(decision_model_settings,field,value)
            decision_model_settings.cruise_speed=min(decision_model_settings.cruise_speed,
                                                     self.configured_behaviors.limits.max_speed_mps)
        reset_decision(settings=decision_model_settings,
                       parking_inputs_provider=self.parking.decision_inputs if self.parking else None,
                       lane_change_inputs_provider=self.lane_changes.decision_inputs if self.lane_changes else None,
                       driving_inputs_provider=self.driving.decision_inputs if self.driving else None,
                       task_inputs_provider=task_inputs_provider)
        _validate_motion_contract(planning_settings, decision_settings())
        self.safety = SafetySupervisor(config, planning_settings=planning_settings,
                                       decision_settings=decision_settings(),parking_guard=(
                                           self.parking.assess if self.parking and parking_actions else None),
                                       lane_change_guard=(self.lane_changes.assess
                                           if self.lane_changes and lane_change_actions else None),
                                       driving_guard=(self.driving.assess
                                           if self.driving and driving_actions else None),
                                       execution_guard=self._assess_execution)
        self.runtime_info = decision_info()
        self.runtime_info.update({"pid": os.getpid(),
                                  "runtime_io_version": "bounded-diagnostics-exact-pose-v1",
                                  "project_dir": os.path.dirname(os.path.abspath(__file__)),
                                  "target_ingestion_version": TARGET_INGESTION_VERSION,
                                  "route_reference_version": ROUTE_REFERENCE_VERSION,
                                  "planning_settings": dict(vars(planning_settings)),
                                  "behavior_model_status":self.configured_behaviors.status,
                                  "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        configure_control(config)
        actions=list(("PATH_STOP","DWELL","FEEDBACK", "LIGHTS")
                     if getattr(config,"control_calibrated",False) else ())
        if task_inputs_provider is not None:
            if getattr(config, 'control_calibrated', False) is not True:
                raise ValueError('task progress requires configured control')
            actions.extend(('TASK_ROUTE', 'PATH'))
        self.behaviors = BehaviorTransport(tuple(dict.fromkeys(
            actions+list(parking_actions)+list(lane_change_actions)+list(driving_actions))))

    def _assess_execution(self,perception,trajectory):
        # All configured bridges check the previous actually sent command,
        # including ordinary paths between their private maneuver stages.
        for bridge in (self.parking,self.lane_changes,self.driving):
            if bridge is not None and hasattr(bridge,'assess_execution'):
                result=bridge.assess_execution(perception,trajectory)
                if result.active:
                    return result
        return SafetyAssessment()

    def request_stop(self, unused_signal=None, unused_frame=None):
        self.stop_requested = True

    def run(self, once=False):
        self._install_signals()
        if not self._acquire_instance_lock():
            self.logger.warning("已有队长进程运行，忽略重复启动 pid=%s", os.getpid())
            return
        try:
            self._write_pid()
            self.logger.info(
                "队长启动 pid=%s send_control=%s safety_brake_enabled=%s 日志=%s",
                os.getpid(), self.config.send_control,
                self.config.safety_brake_enabled,
                os.path.join(self.config.runtime_dir, "captain.log"),
            )
            info = decision_info()
            self.logger.info(
                "决策实现 version=%s engine=%s cruise_speed=%.3f m/s (%.1f km/h)",
                info["version"], info["engine"], info["cruise_speed"],
                info["cruise_speed"] * 3.6,
            )
            self.logger.info("路线接续实现 version=%s", ROUTE_REFERENCE_VERSION)
            self.logger.info('运行诊断 version=%s interval=%.3fs GPS采用同帧回调历史',
                             self.runtime_info['runtime_io_version'],self.config.diagnostic_interval_s)
            try:
                self.adapter.bootstrap()
                self.adapter.initialize()
                self.adapter.load_hdmap()
                route_manager = RouteManager(self.adapter, self.logger)
                builder = PerceptionBuilder(
                    self.adapter, route_manager, self.config.scene_id_override
                )
                info = builder.update_case_info()
                self.logger.info(
                    "案例: %s (scene_id=%s)", info.get("case_name", ""), builder.scene_id
                )
                self._wait_until_running()
                self._loop(builder, once)
            finally:
                try:
                    self._diagnostics.close()
                    self.adapter.shutdown()
                finally:
                    self._remove_pid()
        finally:
            self._release_instance_lock()

    def _loop(self, builder, once):
        period = 1.0 / self.config.loop_hz
        last_frame = None
        last_route_signature = None
        processed = 0
        while not self.stop_requested:
            if self._consume_stop_request():
                break
            start = time.time()
            cycle_start = time.perf_counter()
            if processed == 0:
                self.logger.info("首帧检查: 查询案例状态")
            status = self.adapter.get_case_status()
            if status == self.adapter.CASE_STOP:
                self.logger.info("案例已停止，队长进程退出")
                break
            if status != self.adapter.CASE_RUNNING:
                self._flush_evaluation()
                time.sleep(0.2)
                continue
            if processed == 0:
                self.logger.info("首帧检查: 案例运行中，开始读取感知")
            phase_start = time.perf_counter()
            perception = builder.build()
            timing = {'perception':(time.perf_counter()-phase_start)*1000.}
            lane = perception.lane
            route_signature = (lane.lane_id, tuple(lane.forward_lane_ids),
                               lane.forward_reference_valid, lane.forward_reference_status)
            if route_signature != last_route_signature:
                self.logger.info("前方参考 lane=%s lanes=%s valid=%s status=%s points=%s",
                                 lane.lane_id, ",".join(lane.forward_lane_ids),
                                 lane.forward_reference_valid, lane.forward_reference_status,
                                 len(lane.forward_reference))
                last_route_signature = route_signature
            if processed == 0:
                self.logger.info("首帧检查: 感知读取完成 frame=%s valid=%s",
                                 perception.frame_id, perception.valid)
            repeated = last_frame == perception.frame_id
            last_frame = perception.frame_id
            self.behaviors.prepare(perception)
            if self.parking: self.parking.prepare(perception)
            if self.lane_changes: self.lane_changes.prepare(perception)
            if self.driving: self.driving.prepare(perception)
            phase_start = time.perf_counter()
            try:
                decision = _member_output(decide(perception), DecisionTarget, perception)
            except Exception as exc:
                decision = DecisionTarget().bind(perception)
                decision.errors.append("DECISION_INVALID:" + type(exc).__name__ + ":" + str(exc))
            timing['decision'] = (time.perf_counter()-phase_start)*1000.
            phase_start = time.perf_counter()
            try:
                trajectory = _member_output(plan(perception, decision), Trajectory, decision)
            except Exception as exc:
                trajectory = Trajectory().bind(decision)
                trajectory.errors.append("TRAJECTORY_INVALID:" + type(exc).__name__ + ":" + str(exc))
            timing['planning'] = (time.perf_counter()-phase_start)*1000.
            phase_start = time.perf_counter()
            try:
                control = compute_control(perception, trajectory)
                if control.valid:
                    control = validate_output(control, ControlOut, trajectory)
            except Exception as exc:
                control = ControlOut().bind(trajectory)
                control.errors.append("CONTROL_INVALID:" + type(exc).__name__ + ":" + str(exc))
            timing['control'] = (time.perf_counter()-phase_start)*1000.
            phase_start = time.perf_counter()
            safety = self.safety.evaluate(perception, decision, trajectory, control)
            timing['safety'] = (time.perf_counter()-phase_start)*1000.
            phase_start = time.perf_counter()
            receipt = {"attempted": False, "ok": False, "reason": "observe_mode",
                       "safety_mode": safety.mode, "safety_reason": safety.reason,
                       "safety_candidate": safety.control is not None}
            if safety.active:
                if (self.config.send_control and
                        getattr(self.config, "safety_brake_enabled", False) and
                        safety.control is not None):
                    receipt["attempted"] = True
                    receipt["ok"] = bool(self.adapter.send_control(safety.control))
                    receipt["reason"] = ("safety_sent" if receipt["ok"]
                                         else "safety_send_failed")
                    receipt["adapter"] = to_dict(getattr(self.adapter, "last_send_result", {}))
                elif safety.control is None:
                    receipt["reason"] = "safety_command_unavailable"
                else:
                    receipt["reason"] = "safety_observe_only"
            elif self.config.send_control:
                if not getattr(self.config, "safety_brake_enabled", False):
                    receipt["reason"] = "safety_brake_not_armed"
                elif repeated:
                    receipt["reason"] = "gps_frame_repeated"
                elif decision.valid and trajectory.valid and control.valid:
                    receipt["attempted"] = True
                    receipt["ok"] = bool(self.adapter.send_control(control))
                    receipt["reason"] = "sent" if receipt["ok"] else "send_failed"
                    receipt["adapter"] = to_dict(getattr(self.adapter, "last_send_result", {}))
                    if not receipt["ok"]:
                        self.logger.warning("控制指令发送失败 frame=%s receipt=%s",
                                            perception.frame_id, receipt["adapter"])
                elif not self._warned_no_control:
                    receipt["reason"] = "pipeline_invalid"
                    self.logger.warning("数据链或控制输出无效，不发送控制 frame=%s", perception.frame_id)
                    self._warned_no_control = True
                else:
                    receipt["reason"] = "pipeline_invalid"
            self.behaviors.observe(perception,decision,trajectory,control,safety.active,receipt)
            timing['send_and_feedback'] = (time.perf_counter()-phase_start)*1000.
            for bridge in (self.parking,self.lane_changes,self.driving):
                if bridge is not None:
                    bridge.record_sent(perception,trajectory,receipt,safety.active)
            if self.lane_changes:
                self.lane_changes.observe_lights(perception,decision,trajectory,control,
                                                self.behaviors,receipt,safety.active)
            if self.config.publish_json:
                self._tracking_measurements.record(perception,trajectory,receipt,safety.active,time.monotonic())
            # Flush after sending control, so file I/O cannot expire its frame.
            phase_start = time.perf_counter()
            self._flush_evaluation()
            timing['evaluation'] = (time.perf_counter()-phase_start)*1000.
            phase_start = time.perf_counter()
            timing['before_diagnostics'] = (phase_start-cycle_start)*1000.
            timing['previous_cycle'] = self._loop_timing.get('total',0.)
            timing['previous_diagnostic_enqueue'] = self._loop_timing.get('diagnostic_enqueue',0.)
            self._loop_timing = timing
            if self.config.publish_json and time.monotonic() >= self._diagnostic_next:
                self._diagnostic_next = time.monotonic()+getattr(self.config,'diagnostic_interval_s',.5)
                self._diagnostic_cycle = processed
                self._publish(perception)
                self._publish_pipeline(perception, decision, trajectory, control, receipt,
                                       safety)
            self._trace_braking(perception, decision, trajectory, control, receipt, safety)
            timing['diagnostic_enqueue'] = (time.perf_counter()-phase_start)*1000.
            processed += 1
            if processed == 1 or processed % 100 == 0:
                target_status = perception.source_status.get("targets", {})
                targets_usable = (perception.targets_valid
                                  and isinstance(perception.target_source, str)
                                  and perception.target_source.startswith("sensor:")
                                  and target_status.get("usable", False))
                self.logger.info(
                    "frame=%s valid=%s lane=%s targets=%s target_source=%s targets_usable=%s "
                    "target_reason=%s speed=%.2fm/s decision=%s mode=%s decision_reason=%s "
                    "trajectory=%s control=%s send=%s reason=%s "
                    "stop_distance=%.2f brake=%.3f reference_speed=%s safety_reason=%s",
                    perception.frame_id,
                    perception.valid,
                    perception.lane.lane_id,
                    len(perception.targets),
                    perception.target_source, targets_usable,
                    target_status.get("reason", "unknown"),
                    perception.ego.speed,
                    decision.valid, decision.mode, decision.reason,
                    trajectory.valid, control.valid, receipt["reason"],
                    trajectory.reason + "; " + "; ".join(control.errors),
                    trajectory.stop_distance, control.brake,
                    control.diagnostics.get("reference_speed_mps"), safety.reason,
                )
            timing['total'] = (time.perf_counter()-cycle_start)*1000.
            self._timing_summary['cycles'] += 1
            self._timing_summary['over_budget_cycles'] += int(timing['total']>period*1000.)
            for key,value in timing.items():
                if not key.startswith('previous_'):
                    maxima = self._timing_summary['max_ms']
                    maxima[key] = max(maxima.get(key,0.),value)
            if once:
                break
            self._sleep_remaining(start, period)
        if (processed and self.config.publish_json and self._diagnostic_cycle != processed-1):
            self._publish(perception)
            self._publish_pipeline(perception, decision, trajectory, control, receipt, safety)
        self.flush_diagnostics()

    def flush_diagnostics(self, timeout=2.0):
        return self._diagnostics.flush(timeout)

    def _trace_braking(self, perception, decision, trajectory, control, receipt, safety):
        """Retain brief brake events that fall between periodic log samples.

        This runs after sending and never changes member outputs or receipts.
        Save at most 80 full snapshots per process, with eight preceding brief
        frames and a two-second sample interval during an unchanged brake.
        """
        commanded = safety.control if safety.active else control
        active = (safety.active or not trajectory.valid or trajectory.emergency_stop
                  or (commanded is not None and commanded.valid
                      and (commanded.brake > 0 or commanded.handbrake)))
        signature = (active, safety.mode, safety.reason, trajectory.valid,
                     trajectory.emergency_stop, trajectory.reason,
                     control.valid, control.source, tuple(control.errors), decision.mode)
        now = time.monotonic()
        previous = self._braking_signature
        changed = ((active or (previous is not None and previous[0]))
                   and signature != previous)
        sample = active and now >= self._braking_next_sample
        brief = {"frame_id": perception.frame_id, "speed_mps": perception.ego.speed,
                 "observed_brake": perception.ego.brake, "decision_mode": decision.mode,
                 "decision_reason": decision.reason, "desired_speed_mps": decision.target_speed,
                 "stop_distance_m": trajectory.stop_distance,
                 "trajectory_reason": trajectory.reason, "control_source": control.source,
                 "control_brake": control.brake, "control_throttle": control.throttle,
                 "control_diagnostics": to_dict(control.diagnostics),
                 "safety_reason": safety.reason, "send": dict(receipt),
                 "target_count": len(perception.targets)}
        self._braking_signature = signature
        if changed or sample:
            event = ("brake_start" if active and (previous is None or not previous[0])
                     else "brake_release" if not active else
                     "brake_change" if changed else "brake_sample")
            self._braking_next_sample = now + 2.0
            self.logger.info("braking_event=%s frame=%s decision=%s trajectory=%s "
                             "stop_distance=%.2f control_brake=%.3f reference_speed=%s "
                             "safety_reason=%s send=%s",
                             event, perception.frame_id, decision.reason, trajectory.reason,
                             trajectory.stop_distance, control.brake,
                             control.diagnostics.get("reference_speed_mps"),
                             safety.reason, receipt.get("reason"))
            full_event = event in ('brake_start','brake_release') or now >= self._diagnostic_event_next
            if self.config.publish_json and full_event and self._braking_event_count < 80:
                self._diagnostic_event_next = now+max(.5,getattr(self.config,'diagnostic_interval_s',.5))
                self._braking_event_count += 1
                directory = os.path.join(self.config.runtime_dir, "braking_events")
                path = os.path.join(directory, "{0}-{1:03d}-{2}.json".format(
                    self.runtime_info["pid"], self._braking_event_count, perception.frame_id))
                try:
                    self._diagnostics.submit(path, {
                        "event": event, "runtime": dict(self.runtime_info),
                        "recorded_monotonic": now, "preceding_frames": list(self._braking_history),
                        "summary": brief, "perception": perception_to_dict(perception),
                        "decision": to_dict(decision), "trajectory": to_dict(trajectory),
                        "control": to_dict(control), "send": dict(receipt),
                        "safety": {"mode": safety.mode, "reason": safety.reason,
                                   "candidate": to_dict(safety.control)}})
                except (OSError, TypeError, ValueError) as exc:
                    if not self._braking_io_warned:
                        self.logger.warning("刹车事件快照写入失败，控制循环继续: %s", exc)
                        self._braking_io_warned = True
        self._braking_history.append(brief)
        self._braking_history = self._braking_history[-8:]

    def _wait_until_running(self):
        last_status = None
        self.logger.info("等待案例运行: 开始查询 SDK 状态")
        while not self.stop_requested:
            if self._consume_stop_request():
                return
            status = self.adapter.get_case_status()
            if status == self.adapter.CASE_RUNNING:
                self.logger.info("案例状态已运行，进入感知/控制循环")
                return
            if status != last_status:
                self.logger.info("等待案例运行，当前状态=%s", status)
                last_status = status
            time.sleep(0.5)

    def _flush_evaluation(self):
        flush = getattr(self.adapter, "flush_evaluation", None)
        if flush is not None:
            flush()

    def _evaluation_status(self):
        status = getattr(self.adapter, "evaluation_status", None)
        return status() if status is not None else {}

    def _consume_stop_request(self):
        try:
            with open(self.stop_path, encoding="ascii") as stream:
                requested_pid = stream.read().strip()
        except (OSError, UnicodeError):
            return False
        if requested_pid == str(os.getpid()):
            self.stop_requested = True
            self.logger.info("收到结束脚本停止请求，保存评价并退出")
        return self.stop_requested

    def _publish(self, perception):
        return self._diagnostics.submit(self.snapshot_path, perception_to_dict(perception))

    def _publish_pipeline(self, perception, decision, trajectory, control, receipt, safety):
        return self._diagnostics.submit(self.pipeline_path, {
            "runtime": dict(self.runtime_info),
            "timings_ms": dict(self._loop_timing),
            "timing_summary": to_dict(self._timing_summary),
            "diagnostics_io": self._diagnostics.status(),
            "tracking_measurements": self._tracking_measurements.summary(),
            "behavior_input_readiness": dict(case_id=perception.case_id,task_id=perception.task_id,
                behavior_model_status=self.configured_behaviors.status,
                sensor_visibility_configured=bool(getattr(self.config,'sensor_visibility_file','')),
                gps_pose_callback=self.adapter.gps_poses.status if hasattr(self.adapter,'gps_poses') else 'unavailable'),
            "parking_integration": {"configured": self.parking is not None,
                                    "reason": self.parking.reason if self.parking else "NOT_CONFIGURED"},
            "evaluation": self._evaluation_status(),
            "perception_frame_id": perception.frame_id,
            "target_input": target_input_summary(perception),
            "route_reference": {
                "valid": perception.lane.forward_reference_valid,
                "lane_ids": list(perception.lane.forward_lane_ids),
                "status": perception.lane.forward_reference_status,
                "point_count": len(perception.lane.forward_reference)},
            "map_observations": {
                "parking_count": len(perception.parking_spaces),
                "stop_line_count": len(perception.map_stop_lines),
                "crosswalk_count": len(perception.map_crosswalks),
                "speed_limits": to_dict(perception.speed_limit_observations),
                "status": to_dict(perception.map_observation_status)},
            "maneuver_environment": to_dict(perception.maneuver_environment),
            "decision": to_dict(decision), "trajectory": to_dict(trajectory),
            "control": to_dict(control), "safety": {
                "mode": safety.mode, "reason": safety.reason,
                "candidate": to_dict(safety.control)}, "send": receipt})

    def _publish_json(self, path, data):
        # Snapshots are diagnostics, not the member-to-member transport. On
        # Windows an open reader can temporarily prevent atomic replacement.
        # Never let this optional I/O failure terminate the driving loop.
        try:
            # Installed task-cache reads also belong to the diagnostic worker.
            readiness = data.get('behavior_input_readiness')
            if path == self.pipeline_path and isinstance(readiness,dict):
                task = (readiness['case_id'],readiness['task_id'])
                if task != self._input_task or time.monotonic()>=self._input_retry_at:
                    self._input_task = task
                    self._input_readiness = installed_task_parameters(getattr(self.config,'sdk_dir',''),
                        task[1],getattr(self.config,'vehicle_id','0'))
                    self._input_retry_at = (float('inf') if self._input_readiness['status']=='TASK_PARAMETERS_FOUND'
                                            else time.monotonic()+5.)
                readiness['installed_parameters'] = self._input_readiness
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2,
                          sort_keys=True, allow_nan=False)
            os.replace(temporary, path)
        except OSError as exc:
            if path not in self._snapshot_failures:
                self.logger.warning(
                    "诊断快照写入失败，本帧跳过该快照，控制循环继续: %s: %s",
                    path, exc)
            self._snapshot_failures.add(path)
            return False
        if path in self._snapshot_failures:
            self._snapshot_failures.remove(path)
            self.logger.info("诊断快照写入已恢复: %s", path)
        return True

    @staticmethod
    def _sleep_remaining(start, period):
        remaining = period - (time.time() - start)
        if remaining > 0.0:
            time.sleep(remaining)

    def _install_signals(self):
        signal.signal(signal.SIGINT, self.request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self.request_stop)

    def _acquire_instance_lock(self):
        if not os.path.isdir(self.config.runtime_dir):
            os.makedirs(self.config.runtime_dir)
        stream = open(self.lock_path, "a+b")
        try:
            if os.path.getsize(self.lock_path) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self._lock_stream = stream
        return True

    def _release_instance_lock(self):
        if self._lock_stream is None:
            return
        try:
            self._lock_stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock_stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_stream.close()
            self._lock_stream = None

    def _write_pid(self):
        if not os.path.isdir(self.config.runtime_dir):
            os.makedirs(self.config.runtime_dir)
        with open(self.pid_path, "w", encoding="ascii") as stream:
            stream.write(str(os.getpid()))

    def _remove_pid(self):
        if os.path.exists(self.pid_path):
            try:
                os.remove(self.pid_path)
            except OSError:
                pass
