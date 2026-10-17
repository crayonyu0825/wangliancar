"""Diagnose why a NEVC scene can or cannot score, read-only.

Reads a saved captain Perception snapshot plus the project configuration and
reports, per gate, whether the decision member will drive the task or protect
(stop/brake). This is a diagnosis tool: it never connects to SimOne, never
sends control and never claims a scene passed.

The gates mirror the real code paths, so the report answers "which single
condition blocks this scene" rather than producing a pass/fail impression.

    python scripts/diagnose_scene.py --snapshot runtime_data/latest_perception.json
    python scripts/diagnose_scene.py --directory runtime_data/scene_acceptance
    python scripts/diagnose_scene.py --snapshot x.json --json-out report.json
"""

from __future__ import print_function

import argparse
import glob
import json
import os
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.config import load_config  # noqa: E402
from core.interfaces import EgoState, LaneContext, Perception, Target, TrafficControl  # noqa: E402
from core.scene_requirements import (  # noqa: E402
    AEB_SCENES, FOLLOW_SCENES, SIGNAL_SCENES, requires_targets)
from simone_platform.sensor_catalog import target_sensor_ids  # noqa: E402


DEFAULT_SNAPSHOT = os.path.join(PROJECT_DIR, "runtime_data", "latest_perception.json")

# Scenes whose task needs a behaviour-model capability (parking, lane change,
# passage). Source: config/BEHAVIOR_INPUTS.md and the action tuples in
# runtime.py; a behaviour model is required before these are enabled at all.
PARKING_SCENES = frozenset((7, 8))
LANE_CHANGE_SCENES = frozenset((16, 17, 26, 27, 28, 29))
PASSAGE_SCENES = frozenset((18, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40))
CONTINUOUS_SCENES = frozenset((41,))

PASS = "PASS"
BLOCK = "BLOCK"
UNKNOWN = "UNKNOWN"

NESTED = {"ego": EgoState, "lane": LaneContext, "traffic": TrafficControl,
          "targets": Target}
SCALARS = (int, float, bool, str)


def _assign(obj, data, path):
    if not isinstance(data, dict):
        raise ValueError("{0}: expected object, got {1}".format(path, type(data).__name__))
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if key in NESTED:
            items = value if isinstance(value, list) else [value]
            built = []
            for index, item in enumerate(items):
                child = NESTED[key]()
                _assign(child, item, "{0}.{1}[{2}]".format(path, key, index))
                built.append(child)
            setattr(obj, key, built if isinstance(value, list) else built[0])
        elif isinstance(current, list) and isinstance(value, list):
            setattr(obj, key, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            setattr(obj, key, value)
        elif value is None:
            if isinstance(current, float):
                setattr(obj, key, float("nan"))
        elif isinstance(value, SCALARS):
            setattr(obj, key, value)
    return obj


def load_perception(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or "ego" not in data:
        raise ValueError("{0}: not a captain perception snapshot".format(path))
    return _assign(Perception(), data, os.path.basename(path))


def snapshot_files(target):
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.json")))
        if not files:
            raise ValueError("no .json snapshots in " + target)
        return files
    if not os.path.exists(target):
        raise ValueError("path does not exist: " + target)
    return [target]


def _result(status, detail):
    return {"status": status, "detail": detail}


def _target_gate(perception):
    """Mirror members/decision/protocol.py:targets_usable plus requires_targets."""
    scene = perception.scene_id
    required = requires_targets(scene)
    source = perception.target_source
    status = perception.source_status.get("targets", {}) if isinstance(
        perception.source_status, dict) else {}
    sensor_ok = (isinstance(source, str) and source.startswith("sensor:"))
    usable = isinstance(status, dict) and status.get("usable") is True
    read_ok = isinstance(status, dict) and status.get("sensor_read_ok") is not False
    parsed = perception.targets_valid is True and isinstance(perception.targets, list)
    observed = parsed and sensor_ok and usable and read_ok

    if not required:
        return _result(PASS, "scene {0} does not need target observations".format(scene))
    if not parsed:
        return _result(BLOCK, "target frame missing or unparsable (targets_valid=False)")
    if not sensor_ok:
        if source == "ground_truth":
            return _result(BLOCK, "fell back to Ground Truth; the decision member rejects "
                                  "ground_truth as evidence (Sensor API produced no frame)")
        return _result(BLOCK, "target source is '{0}', not a Sensor API source".format(source))
    if not usable:
        return _result(BLOCK, "source_status.targets.usable is not True (quality={0})".format(
            status.get("quality") if isinstance(status, dict) else "?"))
    if not read_ok:
        return _result(BLOCK, "source_status.targets.sensor_read_ok is False")
    return _result(PASS, "Sensor API target frame usable ({0} object(s))".format(
        len(perception.targets)))


def _sensor_config_gate(perception, config):
    """Whether the platform published a configured detection source at all."""
    configurations = perception.sensor_configurations
    ok = perception.sensor_configurations_valid is True
    candidates = target_sensor_ids(configurations, getattr(config, "sensor_id", "")) \
        if ok and isinstance(configurations, list) else []
    if not requires_targets(perception.scene_id):
        return _result(PASS, "not required for this scene")
    if not ok:
        return _result(BLOCK, "platform sensor configurations not readable; a fresh "
                              "Sensor API frame cannot be discovered")
    if not candidates:
        return _result(BLOCK, "platform published no camera/lidar/fusion/perfect "
                              "detection source for this task")
    return _result(PASS, "configured detection sources: " + ", ".join(candidates))


def _ego_gate(perception):
    from core.validation import current
    if not current(perception):
        return _result(BLOCK, "captain frame expired (valid_until is process-local; a "
                              "saved snapshot always fails this)")
    if perception.ego.valid is not True:
        return _result(BLOCK, "ego state invalid")
    return _result(PASS, "ego frame valid")


def _lane_gate(perception):
    if perception.lane.valid is True:
        return _result(PASS, "lane context valid (id={0})".format(perception.lane.lane_id))
    if perception.scene_id in PARKING_SCENES:
        return _result(PASS, "parking scene: lane context is not the driving reference")
    return _result(BLOCK, "lane context invalid; decision protects instead of driving")


def _signal_gate(perception):
    scene = perception.scene_id
    if scene not in SIGNAL_SCENES:
        return _result(PASS, "scene {0} does not require signal data".format(scene))
    traffic = perception.traffic
    if traffic.observed and not traffic.ambiguous:
        return _result(PASS, "signal observed: {0}".format(traffic.signal_state))
    if traffic.ambiguous:
        return _result(BLOCK, "signal group unresolved; decision stops")
    return _result(UNKNOWN, "no signal observed yet; decision keeps protection until the "
                            "map proves no applicable lamp ahead")


def _behavior_model_gate(perception, config):
    """Parking / lane change / passage need a verified behaviour model file."""
    scene = perception.scene_id
    needs = []
    if scene in PARKING_SCENES:
        needs.append("parking")
    if scene in LANE_CHANGE_SCENES:
        needs.append("lane change")
    if scene in PASSAGE_SCENES:
        needs.append("passage")
    if scene in CONTINUOUS_SCENES:
        needs.append("task route")
    if not needs:
        return _result(PASS, "scene {0} needs no behaviour-model capability".format(scene))
    path = getattr(config, "behavior_model_file", "") or ""
    if not path:
        return _result(BLOCK, "behavior_model_file is empty, so {0} capability is never "
                              "enabled (provider is None)".format(" / ".join(needs)))
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
    if not os.path.isfile(resolved):
        return _result(BLOCK, "behavior_model_file points to a missing file: " + resolved)
    try:
        with open(resolved, "r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (ValueError, OSError) as exc:
        return _result(BLOCK, "behavior_model_file is unreadable: " + type(exc).__name__)
    if not isinstance(document, dict) or document.get("verified") is not True:
        return _result(BLOCK, "behaviour model is not verified=true, so it is disabled")
    scope = document.get("scope", {}) if isinstance(document.get("scope"), dict) else {}
    model_scene = (scope.get("case_id"), scope.get("task_id"))
    actual = (perception.case_id, perception.task_id)
    if model_scene != actual:
        return _result(BLOCK, "behaviour model is scoped to case/task {0}, but this frame is "
                              "{1}; one model file covers one task".format(model_scene, actual))
    return _result(PASS, "verified model covers {0}".format(" / ".join(needs)))


def diagnose(perception, config, source_name=""):
    gates = {
        "ego": _ego_gate(perception),
        "lane": _lane_gate(perception),
        "sensor_configuration": _sensor_config_gate(perception, config),
        "targets": _target_gate(perception),
        "signal": _signal_gate(perception),
        "behaviour_model": _behavior_model_gate(perception, config),
    }
    return {
        "source": source_name,
        "scene_id": perception.scene_id,
        "case_name": perception.case_name,
        "case_id": perception.case_id,
        "task_id": perception.task_id,
        "requires_targets": requires_targets(perception.scene_id),
        "is_aeb": perception.scene_id in AEB_SCENES,
        "is_follow": perception.scene_id in FOLLOW_SCENES,
        "gates": gates,
        "blockers": _blockers(gates),
    }


def _blockers(gates):
    return [name for name, value in gates.items() if value["status"] == BLOCK]


def _print_scene(report, verbose=True):
    scene = report["scene_id"]
    verdict = "CAN SCORE" if not report["blockers"] else "BLOCKED"
    label = report["source"] or (report["case_name"] or "?")
    print("\n场景 {0:02d}  {1}   ({2})".format(scene, verdict, label))
    print("  requires_targets={0} AEB={1} FOLLOW={2}".format(
        report["requires_targets"], report["is_aeb"], report["is_follow"]))
    for name in ("ego", "lane", "sensor_configuration", "targets", "signal",
                 "behaviour_model"):
        gate = report["gates"][name]
        if not verbose and gate["status"] == PASS:
            continue
        print("  [{0:<7}] {1:<22} {2}".format(gate["status"], name, gate["detail"]))
    if report["blockers"]:
        print("  阻塞项: {0}".format(", ".join(report["blockers"])))
    else:
        print("  无结构性阻塞")


def _advice(reports):
    blocking = {}
    for report in reports:
        for name in report["blockers"]:
            blocking.setdefault(name, []).append(report)
    if not blocking:
        print("\n没有结构性阻塞。若实车仍为 0 分，问题在场景行为或判分条件，"
              "需对照评分规则逐条检查车辆动作。")
        return
    print("\n按阻塞类型汇总（修复优先级从高到低）:")
    order = ("behaviour_model", "sensor_configuration", "targets", "lane", "ego", "signal")
    for name in order:
        items = blocking.get(name)
        if not items:
            continue
        labels = []
        for report in items:
            label = "场景%02d" % report["scene_id"]
            if report["source"]:
                label += "(" + report["source"] + ")"
            labels.append(label)
        print("  {0:<22} {1}".format(name, ", ".join(labels)))


def main():
    parser = argparse.ArgumentParser(
        description="诊断 NEVC 场景能否得分（只读，不连接 SimOne）")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help="单个 latest_perception.json")
    parser.add_argument("--directory", help="包含多个 .json 快照的目录")
    parser.add_argument("--config", help="叠加的本机 INI 配置")
    parser.add_argument("--scene", type=int, help="覆盖快照中的场景号后再诊断")
    parser.add_argument("--json-out", help="把诊断结果写入该文件")
    parser.add_argument("--quiet", action="store_true", help="只显示非 PASS 的闸门")
    args = parser.parse_args()

    try:
        config = load_config(PROJECT_DIR, args.config)
        target = args.directory if args.directory else args.snapshot
        files = snapshot_files(target)
    except (ValueError, RuntimeError) as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 2

    reports = []
    for path in files:
        try:
            perception = load_perception(path)
        except (ValueError, OSError) as exc:
            print("ERROR: {0}".format(exc), file=sys.stderr)
            return 2
        if args.scene is not None:
            perception.scene_id = args.scene
        reports.append(diagnose(perception, config, os.path.basename(path)))

    # A saved snapshot carries an expired process-local deadline, so the ego
    # gate would always fail. Report it once instead of drowning every scene.
    expired = all(r["gates"]["ego"]["status"] == BLOCK for r in reports)
    if expired:
        for report in reports:
            report["gates"]["ego"] = _result(
                UNKNOWN, "snapshot is expired by design; freshness is only provable live")
            report["blockers"] = _blockers(report["gates"])
    for report in reports:
        _print_scene(report, verbose=not args.quiet)

    _advice(reports)
    if expired:
        print("\n注意: 快照的 valid_until 是进程内单调时钟，保存后必然过期。本报告把"
              "\n      新鲜度标为 UNKNOWN 而不是 PASS；实车诊断请看最新的 captain.log。")

    if args.json_out:
        payload = {"snapshot_expired_by_design": expired, "scenes": reports}
        with open(args.json_out, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
        print("\nwrote {0}".format(args.json_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
