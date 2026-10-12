"""Offline replay of captain Perception snapshots through the decision member.

Read-only. Reconstructs `Perception` objects from saved JSON snapshots, feeds
them to the decision engine in order and reports the behaviour distribution.

This exists because the decision member cannot be fully judged from synthetic
unit tests: real snapshots show how often lane membership is actually verified,
whether target extents are published and whether the tree collapses into a
single behaviour. It does NOT prove scenario performance.

    python scripts/replay_decision.py --snapshot runtime_data/latest_perception.json
    python scripts/replay_decision.py --directory runtime_data/scene_acceptance --summary-only
"""

from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import sys
import time


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.interfaces import (  # noqa: E402  (path setup must run first)
    EgoState, LaneContext, Perception, Target, TrafficControl, ManeuverEnvironment)
from members.decision.engine import DecisionEngine  # noqa: E402
from members.decision.settings import FIELDS, DecisionSettings  # noqa: E402


DEFAULT_SNAPSHOT = os.path.join(PROJECT_DIR, "runtime_data", "latest_perception.json")

# Rebuilt from the snapshot rather than trusted, because the same object type
# is used for nested members.
SCALARS = (int, float, bool, str)
NESTED = {
    "ego": EgoState,
    "lane": LaneContext,
    "traffic": TrafficControl,
    "targets": Target,
    "maneuver_environment": ManeuverEnvironment,
}


def _assign(obj, data, path):
    """Copy known attributes from a JSON dict onto a contract object."""
    if not isinstance(data, dict):
        raise ValueError("{0}: expected an object, got {1}".format(path, type(data).__name__))
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
            # JSON null stands for a non-finite float that was dropped on write.
            if isinstance(current, float):
                setattr(obj, key, float("nan"))
        elif isinstance(value, SCALARS):
            setattr(obj, key, value)
    return obj


def load_perception(path, revive_ttl=None):
    """Rebuild a Perception from a snapshot; the frame deadline cannot be revived.

    `valid_until` is a process-local monotonic deadline, so a snapshot always
    arrives expired. By default it is reported rather than rewritten, because
    stamping a fresh deadline would assert that a stale frame is current. A
    caller may pass `revive_ttl` to route around that guard and inspect the
    behaviour tree instead; the result then says nothing about frame freshness.
    """
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or "ego" not in data:
        raise ValueError("{0}: not a captain perception snapshot".format(path))
    perception = _assign(Perception(), data, os.path.basename(path))
    if revive_ttl is not None:
        perception.valid_until = time.monotonic() + revive_ttl
    return perception


def snapshot_files(target):
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.json")))
        if not files:
            raise ValueError("no .json snapshots in " + target)
        return files
    if not os.path.exists(target):
        raise ValueError("path does not exist: " + target)
    return [target]


def _frame_signature(perception):
    return (perception.frame_id, perception.timestamp)


def replay(files, settings, stop_on_repeat=True, revive_ttl=None):
    """Run every snapshot through one engine, in order, and collect evidence."""
    engine = DecisionEngine(settings)
    modes = collections.Counter()
    reasons = collections.Counter()
    target_counts = collections.Counter()
    frozen = set()
    frames = []
    for path in files:
        try:
            perception = load_perception(path, revive_ttl=revive_ttl)
        except (ValueError, OSError) as exc:
            frames.append({"file": os.path.basename(path), "error": str(exc)})
            continue
        signature = _frame_signature(perception)
        if stop_on_repeat and signature in frozen:
            # The captain's loop deliberately republishes a snapshot when the
            # GPS frame does not advance. Counting it again would inflate the
            # behaviour distribution with duplicate evidence.
            frozen.add(signature)
            continue
        frozen.add(signature)
        decision = engine.run(perception)
        verified = sum(1 for target in perception.targets
                       if getattr(target, "same_lane_valid", False) is True)
        modes[decision.mode] += 1
        reasons[_reason_key(decision.reason)] += 1
        target_counts["frames"] += 1
        target_counts["targets"] += len(perception.targets)
        target_counts["map_verified"] += verified
        target_counts["targets_usable"] += 1 if perception.targets_valid else 0
        target_counts["lane_valid"] += 1 if perception.lane.valid else 0
        target_counts["ego_valid"] += 1 if perception.valid else 0
        frames.append({
            "file": os.path.basename(path),
            "frame_id": perception.frame_id,
            "snapshot_valid_until": perception.valid_until,
            "signal": decision.mode,
            "target_speed": decision.target_speed,
            "stop_distance": decision.stop_distance,
            "obstacle_clearance_m": decision.obstacle_clearance_m,
            "obstacle_clearances_m": decision.obstacle_clearances_m,
            "precision_stop": decision.precision_stop,
            "target_lane_id": decision.target_lane_id,
            "reason": decision.reason,
            "decision_valid": decision.valid,
            "decision_errors": list(decision.errors),
            "counts": {"targets": len(perception.targets), "map_verified": verified},
        })
    return {"modes": modes, "reasons": reasons, "counts": target_counts, "frames": frames}


def _reason_key(reason):
    """Collapse a reason into its leading clause so counts stay readable."""
    if not isinstance(reason, str) or not reason:
        return "(none)"
    return reason.split(";")[0].strip()


def print_report(report, settings, summary_only=False, revived=False):
    modes, counts, frames = report["modes"], report["counts"], report["frames"]
    print("frames replayed : {0}".format(counts["frames"]))
    print("ego valid       : {0}".format(counts["ego_valid"]))
    print("lane valid      : {0}".format(counts["lane_valid"]))
    print("targets usable  : {0}".format(counts["targets_usable"]))
    print("targets seen    : {0} (map-verified {1})".format(
        counts["targets"], counts["map_verified"]))
    print("settings        : " + ", ".join(
        "{0}={1}".format(name, getattr(settings, name)) for name in ("cruise_speed",
                                                                     "min_gap",
                                                                     "time_headway")))
    print("\nbehaviour distribution:")
    for mode, count in modes.most_common():
        print("  {0:<18} {1}".format(mode, count))
    print("\nreasons:")
    for reason, count in report["reasons"].most_common():
        print("  {0:<64} {1}".format(reason[:64], count))
    if summary_only:
        return
    print("\nframes:")
    header = "  {0:<34} {1:>7} {2:<18} {3:>7} {4:>8}"
    print(header.format("file", "frame", "signal", "speed", "stop_d"))
    for item in frames:
        if "error" in item:
            print("  {0:<34} ERROR {1}".format(item["file"][:34], item["error"]))
            continue
        print(header.format(item["file"][:34], item["frame_id"], item["signal"],
                            "{0:.2f}".format(item["target_speed"]),
                            "{0:.1f}".format(item["stop_distance"])))
        if item["decision_errors"]:
            print("      errors: {0}".format(", ".join(item["decision_errors"])))
    if revived:
        print("\nWARNING: --revive-ttl was used. Frame deadlines were rewritten so the")
        print("         behaviour tree could be inspected. Every freshness guarantee is")
        print("         void in this run: it shows what the decision would have chosen")
        print("         given the saved geometry, not what it would accept as live data.")
    else:
        print("\nNOTE: snapshots arrive with an expired process-local valid_until, so the")
        print("      engine stops at the freshness guard on every frame. That is the")
        print("      safety rule working, not a decision fault, but it means only one")
        print("      behaviour is visible. Use --revive-ttl to inspect the full tree.")


def main():
    parser = argparse.ArgumentParser(
        description="离线回放 Perception 快照并统计决策行为（只读）")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help="单个 latest_perception.json")
    parser.add_argument("--directory", help="包含多个 .json 快照的目录")
    parser.add_argument("--summary-only", action="store_true", help="不打印逐帧明细")
    parser.add_argument("--json-out", help="把完整结果写入该文件")
    parser.add_argument("--keep-repeats", action="store_true",
                        help="重复帧也计入统计（默认跳过）")
    parser.add_argument("--revive-ttl", type=float, default=None,
                        metavar="SECONDS",
                        help="诊断用：把每帧有效期重置为当前时刻后 SECONDS 秒，"
                             "以绕过新鲜度门控观察完整行为树。会破坏新鲜度语义，"
                             "结果不能作为实时数据结论")
    for name in FIELDS:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None,
                            help="覆盖该决策参数（默认读环境变量或内置值）")
    args = parser.parse_args()

    overrides = dict((name, getattr(args, name)) for name in FIELDS
                     if getattr(args, name) is not None)
    try:
        settings = DecisionSettings.from_environment(**overrides)
        target = args.directory if args.directory else args.snapshot
        files = snapshot_files(target)
        report = replay(files, settings, stop_on_repeat=not args.keep_repeats,
                        revive_ttl=args.revive_ttl)
    except ValueError as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 2

    print_report(report, settings, summary_only=args.summary_only,
                 revived=args.revive_ttl is not None)
    if args.json_out:
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "revive_ttl": args.revive_ttl,
            "freshness_valid": args.revive_ttl is None,
            "settings": dict((name, getattr(settings, name)) for name in FIELDS),
            "counts": dict(report["counts"]),
            "modes": dict(report["modes"]),
            "reasons": dict(report["reasons"]),
            "frames": report["frames"],
        }
        with open(args.json_out, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
        print("\nwrote {0}".format(args.json_out))
    if report["counts"]["frames"] == 0:
        print("no frames replayed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
