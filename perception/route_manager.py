"""HD map lookup and lane-context normalization."""

import math
import copy
import time
import heapq

from core.geometry import (nearest_path_error, normalize_angle, project_polyline,
                           projection_within_polyline, DEFAULT_FORWARD_HEADING_ERROR_RAD)
from core.interfaces import LaneContext
from core.route_segments import verified_spans
from core.region_geometry import simple_outline
from perception.maneuver_map import ManeuverMap


def _sdk_string(value):
    if value is None:
        return ""
    if hasattr(value, "GetString"):
        return value.GetString()
    return str(value)


MAX_REFERENCE_POINTS = 20000
FORWARD_HORIZON_M = 120.0  # Beyond the planner's 60 m horizon; never a goal.
MAX_FORWARD_LANES = 32
MAX_ROUTE_SEARCH_STATES = 128
JOIN_POSITION_TOLERANCE_M = 0.2
JOIN_HEIGHT_TOLERANCE_M = 0.1
JOIN_HEADING_TOLERANCE_RAD = 0.35
ROUTE_REFERENCE_VERSION = "successor-continuation-v4"


def _vector_points(vector):
    points = []
    size = int(vector.Size())
    if not 0 <= size <= MAX_REFERENCE_POINTS:
        raise ValueError("map sample point budget exceeded")
    for index in range(size):
        item = vector.GetElement(index)
        points.append((float(item.x), float(item.y), float(item.z)))
    return points


def _clean_points(points):
    clean = []
    for point in points:
        if not all(math.isfinite(v) for v in point):
            raise ValueError("nonfinite map sample")
        if not clean or math.hypot(point[0] - clean[-1][0], point[1] - clean[-1][1]) > 1e-6:
            clean.append(point)
    if len(clean) < 2:
        raise ValueError("degenerate map sample")
    return clean


def _length(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _heading(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _connected_segment(reference, points, diagnostics=None):
    """Orient at the join, not against the ego heading far before a curve.

    Endpoints differing by at most map-sampling tolerance share the existing
    endpoint. No gap, junction chord or extrapolated lane is manufactured.
    """
    tail = reference[-1]
    previous_heading = _heading(reference[-2], tail)
    candidates = []
    for segment, reversed_direction in ((points, False), (list(reversed(points)), True)):
        start = segment[0]
        gap = math.hypot(start[0] - tail[0], start[1] - tail[1])
        height = abs(start[2] - tail[2])
        detail = {"reversed": reversed_direction, "xy_gap_m": gap,
                  "z_gap_m": height, "tail": tail, "start": start}
        if diagnostics is not None:
            diagnostics.append(detail)
        if gap > JOIN_POSITION_TOLERANCE_M or height > JOIN_HEIGHT_TOLERANCE_M:
            detail["reason"] = "position_or_height_gap"
            continue
        next_heading = _heading(start, segment[1])
        joined_heading = _heading(tail, segment[1])
        deltas = [abs(normalize_angle(next_heading - previous_heading)),
                  abs(normalize_angle(joined_heading - previous_heading)),
                  abs(normalize_angle(joined_heading - next_heading))]
        detail["heading_deltas_rad"] = deltas
        if (math.hypot(segment[1][0] - tail[0], segment[1][1] - tail[1]) <= 1e-6
                or any(delta > JOIN_HEADING_TOLERANCE_RAD for delta in deltas)):
            detail["reason"] = "degenerate_or_heading_discontinuity"
            continue
        detail["reason"] = "connected"
        candidates.append((segment, reversed_direction))
    return candidates[0] if len(candidates) == 1 else None


def _forward_links(link_info, reversed_direction):
    """Keep native ID objects for SDK calls; publish strings separately."""
    if link_info is None or not link_info.exists:
        return None
    field = "predecessorLaneIds" if reversed_direction else "successorLaneIds"
    vector = getattr(link_info.laneLink, field, None)
    if vector is None or not 0 <= int(vector.Size()) <= MAX_FORWARD_LANES:
        return None
    result, seen = [], set()
    for i in range(int(vector.Size())):
        native = vector.GetElement(i)
        identity = _sdk_string(native)
        if not identity:
            return None
        if identity not in seen:
            result.append((identity, native))
            seen.add(identity)
    return result


def _orient_forward(points, heading, x=None, y=None):
    if len(points) < 2:
        return points
    projection = project_polyline(points, points[0][0] if x is None else x,
                                  points[0][1] if y is None else y)
    if projection is None:
        return []
    path_heading = projection["heading"]
    if math.cos(path_heading - heading) < 0.0:
        return list(reversed(points))
    return points


class RouteManager(object):
    def __init__(self, adapter, logger, refresh_sec=0.2):
        self.adapter = adapter
        self.logger = logger
        self.refresh_sec = refresh_sec
        self.last_update = 0.0
        self.last_lane = LaneContext()
        self._join_failure_signature = None
        self._route_hint = ()
        self._route_context = None
        self._route_choices = {}
        self._retired_native_lanes = {}
        self._retired_branch_ids = {}
        self._last_ego = None
        self._last_ego_at = None
        self._neighbor_samples = {}
        self._neighbor_sample_at = 0.0
        self._maneuver_map = ManeuverMap(adapter)

    def read_maneuver_map(self,ego,lane,neighbors):
        return self._maneuver_map.observe(ego,lane,neighbors)

    def set_route_hint(self, points, valid, context=None):
        """Static task waypoints select branches; they are not trajectory points."""
        hint = ()
        if valid and isinstance(points, (list,tuple)) and 2 <= len(points) <= 2000:
            if all(isinstance(p,(list,tuple)) and len(p)>=2
                   and all(type(v) in (int,float) and math.isfinite(v) for v in p[:2])
                   for p in points):
                hint = tuple(tuple(p[:2]) for p in points)
        if hint != self._route_hint or context != self._route_context:
            self._route_hint, self._route_context = hint, context
            self._route_choices.clear()
            self.last_lane = LaneContext()
            self.last_update = 0.0
            self._retired_native_lanes.clear()
            self._retired_branch_ids.clear()
            self._last_ego = None
            self._last_ego_at = None
            self._neighbor_samples.clear()

    def read_neighbor_lanes(self, ego, lane):
        """Detached adjacent geometry; no inference that a lane is empty.

        Reuse the existing sample cleaning/orientation and native ID handling.
        Only same-road/section OpenDRIVE lane signs prove nominal direction;
        unknown IDs, road type or shared boundary remain explicitly unknown.
        A marking query applies at the ego position, not to the full corridor.
        """
        hdmap = getattr(self.adapter, "hdmap", None)
        if (not ego.valid or not lane.valid or not self.adapter.map_loaded
                or hdmap is None or not hasattr(hdmap, "pySimString")):
            return []
        result = []
        now = time.monotonic()
        if now-self._neighbor_sample_at >= self.refresh_sec:
            self._neighbor_samples.clear()
            self._neighbor_sample_at = now
        for side, identity, mark in (("left",lane.left_lane_id,lane.left_mark_type),
                                     ("right",lane.right_lane_id,lane.right_mark_type)):
            if not identity:
                continue
            item = dict(lane_id=identity, side=side, source="hdmap", geometry_valid=False,
                        center_line=[], left_boundary=[], right_boundary=[], width_m=None,
                        direction_verified=False, same_direction=False, lane_type="unknown",
                        shared_marking=str(mark), marking_scope="ego_position_only",
                        crossing_allowed_at_ego=False, crossing_range_verified=False,
                        reason="unavailable")
            try:
                key = (id(hdmap),lane.lane_id,identity)
                native = hdmap.pySimString(identity)
                if key not in self._neighbor_samples:
                    sample = hdmap.getLaneSample(native)
                    if not sample.exists:
                        raise ValueError("neighbor sample unavailable")
                    raw = tuple(_clean_points(_vector_points(v)) for v in
                        (sample.laneInfo.centerLine,sample.laneInfo.leftBoundary,
                         sample.laneInfo.rightBoundary))
                    kind = "unknown"
                    if hasattr(hdmap,"getLaneType"):
                        lane_type = hdmap.getLaneType(native)
                        if getattr(lane_type,"exists",False):
                            kind = _sdk_string(lane_type.laneType)
                    self._neighbor_samples[key] = raw+(kind,)
                center,left,right,kind = self._neighbor_samples[key]
                center = _orient_forward(center,ego.heading,ego.x,ego.y)
                reversed_sample = center[0] != self._neighbor_samples[key][0][0]
                if reversed_sample:
                    left,right = list(reversed(right)),list(reversed(left))
                simple_outline([tuple(v[:2]) for v in left]+[tuple(v[:2]) for v in reversed(right)])
                projection = project_polyline(center,ego.x,ego.y)
                if not projection_within_polyline(projection,len(center)):
                    raise ValueError("ego outside neighbor longitudinal coverage")
                i,t = projection["index"],projection["ratio"]
                z = center[i][2]+t*(center[i+1][2]-center[i][2])
                if abs(z-ego.z) > JOIN_HEIGHT_TOLERANCE_M:
                    raise ValueError("neighbor on another height level")
                width = hdmap.getLaneWidth(native,hdmap.pySimPoint3D(*center[i]))
                if not width.exists or not math.isfinite(float(width.width)) or width.width <= .1:
                    raise ValueError("neighbor width unavailable")
                item.update(center_line=center,left_boundary=left,right_boundary=right,
                            width_m=float(width.width),geometry_valid=True,lane_type=kind,
                            native_sample_reversed=bool(reversed_sample),reason="geometry_observed")
                try:
                    a,b = tuple(int(v) for v in lane.lane_id.split("_")),tuple(int(v) for v in identity.split("_"))
                    item["direction_verified"] = len(a)==len(b)==3 and a[:2]==b[:2] and a[2]!=0 and b[2]!=0
                    item["same_direction"] = item["direction_verified"] and a[2]*b[2]>0
                except (ValueError,TypeError):
                    pass
                shared = right if side=="left" else left
                current = lane.left_boundary if side=="left" else lane.right_boundary
                p,q = project_polyline(shared,ego.x,ego.y),project_polyline(current,ego.x,ego.y)
                boundary_verified = (p is not None and q is not None
                    and math.hypot(p["point"][0]-q["point"][0],p["point"][1]-q["point"][1]) <= JOIN_POSITION_TOLERANCE_M)
                item["shared_boundary_verified"] = boundary_verified
                label = str(mark).lower().split(".")[-1]
                item["crossing_allowed_at_ego"] = (item["same_direction"] and boundary_verified
                    and kind.lower().split(".")[-1]=="driving" and label in ("broken","dashed","dashline"))
                # Publish native marking extent facts, without confusing the
                # lane-centre s with OpenDRIVE road/section s. The legacy SDK
                # does not expose section origin or permitted crossing spans.
                if hasattr(hdmap,"getRoadMark"):
                    observed = hdmap.getRoadMark(hdmap.pySimPoint3D(ego.x,ego.y,ego.z),
                                                hdmap.pySimString(lane.lane_id))
                    raw_current = _clean_points(_vector_points(hdmap.getLaneSample(
                        hdmap.pySimString(lane.lane_id)).laneInfo.centerLine))
                    origin = project_polyline(raw_current,ego.x,ego.y)
                    reverse_current = origin and math.cos(origin["heading"]-ego.heading)<0
                    native_side = ("right" if side=="left" else "left") if reverse_current else side
                    value = getattr(observed,native_side,None) if observed.exists else None
                    if value is not None:
                        extent = [getattr(value,n,None) for n in ("sOffset","length")]
                        extent = [v if type(v) in (int,float) and math.isfinite(v) and v>=0 else None for v in extent]
                        item["marking_observation"] = dict(type=_sdk_string(value.type),
                            section_s_offset_m=extent[0],native_length_m=extent[1],
                            native_side=native_side,
                            semantics="native_extent_not_crossing_authority")
            except (AttributeError,TypeError,ValueError,OverflowError,RuntimeError) as exc:
                item["reason"] = "NEIGHBOR_QUERY:"+str(exc)
            result.append(item)
        return result

    def update(self, ego, force=False):
        if not ego.valid or not self.adapter.map_loaded:
            return LaneContext()
        now = time.monotonic()
        cached_projection = project_polyline(self.last_lane.center_line, ego.x, ego.y)
        crossed_end = (cached_projection is not None
                       and cached_projection["index"] == len(self.last_lane.center_line) - 2
                       and cached_projection["raw_ratio"] >= 1.0)
        if (not force and not crossed_end and self.last_lane.valid
                and now - self.last_update < self.refresh_sec):
            lane = copy.deepcopy(self.last_lane)
            lane.lateral_offset, lane.heading_error = nearest_path_error(
                lane.center_line, ego.x, ego.y, ego.heading
            )
            self._last_ego, self._last_ego_at = copy.deepcopy(ego), now
            return lane
        try:
            lane = self._lookup(ego)
            self.last_lane = copy.deepcopy(lane)
            self.last_update = now
            self._last_ego, self._last_ego_at = copy.deepcopy(ego), now
            return lane
        except Exception as exc:
            self.logger.warning("车道查询失败: %s", exc)
            return LaneContext()

    def _lookup(self, ego):
        hdmap = self.adapter.hdmap
        position = hdmap.pySimPoint3D(ego.x, ego.y, ego.z)
        nearest = hdmap.getNearMostLane(position)
        if not nearest.exists:
            return LaneContext()
        lane = LaneContext()
        lane.lane_id = _sdk_string(nearest.laneId)
        sample = hdmap.getLaneSample(nearest.laneId)
        if not sample.exists:
            return lane
        points = _clean_points(_vector_points(sample.laneInfo.centerLine))
        native_id = nearest.laneId
        successor = self._ego_successor(ego, _sdk_string(native_id), points)
        if successor is not None:
            native_id, joined_points = successor
            sample = hdmap.getLaneSample(native_id)
            if not sample.exists:
                return lane
            lane.lane_id = _sdk_string(native_id)
            points = _clean_points(_vector_points(sample.laneInfo.centerLine))
        else:
            joined_points = None
        lane.center_line = _orient_forward(points, ego.heading, ego.x, ego.y)
        reversed_direction = bool(points and lane.center_line and lane.center_line[0] != points[0])
        if joined_points is not None:
            lane.center_line = joined_points
        lane.left_boundary = _vector_points(sample.laneInfo.leftBoundary)
        lane.right_boundary = _vector_points(sample.laneInfo.rightBoundary)
        for name in ("left_boundary", "right_boundary"):
            if not all(math.isfinite(v) for point in getattr(lane, name) for v in point):
                setattr(lane, name, [])
        link_info = None
        if hasattr(hdmap, "getLaneLink"):
            try:
                link_info = hdmap.getLaneLink(native_id)
            except Exception as exc:
                self.logger.warning("车道连通查询失败: %s", exc)
            if link_info is not None and link_info.exists:
                lane.left_lane_id = _sdk_string(link_info.laneLink.leftNeighborLaneId)
                lane.right_lane_id = _sdk_string(link_info.laneLink.rightNeighborLaneId)
                for name, field in (("predecessor_lane_ids", "predecessorLaneIds"),
                                    ("successor_lane_ids", "successorLaneIds")):
                    vector = getattr(link_info.laneLink, field, None)
                    if vector is not None:
                        setattr(lane, name, [_sdk_string(vector.GetElement(i)) for i in range(vector.Size())])
        if hasattr(hdmap, "getLaneWidth"):
            width_info = hdmap.getLaneWidth(native_id, position)
            if width_info.exists and math.isfinite(float(width_info.width)) and float(width_info.width) > 0.1:
                lane.lane_width = float(width_info.width)
                lane.lane_width_valid = True
        if hasattr(hdmap, "getRoadMark"):
            try:
                mark_info = hdmap.getRoadMark(position, native_id)
                if mark_info.exists:
                    lane.left_mark_type = _sdk_string(mark_info.left.type)
                    lane.right_mark_type = _sdk_string(mark_info.right.type)
            except (AttributeError, TypeError, ValueError) as exc:
                self.logger.warning("车道标线查询失败: %s", exc)
        if reversed_direction:
            lane.left_boundary, lane.right_boundary = list(reversed(lane.right_boundary)), list(reversed(lane.left_boundary))
            lane.left_mark_type, lane.right_mark_type = lane.right_mark_type, lane.left_mark_type
            lane.left_lane_id, lane.right_lane_id = lane.right_lane_id, lane.left_lane_id
            lane.predecessor_lane_ids, lane.successor_lane_ids = lane.successor_lane_ids, lane.predecessor_lane_ids
        lane.lateral_offset, lane.heading_error = nearest_path_error(
            lane.center_line, ego.x, ego.y, ego.heading
        )
        lane.valid = project_polyline(lane.center_line, ego.x, ego.y) is not None
        lane.source = "hdmap"
        if hasattr(hdmap,"getLaneType"):
            try:
                kind = hdmap.getLaneType(native_id)
                if getattr(kind,"exists",False):
                    lane.lane_type = _sdk_string(kind.laneType)
                    lane.lane_type_valid = lane.lane_type.lower().split(".")[-1]!="unknown"
            except (AttributeError,TypeError,ValueError,RuntimeError):
                pass
        if lane.valid:
            self._extend_reference(lane, ego, link_info, reversed_direction)
        return lane

    def _ego_successor(self, ego, nearest_id, native_points):
        """Keep bounded progress on the previously verified selected route.

        An already retired native ID cannot undo a verified promotion. Several
        short segments may be crossed, but candidate progress must be reachable
        from the preceding ego sample; competing distant route matches remain
        ambiguous. Fresh width, height and native geometry are still required.
        """
        lane, hdmap = self.last_lane, self.adapter.hdmap
        if (not lane.valid or not lane.forward_reference_valid
                or not hasattr(hdmap, 'pySimString') or not hasattr(hdmap, 'getLaneWidth')):
            return None
        retired = self._retired_native_lanes.get(nearest_id)
        if nearest_id != lane.lane_id and retired != native_points:
            return None
        old = project_polyline(lane.center_line, ego.x, ego.y)
        native = project_polyline(_orient_forward(native_points, ego.heading, ego.x, ego.y),
                                  ego.x, ego.y)
        if old is None or native is None:
            return None
        spans = verified_spans(len(lane.center_line), len(lane.forward_reference),
                               lane.forward_lane_ids, lane.forward_lane_spans)
        if not spans:
            return None
        on_current = projection_within_polyline(old, len(lane.center_line))
        if not on_current and (old['index'] != len(lane.center_line)-2 or old['raw_ratio'] <= 1):
            return None
        if (not on_current and nearest_id == lane.lane_id
                and (native['index'] != len(native_points)-2 or native['raw_ratio'] <= 1)):
            return None
        progress_limit = None
        if not on_current:
            previous = self._last_ego
            anchor = (project_polyline(lane.center_line, previous.x, previous.y)
                      if previous is not None else None)
            if (not projection_within_polyline(anchor, len(lane.center_line))
                    or not lane.lane_width_valid or anchor['distance'] > lane.lane_width*.5):
                return None
            elapsed = max(0.0, time.monotonic()-self._last_ego_at)
            speeds = [v for v in (previous.speed, ego.speed)
                      if type(v) in (int,float) and math.isfinite(v) and v >= 0]
            travel = max(math.hypot(ego.x-previous.x, ego.y-previous.y),
                         max(speeds or [0.0])*elapsed)
            progress_limit = anchor['s'] + travel + JOIN_POSITION_TOLERANCE_M
        candidates = []
        for identity, (start, end) in spans.items():
            if (on_current and identity != lane.lane_id) or (not on_current and start == 0):
                continue
            points = lane.forward_reference[start:end+1]
            p = project_polyline(points, ego.x, ego.y)
            progress = _length(lane.forward_reference[:start+1]) + p['s'] if p else None
            if (not projection_within_polyline(p, len(points))
                    or (progress_limit is not None and progress > progress_limit)
                    or abs(normalize_angle(ego.heading-p['heading'])) > DEFAULT_FORWARD_HEADING_ERROR_RAD):
                continue
            native_id = hdmap.pySimString(identity)
            sample = hdmap.getLaneSample(native_id)
            if not sample.exists:
                continue
            fresh = _orient_forward(_clean_points(_vector_points(sample.laneInfo.centerLine)),
                                    ego.heading, ego.x, ego.y)
            if (len(fresh) != len(points) or fresh[1:] != points[1:]
                    or math.hypot(fresh[0][0]-points[0][0], fresh[0][1]-points[0][1]) > JOIN_POSITION_TOLERANCE_M
                    or abs(fresh[0][2]-points[0][2]) > JOIN_HEIGHT_TOLERANCE_M):
                continue
            width = hdmap.getLaneWidth(native_id, hdmap.pySimPoint3D(ego.x, ego.y, ego.z))
            i, r = p['index'], p['ratio']
            z = points[i][2] + r*(points[i+1][2]-points[i][2])
            if (width.exists and math.isfinite(width.width) and width.width > 0
                    and p['distance'] <= width.width*.5
                    and abs(ego.z-z) <= JOIN_HEIGHT_TOLERANCE_M
                    and not self._unselected_branch_is_closer(lane, ego, nearest_id, p['distance'])):
                candidates.append((progress, p['distance'], native_id, points))
        candidates.sort(key=lambda value:(value[0],value[1]))
        if (not candidates or any(abs(c[0]-candidates[0][0]) > JOIN_POSITION_TOLERANCE_M
                                  for c in candidates[1:])):
            return None
        selected = candidates[0]
        if _sdk_string(selected[2]) != lane.lane_id and nearest_id == lane.lane_id:
            self._retired_native_lanes[nearest_id] = list(native_points)
            self._retired_branch_ids[nearest_id] = list(lane.successor_lane_ids)
            while len(self._retired_native_lanes) > MAX_FORWARD_LANES:
                retired_id = next(iter(self._retired_native_lanes))
                del self._retired_native_lanes[retired_id]
                self._retired_branch_ids.pop(retired_id, None)
        return selected[2:]

    def _unselected_branch_is_closer(self, lane, ego, nearest_id, distance):
        """A task choice cannot override clear membership in another branch.

        Width corridors overlap at the common join. Within join sampling
        tolerance retain the selected task branch, but reject recovery if an
        unselected branch explains the pose substantially better.
        """
        hdmap = self.adapter.hdmap
        alternatives = (lane.successor_lane_ids if nearest_id == lane.lane_id
                        else self._retired_branch_ids.get(nearest_id, []))
        for identity in alternatives:
            if identity in lane.forward_lane_ids:
                continue
            native = hdmap.pySimString(identity)
            sample = hdmap.getLaneSample(native)
            if not sample.exists:
                return True
            points = _orient_forward(_clean_points(_vector_points(sample.laneInfo.centerLine)),
                                     ego.heading, ego.x, ego.y)
            projection = project_polyline(points, ego.x, ego.y)
            if not projection_within_polyline(projection, len(points)):
                continue
            width = hdmap.getLaneWidth(native, hdmap.pySimPoint3D(ego.x, ego.y, ego.z))
            if not width.exists or not math.isfinite(width.width) or width.width <= 0:
                return True
            i,r = projection['index'],projection['ratio']
            z = points[i][2]+r*(points[i+1][2]-points[i][2])
            if (projection['distance'] <= width.width*.5
                    and abs(ego.z-z) <= JOIN_HEIGHT_TOLERANCE_M
                    and abs(normalize_angle(ego.heading-projection['heading'])) <= DEFAULT_FORWARD_HEADING_ERROR_RAD
                    and projection['distance'] + JOIN_POSITION_TOLERANCE_M < distance):
                return True
        return False

    def _extend_reference(self, lane, ego, link_info, reversed_direction):
        """Bounded map expansion; only a unique, connected forward lane is used.

        A partial reference is still usable geometry. Its end remains a stop
        boundary until the map/route supplies a verified continuation.
        """
        reference = list(lane.center_line)
        lane.forward_reference = reference
        lane.forward_lane_ids = [lane.lane_id]
        lane.forward_lane_spans = [{"lane_id": lane.lane_id,
                                    "start_index": 0,
                                    "end_index": len(reference) - 1}]
        lane.forward_reference_valid = True
        projection = project_polyline(reference, ego.x, ego.y)
        remaining = _length(reference) - projection["s"]
        visited = {lane.lane_id}
        hdmap = self.adapter.hdmap
        while True:
            try:
                links = _forward_links(link_info, reversed_direction)
                if links is None:
                    lane.forward_reference_status = "link_unavailable"
                    return
                if not links:
                    lane.forward_reference_status = "map_end"
                    return
                if remaining >= FORWARD_HORIZON_M:
                    lane.forward_reference_status = "lookahead_limit"
                    return
                if len(links) != 1:
                    selection = self._select_successor(reference, links, lane, ego)
                    if selection is None:
                        lane.forward_reference_status = "ambiguous_successor"
                        return
                    links = [selection]
                identity, native = links[0]
                if identity in visited:
                    lane.forward_reference_status = "cycle"
                    return
                if len(visited) >= MAX_FORWARD_LANES:
                    lane.forward_reference_status = "lane_limit"
                    return
                sample = hdmap.getLaneSample(native)
                if not sample.exists:
                    lane.forward_reference_status = "successor_unavailable"
                    return
                points = _clean_points(_vector_points(sample.laneInfo.centerLine))
                diagnostics = []
                connection = _connected_segment(reference, points, diagnostics)
                if connection is None:
                    lane.forward_reference_status = "disconnected_successor"
                    signature = (lane.forward_lane_ids[-1], identity, repr(diagnostics))
                    if signature != self._join_failure_signature:
                        self.logger.warning("车道接续拒绝 from=%s to=%s checks=%s",
                                            signature[0], identity, diagnostics)
                        self._join_failure_signature = signature
                    return
                points, reversed_direction = connection
                if len(reference) + len(points) - 1 > MAX_REFERENCE_POINTS:
                    lane.forward_reference_status = "point_limit"
                    return
                previous_end = reference[-1]
                start_index = len(reference) - 1
                reference.extend(points[1:])
                remaining += _length([previous_end] + points[1:])
                lane.forward_lane_ids.append(identity)
                lane.forward_lane_spans.append({"lane_id": identity,
                    "start_index": start_index, "end_index": len(reference) - 1})
                visited.add(identity)
                link_info = hdmap.getLaneLink(native) if hasattr(hdmap, "getLaneLink") else None
            except Exception as exc:
                # Optional successor failure must not discard the usable
                # current lane, its traffic association or target membership.
                lane.forward_reference_status = "successor_query_failed"
                self.logger.warning("前方车道接续查询失败: %s", exc)
                return

    def _select_successor(self, reference, links, lane, ego):
        """Shortest verified continuation to the next off-prefix waypoint.

        All graph edges pass the same position/height/direction checks as normal
        continuation. A bounded, incomplete search or effectively equal choices
        stays unresolved; a heading or Euclidean-nearest guess cannot select it.
        """
        if not self._route_hint or not hasattr(self.adapter.hdmap, 'getLaneWidth'):
            return None
        progress = project_polyline(self._route_hint, ego.x, ego.y)
        remaining = self._route_hint[progress['index']+1:] if progress else self._route_hint[-1:]
        goal = None
        for point in remaining:
            # Every prefix segment has already been selected and joined.
            # Use its own measured width, not the current lane's width.
            covered = False
            for span in lane.forward_lane_spans:
                points = reference[span['start_index']:span['end_index']+1]
                projected = project_polyline(points, point[0], point[1])
                if not projection_within_polyline(projected, len(points)):
                    continue
                i,r = projected['index'],projected['ratio']
                z = points[i][2]+r*(points[i+1][2]-points[i][2])
                native = self.adapter.hdmap.pySimString(span['lane_id']) if hasattr(self.adapter.hdmap,'pySimString') else None
                width = (self.adapter.hdmap.getLaneWidth(native,
                         self.adapter.hdmap.pySimPoint3D(point[0],point[1],z)) if native is not None else None)
                if (width is not None and width.exists and math.isfinite(width.width)
                        and width.width > 0 and projected['distance'] <= width.width*.5):
                    covered = True
                    break
            if not covered:
                goal = point
                break
        if goal is None:
            return None
        key = (lane.forward_lane_ids[-1], tuple(identity for identity,native in links), goal)
        cached = self._route_choices.get(key)
        if cached is not None:
            return next((link for link in links if link[0] == cached), None)
        hdmap, queue, sequence = self.adapter.hdmap, [], 0
        def enqueue(identity, native, anchor, cost, first, visited):
            sample = hdmap.getLaneSample(native)
            if not sample.exists:
                raise ValueError('competitive route branch sample unavailable')
            connection = _connected_segment(anchor,_clean_points(_vector_points(sample.laneInfo.centerLine)))
            if connection is None:
                return
            points, reverse = connection
            points = [anchor[-1]] + points[1:]
            heapq.heappush(queue,(cost,sequence,identity,native,points,reverse,first,visited))
        try:
            for identity,native in links:
                enqueue(identity,native,reference,0.0,identity,frozenset(lane.forward_lane_ids+[identity]))
                sequence += 1
            solutions, expanded, best_cost = [], 0, float('inf')
            seen = {}
            while queue:
                cost, unused, identity,native,points,reverse,first,visited = heapq.heappop(queue)
                if cost > best_cost + JOIN_POSITION_TOLERANCE_M:
                    break
                expanded += 1
                if expanded > MAX_ROUTE_SEARCH_STATES:
                    return None
                state = (identity,reverse,first)
                if cost > seen.get(state,float('inf')) + 1e-6:
                    continue
                seen[state] = cost
                projected = project_polyline(points,goal[0],goal[1])
                if projection_within_polyline(projected, len(points)):
                    i,r = projected['index'],projected['ratio']
                    z = points[i][2]+r*(points[i+1][2]-points[i][2])
                    width = hdmap.getLaneWidth(native,hdmap.pySimPoint3D(goal[0],goal[1],z))
                    if (width.exists and math.isfinite(width.width) and width.width>0
                            and projected['distance'] <= width.width*.5):
                        reached = cost+projected['s']
                        solutions.append((reached,first))
                        best_cost = min(best_cost,reached)
                        continue
                successors = _forward_links(hdmap.getLaneLink(native),reverse)
                if successors is None:
                    # An unreadable competitive branch cannot be ruled out.
                    return None
                if successors and len(visited)>=MAX_FORWARD_LANES:
                    return None
                for next_id,next_native in successors:
                    if next_id not in visited:
                        enqueue(next_id,next_native,points,cost+_length(points),first,visited|{next_id})
                        sequence += 1
            if not solutions:
                return None
            choices = {first for cost,first in solutions if cost<=best_cost+JOIN_POSITION_TOLERANCE_M}
            if len(choices)!=1:
                return None
            selected = choices.pop()
            self._route_choices[key] = selected
            if hasattr(self.logger,'info'):
                self.logger.info("任务路径选择后继 from=%s to=%s goal=%s",key[0],selected,goal)
            return next(link for link in links if link[0]==selected)
        except (AttributeError,TypeError,ValueError,RuntimeError) as exc:
            self.logger.warning("任务路线分叉查询失败: %s",exc)
            return None

    def locate_target(self, target):
        """Map membership requires bounded projection and a measured lane width."""
        target.lane_width_m = None
        if not self.adapter.map_loaded:
            return "", False
        try:
            hdmap = self.adapter.hdmap
            pos = hdmap.pySimPoint3D(target.x, target.y, target.z)
            nearest = hdmap.getNearMostLane(pos)
            if not nearest.exists:
                return "", False
            sample = hdmap.getLaneSample(nearest.laneId)
            width = hdmap.getLaneWidth(nearest.laneId, pos)
            if not sample.exists or not width.exists or not math.isfinite(width.width) or width.width <= 0:
                return "", False
            points = _vector_points(sample.laneInfo.centerLine)
            if not all(math.isfinite(v) for p in points for v in p):
                return "", False
            projection = project_polyline(points, target.x, target.y)
            if (not projection_within_polyline(projection, len(points)) or
                    projection["distance"] > width.width * 0.5):
                return self._locate_on_verified_reference(target, pos)
            i, r = projection["index"], projection["ratio"]
            road_z = points[i][2] + r * (points[i + 1][2] - points[i][2])
            if abs(target.z - road_z) > max(2.0, target.height):
                return "", False
            target.lane_width_m = float(width.width)
            return _sdk_string(nearest.laneId), True
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return "", False

    def _locate_on_verified_reference(self, target, position):
        """Handle nearest-lane endpoint ambiguity using verified joined spans.

        Width is queried for the candidate lane at this target position. The
        joined geometry is already bounded and height checked; no extrapolated
        membership or arbitrary neighbouring lane is manufactured.
        """
        lane, hdmap = self.last_lane, self.adapter.hdmap
        if not lane.valid or not lane.forward_reference_valid or not hasattr(hdmap, 'pySimString'):
            return "", False
        spans = verified_spans(len(lane.center_line), len(lane.forward_reference),
                               lane.forward_lane_ids, lane.forward_lane_spans)
        candidates = []
        for identity, (start, end) in spans.items():
            points = lane.forward_reference[start:end+1]
            projection = project_polyline(points, target.x, target.y)
            if not projection_within_polyline(projection, len(points)):
                continue
            width = hdmap.getLaneWidth(hdmap.pySimString(identity), position)
            if (not width.exists or not math.isfinite(width.width) or width.width <= 0
                    or projection['distance'] > width.width*.5):
                continue
            i, ratio = projection['index'], projection['ratio']
            road_z = points[i][2] + ratio*(points[i+1][2]-points[i][2])
            if abs(target.z-road_z) <= max(2.0, target.height):
                candidates.append((projection['distance'], identity, float(width.width)))
        candidates.sort()
        if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0] <= 1e-3):
            return "", False
        target.lane_width_m = candidates[0][2]
        return candidates[0][1], True
