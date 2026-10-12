"""Incrementally bind marking intervals to a bounded SDK boundary window.

The nominal geometry is the SDK's piecewise-linear lane sample, in world XYZ.
Results describe this local border only; they never authorize a whole vehicle
trajectory, establish visibility, or manufacture a road/free-space polygon.
"""
import copy
import math
import re
import time
from collections import OrderedDict

from core.geometry import project_polyline,projection_within_polyline
from core.validation import number
from core.boundary_lineage import native_points3,window_lineage_steps,densify_lineage_steps

MAX_NATIVE_POINTS = 20000
MAX_LOCAL_POINTS = 4096
MAX_ATOMIC_STEPS = 16
MAX_FRAME_SECONDS = .005
SAMPLE_SPACING_M = 1.
ROAD_S_GUARD_M = .25
XY_TOLERANCE_M = .2
Z_TOLERANCE_M = .1


def points3(raw):
    return native_points3(raw)


def length3(points):
    return sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(points,points[1:]))


def window3(points,start,end):
    """Clip existing segments by their own XY arc length; no extrapolation.

    This coordinate is explicitly boundary arc length, never OpenDRIVE road-s.
    All original vertices within the window and linearly interpolated Z survive.
    """
    if not all(number(v) for v in (start,end)) or not start<end:
        raise ValueError('BOUNDARY_WINDOW_INVALID')
    result,along=[],0.
    start=max(0.,start)
    for a,b in zip(points,points[1:]):
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if not number(length) or length<=0: raise ValueError('BOUNDARY_SEGMENT_INVALID')
        low,high=max(start,along),min(end,along+length)
        if high>low+1e-8:
            for at in (low,high):
                ratio=(at-along)/length
                point=(a if at==along else b if at==along+length else
                       tuple(a[i]+ratio*(b[i]-a[i]) for i in range(3)))
                if not result or math.hypot(point[0]-result[-1][0],point[1]-result[-1][1])>1e-6:
                    result.append(point)
            if len(result)>MAX_LOCAL_POINTS: raise ValueError('LOCAL_BOUNDARY_POINT_BUDGET')
        along+=length
        if along>=end: break
    if len(result)<2: raise ValueError('BOUNDARY_WINDOW_EMPTY')
    return result


def densify3(points):
    result = [points[0]]
    for a,b in zip(points,points[1:]):
        steps = max(1,int(math.ceil(math.hypot(b[0]-a[0],b[1]-a[1])/SAMPLE_SPACING_M)))
        if len(result)+steps>MAX_LOCAL_POINTS:
            raise ValueError("LOCAL_BOUNDARY_POINT_BUDGET")
        for index in range(1,steps+1):
            result.append(b if index==steps else tuple(a[i]+index/float(steps)*(b[i]-a[i]) for i in range(3)))
    return result


def _sdk_text(value):
    if hasattr(value,"GetString"):
        value = value.GetString()
    return str(value).lower().split(".")[-1]


def _reason(error):
    value = str(error)
    return value if isinstance(error,ValueError) and re.fullmatch(r"[A-Z_]+",value) else "BOUNDARY_QUERY_FAILED"


def _project3(points,point):
    value = project_polyline(points,point[0],point[1])
    if not projection_within_polyline(value,len(points)):
        raise ValueError("SHARED_BOUNDARY_COVERAGE_MISMATCH")
    i,t = value["index"],value["ratio"]
    z = points[i][2]+t*(points[i+1][2]-points[i][2])
    if value["distance"]>XY_TOLERANCE_M or abs(z-point[2])>Z_TOLERANCE_M:
        raise ValueError("SHARED_BOUNDARY_SHAPE_OR_HEIGHT_MISMATCH")
    return value["s"]


class _Job(object):
    def __init__(self,hdmap,identity,road,section,current,neighbor,native_side,intervals,ego,clock,logical_side,source_boundaries):
        self.clock,self.hdmap = clock,hdmap
        self.current,self.neighbor,self.intervals = current,neighbor,copy.deepcopy(intervals)
        self.identity,self.road,self.section,self.native_side = identity,road,section,native_side
        self.logical_side=logical_side
        self.source_boundaries=source_boundaries
        projection = project_polyline(current,ego.x,ego.y)
        other = project_polyline(neighbor,ego.x,ego.y)
        if not projection_within_polyline(projection,len(current)) or not projection_within_polyline(other,len(neighbor)):
            raise ValueError("EGO_OUTSIDE_SHARED_BOUNDARY")
        # Fixed window while pending: vehicle movement cannot restart a job
        # every frame and prevent it ever finishing. Later windows overlap.
        self.origin_s = projection["s"]
        self.window_start,self.window_end = max(0.,self.origin_s-25.),min(length3(current),self.origin_s+75.)
        self.local,self.source_positions=None,None
        # Wider other-side window avoids confusing crop endpoints with an
        # actual mismatch. It is evidence for shape, not a permitted border.
        self.other = window3(neighbor,other["s"]-40.,other["s"]+90.)
        self.result,self.previous,self.reason,self.retry_after = None,None,"BOUNDARY_BINDING_PENDING",0.
        self.iterator = self._build()

    def _build(self):
        clipped,positions=yield from window_lineage_steps(self.current,self.window_start,self.window_end)
        self.local,self.source_positions=yield from densify_lineage_steps(self.current,clipped,positions,SAMPLE_SPACING_M)
        # Require bidirectional, ordered coincidence over the chosen window.
        prior = -1.
        for point in self.local:
            at = _project3(self.other,point)
            if at<prior-XY_TOLERANCE_M:
                raise ValueError("SHARED_BOUNDARY_ORDER_AMBIGUOUS")
            prior = at
            yield
        low = _project3(self.other,self.local[0]); high = _project3(self.other,self.local[-1])
        reverse_window = window3(self.other,low,high)
        for point in reverse_window:
            _project3(self.local,point)
            yield
        samples = []
        native_road,native_lane = self.hdmap.pySimString(self.road["id"]),self.hdmap.pySimString(self.identity)
        for point in self.local:
            st = self.hdmap.getRoadST(native_road,self.hdmap.pySimPoint3D(*point))
            if getattr(st,"exists",False) is not True or not number(getattr(st,"s",None)):
                raise ValueError("BOUNDARY_ROAD_S_UNAVAILABLE")
            s = float(st.s)
            if not self.section["start_m"]-1e-6<=s<=self.section["end_m"]+1e-6:
                raise ValueError("BOUNDARY_OUTSIDE_LANE_SECTION")
            yield
            active = next((v for v in self.intervals if v["road_s_start_m"]<=s<v["road_s_end_m"]),None)
            # A section's last point is a map gate, not marking authority.
            if active is None:
                samples.append((point,s,None))
                continue
            observed = self.hdmap.getRoadMark(self.hdmap.pySimPoint3D(*point),native_lane)
            mark = getattr(observed,self.native_side,None)
            offset = getattr(mark,"sOffset",None)
            matched = (getattr(observed,"exists",False) is True and mark is not None and number(offset)
                and _sdk_text(getattr(mark,"type","unknown"))==active["type"]
                and abs(offset-(active["road_s_start_m"]-self.section["start_m"]))<.001
                and active["semantic_verified"] is True)
            if not matched:
                samples.append((point,s,None))
                yield
                continue
            samples.append((point,s,active))
            yield
        deltas = [b[1]-a[1] for a,b in zip(samples,samples[1:])]
        if not deltas or any(abs(v)<=1e-8 for v in deltas) or not (all(v>0 for v in deltas) or all(v<0 for v in deltas)):
            raise ValueError("BOUNDARY_ROAD_S_NONMONOTONIC")
        member = self.section["lanes"][int(self.identity.split("_")[-1])]
        known_direction = member["permission_scope_verified"] and self.road["rule"] in ("RHT","LHT")
        expected_sign = (1 if int(self.identity.split("_")[-1])<0 else -1)*(1 if self.road["rule"]=="RHT" else -1)
        matches_direction = bool(known_direction and deltas[0]*expected_sign>0)
        ranges,run = [],None
        for index,(a,b) in enumerate(zip(samples,samples[1:])):
            active = a[2]
            # Whole nominal border edges are included only within one matched
            # record, inset from transitions. No road-s interpolation or
            # fabricated transition point extends permission across a solid.
            permitted = (matches_direction and active is not None and active==b[2] and active["crossing_permitted"]
                and min(a[1],b[1])>=active["road_s_start_m"]+ROAD_S_GUARD_M
                and max(a[1],b[1])<=active["road_s_end_m"]-ROAD_S_GUARD_M)
            if permitted:
                if run is None:
                    run = dict(shared_boundary_world=[a[0]],road_s_samples_m=[a[1]],marking=copy.deepcopy(active),
                               source_positions=[self.source_positions[index]])
                    ranges.append(run)
                run["shared_boundary_world"].append(b[0]); run["road_s_samples_m"].append(b[1])
                run['source_positions'].append(self.source_positions[index+1])
            else:
                run = None
        return dict(ranges=ranges,shared_boundary_world=list(self.local),
                    source_lineage_model='native_edge_rational_v1',source_lane_id=self.identity,
                    source_boundary_side=self.logical_side,source_boundary_world=list(self.current),
                    source_lane_boundaries={k:list(v) for k,v in self.source_boundaries.items()},
                    source_positions=list(self.source_positions),
                    road_s_samples_m=[v[1] for v in samples],boundary_arc_start_m=self.window_start,
                    boundary_arc_end_m=self.window_end,verification_model="sdk_piecewise_linear_boundary_v1",
                    complete_marking_coverage=all(v[2] is not None for v in samples[:-1]),
                    travel_direction_verified=known_direction,travel_matches_declared_direction=matches_direction,
                    declared_road_s_direction=expected_sign if known_direction else None,
                    sample_spacing_m=SAMPLE_SPACING_M,road_s_transition_guard_m=ROAD_S_GUARD_M,
                    xy_alignment_tolerance_m=XY_TOLERANCE_M,z_alignment_tolerance_m=Z_TOLERANCE_M)

    def advance(self):
        try:
            next(self.iterator)
        except StopIteration as completed:
            self.result,self.reason = completed.value,"LOCAL_BOUNDARY_RANGE_VERIFIED"
        except Exception as error:
            self.reason = _reason(error)
            self.iterator,self.retry_after = None,self.clock()+.2


class CrossingRanges(object):
    def __init__(self,clock=None):
        self.clock = clock or time.monotonic
        self.jobs = OrderedDict()
        self.cursor = 0

    def clear(self):
        self.jobs.clear()

    def observe(self,ego,lane,items,document,digest,hdmap,prior_sources=()):
        deadline,steps,pending = self.clock()+MAX_FRAME_SECONDS,0,[]
        # Refresh prepared original directions before preparing the new SDK
        # current-lane directions. Both still share the same bounded budget.
        directed=[(source,item,True) for source,item in prior_sources]+[(lane,item,False) for item in items]
        for source,item,prior_source in directed:
            item.update(crossing_ranges_world=[],crossing_boundary_window={},crossing_range_verified=False,
                        crossing_geometry_bound=False)
            if (item.get("geometry_valid") is not True or item.get("shared_boundary_verified") is not True
                    or item.get("same_direction") is not True
                    or (not prior_source and item.get("marking_semantics_verified") is not True)
                    or str(item.get("lane_type","unknown")).lower().split(".")[-1]!="driving"):
                continue
            try:
                side = item["side"]
                source_boundaries=dict(left=points3(source.left_boundary),right=points3(source.right_boundary))
                current = source_boundaries[side]
                neighbor = points3(item["right_boundary"] if side=="left" else item["left_boundary"])
                native_side = item.get("marking_observation",{}).get("native_side")
                if native_side not in ("left","right"):
                    raise ValueError("NATIVE_SHARED_MARKING_SIDE_UNAVAILABLE")
                key = (id(hdmap),digest,source.lane_id,item["lane_id"],native_side,
                       source_boundaries['left'],source_boundaries['right'],neighbor)
                job = self.jobs.get(key)
                at = project_polyline(current,ego.x,ego.y)
                if not projection_within_polyline(at,len(current)):
                    raise ValueError("EGO_OUTSIDE_SHARED_BOUNDARY")
                if prior_source:
                    if job is None:
                        raise ValueError('ORIGINAL_DIRECTION_BINDING_UNAVAILABLE')
                    if steps+2>MAX_ATOMIC_STEPS or self.clock()>=deadline:
                        raise ValueError('SOURCE_NATIVE_RECHECK_BUDGET')
                    # Query the original border, as in _Job._build; ego can
                    # already be outside its old lane. Requery road-s there
                    # rather than borrowing the target lane's current road-s.
                    index,ratio=at['index'],at['ratio']
                    a,b=current[index:index+2]
                    point=tuple(a[k]+ratio*(b[k]-a[k]) for k in range(3))
                    road,section,unused=document.lane(source.lane_id)
                    native_point=hdmap.pySimPoint3D(*point)
                    steps+=1
                    st=hdmap.getRoadST(hdmap.pySimString(road['id']),native_point)
                    at_s=getattr(st,'s',None)
                    if not (getattr(st,'exists',False) is True and number(at_s)
                            and section['start_m']<=at_s<section['end_m']):
                        raise ValueError('SOURCE_NATIVE_ROAD_S_UNVERIFIED')
                    if self.clock()>=deadline:
                        raise ValueError('SOURCE_NATIVE_RECHECK_BUDGET')
                    steps+=1
                    observed=hdmap.getRoadMark(native_point,
                                               hdmap.pySimString(source.lane_id))
                    if self.clock()>=deadline:
                        raise ValueError('SOURCE_NATIVE_RECHECK_BUDGET')
                    mark=getattr(observed,native_side,None)
                    active=next((v for v in item.get('marking_intervals_road_s',[])
                                 if number(at_s) and v['road_s_start_m']<=at_s<v['road_s_end_m']),None)
                    offset=getattr(mark,'sOffset',None)
                    if not (getattr(observed,'exists',False) is True and active is not None
                            and active['semantic_verified'] is True and number(offset)
                            and _sdk_text(getattr(mark,'type','unknown'))==active['type']
                            and abs(offset-(active['road_s_start_m']-section['start_m']))<.001):
                        raise ValueError('SOURCE_NATIVE_MARKING_UNVERIFIED')
                    item['marking_semantics_verified']=True
                    item['crossing_allowed_at_source_probe']=bool(active['crossing_permitted'])
                    item['source_marking_recheck']=dict(point_world=point,road_s_m=float(at_s),
                        native_side=native_side,model='original_border_projection_v1')
                total = length3(current)
                replace = (job is None or (job.iterator is None and job.result is None and self.clock()>=job.retry_after)
                    or (job is not None and (at["s"]<job.window_start or at["s"]>job.window_end
                        or (job.result is not None and at["s"]>job.window_end-40. and job.window_end<total-1e-6))))
                if replace:
                    previous = job.result if job is not None and job.result is not None else (job.previous if job is not None else None)
                    road,section,unused = document.lane(source.lane_id)
                    job = _Job(hdmap,source.lane_id,road,section,current,neighbor,native_side,
                               item["marking_intervals_road_s"],ego,self.clock,side,source_boundaries)
                    job.previous = previous
                    self.jobs[key] = job
                self.jobs.move_to_end(key)
                while len(self.jobs)>8:
                    self.jobs.popitem(last=False)
                pending.append((item,job,at["s"]))
            except (ValueError,TypeError,KeyError,IndexError,AttributeError,OverflowError) as error:
                item["crossing_scope_reason"] = _reason(error)
        # Rotate between both neighbors. One expensive/retrying side cannot
        # monopolize the whole frame or prevent the other side from progressing.
        while pending and steps<MAX_ATOMIC_STEPS and self.clock()<deadline:
            available = [v for v in pending if v[1].result is None and v[1].iterator is not None]
            if not available:
                break
            item,job,at = available[self.cursor%len(available)]
            job.advance(); steps += 1; self.cursor += 1
        for item,job,at in pending:
            item["crossing_scope_reason"] = job.reason
            ready = job.result if job.result is not None else job.previous
            if ready is not None and ready["boundary_arc_start_m"]<=at<=ready["boundary_arc_end_m"]:
                result = copy.deepcopy(ready)
                item.update(crossing_ranges_world=result.pop("ranges"),crossing_boundary_window=result,
                            crossing_range_verified=True,crossing_geometry_bound=True,
                            travel_direction_verified=result["travel_direction_verified"],
                            travel_matches_declared_direction=result["travel_matches_declared_direction"],
                            crossing_allowed_at_ego=bool(item["crossing_allowed_at_ego"] and result["travel_matches_declared_direction"]),
                            marking_scope="verified_local_boundary_fragments")
        return dict(atomic_steps=steps,max_atomic_steps=MAX_ATOMIC_STEPS,time_budget_s=MAX_FRAME_SECONDS)
