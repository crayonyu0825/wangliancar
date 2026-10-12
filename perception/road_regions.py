"""Local pieces of actual SDK lanes; no hull, extrapolation or free-space guess."""
import copy
import math
from collections import OrderedDict

from core.geometry import project_polyline,projection_within_polyline
from core.region_geometry import simple_outline,outline_contains_segment,outline_contains_path,strip_cells
from perception.crossing_ranges import points3,window3


def _at(points,projection):
    if not projection_within_polyline(projection,len(points)):
        raise ValueError("LOCAL_CUT_OUTSIDE_BOUNDARY")
    i,t = projection["index"],projection["ratio"]
    return tuple(points[i][j]+t*(points[i+1][j]-points[i][j]) for j in range(3))


def _cut_station(center,a,b):
    """Unique XY intersection, in the centre polyline's own arc coordinate."""
    values,along = [],0.
    ux,uy = b[0]-a[0],b[1]-a[1]
    for c,d in zip(center,center[1:]):
        dx,dy = d[0]-c[0],d[1]-c[1]
        length = math.hypot(dx,dy)
        denominator = dx*uy-dy*ux
        if abs(denominator)>1e-10:
            t = ((a[0]-c[0])*uy-(a[1]-c[1])*ux)/denominator
            v = ((a[0]-c[0])*dy-(a[1]-c[1])*dx)/denominator
            if -1e-8<=t<=1+1e-8 and -1e-8<=v<=1+1e-8:
                values.append(along+max(0.,min(1.,t))*length)
        along += length
    unique = []
    for value in sorted(values):
        if not unique or value-unique[-1]>1e-6: unique.append(value)
    if len(unique)!=1: raise ValueError("LOCAL_CUT_CENTER_INTERSECTION_AMBIGUOUS")
    return unique[0]


def local_region(center,left,right,reference):
    """Crop native borders with cuts proven to stay in the original ring."""
    source = simple_outline([v[:2] for v in left]+[v[:2] for v in reversed(right)])
    pieces = []
    for boundary in (left,right):
        a = project_polyline(boundary,*reference[0][:2]); b = project_polyline(boundary,*reference[-1][:2])
        _at(boundary,a); _at(boundary,b)
        pieces.append(window3(boundary,a["s"],b["s"]))
    a,b = pieces
    for start,end in ((a[0],b[0]),(a[-1],b[-1])):
        if not outline_contains_segment(start,end,source):
            raise ValueError("LOCAL_END_CUT_LEAVES_SOURCE_LANE")
    outline = simple_outline([v[:2] for v in a]+[v[:2] for v in reversed(b)])
    low,high = _cut_station(center,a[0],b[0]),_cut_station(center,a[-1],b[-1])
    center_local = window3(center,low,high)
    if not outline_contains_path(center_local,outline):
        raise ValueError("LOCAL_CENTER_LEAVES_BOUNDARIES")
    cells=strip_cells([v[:2] for v in a],[v[:2] for v in b])
    return dict(center_line=center_local,left_boundary=a,right_boundary=b,polygon=outline,
                convex_cells=cells,center_arc_start_m=low,center_arc_end_m=high,
                cell_bounds_xy=[(min(v[0] for v in cell),min(v[1] for v in cell),max(v[0] for v in cell),max(v[1] for v in cell)) for cell in cells],
                coordinate_system="world_xyz_metres",geometry_model="sdk_piecewise_linear_lane_v1")


class RoadRegions(object):
    def __init__(self): self.cache=OrderedDict()
    def clear(self): self.cache.clear()

    def observe(self,ego,lane,neighbors,document,digest):
        result = []
        sources = [(lane.lane_id,lane.center_line,lane.left_boundary,lane.right_boundary,
                    lane.lane_type if lane.lane_type_valid else "unknown","current")]
        sources += [(v["lane_id"],v["center_line"],v["left_boundary"],v["right_boundary"],
                     v.get("lane_type","unknown"),"neighbor") for v in neighbors if v.get("geometry_valid") is True]
        for identity,raw_center,raw_left,raw_right,kind,role in sources:
            item = dict(lane_id=identity,role=role,source="sdk_matched_opendrive",geometry_valid=False,
                        drivable_verified=False,polygon=[],convex_cells=[],reason="SOURCE_LANE_UNAVAILABLE")
            try:
                unused,unused,member = document.lane(identity)
                center,left,right = points3(raw_center),points3(raw_left),points3(raw_right)
                projection = project_polyline(center,ego.x,ego.y)
                if not projection_within_polyline(projection,len(center)):
                    raise ValueError("EGO_OUTSIDE_LOCAL_LANE")
                bucket = math.floor(projection["s"]/20.)*20.
                reference = window3(center,bucket-25.,bucket+75.)
                key = (digest,identity,center,left,right,bucket)
                if key not in self.cache:
                    try: self.cache[key]=local_region(center,left,right,reference)
                    except (ValueError,TypeError,KeyError,IndexError,OverflowError) as error:
                        self.cache[key]=dict(error=str(error) if isinstance(error,ValueError) else "LOCAL_GEOMETRY_UNAVAILABLE")
                self.cache.move_to_end(key)
                while len(self.cache)>16: self.cache.popitem(last=False)
                value = self.cache[key]
                if "error" in value: raise ValueError(value["error"])
                item.update(copy.deepcopy(value),geometry_valid=True,
                    drivable_verified=member["type"]=="driving" and member["permission_scope_verified"]
                        and str(kind).lower().split(".")[-1]=="driving",
                    reason="LOCAL_SOURCE_LANE_VERIFIED",map_digest=digest)
            except (ValueError,TypeError,KeyError,IndexError,AttributeError,OverflowError) as error:
                item["reason"] = str(error) if isinstance(error,ValueError) else "LOCAL_GEOMETRY_UNAVAILABLE"
            result.append(item)
        return result
