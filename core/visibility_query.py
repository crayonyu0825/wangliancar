"""Bounded queries over one original Sensor packet's verified XY regions.

No sensor fusion, calibration or authority is created here. Exact union
topology and convex-body queries reuse the existing corridor engine.
"""
import time
from fractions import Fraction

from core.corridor_region import CorridorRegion,NumericResolution,PreparationLimit
from core.region_geometry import convex_polygon,simple_outline,strip_cells
from core.validation import number


class VisibilityQuery(object):
    def __init__(self,source,source_frame_id,observed_at_s,valid_until_s,clock=None,max_checks=100000):
        if (not isinstance(source,str) or not source.startswith('sensor:') or not source[7:]
                or type(source_frame_id) is not int or source_frame_id<0
                or not all(number(v) for v in (observed_at_s,valid_until_s))
                or not 0<=observed_at_s<valid_until_s or type(max_checks) is not int or max_checks<=0):
            raise ValueError('COVERAGE_QUERY_SOURCE_OR_BUDGET_INVALID')
        self.source,self.frame,self.observed=source,source_frame_id,observed_at_s
        self.deadline,self.query_deadline=valid_until_s,valid_until_s
        self.clock,self.max_checks=clock or time.monotonic,max_checks
        self.checks,self.reason,self.regions=0,'NOT_CHECKED',{}

    def step(self):
        self.checks+=1
        now=self.clock()
        if (self.checks>self.max_checks or not number(now)
                or not self.observed<=now<self.query_deadline):
            raise ValueError('COVERAGE_QUERY_BUDGET_OR_DEADLINE')

    def _exact_convex(self,points):
        exact=[]
        for point in points:
            self.step(); exact.append(tuple(Fraction(v) for v in point))
        sign=None
        for index,b in enumerate(exact):
            self.step(); a,c=exact[index-1],exact[(index+1)%len(exact)]
            turn=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
            if not turn: continue
            if sign is not None and (turn>0)!=sign: return False
            sign=turn>0
        return sign is not None

    def contains(self,outline,records,left=None,right=None):
        """Check the full region, including holes between its vertices.

        For a lane, supply its original left/right borders. Their existing
        oriented strip mesh covers the exact simple outline, without a hull.
        Single convex profiles retain closed-boundary containment; a union
        query needs positive clearance under the corridor numeric guard.
        """
        try:
            self.query_deadline=self.deadline; self.step()
            if not isinstance(records,(tuple,list)) or not 1<=len(records)<=32:
                raise ValueError('COVERAGE_QUERY_REGIONS_UNAVAILABLE')
            polygons=[]; binding=None
            for record in records:
                self.step()
                if (not isinstance(record,dict) or record.get('source')!=self.source
                        or record.get('sensor_id')!=self.source[7:] or record.get('source_kind')!='sensor'
                        or record.get('complete_detections') is not True or record.get('coverage_verified') is not True
                        or record.get('clock_id')!='process_monotonic'
                        or type(record.get('source_frame_id')) is not int or record['source_frame_id']!=self.frame
                        or type(record.get('pose_frame_id')) is not int or record['pose_frame_id']!=self.frame
                        or record.get('pose_source') not in ('current_frame_pose','sensor_pose_history_exact_frame')
                        or not number(record.get('observed_at_s')) or record['observed_at_s']!=self.observed
                        or not number(record.get('reference_z_m'))
                        or not number(record.get('valid_until_s')) or record['valid_until_s']>self.deadline
                        or not isinstance(record.get('verification_reference'),str)
                        or not record['verification_reference'].strip()):
                    raise ValueError('COVERAGE_QUERY_PACKET_OR_POSE_MISMATCH')
                key=(record['pose_source'],record['reference_z_m'])
                if binding is not None and key!=binding:
                    raise ValueError('COVERAGE_QUERY_PACKET_OR_POSE_MISMATCH')
                binding=key
                self.query_deadline=min(self.query_deadline,record['valid_until_s']); self.step()
                polygon=record.get('polygon')
                if not isinstance(polygon,(list,tuple)) or not 3<=len(polygon)<=128:
                    raise ValueError('COVERAGE_QUERY_POLYGON_INVALID')
                for unused in range(len(polygon)**2): self.step()
                polygons.append(tuple(convex_polygon(polygon)))
            if not isinstance(outline,(list,tuple)) or not 3<=len(outline)<=40000:
                raise ValueError('COVERAGE_QUERY_OUTLINE_INVALID')
            points=[]
            for point in outline:
                self.step()
                if (not isinstance(point,(list,tuple)) or len(point)!=2
                        or not all(number(v) for v in point)):
                    raise ValueError('COVERAGE_QUERY_OUTLINE_INVALID')
                points.append(tuple(point))
            key=tuple(polygons)
            if key not in self.regions:
                if len(self.regions)>=32: raise ValueError('COVERAGE_QUERY_CACHE_LIMIT')
                self.regions[key]=CorridorRegion(key)
            region=self.regions[key]
            prepared=region.prepare(self.step,self.max_checks)
            # Every convex profile denotes its entire interior. Exact states
            # avoid tolerance filling a positive but tiny unknown gap.
            for polygon in prepared.polygons:
                if not self._exact_convex(polygon.exact):
                    raise ValueError('COVERAGE_QUERY_PROFILE_NOT_EXACT_CONVEX')
            for polygon in prepared.polygons:
                covered=True
                for point in points:
                    self.step()
                    if polygon.state(tuple(Fraction(v) for v in point),self.step)==-1:
                        covered=False; break
                if covered:
                    self.reason='COVERAGE_COMPLETE_SINGLE_REGION'; self.step(); return True
            if left is not None or right is not None:
                if (not isinstance(left,(list,tuple)) or not isinstance(right,(list,tuple))
                        or points!=list(left)+list(reversed(right))):
                    raise ValueError('COVERAGE_QUERY_LANE_OUTLINE_MISMATCH')
                simple_outline(points,self.max_checks,self.step)
                cells=strip_cells(left,right,check_step=self.step)
            else:
                if len(points)>64: raise ValueError('COVERAGE_QUERY_CONVEX_BODY_LIMIT')
                if not self._exact_convex(points):
                    raise ValueError('COVERAGE_QUERY_BODY_NOT_EXACT_CONVEX')
                cells=[points]
            for cell in cells:
                if region.body_clearance(cell,self.step,self.max_checks)<=0:
                    self.reason='COVERAGE_REGION_HAS_UNOBSERVED_PART'; return False
            self.reason='COVERAGE_COMPLETE_PACKET_UNION'; self.step(); return True
        except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError,NumericResolution,PreparationLimit) as error:
            self.reason=str(error); return False
