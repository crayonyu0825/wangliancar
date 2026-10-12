"""Whole-body legal-crossing geometry through the existing P02 sweep.

Keep the complete source lane perimeter except exact source-bound openings.
This stops bypasses through unverified windows or caps, and retains the ends
of each opening. Visibility, law evidence and live deadlines remain caller
requirements; this component gives no behavior or driving authority.
"""
import math
from fractions import Fraction

from core.boundary_lineage import decode_position,exact_point
from core.validation import number
from members.planning.corridor_region import (CorridorRegion,NumericResolution,_tree_steps,_box,_sort_steps)


class CrossingCorridor(CorridorRegion):
    def __init__(self,polygons,source_descriptor,preparation_max_checks):
        # A tuple descriptor is created from the freshly verified formal scene.
        # No references to mutable source dictionaries survive preparation.
        if type(source_descriptor) is not tuple or len(source_descriptor)!=4:
            raise ValueError('immutable source crossing descriptor required')
        self._source_descriptor=source_descriptor
        super(CrossingCorridor,self).__init__(polygons,preparation_max_checks)

    def _compile_steps(self,snapshot):
        prepared=yield from super(CrossingCorridor,self)._compile_steps(snapshot)
        left,right,side,openings=self._source_descriptor
        if (side not in ('left','right') or type(openings) is not tuple or len(openings)>4096
                or any(type(v) is not tuple or not 2<=len(v)<=20000 for v in (left,right))):
            raise ValueError('bounded immutable source boundaries required')
        for border in (left,right):
            for point in border:
                yield
                if type(point) is not tuple or len(point)!=3 or not all(number(v) for v in point):
                    raise ValueError('finite immutable source XYZ points required')
        native=left if side=='left' else right
        intervals=[]
        for part in openings:
            yield
            if type(part) is not tuple or len(part)!=2 or any(type(v) is not tuple for v in part):
                raise ValueError('immutable bound source interval required')
            a,ta=decode_position(part[0],len(native)); b,tb=decode_position(part[1],len(native))
            low,high=a+ta,b+tb
            if not low<high: raise ValueError('ordered source interval required')
            intervals.append((low,high))
        # Intervals may overlap. Exact union here subtracts no positive gap.
        intervals=yield from _sort_steps(intervals,lambda v:v[0])
        merged=[]
        for low,high in intervals:
            yield
            if merged and low<=merged[-1][1]: merged[-1]=(merged[-1][0],max(high,merged[-1][1]))
            else: merged.append((low,high))
        barriers=[]
        cursor=0
        for name,border in (('left',left),('right',right)):
            for index,(a,b) in enumerate(zip(border,border[1:])):
                yield
                allowed=[]
                if name==side:
                    while cursor<len(merged) and merged[cursor][1]<=index:
                        yield
                        cursor+=1
                    current=cursor
                    while current<len(merged) and merged[current][0]<index+1:
                        yield
                        low,high=merged[current]
                        if high>index: allowed.append((max(Fraction(0),low-index),min(Fraction(1),high-index)))
                        current+=1
                start=Fraction(0)
                for low,high in allowed:
                    yield
                    if start<low: barriers.append(self._edge(a,b,start,low))
                    start=max(start,high)
                if start<1: barriers.append(self._edge(a,b,start,Fraction(1)))
        # End caps remain prohibited even where the neighboring lane extends
        # beyond the source lane. An open finite paint window is not an exit.
        for a,b in ((left[0],right[0]),(left[-1],right[-1])):
            yield
            barriers.append(self._edge(a,b,Fraction(0),Fraction(1)))
        for edge,error in barriers:
            yield
            prepared.roundoff=max(prepared.roundoff,error)
        prepared.barrier_count=len(barriers)
        segments=prepared.segments+[v[0] for v in barriers]
        prepared.root=yield from _tree_steps(segments)
        return prepared

    def _edge(self,a,b,start,end):
        native=(a,b)
        points=[exact_point(native,(0,str(t.numerator),str(t.denominator))) for t in (start,end)]
        exact=[p[:2] for p in points]
        converted=[tuple(float(v) for v in p) for p in exact]
        if converted[0]==converted[1]: raise NumericResolution('CROSSING_BOUNDARY_EDGE_COLLAPSED')
        error=0.
        for original,value in zip(exact,converted):
            error=max(error,math.hypot(*(float(abs(original[i]-Fraction(value[i]))) for i in (0,1)))
                      +32.*2.**-52*max(1.,*(abs(v) for v in value)))
        dx,dy=converted[1][0]-converted[0][0],converted[1][1]-converted[0][1]
        length=math.hypot(dx,dy)
        if not number(length) or length<=0: raise ValueError('invalid source boundary length')
        return dict(points=converted,exact=exact,box=_box(converted),normal=(-dy/length,dx/length)),error
