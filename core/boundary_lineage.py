"""Exact native-edge provenance; XYZ output is rounded only for SDK queries.

Positions use a native edge index and canonical rational numerator/denominator
strings, so JSON replay does not round them through a binary64 number.
"""
import math
from fractions import Fraction

from core.validation import number

MAX_NATIVE_POINTS=20000
MAX_LOCAL_POINTS=4096
MAX_RATIO_DIGITS=400


def native_points3(raw):
    if not isinstance(raw,(tuple,list)) or not 2<=len(raw)<=MAX_NATIVE_POINTS:
        raise ValueError('BOUNDARY_POINT_BUDGET_OR_MISSING')
    result=[]
    for point in raw:
        if not isinstance(point,(tuple,list)) or len(point)!=3 or not all(number(v) for v in point):
            raise ValueError('BOUNDARY_COORDINATE_UNAVAILABLE')
        value=tuple(float(v) for v in point)
        if result and math.hypot(value[0]-result[-1][0],value[1]-result[-1][1])<=1e-6:
            if abs(value[2]-result[-1][2])>1e-6:
                raise ValueError('VERTICAL_BOUNDARY_SEGMENT_UNSUPPORTED')
            continue
        result.append(value)
    if len(result)<2: raise ValueError('BOUNDARY_DEGENERATE')
    return tuple(result)


def encode_position(index,ratio):
    return (index,str(ratio.numerator),str(ratio.denominator))


def decode_position(raw,count):
    if (not isinstance(raw,(list,tuple)) or len(raw)!=3 or type(raw[0]) is not int
            or not 0<=raw[0]<count-1 or any(type(v) is not str or not 1<=len(v)<=MAX_RATIO_DIGITS
            or any(c not in '0123456789' for c in v) for v in raw[1:])):
        raise ValueError('BOUNDARY_SOURCE_POSITION_INVALID')
    numerator,denominator=int(raw[1]),int(raw[2])
    if denominator<=0 or numerator>denominator:
        raise ValueError('BOUNDARY_SOURCE_POSITION_INVALID')
    ratio=Fraction(numerator,denominator)
    if encode_position(raw[0],ratio)!=tuple(raw):
        raise ValueError('BOUNDARY_SOURCE_POSITION_NONCANONICAL')
    return raw[0],ratio


def exact_point(points,position):
    index,ratio=decode_position(position,len(points))
    return tuple(Fraction(points[index][i])+ratio*(Fraction(points[index+1][i])-Fraction(points[index][i]))
                 for i in range(3))


def window_with_lineage(points,start,end):
    return _drain(window_lineage_steps(points,start,end))


def _drain(steps):
    while True:
        try: next(steps)
        except StopIteration as completed: return completed.value


def window_lineage_steps(points,start,end):
    if not all(number(v) for v in (start,end)) or not start<end:
        raise ValueError('BOUNDARY_WINDOW_INVALID')
    result,positions,along=[],[],0.
    start=max(0.,start)
    for index,(a,b) in enumerate(zip(points,points[1:])):
        yield
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if not number(length) or length<=0: raise ValueError('BOUNDARY_SEGMENT_INVALID')
        low,high=max(start,along),min(end,along+length)
        if high>low:
            for at in (low,high):
                yield
                ratio=Fraction(0 if at==along else 1 if at==along+length else (at-along)/length)
                position=encode_position(index,ratio)
                point=tuple(float(v) for v in exact_point(points,position))
                if not positions or Fraction(positions[-1][0])+decode_position(positions[-1],len(points))[1]!=index+ratio:
                    result.append(point); positions.append(position)
            if len(result)>MAX_LOCAL_POINTS: raise ValueError('LOCAL_BOUNDARY_POINT_BUDGET')
        along+=length
        if along>=end: break
    if len(result)<2: raise ValueError('BOUNDARY_WINDOW_EMPTY')
    return result,positions


def densify_with_lineage(native,points,positions,spacing):
    return _drain(densify_lineage_steps(native,points,positions,spacing))


def densify_lineage_steps(native,points,positions,spacing):
    if not number(spacing) or spacing<=0 or len(points)!=len(positions):
        raise ValueError('BOUNDARY_DENSIFY_INVALID')
    result,output=[points[0]],[positions[0]]
    for a,b,left,right in zip(points,points[1:],positions,positions[1:]):
        yield
        i,low=decode_position(left,len(native)); j,high=decode_position(right,len(native))
        if low==1: i+=1; low=Fraction(0)
        if j==i+1 and high==0: high=Fraction(1); j=i
        if j!=i or not low<high: raise ValueError('BOUNDARY_SOURCE_EDGE_SKIPPED')
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if not number(length): raise ValueError('BOUNDARY_SEGMENT_INVALID')
        steps=max(1,int(math.ceil(length/spacing)))
        if len(result)+steps>MAX_LOCAL_POINTS: raise ValueError('LOCAL_BOUNDARY_POINT_BUDGET')
        for index in range(1,steps+1):
            yield
            ratio=low+(high-low)*Fraction(index,steps)
            position=encode_position(i,ratio)
            result.append(tuple(float(v) for v in exact_point(native,position)))
            output.append(position)
    return result,output
