"""SDK-independent finite reference position; no path extrapolation."""
import math


def point_at_distance(points,value):
    if value<0:raise ValueError('negative route distance')
    walked=0.0
    for a,b in zip(points,points[1:]):
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if length<=1e-8:continue
        if walked+length+1e-8>=value:
            u=max(0.0,min(1.0,(value-walked)/length))
            return a[0]+u*(b[0]-a[0]),a[1]+u*(b[1]-a[1]),math.atan2(b[1]-a[1],b[0]-a[0])
        walked+=length
    raise ValueError('stop boundary outside verified route')
