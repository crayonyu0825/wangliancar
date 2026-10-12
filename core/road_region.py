"""Verified simple road outline; concavities are never filled by a hull."""
import math
from core.behavior_protocol import finite, require


def cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def distance(point,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    square=dx*dx+dy*dy
    u=max(0.0,min(1.0,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/square)) if square else 0.0
    return math.hypot(point[0]-a[0]-u*dx,point[1]-a[1]-u*dy)


class RoadRegion(object):
    def __init__(self,points):
        require(isinstance(points,(list,tuple)) and 3<=len(points)<=512,'road outline unavailable or over budget')
        clean=[]
        for p in points:
            require(isinstance(p,(list,tuple)) and len(p)==2 and all(finite(v) for v in p),'road outline invalid')
            q=tuple(p)
            if not clean or q!=clean[-1]:clean.append(q)
        if clean and clean[0]==clean[-1]:clean.pop()
        require(len(clean)>=3 and len(set(clean))==len(clean),'road outline repeated vertex or degenerate')
        area=sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(clean,clean[1:]+clean[:1]))
        require(abs(area)>1e-8,'road outline zero area')
        edges=list(zip(clean,clean[1:]+clean[:1]))
        for i,(a,b) in enumerate(edges):
            for j,(c,d) in enumerate(edges[i+2:],i+2):
                if i==0 and j==len(edges)-1:continue
                require(not (cross(a,b,c)*cross(a,b,d)<-1e-8 and cross(c,d,a)*cross(c,d,b)<-1e-8)
                        and min(distance(a,c,d),distance(b,c,d),distance(c,a,b),distance(d,a,b))>1e-8,
                        'road outline self-intersects')
        self.points=clean
        self.edges=edges
        self.bounds=[(min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])) for a,b in edges]

    def contains(self,point):
        contained=False
        for a,b in self.edges:
            if distance(point,a,b)<=1e-8:return True
            if (a[1]>point[1])!=(b[1]>point[1]):
                x=a[0]+(point[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
                if x>point[0]:contained=not contained
        return contained

    def intersects(self,body):
        """Exact overlap with a convex footprint, including a concave outline."""
        if any(self.contains(p) for p in body):return True
        for p in self.points:
            signs=[cross(a,b,p) for a,b in zip(body,body[1:]+body[:1])]
            if all(v>=-1e-8 for v in signs) or all(v<=1e-8 for v in signs):return True
        for a,b in zip(body,body[1:]+body[:1]):
            for c,d in self.edges:
                if (cross(a,b,c)*cross(a,b,d)<-1e-8
                        and cross(c,d,a)*cross(c,d,b)<-1e-8):return True
                if min(distance(a,c,d),distance(b,c,d),distance(c,a,b),distance(d,a,b))<1e-8:return True
        return False

    def body_gap(self,body):
        if not all(self.contains(p) for p in body):return 0.0
        minimum=float('inf')
        bbox=(min(p[0] for p in body),min(p[1] for p in body),max(p[0] for p in body),max(p[1] for p in body))
        candidates=[]
        for (c,d),bound in zip(self.edges,self.bounds):
            dx=max(0.0,bbox[0]-bound[2],bound[0]-bbox[2])
            dy=max(0.0,bbox[1]-bound[3],bound[1]-bbox[3])
            candidates.append((math.hypot(dx,dy),c,d))
        candidates.sort(key=lambda value:value[0])
        for a,b in zip(body,body[1:]+body[:1]):
            for lower,c,d in candidates:
                if lower>minimum:break
                if cross(a,b,c)*cross(a,b,d)<-1e-8 and cross(c,d,a)*cross(c,d,b)<-1e-8:return 0.0
                minimum=min(minimum,distance(a,c,d),distance(b,c,d),distance(c,a,b),distance(d,a,b))
        for point in self.points:
            if not (bbox[0]<point[0]<bbox[2] and bbox[1]<point[1]<bbox[3]):continue
            signs=[cross(a,b,point) for a,b in zip(body,body[1:]+body[:1])]
            if all(v>1e-8 for v in signs) or all(v<-1e-8 for v in signs):return 0.0
        return minimum

    def to_dict(self):
        return dict(kind='verified_simple_road_outline',points=self.points)
