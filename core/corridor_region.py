"""Exact topology of a bounded union of explicit simple XY polygons.

This is geometry, not map/visibility authority. Every input polygon denotes
only its own interior; gaps, holes and disconnected components are retained.
Float inputs are treated as exact rational coordinates for topology. Exposed
edge endpoints return to float only for the existing continuous distance test,
with a conservative conversion/roundoff allowance subtracted from clearance.
"""
import heapq
import math
from fractions import Fraction

from core.region_geometry import simple_outline,convex_polygon
from core.validation import number

MAX_POLYGONS=32
MAX_VERTICES=20000
MAX_FRACTION_BITS=8192


class NumericResolution(Exception):
    pass


class PreparationLimit(Exception):
    pass


def _drain(steps,check):
    while True:
        # The callback is outside the generator: a frame budget interrupts the
        # driver without closing or losing the partially built static geometry.
        check()
        try: next(steps)
        except StopIteration as completed: return completed.value


def _sort_steps(values,key):
    records=[]
    for index,value in enumerate(values):
        yield
        records.append((key(value),index,value))
    width=1
    while width<len(records):
        merged=[]
        for start in range(0,len(records),2*width):
            a,b,end=start,min(start+width,len(records)),min(start+2*width,len(records))
            middle=b
            while a<middle or b<end:
                yield
                if b==end or (a<middle and records[a][:2]<=records[b][:2]):
                    merged.append(records[a]); a+=1
                else: merged.append(records[b]); b+=1
        records=merged; width*=2
    result=[]
    for record in records:
        yield
        result.append(record[2])
    return result


def _bounded(value):
    if max(value.numerator.bit_length(),value.denominator.bit_length())>MAX_FRACTION_BITS:
        raise NumericResolution("REGION_NUMERIC_RESOLUTION")
    return value


def _cross(a,b): return _bounded(a[0]*b[1]-a[1]*b[0])
def _sub(a,b): return (a[0]-b[0],a[1]-b[1])
def _dot(a,b): return _bounded(a[0]*b[0]+a[1]*b[1])
def _at(edge,t): return tuple(_bounded(edge['a'][i]+t*edge['vector'][i]) for i in (0,1))


def _box(points):
    return (min(p[0] for p in points),min(p[1] for p in points),
            max(p[0] for p in points),max(p[1] for p in points))


def _box_gap(a,b):
    return math.hypot(max(0.,a[0]-b[2],b[0]-a[2]),max(0.,a[1]-b[3],b[1]-a[3]))


class _BodyBoxes(object):
    def __init__(self,body):
        self.body,self.box,self.origin,self.axes=body,_box(body),body[0],[]
        for a,b in zip(body,body[1:]+body[:1]):
            dx,dy=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dy)
            if not number(length) or length<=0: raise ValueError('invalid body edge')
            nx,ny=-dy/length,dx/length
            values=[(p[0]-self.origin[0])*nx+(p[1]-self.origin[1])*ny for p in body]
            if not all(number(v) for v in values): raise ValueError('body projection overflow')
            self.axes.append((nx,ny,min(values),max(values)))

    def lower_bound(self,box):
        lower=_box_gap(self.box,box)
        for nx,ny,low,high in self.axes:
            xmin,xmax=(box[0],box[2]) if nx>=0 else (box[2],box[0])
            ymin,ymax=(box[1],box[3]) if ny>=0 else (box[3],box[1])
            first=(xmin-self.origin[0])*nx+(ymin-self.origin[1])*ny
            last=(xmax-self.origin[0])*nx+(ymax-self.origin[1])*ny
            lower=max(lower,first-high,low-last)
        return lower

    def segment_gap(self,edge):
        # Separating-axis distance is a lower bound, sufficient for a road
        # clearance certificate. It avoids repeated vertex/edge distances.
        # Zero/unresolved separation is checked with exact contact below.
        a,b=edge['points']
        lower=0.
        for nx,ny,low,high in self.axes:
            first=(a[0]-self.origin[0])*nx+(a[1]-self.origin[1])*ny
            last=(b[0]-self.origin[0])*nx+(b[1]-self.origin[1])*ny
            if not number(first) or not number(last): raise ValueError('edge projection overflow')
            lower=max(lower,min(first,last)-high,low-max(first,last))
        nx,ny=edge['normal']
        values=[(p[0]-a[0])*nx+(p[1]-a[1])*ny for p in self.body]
        if not all(number(v) for v in values): raise ValueError('edge projection overflow')
        return max(lower,min(values),-max(values))


def _cut(edge,t):
    edge['cuts'].add(_bounded(t))
    if t==0: edge['start_contact']=True
    if t==1: edge['end_contact']=True


def _contacts(first,second):
    shared=False
    for t,a in ((0,first['a']),(1,first['b'])):
        for u,b in ((0,second['a']),(1,second['b'])):
            if a==b:
                _cut(first,Fraction(t)); _cut(second,Fraction(u)); shared=True
    r,s=first['vector'],second['vector']
    offset=_sub(second['a'],first['a'])
    denominator=_cross(r,s)
    same=(first['a']==second['a'] and first['b']==second['b'] or
          first['a']==second['b'] and first['b']==second['a'])
    if shared and (denominator or same): return True
    if denominator:
        t=_bounded(_cross(offset,s)/denominator)
        u=_bounded(_cross(offset,r)/denominator)
        if 0<=t<=1 and 0<=u<=1:
            _cut(first,t); _cut(second,u)
            return True
    elif _cross(offset,r)==0:
        square=_dot(r,r)
        low,high=sorted((_bounded(_dot(offset,r)/square),
                         _bounded(_dot(_sub(second['b'],first['a']),r)/square)))
        low,high=max(Fraction(0),low),min(Fraction(1),high)
        if low<=high:
            for t in (low,high):
                _cut(first,t)
                u=_bounded(_dot(_sub(_at(first,t),second['a']),s)/_dot(s,s))
                _cut(second,u)
            return True
    return shared


class _Polygon(object):
    def __init__(self,points,check):
        _drain(self.initialize_steps(points),check)

    def initialize_steps(self,points):
        self.points=points
        self.exact=[]
        lowx,lowy=points[0]; highx,highy=points[0]
        for p in points:
            yield
            lowx,lowy=min(lowx,p[0]),min(lowy,p[1])
            highx,highy=max(highx,p[0]),max(highy,p[1])
            self.exact.append(tuple(Fraction(v) for v in p))
        self.box=(lowx,lowy,highx,highy)
        if not number(highx-lowx) or not number(highy-lowy):
            raise ValueError('region arithmetic overflow')
        area=Fraction(0)
        for a,b in zip(self.exact,self.exact[1:]+self.exact[:1]):
            yield
            area=_bounded(area+_cross(_sub(a,self.exact[0]),_sub(b,self.exact[0])))
        if area==0: raise ValueError('region has no exact area')
        if area<0:
            self.points=list(reversed(points)); self.exact.reverse()
        self.edges=list(zip(self.exact,self.exact[1:]+self.exact[:1]))
        for index,b in enumerate(self.exact):
            yield
            a,c=self.exact[index-1],self.exact[(index+1)%len(self.exact)]
            if _cross(_sub(a,b),_sub(c,b))==0 and _dot(_sub(a,b),_sub(c,b))>0:
                raise ValueError('exact adjacent region edges overlap')
        self.low,self.high=min(p[1] for p in self.exact),max(p[1] for p in self.exact)
        self.bands=[[] for unused in range(min(128,len(points)))]
        for edge in self.edges:
            a,b=edge
            coefficients=(a[1]-b[1],b[0]-a[0],_cross(a,b))
            denominator=max(v.denominator for v in coefficients)
            # Input int/float denominators are powers of two. Integer line
            # coefficients avoid repeated rational divisions at every query.
            integers=tuple(v.numerator*(denominator//v.denominator) for v in coefficients)
            record=(a,b,_sub(b,a),integers)
            for index in range(self.band(min(a[1],b[1])),self.band(max(a[1],b[1]))+1):
                yield
                self.bands[index].append(record)

    def band(self,y):
        return max(0,min(len(self.bands)-1,int((y-self.low)*len(self.bands)/(self.high-self.low))))

    def state(self,point,check):
        return _drain(self.state_steps(point),check)

    def state_steps(self,point):
        # -1 outside, 1 strictly inside, or the oriented boundary edge.
        x,y=point
        if not self.box[0]<=x<=self.box[2] or not self.low<=y<=self.high: return -1
        denominator=x.denominator*y.denominator//math.gcd(x.denominator,y.denominator)
        if denominator.bit_length()>MAX_FRACTION_BITS: raise NumericResolution('REGION_NUMERIC_RESOLUTION')
        ix,iy=x.numerator*(denominator//x.denominator),y.numerator*(denominator//y.denominator)
        contained=False
        for a,b,vector,coefficients in self.bands[self.band(y)]:
            yield
            orientation=coefficients[0]*ix+coefficients[1]*iy+coefficients[2]*denominator
            if orientation.bit_length()>2*MAX_FRACTION_BITS: raise NumericResolution('REGION_NUMERIC_RESOLUTION')
            if (orientation==0 and min(a[0],b[0])<=x<=max(a[0],b[0])
                    and min(a[1],b[1])<=y<=max(a[1],b[1])):
                return vector
            if (a[1]>y)!=(b[1]>y):
                if (orientation>0)==(b[1]>a[1]): contained=not contained
        return 1 if contained else -1


def _tree(segments,check):
    return _drain(_tree_steps(segments),check)


def _tree_steps(segments):
    box=list(segments[0]['box'])
    for segment in segments:
        yield
        for i in (0,1): box[i]=min(box[i],segment['box'][i])
        for i in (2,3): box[i]=max(box[i],segment['box'][i])
    box=tuple(box)
    if len(segments)<=8: return (box,segments,None)
    axis=0 if box[2]-box[0]>=box[3]-box[1] else 1
    ordered=yield from _sort_steps(segments,lambda v:v['box'][axis]+(v['box'][axis+2]-v['box'][axis])*.5)
    middle=len(ordered)//2
    left=yield from _tree_steps(ordered[:middle])
    right=yield from _tree_steps(ordered[middle:])
    return (box,None,(left,right))


def _exact_contact(body,edge,check):
    shapes=([tuple(Fraction(v) for v in p) for p in body],edge['exact'])
    for shape in shapes:
        for a,b in zip(shape,shape[1:]+shape[:1]):
            check(); vector=_sub(b,a)
            values=[[_cross(vector,_sub(p,a)) for p in vertices] for vertices in shapes]
            if max(values[0])<min(values[1]) or max(values[1])<min(values[0]): return False
    return True


class _PreparedRegion(object):
    def __init__(self,polygons,segments,roundoff,check):
        self.polygons,self.roundoff=polygons,roundoff
        self.root=_tree(segments,check)
        self.edge_count=len(segments)

    def gap(self,body,point,check):
        exact=tuple(Fraction(v) for v in point)
        if not any(poly.state(exact,check)!=-1 for poly in self.polygons): return -1.
        boxes=_BodyBoxes(body)
        queue=[(boxes.lower_bound(self.root[0]),0,self.root)]
        serial,minimum=0,float('inf')
        while queue:
            check()
            lower,unused,node=heapq.heappop(queue)
            if lower-self.roundoff>=minimum: break
            if node[1] is not None:
                for edge in node[1]:
                    check()
                    if boxes.lower_bound(edge['box'])-self.roundoff>=minimum: continue
                    gap=boxes.segment_gap(edge)
                    if gap<=self.roundoff:
                        if _exact_contact(body,edge,check): return 0.
                        raise NumericResolution('REGION_CLEARANCE_WITHIN_ROUNDOFF')
                    minimum=min(minimum,gap)
            else:
                for child in node[2]:
                    serial+=1
                    heapq.heappush(queue,(boxes.lower_bound(child[0]),serial,child))
        return minimum-self.roundoff


class CorridorRegion(object):
    """Explicit union input; constructing this object proves no source facts.

    Use a list of simple polygon XY rings. A ring may be concave. The caller
    must separately establish drivable area, complete visibility, permission,
    source identity and deadlines. A hull or guessed corridor is never added.
    """
    def __init__(self,polygons,preparation_max_checks=None):
        if preparation_max_checks is not None and (type(preparation_max_checks) is not int or preparation_max_checks<=0):
            raise ValueError('positive total preparation budget required')
        self.polygons=polygons
        self._fingerprint,self._prepared=None,None
        self.preparation_max_checks=preparation_max_checks
        self._job_raw,self._job,self._job_steps,self._job_error=None,None,0,None

    def prepare(self,check,max_checks):
        if self.preparation_max_checks is not None:
            if type(self.preparation_max_checks) is not int or self.preparation_max_checks<=0:
                raise ValueError('positive total preparation budget required')
            return self._prepare_incrementally(check)
        fingerprint=_drain(self._snapshot_steps(self.polygons),check)
        if fingerprint==self._fingerprint and self._prepared is not None:
            check(); return self._prepared
        # The legacy mutable-input path retains its conservative float guard.
        normalized=[]
        for ring in fingerprint:
            normalized.append(simple_outline(ring,max_checks=max_checks,check_step=check))
        prepared=_drain(self._compile_steps(normalized),check)
        check()
        self._fingerprint,self._prepared=fingerprint,prepared
        return prepared

    def body_clearance(self,body,check,max_checks):
        """Nominal convex body clearance in this exact union, without authority.

        Reuses the same exterior/hole/contact query as continuous planning.
        The caller supplies its own shared work/deadline callback and must
        check current map/coverage/objects independently. Zero is contact.
        """
        check()
        if not isinstance(body,(list,tuple)) or not 3<=len(body)<=64:
            raise ValueError('bounded convex body required')
        for unused in range(len(body)**2): check()
        body=convex_polygon(body)
        prepared=self.prepare(check,max_checks)
        point=tuple(sum(p[axis]/len(body) for p in body) for axis in (0,1))
        check()
        gap=prepared.gap(body,point,check)
        check()
        return gap

    def _prepare_incrementally(self,check):
        raw=self.polygons
        if raw is not self._job_raw:
            self._fingerprint,self._prepared=None,None
            self._job_raw,self._job_steps,self._job_error=raw,0,None
            self._job=self._incremental_steps(raw)
        if self._job_error is not None:
            check()
            kind,message=self._job_error
            raise kind(message)
        if self._prepared is not None:
            check(); return self._prepared
        while True:
            # An exception from the caller's per-frame budget must not enter
            # the generator; otherwise Python closes it and loses progress.
            check()
            if self._job_steps>=self.preparation_max_checks:
                self._job_error=(PreparationLimit,'REGION_TOTAL_PREPARATION_LIMIT')
                self._job.close(); self._job=None
                raise PreparationLimit(self._job_error[1])
            self._job_steps+=1
            try: next(self._job)
            except StopIteration as completed:
                self._fingerprint,self._prepared=completed.value
                self._job=None
                return self._prepared
            except (ValueError,TypeError,OverflowError,ZeroDivisionError,NumericResolution) as error:
                # Retain a detached reason, not an exception/traceback which
                # grows on repeated raises and keeps a failed job alive.
                self._job_error=(type(error),str(error)); self._job=None
                raise

    def _incremental_steps(self,raw):
        fingerprint=yield from self._snapshot_steps(raw,immutable=True)
        prepared=yield from self._compile_steps(fingerprint)
        # No source deadline or safety result is retained with this geometry.
        yield
        return fingerprint,prepared

    def _snapshot_steps(self,raw,immutable=False):
        if (not isinstance(raw,(tuple,list)) or not 1<=len(raw)<=MAX_POLYGONS
                or any(not isinstance(v,(tuple,list)) for v in raw)
                or sum(len(v) for v in raw)>MAX_VERTICES):
            raise ValueError('bounded explicit polygon regions required')
        if immutable and (type(raw) is not tuple or any(type(v) is not tuple for v in raw)):
            raise ValueError('incremental preparation requires immutable tuple rings')
        snapshot=[]
        for ring in raw:
            values,seen=[],set()
            if len(ring)<3: raise ValueError('region needs three vertices')
            for point in ring:
                yield
                if (not isinstance(point,(tuple,list)) or len(point)!=2
                        or not all(number(v) for v in point) or immutable and type(point) is not tuple):
                    raise ValueError('explicit finite region XY vertices required')
                point=tuple(point)
                if point in seen: raise ValueError('repeated outline vertex')
                values.append(point); seen.add(point)
            snapshot.append(tuple(values))
        return tuple(snapshot)

    def _compile_steps(self,snapshot):
        polygons=[]
        for ring in snapshot:
            poly=_Polygon.__new__(_Polygon)
            yield from poly.initialize_steps(list(ring))
            polygons.append(poly)
        edges=[]
        for index,poly in enumerate(polygons):
            for edge_index,(a,b) in enumerate(poly.edges):
                yield
                edge=dict(polygon=index,a=a,b=b,vector=_sub(b,a),cuts={Fraction(0),Fraction(1)},
                          start_contact=False,end_contact=False,box=_box((a,b)),internal=False,index=edge_index)
                edges.append(edge)
        reversed_edges={}
        def point_key(point):
            # Fraction.__hash__ in Python 3.6 repeatedly computes a modular
            # inverse. Canonical numerator/denominator pairs are exact keys.
            return tuple((v.numerator,v.denominator) for v in point)
        for edge in edges:
            yield
            akey,bkey=point_key(edge['a']),point_key(edge['b'])
            for prior in reversed_edges.get((bkey,akey),[]):
                yield
                if prior['polygon']!=edge['polygon']:
                    prior['internal']=edge['internal']=True
            reversed_edges.setdefault((akey,bkey),[]).append(edge)
        active=[]
        ordered=yield from _sort_steps(range(len(edges)),lambda i:edges[i]['box'][0])
        for index in ordered:
            yield
            edge=edges[index]
            retained=[]
            for prior_index in active:
                yield
                if edges[prior_index]['box'][2]>=edge['box'][0]: retained.append(prior_index)
            active=retained
            for prior_index in active:
                yield
                prior=edges[prior_index]
                same=prior['polygon']==edge['polygon']
                if same and abs(prior['index']-edge['index']) in (1,len(polygons[edge['polygon']].edges)-1):
                    continue
                if prior['box'][3]<edge['box'][1] or edge['box'][3]<prior['box'][1]: continue
                if _contacts(prior,edge) and same:
                    raise ValueError('exact region self intersection')
            active.append(index)
        segments,roundoff,cache,previous_polygon=[],0.,None,None
        for edge in edges:
            yield
            own=edge['polygon']
            if previous_polygon!=own: cache=None
            if edge['internal']:
                cache=None; previous_polygon=own
                continue
            cuts=yield from _sort_steps(edge['cuts'],lambda v:v)
            for low,high in zip(cuts,cuts[1:]):
                yield
                if high==low: continue
                a,b=_at(edge,low),_at(edge,high)
                midpoint=tuple(_bounded((a[i]+b[i])/2) for i in (0,1))
                reuse=low==0 and not edge['start_contact'] and cache is not None
                states,internal={},False
                for other,poly in enumerate(polygons):
                    if other==own: continue
                    yield
                    state=cache.get(other) if reuse else None
                    if state is None: state=yield from poly.state_steps(midpoint)
                    states[other]=state if state in (-1,1) else None
                    if state==1:
                        internal=True
                    elif state!=-1 and _cross(edge['vector'],state)==0 and _dot(edge['vector'],state)<0:
                        internal=True  # The other interior covers our outward side.
                cache=states if high==1 and not edge['end_contact'] else None
                if internal: continue
                converted=[]
                for point in (a,b):
                    value=tuple(float(v) for v in point)
                    error=math.hypot(*(float(abs(point[i]-Fraction(value[i]))) for i in (0,1)))
                    # Include arithmetic roundoff of the subsequent distance
                    # and box tests; it can only reduce a clearance proof.
                    allowance=32.*(2.**-52)*max(1.,*(abs(v) for v in value))
                    roundoff=max(roundoff,error+allowance)
                    converted.append(value)
                if converted[0]==converted[1]:
                    raise NumericResolution('REGION_EDGE_COLLAPSED')
                dx,dy=converted[1][0]-converted[0][0],converted[1][1]-converted[0][1]
                length=math.hypot(dx,dy)
                if not number(length) or length<=0: raise ValueError('region edge arithmetic overflow')
                segments.append(dict(points=converted,exact=[a,b],box=_box(converted),normal=(-dy/length,dx/length)))
            previous_polygon=own
        if not segments: raise ValueError('region has no exterior boundary')
        prepared=_PreparedRegion.__new__(_PreparedRegion)
        prepared.polygons,prepared.roundoff=polygons,roundoff
        prepared.edge_count=len(segments)
        prepared.segments=segments
        prepared.root=yield from _tree_steps(segments)
        return prepared
