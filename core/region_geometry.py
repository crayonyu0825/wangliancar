"""Shared polygon helpers lifted from the existing decision observations.

No sensing, behavior selection or SDK dependency. Touching is intersection.
"""
import math


def _finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def convex_polygon(raw):
    if (not isinstance(raw, (tuple, list)) or not 3 <= len(raw) <= 128
            or any(not isinstance(p, (tuple, list)) or len(p) != 2
                   or not all(_finite(v) for v in p) for p in raw)):
        raise ValueError("polygon coverage missing or invalid")
    points = [(float(p[0]), float(p[1])) for p in raw]
    if len(set(points)) != len(points):
        raise ValueError("repeated polygon vertex")
    area = sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points, points[1:]+points[:1]))
    signs = [cross(points[i-1], points[i], points[(i+1)%len(points)])
             for i in range(len(points))]
    signs = [v > 0 for v in signs if abs(v) > 1e-8]
    if abs(area) <= 1e-8 or not signs or not all(v == signs[0] for v in signs):
        raise ValueError("nonconvex or degenerate polygon")
    # Consistent turn signs alone can admit a star with multiple windings.
    edges = list(zip(points, points[1:]+points[:1]))
    for i, (a,b) in enumerate(edges):
        for j, (c,d) in enumerate(edges[i+1:], i+1):
            if j == i+1 or (i == 0 and j == len(edges)-1):
                continue
            if segments_intersect(a,b,c,d):
                raise ValueError("self-intersecting polygon")
    return points


def simple_outline(raw, max_checks=100000, check_step=None):
    """Validate a possibly concave map outline without inventing a hull.

    Bound work for long SDK samples. Exhaustion is unknown geometry, never a
    certificate. Sorting edge boxes avoids quadratic work on straight lanes.
    """
    if (not isinstance(raw, (tuple, list)) or not 3 <= len(raw) <= 40000
            or any(not isinstance(p, (tuple, list)) or len(p) != 2
                   or not all(_finite(v) for v in p) for p in raw)):
        raise ValueError("invalid map outline")
    points = [tuple(p) for p in raw]
    if len(set(points)) != len(points):
        raise ValueError("repeated outline vertex")
    for index,b in enumerate(points):
        if check_step is not None: check_step()
        a,c=points[index-1],points[(index+1)%len(points)]
        if (abs(cross(a,b,c))<=1e-8
                and (a[0]-b[0])*(c[0]-b[0])+(a[1]-b[1])*(c[1]-b[1])>1e-10):
            raise ValueError("overlapping adjacent outline edges")
    if abs(sum(cross(points[0], a, b) for a, b in zip(points, points[1:]+points[:1]))) <= 1e-8:
        raise ValueError("degenerate outline")
    boxes = sorted((min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]), i, a, b)
                   for i, (a,b) in enumerate(zip(points, points[1:]+points[:1])))
    active, checks = [], 0
    for edge in boxes:
        if check_step is not None: check_step()
        active = [v for v in active if v[1]+1e-8 >= edge[0]]
        for prior in active:
            if check_step is not None: check_step()
            checks += 1
            if checks > max_checks:
                raise ValueError("outline validation budget exhausted")
            if abs(prior[4]-edge[4]) in (1, len(points)-1):
                continue
            if prior[3]+1e-8 < edge[2] or edge[3]+1e-8 < prior[2]:
                continue
            if segments_intersect(prior[5], prior[6], edge[5], edge[6]):
                raise ValueError("self-intersecting outline")
        active.append(edge)
    return points


def segments_intersect(a,b,c,d):
    signs = (cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b))
    if signs[0]*signs[1] < -1e-8 and signs[2]*signs[3] < -1e-8:
        return True
    for p,u,v,value in ((c,a,b,signs[0]), (d,a,b,signs[1]),
                         (a,c,d,signs[2]), (b,c,d,signs[3])):
        if (abs(value) <= 1e-8 and min(u[0],v[0])-1e-8 <= p[0] <= max(u[0],v[0])+1e-8
                and min(u[1],v[1])-1e-8 <= p[1] <= max(u[1],v[1])+1e-8):
            return True
    return False


def inside(point, region):
    values = [cross(a,b,point) for a,b in zip(region, region[1:]+region[:1])]
    return all(v >= -1e-8 for v in values) or all(v <= 1e-8 for v in values)


def intersects(left, right):
    for shape in (left, right):
        for a,b in zip(shape, shape[1:]+shape[:1]):
            nx,ny = -(b[1]-a[1]), b[0]-a[0]
            l = [nx*p[0]+ny*p[1] for p in left]
            r = [nx*p[0]+ny*p[1] for p in right]
            if max(l) < min(r)-1e-8 or max(r) < min(l)-1e-8:
                return False
    return True


def hull(points):
    ordered = sorted(set(points))
    if len(ordered) < 3:
        raise ValueError("insufficient swept footprint")
    lower,upper = [],[]
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2],lower[-1],p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2],upper[-1],p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def rectangle(x,y,heading,front,rear,half_width):
    c,s = math.cos(heading),math.sin(heading)
    return [(x+dx*c-dy*s,y+dx*s+dy*c) for dx,dy in
            ((front,half_width),(front,-half_width),(-rear,-half_width),(-rear,half_width))]


def circle_intersects(region,x,y,radius):
    if inside((x,y),region):
        return True
    for a,b in zip(region,region[1:]+region[:1]):
        dx,dy = b[0]-a[0],b[1]-a[1]
        length = dx*dx+dy*dy
        t = max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/length)) if length > 1e-12 else 0.
        if math.hypot(x-a[0]-t*dx,y-a[1]-t*dy) <= radius+1e-8:
            return True
    return False


def outline_contains_point(point,outline):
    """Ray membership of a previously validated simple, possibly concave ring."""
    contained = False
    for a,b in zip(outline,outline[1:]+outline[:1]):
        if (abs(cross(a,b,point))<=1e-8 and min(a[0],b[0])-1e-8<=point[0]<=max(a[0],b[0])+1e-8
                and min(a[1],b[1])-1e-8<=point[1]<=max(a[1],b[1])+1e-8):
            return True
        if (a[1]>point[1])!=(b[1]>point[1]) and point[0]<a[0]+(point[1]-a[1])*(b[0]-a[0])/(b[1]-a[1]):
            contained = not contained
    return contained


def outline_contains_segment(a,b,outline,max_checks=100000):
    """Split at all ring contacts, including a concave vertex or collinear edge.

    Checking only endpoints/midpoint can miss a cutout. Budget exhaustion is
    an unknown result (ValueError), not an inside certificate.
    """
    if len(outline)<3 or len(outline)>40000:
        raise ValueError("invalid containment outline")
    dx,dy = b[0]-a[0],b[1]-a[1]
    square = dx*dx+dy*dy
    if square<=1e-12:
        return outline_contains_point(a,outline)
    cuts,checks = [0.,1.],0
    for c,d in zip(outline,outline[1:]+outline[:1]):
        checks += 1
        if checks>max_checks: raise ValueError("segment containment budget")
        ux,uy = d[0]-c[0],d[1]-c[1]
        denominator = dx*uy-dy*ux
        if abs(denominator)>1e-10:
            t = ((c[0]-a[0])*uy-(c[1]-a[1])*ux)/denominator
            u = ((c[0]-a[0])*dy-(c[1]-a[1])*dx)/denominator
            if -1e-8<=t<=1+1e-8 and -1e-8<=u<=1+1e-8:
                cuts.append(max(0.,min(1.,t)))
        elif abs(cross(a,b,c))<=1e-8:
            for point in (c,d):
                t = ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/square
                if 0<=t<=1: cuts.append(t)
    cuts = sorted(set(cuts))
    for low,high in zip(cuts,cuts[1:]):
        if high-low<=1e-8: continue
        checks += len(outline)
        if checks>max_checks: raise ValueError("segment containment budget")
        t = (low+high)/2.
        if not outline_contains_point((a[0]+t*dx,a[1]+t*dy),outline): return False
    return True


def outline_contains_path(points,outline,max_checks=100000):
    """Sweep native path/outline edge boxes; endpoints may lie on end cuts.

    A connected path with no other boundary contact stays in the component of
    its interior sample. This avoids a full ray test for every dense vertex.
    """
    if len(points)<2 or len(outline)<3: return False
    distinct=[points[0]]
    for point in points[1:]:
        if point[:2]!=distinct[-1][:2]: distinct.append(point)
    points=distinct
    if len(points)==1: return outline_contains_point(points[0],outline)
    boxes=[]
    for kind,edges in ((0,zip(points,points[1:])),(1,zip(outline,outline[1:]+outline[:1]))):
        for index,(a,b) in enumerate(edges):
            boxes.append((min(a[0],b[0]),max(a[0],b[0]),min(a[1],b[1]),max(a[1],b[1]),kind,index,a,b))
    active,checks=[],0
    for edge in sorted(boxes):
        active=[v for v in active if v[1]+1e-8>=edge[0]]
        for prior in active:
            if prior[4]==edge[4]: continue
            checks+=1
            if checks>max_checks: raise ValueError("path containment budget")
            if prior[3]+1e-8<edge[2] or edge[3]+1e-8<prior[2]: continue
            path,border=(prior,edge) if prior[4]==0 else (edge,prior)
            a,b,c,d=path[6],path[7],border[6],border[7]
            if not segments_intersect(a,b,c,d): continue
            dx,dy,ux,uy=b[0]-a[0],b[1]-a[1],d[0]-c[0],d[1]-c[1]
            denominator=dx*uy-dy*ux
            if abs(denominator)<=1e-10:
                # A positive-length overlap is not merely an end-cut contact.
                square=dx*dx+dy*dy
                low=max(0.,min(((c[0]-a[0])*dx+(c[1]-a[1])*dy)/square,((d[0]-a[0])*dx+(d[1]-a[1])*dy)/square))
                high=min(1.,max(((c[0]-a[0])*dx+(c[1]-a[1])*dy)/square,((d[0]-a[0])*dx+(d[1]-a[1])*dy)/square))
                if high-low>1e-8: return False
                t=(low+high)/2.
            else: t=((c[0]-a[0])*uy-(c[1]-a[1])*ux)/denominator
            contact=(a[0]+t*dx,a[1]+t*dy)
            if all(math.hypot(contact[0]-end[0],contact[1]-end[1])>1e-7 for end in (points[0],points[-1])): return False
        active.append(edge)
    a,b=points[(len(points)-2)//2],points[(len(points)-2)//2+1]
    return outline_contains_point(((a[0]+b[0])/2.,(a[1]+b[1])/2.),outline)


def strip_cells(left,right,check_step=None):
    """Oriented zipper mesh of a validated simple lane strip in linear work.

    Every source border vertex participates; interior edges cancel pairwise.
    Uniform triangle orientation plus the simple exterior ring proves that
    cells cover that ring once, without overlap or filling its concavities.
    A folded zipper is rejected rather than repaired with a convex hull.
    """
    outline=list(left)+list(reversed(right))
    if len(left)<2 or len(right)<2: raise ValueError("insufficient strip boundary")
    sign=1 if sum(cross(outline[0],a,b) for a,b in zip(outline,outline[1:]+outline[:1]))>0 else -1
    lengths=[]
    for boundary in (left,right):
        values=[0.]
        for a,b in zip(boundary,boundary[1:]):
            if check_step is not None: check_step()
            values.append(values[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
        if values[-1]<=1e-8: raise ValueError("degenerate strip boundary")
        lengths.append([v/values[-1] for v in values])
    i,j,cells=0,0,[]
    while i+1<len(left) or j+1<len(right):
        if check_step is not None: check_step()
        advance_left=j+1==len(right) or (i+1<len(left) and lengths[0][i+1]<=lengths[1][j+1])
        options=[advance_left,not advance_left]
        selected=None
        for option in options:
            if check_step is not None: check_step()
            if option and i+1<len(left): triangle=[left[i],left[i+1],right[j]]
            elif not option and j+1<len(right): triangle=[left[i],right[j+1],right[j]]
            else: continue
            if sign*cross(*triangle)>1e-10: selected=(option,triangle); break
        if selected is None: raise ValueError("strip mesh folds or degenerates")
        option,triangle=selected; cells.append(triangle)
        if option: i+=1
        else: j+=1
    return cells


def convex_cells(outline,max_checks=100000):
    """Exact nominal ear decomposition; never fill a concavity with its hull."""
    points = simple_outline(outline,max_checks)
    if len(points)>4096: raise ValueError("local triangulation vertex budget")
    # Collinear map sampling vertices add no XY area and can block an ear.
    changed,checks = True,0
    while changed and len(points)>3:
        checks += len(points)
        if checks>max_checks: raise ValueError("local triangulation budget")
        changed = False
        result = []
        for i,p in enumerate(points):
            a,b = points[i-1],points[(i+1)%len(points)]
            if abs(cross(a,p,b))<=1e-10 and min(a[0],b[0])-1e-10<=p[0]<=max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10<=p[1]<=max(a[1],b[1])+1e-10:
                changed = True
            else: result.append(p)
        points = result
    if len(points)<3: raise ValueError("degenerate local triangulation")
    sign = 1 if sum(cross(points[0],a,b) for a,b in zip(points,points[1:]+points[:1]))>0 else -1
    cells = []
    while len(points)>3:
        found = False
        for i,p in enumerate(points):
            checks += 1
            if checks>max_checks: raise ValueError("local triangulation budget")
            a,b = points[i-1],points[(i+1)%len(points)]
            if sign*cross(a,p,b)<=1e-10: continue
            triangle = [a,p,b]
            blocked = False
            for j,v in enumerate(points):
                if j in ((i-1)%len(points),i,(i+1)%len(points)): continue
                checks += 1
                if checks>max_checks: raise ValueError("local triangulation budget")
                if inside(v,triangle): blocked = True; break
            if blocked: continue
            cells.append(triangle); del points[i]; found = True; break
        if not found: raise ValueError("local triangulation unresolved")
    cells.append(points)
    return cells


def intersect_convex(left,right):
    """Clip two validated convex polygons. An empty intersection stays empty."""
    subject = list(left)
    orientation = 1 if sum(cross(right[0],a,b) for a,b in zip(right,right[1:]+right[:1]))>0 else -1
    for a,b in zip(right,right[1:]+right[:1]):
        if not subject: return []
        result = []
        previous = subject[-1]; previous_value = orientation*cross(a,b,previous)
        for current in subject:
            value = orientation*cross(a,b,current)
            if (value>=-1e-8)!=(previous_value>=-1e-8):
                ratio = previous_value/(previous_value-value)
                result.append(tuple(previous[i]+ratio*(current[i]-previous[i]) for i in range(2)))
            if value>=-1e-8: result.append(current)
            previous,previous_value = current,value
        subject = []
        for point in result:
            if not subject or math.hypot(point[0]-subject[-1][0],point[1]-subject[-1][1])>1e-8: subject.append(point)
        if len(subject)>1 and math.hypot(subject[0][0]-subject[-1][0],subject[0][1]-subject[-1][1])<=1e-8: subject.pop()
    if len(subject)<3 or abs(sum(cross(subject[0],a,b) for a,b in zip(subject,subject[1:]+subject[:1])))<=1e-8: return []
    return subject
