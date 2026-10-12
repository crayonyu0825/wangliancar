"""Strict source-bound junction/crosswalk geometry, never crossing permission."""
import copy
import math
import re

from core.behavior_contract import require
from core.corridor_region import CorridorRegion
from core.maneuver_facts import ManeuverFacts
from core.region_geometry import (simple_outline, rectangle, convex_cells, intersects,
                                  convex_polygon, inside, segments_intersect, circle_intersects)
from core.validation import number
from core.geometry import project_polyline, projection_within_polyline


def _catalogue(perception, name, api):
    """A positive associated SDK observation is not a complete map catalogue."""
    status = perception.map_observation_status
    meta = status.get(name) if isinstance(status, dict) else None
    require(getattr(perception, name + '_valid') is True and isinstance(meta, dict)
            and meta.get('read_ok') is True and meta.get('source') == 'hdmap'
            and meta.get('clock') == 'static_map' and meta.get('api') == api
            and meta.get('version') == 'map-observations-v1'
            and meta.get('query_scope') == 'signal_associated_lanes'
            and type(meta.get('invalid_records')) is int and meta['invalid_records'] == 0
            and meta.get('errors') == [], 'CROSSWALK_SDK_SOURCE_UNAVAILABLE')
    values = getattr(perception, name)
    require(isinstance(values, list) and len(values) <= 4096,
            'CROSSWALK_SDK_CATALOGUE_INVALID')
    identities = [v.get('id') if isinstance(v, dict) else None for v in values]
    require(all(type(v) is int and v >= 0 for v in identities)
            and len(set(identities)) == len(identities), 'CROSSWALK_SDK_IDENTITY_INVALID')
    return values


def _native_outline(record, kind, ego_z):
    require(isinstance(record, dict) and record.get('valid') is True
            and record.get('source') == 'hdmap' and record.get('kind') == kind
            and all(number(record.get(v)) for v in ('x', 'y', 'z')),
            'CROSSWALK_SDK_RECORD_INVALID')
    for name in ('lane_ids', 'association_signal_ids'):
        values = record.get(name)
        require(isinstance(values, list) and 1 <= len(values) <= 4096
                and all((isinstance(v, str) and bool(v)) if name == 'lane_ids'
                        else (type(v) is int and v >= 0) for v in values)
                and len(set(values)) == len(values),
                'CROSSWALK_SDK_ASSOCIATION_INVALID')
    knots = record.get('boundary_knots')
    require(isinstance(knots, (list, tuple)) and (3 if kind == 'crosswalk' else 2) <= len(knots) <= 128
            and all(isinstance(v, (list, tuple)) and len(v) == 3
                    and all(number(x) for x in v) and abs(v[2] - ego_z) <= .1 for v in knots),
            'CROSSWALK_SDK_BOUNDARY_INVALID')
    return [tuple(v) for v in knots]


def _line_station(reference, boundary, check):
    """Unique actual line/reference intersection; no nearest-point snapping."""
    stations, along = [], 0.
    for a, b in zip(reference, reference[1:]):
        check()
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        require(length > 1e-6, 'CROSSWALK_REFERENCE_DEGENERATE')
        for c, d in zip(boundary, boundary[1:]):
            check()
            ux, uy = d[0] - c[0], d[1] - c[1]
            determinant = dx * uy - dy * ux
            if abs(determinant) <= 1e-9:
                # A line overlapping the reference has no unique stop station.
                require(not segments_intersect(a[:2], b[:2], c[:2], d[:2]),
                        'CROSSWALK_STOP_LINE_AMBIGUOUS')
                continue
            cx, cy = c[0] - a[0], c[1] - a[1]
            ratio = (cx * uy - cy * ux) / determinant
            other = (cx * dy - cy * dx) / determinant
            if -1e-8 <= ratio <= 1 + 1e-8 and -1e-8 <= other <= 1 + 1e-8:
                z = a[2] + ratio * (b[2] - a[2])
                line_z = c[2] + other * (d[2] - c[2])
                require(abs(z - line_z) <= .1, 'CROSSWALK_STOP_LINE_LEVEL_MISMATCH')
                station = along + max(0., min(1., ratio)) * length
                if not any(abs(station - old) <= .01 for old in stations):
                    stations.append(station)
        along += length
    require(len(stations) <= 1, 'CROSSWALK_STOP_LINE_AMBIGUOUS')
    return stations[0] if stations else None


def _at_station(reference, station):
    along = 0.
    for a, b in zip(reference, reference[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if station <= along + length + 1e-8:
            ratio = max(0., min(1., (station - along) / length))
            return tuple(a[i] + ratio * (b[i] - a[i]) for i in range(3)), math.atan2(b[1] - a[1], b[0] - a[0])
        along += length
    raise ValueError('CROSSWALK_GATE_OUTSIDE_REFERENCE')


def _exit_boundary(boundary, gate, heading, check):
    """Cut a source boundary once; retain its original downstream vertices."""
    axis = (math.cos(heading), math.sin(heading))
    values = [(p[0] - gate[0]) * axis[0] + (p[1] - gate[1]) * axis[1] for p in boundary]
    require(values[0] <= 1e-8 and values[-1] > 1e-8,
            'CROSSWALK_OBSERVED_EXIT_UNAVAILABLE')
    index = next((i for i, value in enumerate(values) if value > 1e-8), None)
    require(index is not None and index > 0 and all(v >= -1e-8 for v in values[index:]),
            'CROSSWALK_EXIT_BOUNDARY_AMBIGUOUS')
    a, b = boundary[index - 1], boundary[index]
    ratio = -values[index - 1] / (values[index] - values[index - 1])
    require(0 <= ratio <= 1 + 1e-8, 'CROSSWALK_EXIT_BOUNDARY_AMBIGUOUS')
    result = [tuple(a[i] + ratio * (b[i] - a[i]) for i in range(2))]
    for point in boundary[index:]:
        check()
        if math.hypot(point[0] - result[-1][0], point[1] - result[-1][1]) > 1e-8:
            result.append(tuple(point[:2]))
    require(len(result) >= 2, 'CROSSWALK_OBSERVED_EXIT_UNAVAILABLE')
    return result


def read_crosswalk_facts(perception, identifier, vehicle, clock):
    """Use existing SDK + matched-road + Sensor facts, not a new captain record.

    Only a selected, positively observed crosswalk is supported. The associated
    map query is not assumed exhaustive. The returned local view never supplies
    right of way, permitted time windows, or real execution-error tolerances.
    The downstream road cut is deliberately rejected if it crosses the gate
    more than once; no hull fills road holes or an ambiguous turning exit.
    """
    require(isinstance(identifier, str) and re.fullmatch(r'crosswalk:(0|[1-9][0-9]{0,18})', identifier),
            'CROSSWALK_IDENTIFIER_INVALID')
    native_id = int(identifier.split(':')[1])
    facts = ManeuverFacts(perception, clock)
    digest, objects = facts.map_digest(), facts.objects()
    check = lambda: facts.check_current(True)
    raw = _catalogue(perception, 'map_crosswalks', 'getCrosswalkList')
    matches = [v for v in raw if v['id'] == native_id]
    require(len(matches) == 1, 'CROSSWALK_SELECTED_SOURCE_UNAVAILABLE')
    record = copy.deepcopy(matches[0])
    knots = _native_outline(record, 'crosswalk', perception.ego.z)
    lane_id = perception.lane.lane_id
    require(lane_id in record['lane_ids'], 'CROSSWALK_CURRENT_ROAD_ASSOCIATION_UNKNOWN')
    polygon = convex_polygon([v[:2] for v in knots])
    published = facts.env.crossing_regions
    require(isinstance(published, list) and len(published) <= 4096, 'CROSSWALK_PUBLISHED_SET_INVALID')
    matches = [v for v in published if isinstance(v, dict) and v.get('id') == identifier]
    require(len(matches) == 1, 'CROSSWALK_PUBLISHED_SOURCE_UNAVAILABLE')
    item = copy.deepcopy(matches[0])
    require(item.get('kind') == 'crosswalk' and item.get('geometry_valid') is True
            and tuple(tuple(v) for v in item.get('polygon', [])) == tuple(polygon),
            'CROSSWALK_NATIVE_BOUNDARY_MISMATCH')
    evidence = facts._evidence(item.get('evidence'), 'verified_fusion', covered=True)
    require(item.get('coverage_verified') is True and item.get('occupancy') in ('clear', 'occupied'),
            'CROSSWALK_DYNAMIC_COVERAGE_UNKNOWN')
    road = facts.road(lane_id, require_coverage=True)
    geometry = knots + road['center_line'] + road['left_boundary'] + road['right_boundary']
    views = facts.coverage(geometry)
    require(bool(views) and facts._covered_region(polygon, views), 'CROSSWALK_VISIBILITY_UNKNOWN')
    reference = road['center_line']
    require(2 <= len(reference) <= 3000 and all(math.hypot(b[0]-a[0], b[1]-a[1]) > 1e-6
            for a,b in zip(reference, reference[1:])), 'CROSSWALK_REFERENCE_INVALID')
    require(any(inside(v[:2], polygon) for v in reference)
            or any(segments_intersect(a[:2], b[:2], c, d)
                   for a,b in zip(reference, reference[1:])
                   for c,d in zip(polygon, polygon[1:] + polygon[:1])),
            'CROSSWALK_REFERENCE_DOES_NOT_CROSS')
    current = project_polyline(reference, perception.ego.x, perception.ego.y)
    corners = [project_polyline(reference, *v) for v in polygon]
    require(projection_within_polyline(current, len(reference))
            and all(projection_within_polyline(v, len(reference)) for v in corners),
            'CROSSWALK_GATE_OUTSIDE_OBSERVED_REFERENCE')
    near, far = min(v['s'] for v in corners), max(v['s'] for v in corners)
    require(far - near > 1e-6, 'CROSSWALK_GATE_DEGENERATE')

    # Preserve the earlier real stopline even after the producer's ahead-only
    # association stops publishing it. Do not treat an unreadable catalogue as
    # confirmation that no mandatory stopline exists.
    stop, entry = None, near
    require(perception.map_stop_lines_valid is True, 'CROSSWALK_STOP_LINE_SOURCE_UNKNOWN')
    stops = _catalogue(perception, 'map_stop_lines', 'getStoplineList')
    candidates = []
    for candidate in stops:
        check()
        boundary = _native_outline(candidate, 'stop_line', perception.ego.z)
        if (lane_id not in candidate['lane_ids'] or not set(record['association_signal_ids']).intersection(
                candidate['association_signal_ids'])):
            continue
        station = _line_station(reference, boundary, check)
        if station is not None and station <= near + 1e-8:
            candidates.append((station, candidate))
    if candidates:
        entry, stop = max(candidates, key=lambda pair: pair[0])
        require(all(v[1]['boundary_knots'] == stop['boundary_knots'] for v in candidates
                    if abs(v[0] - entry) <= .01), 'CROSSWALK_STOP_LINE_AMBIGUOUS')
        stop = copy.deepcopy(stop)
    if item.get('entry_stop_line') is not None:
        require(item['entry_stop_line'] == stop, 'CROSSWALK_STOP_LINE_SOURCE_MISMATCH')

    # Occupancy is checked against every original target; non-pedestrians and
    # unknown yaw cannot disappear from either facts or downstream prediction.
    hits = []
    for obj in objects:
        check()
        touches = (circle_intersects(polygon, obj['x'], obj['y'], .5*math.hypot(obj['length'], obj['width']))
                   if obj['heading'] is None else intersects(polygon, rectangle(
                       obj['x'], obj['y'], obj['heading'], obj['length']/2, obj['length']/2, obj['width']/2)))
        if touches: hits.append(obj['id'])
    require(isinstance(item.get('object_ids'), list) and all(type(v) is int for v in item['object_ids'])
            and len(item['object_ids']) == len(set(item['object_ids'])) and sorted(item['object_ids']) == sorted(hits)
            and item['occupancy'] == ('occupied' if hits else 'clear'), 'CROSSWALK_OCCUPANCY_SOURCE_MISMATCH')
    predicted = item.get('predicted_object_ids')
    require(isinstance(predicted, list) and all(type(v) is int and v in [o['id'] for o in objects] for v in predicted)
            and len(predicted) == len(set(predicted)),
            'CROSSWALK_PREDICTED_OBJECT_SET_INVALID')
    gate, heading = _at_station(reference, far)
    left = _exit_boundary(road['left_boundary'], gate, heading, check)
    right = _exit_boundary(road['right_boundary'], gate, heading, check)
    exit_polygon = simple_outline(left + list(reversed(right)), max_checks=100000, check_step=check)
    require(all(project_polyline(reference, *v)['s'] >= far - 1e-6 for v in exit_polygon),
            'CROSSWALK_EXIT_REFERENCE_AMBIGUOUS')
    require(all(number(getattr(vehicle, name, None)) for name in
                ('front_offset_m', 'rear_offset_m', 'half_width_m'))
            and vehicle.front_offset_m > 0 and vehicle.rear_offset_m >= 0 and vehicle.half_width_m > 0,
            'CROSSWALK_VEHICLE_GEOMETRY_UNKNOWN')
    body = rectangle(perception.ego.x, perception.ego.y, perception.ego.heading,
                     vehicle.front_offset_m, vehicle.rear_offset_m, vehicle.half_width_m)
    body_progress = [project_polyline(reference, *v) for v in body]
    exit_progress = all(projection_within_polyline(v, len(reference)) and v['s'] > far + 1e-8
                        for v in body_progress)
    entry_point, unused_heading = _at_station(reference, entry)
    updated = copy.deepcopy(item)
    exit_hits=[]
    for obj in objects:
        check()
        if (circle_intersects(exit_polygon,obj['x'],obj['y'],.5*math.hypot(obj['length'],obj['width']))
                if obj['heading'] is None else intersects(exit_polygon,rectangle(obj['x'],obj['y'],
                    obj['heading'],obj['length']/2,obj['length']/2,obj['width']/2))):
            exit_hits.append(obj['id'])
    updated.update(source_lane_id=lane_id, map_digest=digest, entry_stop_line=stop,
        entry_kind='sdk_stop_line' if stop is not None else 'crosswalk_boundary',
        entry_reference_point=entry_point, exit_reference_point=gate,
        entry_distance_m=entry-current['s'], exit_distance_m=far-current['s'],
        corridor_geometry_verified=True, entry_gate_verified=True, exit_gate_verified=True,
        selected_route_chain_verified=True, conflict_coverage_complete=True, exit_coverage_verified=True,
        exit_occupancy='occupied' if exit_hits else 'clear',exit_object_ids=exit_hits)
    deadlines = [facts.env.valid_until, facts.env.dynamic_valid_until, evidence['valid_until_s'],
                 facts.env.map_semantics['evidence']['valid_until_s'],
                 road['geometry_evidence']['valid_until_s'], road['evidence']['valid_until_s']]
    deadlines.extend(v['valid_until_s'] for v in views)
    deadlines.extend(o['evidence']['valid_until_s'] for o in objects)
    check()
    require(record == next((v for v in perception.map_crosswalks if v.get('id') == native_id), None)
            and item == next((v for v in facts.env.crossing_regions if v.get('id') == identifier), None),
            'CROSSWALK_SOURCE_CHANGED_DURING_READ')
    return dict(kind='CROSSWALK', junction=updated, road=road, corridor=CorridorRegion([road['polygon']]),
                conflict_polygons=[polygon], exit_region=CorridorRegion([exit_polygon]),
                coverage=CorridorRegion([v['polygon'] for v in views]), objects=objects,
                already_inside=intersects(body, polygon), exit_progress_reached=exit_progress,
                source_valid_until_s=min(deadlines), map_digest=digest, frame_id=perception.frame_id)


def read_passage_facts(perception, identifier, vehicle, clock):
    if isinstance(identifier, str) and identifier.startswith('crosswalk:'):
        return read_crosswalk_facts(perception, identifier, vehicle, clock)
    facts=ManeuverFacts(perception,clock)
    digest=facts.map_digest()
    objects=facts.objects()
    matches=[v for v in facts.env.junctions if isinstance(v,dict) and str(v.get('id'))==identifier]
    require(len(matches)==1,'PASSAGE_SELECTED_JUNCTION_UNAVAILABLE')
    item=matches[0]
    require(item.get('map_digest')==digest and item.get('corridor_geometry_verified') is True
            and item.get('entry_gate_verified') is True and item.get('exit_gate_verified') is True
            and item.get('selected_route_chain_verified') is True,'PASSAGE_SELECTED_GEOMETRY_UNVERIFIED')
    facts._evidence(item.get('evidence'),'map')
    regions=item.get('corridor_regions')
    require(isinstance(regions,list) and 2<=len(regions)<=12
            and [v.get('lane_id') for v in regions]==item.get('selected_lane_ids',[])+[item.get('exit_lane_id')],
            'PASSAGE_CORRIDOR_CHAIN_MISMATCH')
    incoming=item.get('incoming_region')
    require(isinstance(incoming,dict) and incoming.get('lane_id')==item.get('source_lane_id'),
            'PASSAGE_INCOMING_SOURCE_BOUNDARIES_UNKNOWN')
    raw=[]; all_polygons=[]
    for region in [incoming]+regions:
        require(isinstance(region,dict),'PASSAGE_REGION_INVALID')
        for name in ('center_line','left_boundary','right_boundary'):
            line=region.get(name)
            require(isinstance(line,(list,tuple)) and 2<=len(line)<=3000
                    and all(len(v)==3 and all(number(x) for x in v)
                            and abs(v[2]-perception.ego.z)<=.1 for v in line),'PASSAGE_REGION_GEOMETRY_INVALID')
            raw.extend(line)
        outline=simple_outline([v[:2] for v in region['left_boundary']]
                               +[v[:2] for v in reversed(region['right_boundary'])])
        require(tuple(outline)==tuple(tuple(v) for v in region.get('polygon',[])),
                'PASSAGE_BOUNDARY_POLYGON_MISMATCH')
        all_polygons.append(outline)
    polygons=all_polygons[1:]
    require([tuple(tuple(p) for p in v) for v in item.get('conflict_polygons',[])]
            ==[tuple(v) for v in polygons[:-1]] and tuple(tuple(v) for v in item.get('exit_region',[]))
            ==tuple(polygons[-1]),'PASSAGE_CONFLICT_OR_EXIT_MISMATCH')
    road=facts.road(perception.lane.lane_id,require_coverage=False)
    raw.extend(road['center_line']+road['left_boundary']+road['right_boundary'])
    views=facts.coverage(raw)
    require(bool(views),'PASSAGE_VISIBILITY_UNKNOWN')
    dynamic=item.get('dynamic_evidence')
    facts._evidence(dynamic,'sensor',covered=True)
    require(item.get('conflict_coverage_complete') is True and item.get('exit_coverage_verified') is True,
            'PASSAGE_CONFLICT_OR_EXIT_VISIBILITY_UNKNOWN')
    body=rectangle(perception.ego.x,perception.ego.y,perception.ego.heading,
                   vehicle.front_offset_m,vehicle.rear_offset_m,vehicle.half_width_m)
    already_inside=any(intersects(body,cell) for polygon in polygons[:-1] for cell in convex_cells(polygon))
    # The static entry gate is retained across junction lanes. Its distance is
    # always projected from current measured GPS, never copied from an old frame.
    entry=item.get('entry_reference_point')
    require(isinstance(entry,(list,tuple)) and len(entry)>=2,'PASSAGE_ENTRY_REFERENCE_UNKNOWN')
    chain=list(road['center_line'])
    for region in regions:
        if region['lane_id']==perception.lane.lane_id:continue
        if math.hypot(chain[-1][0]-region['center_line'][0][0],chain[-1][1]-region['center_line'][0][1])<=.3:
            chain.extend(region['center_line'][1:])
    current=project_polyline(chain,perception.ego.x,perception.ego.y)
    gate=project_polyline(chain,entry[0],entry[1])
    updated=copy.deepcopy(item)
    updated['entry_distance_m']=gate['s']-current['s']
    deadlines=[facts.env.valid_until,facts.env.dynamic_valid_until,road['geometry_evidence']['valid_until_s'],
               item['evidence']['valid_until_s'],dynamic['valid_until_s']]+[v['valid_until_s'] for v in views]
    facts.check_current(True)
    return dict(junction=updated,corridor=CorridorRegion([road['polygon']]+all_polygons),
                conflict_polygons=polygons[:-1],exit_region=CorridorRegion([polygons[-1]]),
                coverage=CorridorRegion([v['polygon'] for v in views]),objects=objects,
                already_inside=already_inside,source_valid_until_s=min(deadlines),map_digest=digest,
                frame_id=perception.frame_id)
