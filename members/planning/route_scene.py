"""SDK-free selected-route scene; native source geometry is never extended."""
import copy
import math

from core.behavior_contract import require
from core.maneuver_facts import ManeuverFacts
from core.region_geometry import simple_outline
from core.route_segments import verified_spans
from core.validation import number
from members.planning.candidate_validation import CorridorRegion
from members.planning.maneuver_scene import read_maneuver_scene, ManeuverGeometryCache


def _reference_window(points, start, end):
    """Existing source edges only, preserving every internal XYZ vertex."""
    require(number(start) and number(end) and 0 <= start < end, 'ROUTE_SCENE_REFERENCE_WINDOW_INVALID')
    values, along = [], 0.
    for a,b in zip(points,points[1:]):
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        require(length > 1e-6, 'ROUTE_SCENE_REFERENCE_INVALID')
        low,high = max(start,along),min(end,along+length)
        if high > low+1e-8:
            for station in (low,high):
                ratio = (station-along)/length
                value = (tuple(a) if station==along else tuple(b) if station==along+length else
                         tuple(a[i]+ratio*(b[i]-a[i]) for i in range(3)))
                if not values or math.hypot(value[0]-values[-1][0],value[1]-values[-1][1]) > 1e-6:
                    values.append(value)
        along += length
    require(end <= along+1e-8 and len(values)>=2, 'ROUTE_SCENE_REFERENCE_WINDOW_OUTSIDE_SOURCE')
    return values


def read_route_scene(perception, clock=None, geometry_cache=None):
    facts = ManeuverFacts(perception, clock)
    digest, objects = facts.map_digest(), facts.objects()
    route = facts.env.map_semantics.get('route_geometry')
    if not isinstance(route, dict) or route.get('geometry_valid') is not True:
        require(not isinstance(route,dict) or route.get('current_source_invalid') is not True,
                'ROUTE_SCENE_CURRENT_NATIVE_SOURCE_INVALID')
        # An unknown continuation remains a finite, source-observed stop region.
        # No guessed predecessor repairs a rear body outside that current area.
        scene = read_maneuver_scene(perception, (perception.lane.lane_id,), clock, geometry_cache)
        scene.update(reference=scene['roads'][0]['center_line'], route_lane_ids=[perception.lane.lane_id],
                     route_status='current_only_protected:'+str(route.get('reason') if isinstance(route,dict) else 'not_published'))
        return scene
    lane = perception.lane
    spans = verified_spans(len(lane.center_line), len(lane.forward_reference), lane.forward_lane_ids, lane.forward_lane_spans)
    selected, prior = route.get('selected_lane_ids'), route.get('predecessor_lane_ids')
    require(lane.forward_reference_valid is True
            and list(map(tuple,lane.center_line)) == list(map(tuple,lane.forward_reference[:len(lane.center_line)]))
            and isinstance(selected,list) and 1 <= len(selected) <= 12 and spans
            and selected == list(lane.forward_lane_ids[:len(selected)]) and selected[0] == lane.lane_id
            and isinstance(prior,list) and len(prior) <= 2 and all(isinstance(v,str) and v for v in prior)
            and len(set(prior+selected)) == len(prior+selected), 'ROUTE_SCENE_SELECTION_MISMATCH')
    require(route.get('source_lane_id') == lane.lane_id and route.get('map_digest') == digest
            and route.get('selection_context') == [perception.case_id, perception.task_id, perception.scene_id]
            and route.get('selected_route_verified') is True and route.get('native_sample_rechecked') is True
            and route.get('sensor_source_verified') is True, 'ROUTE_SCENE_SOURCE_UNVERIFIED')
    facts._evidence(route.get('evidence'), 'map')
    facts._evidence(route.get('dynamic_evidence'), 'sensor')
    raw = route.get('roads')
    require(isinstance(raw,list) and 1 <= len(raw) <= 12
            and [v.get('lane_id') for v in raw if isinstance(v,dict)] == prior+selected,
            'ROUTE_SCENE_ROAD_CHAIN_MISMATCH')
    roads, geometry, deadlines = [], [], [facts.env.valid_until, facts.env.dynamic_valid_until,
        route['evidence']['valid_until_s'], route['dynamic_evidence']['valid_until_s']]
    for item in raw:
        facts.check_current(True)
        identity = item['lane_id']
        require(item.get('role') == ('predecessor' if identity in prior else 'current' if identity==lane.lane_id else 'forward')
                and item.get('source') == 'sdk_matched_opendrive' and item.get('map_digest') == digest
                and all(item.get(v) is True for v in ('geometry_valid','drivable_verified','native_sample_rechecked',
                                                       'travel_direction_verified','travel_matches_declared_direction'))
                and item.get('coordinate_system') == 'world_xyz_metres'
                and item.get('geometry_model') == 'sdk_piecewise_linear_lane_v1'
                and type(item.get('declared_road_s_direction')) is int and item['declared_road_s_direction'] in (-1,1),
                'ROUTE_SCENE_ROAD_SOURCE_INVALID')
        for name in ('center_line','left_boundary','right_boundary'):
            line = item.get(name)
            require(isinstance(line,(list,tuple)) and 2 <= len(line) <= 3000
                    and all(isinstance(v,(list,tuple)) and len(v)==3 and all(number(x) for x in v)
                            and abs(v[2]-perception.ego.z) <= .1 for v in line), 'ROUTE_SCENE_GEOMETRY_INVALID')
            geometry.extend(line)
        polygon = simple_outline([v[:2] for v in item['left_boundary']]
                                 +[v[:2] for v in reversed(item['right_boundary'])], check_step=lambda:facts.check_current(True))
        require(tuple(tuple(v) for v in item.get('polygon',[])) == tuple(polygon), 'ROUTE_SCENE_BOUNDARY_POLYGON_MISMATCH')
        samples = item.get('road_s_samples_m')
        require(isinstance(samples,list) and len(samples)==len(item['center_line']) and all(number(v) for v in samples)
                and all((b-a)*item['declared_road_s_direction']>1e-6 for a,b in zip(samples,samples[1:])),
                'ROUTE_SCENE_TRAVEL_DIRECTION_MISMATCH')
        facts._evidence(item.get('geometry_evidence'), 'map')
        facts._evidence(item.get('evidence'), 'verified_fusion')
        deadlines.extend(item[v]['valid_until_s'] for v in ('geometry_evidence','evidence'))
        roads.append(copy.deepcopy(item))
    joins = route.get('verified_joins')
    require(isinstance(joins,list) and len(joins)==len(roads)-1, 'ROUTE_SCENE_JOIN_SET_INVALID')
    # Published crops may omit distant source join endpoints. Original endpoints
    # are retained in this dedicated, fresh native/ODR join evidence.
    for before,after,join in zip(roads,roads[1:],joins):
        facts.check_current(True)
        require(isinstance(join,dict) and join.get('source_lane_id')==before['lane_id']
                and join.get('target_lane_id')==after['lane_id'] and join.get('native_link_rechecked') is True
                and join.get('odr_link_verified') is True, 'ROUTE_SCENE_JOIN_UNVERIFIED')
        a,b = join.get('source_point'),join.get('target_point')
        require(all(isinstance(v,(list,tuple)) and len(v)==3 and all(number(x) for x in v) for v in (a,b))
                and math.hypot(a[0]-b[0],a[1]-b[1]) <= .3 and abs(a[2]-b[2]) <= .1,
                'ROUTE_SCENE_JOIN_GEOMETRY_MISMATCH')
    reference = route.get('reference')
    require(isinstance(reference,(list,tuple)) and 2 <= len(reference) <= 20000
            and all(isinstance(v,(list,tuple)) and len(v)==3 and all(number(x) for x in v)
                    and abs(v[2]-perception.ego.z)<=.1 for v in reference), 'ROUTE_SCENE_REFERENCE_INVALID')
    expected = _reference_window(lane.forward_reference, route.get('reference_window_start_m'), route.get('reference_window_end_m'))
    require(list(map(tuple,reference)) == expected, 'ROUTE_SCENE_REFERENCE_SOURCE_MISMATCH')
    views = facts.coverage(geometry)
    require(bool(views), 'ROUTE_SCENE_CURRENT_VISIBILITY_UNAVAILABLE')
    deadlines.extend(v['valid_until_s'] for v in views)
    deadlines.extend(o['evidence']['valid_until_s'] for o in objects)
    require(number(route.get('source_observed_at_s')) and route['source_observed_at_s']==facts.env.dynamic_observed_at_s
            and number(route.get('source_valid_until_s')) and route['source_valid_until_s']<=min(deadlines),
            'ROUTE_SCENE_SOURCE_DEADLINE_MISMATCH')
    deadlines.append(route['source_valid_until_s'])
    polygons = [v['polygon'] for v in roads]
    if geometry_cache is None:
        corridor,coverage = CorridorRegion(polygons),CorridorRegion([v['polygon'] for v in views])
    else:
        require(isinstance(geometry_cache,ManeuverGeometryCache), 'ROUTE_SCENE_CACHE_INVALID')
        scope = (perception.case_id,perception.task_id,perception.scene_id,digest)
        corridor = geometry_cache.region(scope,('selected_route',tuple(prior+selected)),polygons)
        coverage = geometry_cache.region(scope,('selected_route_visibility',tuple(prior+selected)),[v['polygon'] for v in views])
    facts.check_current(True)
    return dict(roads=roads,corridor=corridor,coverage=coverage,coverage_records=views,objects=objects,
                crossings=[],legal_crossing_corridor=None,map_digest=digest,reference=[tuple(v) for v in reference],
                route_lane_ids=prior+selected,route_status=route['reason'],frame_id=perception.frame_id,
                source_observed_at_s=facts.env.dynamic_observed_at_s,source_valid_until_s=min(deadlines),
                authority='geometry_and_current_visibility_only')
