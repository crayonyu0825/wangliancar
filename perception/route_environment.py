"""Bounded actual selected-route boundaries and freshly rechecked incoming tail.

No neighbor becomes a successor. Native polygons are kept as separate source
areas, so a gap between two SDK roads is never filled by a join or a hull.
Static preparation may be cached; Sensor occupancy/coverage is always current.
"""
import math
import time
from collections import OrderedDict

from core.behavior_contract import require
from core.geometry import project_polyline, projection_within_polyline, normalize_angle
from core.region_geometry import simple_outline, outline_contains_path
from core.route_segments import verified_spans
from core.validation import number
from perception.crossing_ranges import points3, length3, window3
from perception.opendrive_semantics import lane_identity
from perception.road_regions import local_region


def _public_geometry(value):
    """Detach cached geometry into its JSON-native public representation."""
    if isinstance(value, dict):
        return {key: _public_geometry(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_geometry(item) for item in value]
    return value


class _State(object):
    def __init__(self):
        self.scope = None
        self.previous_ids = ()
        self.previous_native = {}
        self.geometry = OrderedDict()

    def reset(self, scope):
        self.scope = scope
        self.previous_ids = ()
        self.previous_native.clear()
        self.geometry.clear()


class _Work(object):
    def __init__(self, clock, observed_at_s, source_deadline, max_queries, max_checks, max_seconds):
        self.clock, self.source_deadline = clock, source_deadline
        self.observed_at_s = observed_at_s
        self.max_queries, self.max_checks = max_queries, max_checks
        # Python 3.6 on Windows can expose a 15.6 ms monotonic tick. The
        # performance counter is monotonic with sufficient resolution for this
        # independent work budget; source TTL still uses the supplied clock.
        self.started = time.perf_counter()
        self.deadline = self.started + max_seconds
        self.queries, self.checks = 0, 0

    def check(self):
        self.checks += 1
        require(self.checks <= self.max_checks and time.perf_counter() < self.deadline,
                'ROUTE_GEOMETRY_BUDGET_EXHAUSTED')
        now = self.clock()
        require(number(now) and number(self.observed_at_s)
                and self.observed_at_s <= now < self.source_deadline, 'ROUTE_SOURCE_EXPIRED')

    def query(self, function, *args):
        self.check()
        require(self.queries < self.max_queries, 'ROUTE_SDK_QUERY_BUDGET_EXHAUSTED')
        self.queries += 1
        value = function(*args)
        self.check()
        return value


def _line(raw, work):
    from perception.route_manager import _vector_points, _clean_points
    size = int(raw.Size()) if hasattr(raw, 'Size') else len(raw)
    require(2 <= size <= 3000, 'ROUTE_NATIVE_POINT_LIMIT')
    work.check()
    value = points3(_clean_points(_vector_points(raw)))
    require(2 <= len(value) <= 3000, 'ROUTE_NATIVE_GEOMETRY_INVALID')
    return value


def _native(route, identity, selected, allow_snapped_start, work):
    mapper, hdmap = route._maneuver_map, route.adapter.hdmap
    road, section, member = mapper.document.lane(identity)
    require(member['type'] == 'driving' and member['permission_scope_verified'] is True
            and road['rule'] in ('RHT', 'LHT'), 'ROUTE_ODR_TRAVEL_PERMISSION_UNKNOWN')
    native = hdmap.pySimString(identity)
    sample = work.query(hdmap.getLaneSample, native)
    require(getattr(sample, 'exists', False) is True, 'ROUTE_NATIVE_SAMPLE_UNAVAILABLE')
    center, left, right = [_line(getattr(sample.laneInfo, name), work)
                           for name in ('centerLine', 'leftBoundary', 'rightBoundary')]
    expected = tuple(tuple(v) for v in selected)
    forward = (center == expected or allow_snapped_start and center[1:] == expected[1:]
               and math.hypot(center[0][0]-expected[0][0], center[0][1]-expected[0][1]) <= .3)
    reversed_center = tuple(reversed(center))
    backward = (reversed_center == expected or allow_snapped_start and reversed_center[1:] == expected[1:]
                and math.hypot(reversed_center[0][0]-expected[0][0], reversed_center[0][1]-expected[0][1]) <= .3)
    require(bool(forward) != bool(backward), 'ROUTE_SELECTED_NATIVE_REFERENCE_MISMATCH')
    reversed_sample = bool(backward)
    if reversed_sample:
        center, left, right = reversed_center, tuple(reversed(right)), tuple(reversed(left))
    require(all(abs(v[2]-route._route_publish_ego_z) <= .1 for line in (center, left, right) for v in line),
            'ROUTE_NATIVE_HEIGHT_UNKNOWN')
    observed = work.query(hdmap.getLaneType, native)
    kind = getattr(observed, 'laneType', 'unknown')
    if hasattr(kind, 'GetString'): kind = kind.GetString()
    require(getattr(observed, 'exists', False) is True and str(kind).lower().split('.')[-1] == 'driving',
            'ROUTE_NATIVE_DRIVING_TYPE_UNKNOWN')
    road_native = hdmap.pySimString(road['id'])
    samples = {}
    def road_s(point):
        if point not in samples:
            value = work.query(hdmap.getRoadST, road_native, hdmap.pySimPoint3D(*point))
            require(getattr(value, 'exists', False) is True and number(getattr(value, 's', None))
                    and section['start_m']-1e-6 <= value.s <= section['end_m']+1e-6,
                    'ROUTE_NATIVE_ROAD_S_UNKNOWN')
            samples[point] = float(value.s)
        return samples[point]
    first, last = road_s(center[0]), road_s(center[-1])
    sign = (1 if lane_identity(identity)[2] < 0 else -1) * (1 if road['rule'] == 'RHT' else -1)
    require((last-first)*sign > 1e-6, 'ROUTE_NATIVE_TRAVEL_DIRECTION_MISMATCH')
    require(abs(first-(section['start_m'] if sign > 0 else section['end_m'])) <= .2
            and abs(last-(section['end_m'] if sign > 0 else section['start_m'])) <= .2,
            'ROUTE_NATIVE_SECTION_EXTENT_MISMATCH')
    link = work.query(hdmap.getLaneLink, native)
    field = 'predecessorLaneIds' if reversed_sample else 'successorLaneIds'
    vector = getattr(getattr(link, 'laneLink', None), field, None)
    require(getattr(link, 'exists', False) is True and vector is not None and 0 <= int(vector.Size()) <= 32,
            'ROUTE_NATIVE_LINK_UNKNOWN')
    links = []
    for i in range(int(vector.Size())):
        value = vector.GetElement(i)
        links.append(value.GetString() if hasattr(value, 'GetString') else str(value))
    return dict(lane_id=identity, center_line=center, left_boundary=left, right_boundary=right,
                native_reversed=reversed_sample, road=road, section=section, member=member,
                declared_road_s_direction=sign, forward_links=links, road_s=road_s)


def _join(source, target, document, work):
    work.check()
    require(target['lane_id'] in source['forward_links'], 'ROUTE_SELECTED_NATIVE_LINK_MISMATCH')
    a, b = source['center_line'][-2:]
    c, d = target['center_line'][:2]
    require(math.hypot(b[0]-c[0], b[1]-c[1]) <= .3 and abs(b[2]-c[2]) <= .1
            and abs(normalize_angle(math.atan2(b[1]-a[1], b[0]-a[0])
                -math.atan2(d[1]-c[1], d[0]-c[0]))) < .35, 'ROUTE_NATIVE_JOIN_DISCONNECTED')
    before, after = lane_identity(source['lane_id']), lane_identity(target['lane_id'])
    direction = 'successor' if source['declared_road_s_direction'] > 0 else 'predecessor'
    lane_link = source['member']['links'][direction]
    if before[0] == after[0]:
        require(after[1] == before[1]+source['declared_road_s_direction'] and lane_link == after[2],
                'ROUTE_ODR_SECTION_LINK_MISMATCH')
    else:
        link = source['road'][direction]
        require(isinstance(link, dict), 'ROUTE_ODR_ROAD_LINK_UNKNOWN')
        if link['element_type'] == 'road':
            require(link['element_id'] == after[0] and lane_link == after[2]
                    and link['contact_point'] == ('start' if target['declared_road_s_direction'] > 0 else 'end'),
                    'ROUTE_ODR_ROAD_LINK_MISMATCH')
        else:
            junction = document.junctions.get(link['element_id'])
            require(link['element_type'] == 'junction' and isinstance(junction, dict)
                    and target['road']['junction_id'] == link['element_id'], 'ROUTE_ODR_JUNCTION_UNKNOWN')
            selected = [v for v in junction['connections'] if v['incoming_road_id'] == before[0]
                        and v['connecting_road_id'] == after[0]
                        and v['contact_point'] == ('start' if target['declared_road_s_direction'] > 0 else 'end')
                        and any(p['from_lane_id'] == before[2] and p['to_lane_id'] == after[2]
                                for p in v['lane_links'])]
            require(len(selected) == 1, 'ROUTE_ODR_JUNCTION_SELECTION_MISMATCH')
    return dict(source_lane_id=source['lane_id'], target_lane_id=target['lane_id'],
                source_point=list(b), target_point=list(c), native_link_rechecked=True, odr_link_verified=True)


def publish_route(perception, environment, route_manager, coverage, query, evidence_factory,
                  clock=None, max_queries=96, max_checks=10000, max_seconds=.005):
    """Attach only a bounded observed prefix, preserving a selected incoming tail.

    Budgets bound calls/work and reject after overdue native calls; they cannot
    preempt a blocked SDK call and are not a calibrated realtime guarantee.
    """
    clock = clock or time.monotonic
    require(type(max_queries) is int and 1 <= max_queries <= 4096
            and type(max_checks) is int and 1 <= max_checks <= 1000000
            and number(max_seconds) and 0 < max_seconds <= 1., 'ROUTE_PUBLISH_BUDGET_INVALID')
    result = dict(geometry_valid=False, selected_route_verified=False, native_sample_rechecked=False,
                  sensor_source_verified=False, coverage_verified=False, roads=[], reference=[],
                  selected_lane_ids=[], predecessor_lane_ids=[], verified_joins=[],
                  reason='ROUTE_SELECTED_SOURCE_UNAVAILABLE', query_count=0, check_count=0,
                  current_source_invalid=False)
    environment.map_semantics['route_geometry'] = result
    work = _Work(clock, environment.observed_at_s, min(environment.valid_until, environment.dynamic_valid_until),
                 max_queries, max_checks, max_seconds)
    try:
        work.check()
        mapper = route_manager._maneuver_map
        digest = environment.map_semantics.get('digest')
        require(environment.map_semantics.get('geometry_bound') is True
                and environment.map_semantics.get('semantic_verified') is True
                and mapper.document is not None and mapper.digest == digest,
                'ROUTE_CURRENT_MAP_UNVERIFIED')
        current_map = work.query(mapper._read)
        require(current_map.get('verified') is True and current_map.get('semantic_verified') is True
                and current_map.get('digest') == digest, 'ROUTE_CURRENT_MAP_CHANGED')
        lane = perception.lane
        reference = points3(lane.forward_reference)
        spans = verified_spans(len(lane.center_line), len(reference), lane.forward_lane_ids, lane.forward_lane_spans)
        require(lane.forward_reference_valid is True and spans and lane.forward_lane_ids[0] == lane.lane_id
                and points3(lane.center_line) == reference[:len(lane.center_line)], 'ROUTE_SELECTED_SPANS_INVALID')
        position = project_polyline(reference, perception.ego.x, perception.ego.y)
        require(projection_within_polyline(position, len(reference)), 'ROUTE_GPS_OUTSIDE_REFERENCE')
        state = getattr(route_manager, '_route_geometry_state', None)
        if not isinstance(state, _State):
            state = _State()
            route_manager._route_geometry_state = state
        scope = (perception.case_id, perception.task_id, perception.scene_id, digest,
                 id(route_manager.adapter.hdmap), getattr(route_manager, '_route_hint', ()))
        if state.scope != scope: state.reset(scope)
        prior = []
        if lane.lane_id in state.previous_ids:
            index = state.previous_ids.index(lane.lane_id)
            prior = list(state.previous_ids[max(0, index-2):index])
        route_manager._route_publish_ego_z = perception.ego.z
        # Only lanes intersecting the bounded ahead window are fetched. A lane
        # count bound shortens usable source extent rather than selecting a road.
        bucket = math.floor(position['s']/20.)*20.
        low, high = bucket-25., bucket+75.
        if low >= 0:
            prior = []
        else:
            retained, behind = [], 0.
            for identity in reversed(prior):
                start = behind-length3(state.previous_native[identity]['center_line'])
                if behind > low+1e-6: retained.append(identity)
                behind = start
            prior = list(reversed(retained))
        distances = [0.]
        for a,b in zip(reference, reference[1:]):
            work.check()
            distances.append(distances[-1]+math.hypot(b[0]-a[0], b[1]-a[1]))
        chosen = []
        for identity in lane.forward_lane_ids:
            if len(chosen)+len(prior) >= 12: break
            start, end = spans[identity]
            if distances[start] >= high and chosen: break
            chosen.append(identity)
        models = []
        for identity in prior:
            result['failure_lane_id'] = identity
            saved = state.previous_native[identity]
            model = _native(route_manager, identity, saved['center_line'], False, work)
            require(all(model[k] == saved[k] for k in ('center_line', 'left_boundary', 'right_boundary')),
                    'ROUTE_ORIGINAL_INCOMING_SOURCE_CHANGED')
            models.append(model)
        for identity in chosen:
            result['failure_lane_id'] = identity
            start, end = spans[identity]
            models.append(_native(route_manager, identity, reference[start:end+1], identity != lane.lane_id, work))
        joins = [_join(a,b,mapper.document,work) for a,b in zip(models, models[1:])]
        offsets, at = {}, 0.
        for model in reversed(models[:len(prior)]):
            at -= length3(model['center_line'])
            offsets[model['lane_id']] = at
        for identity in chosen:
            offsets[identity] = distances[spans[identity][0]]
        roads = []
        for model in models:
            identity = model['lane_id']
            offset = offsets[identity]
            start, end = max(0., low-offset), min(length3(model['center_line']), high-offset)
            if end <= start+1e-6: continue
            local_reference = window3(model['center_line'], start, end)
            key = (digest, identity, model['center_line'], model['left_boundary'], model['right_boundary'],
                   tuple(local_reference))
            if key not in state.geometry:
                geometry = local_region(model['center_line'], model['left_boundary'], model['right_boundary'], local_reference)
                samples = [model['road_s'](tuple(v)) for v in geometry['center_line']]
                require(all((b-a)*model['declared_road_s_direction'] > 1e-6 for a,b in zip(samples,samples[1:])),
                        'ROUTE_NATIVE_ROAD_S_NONMONOTONIC')
                geometry['road_s_samples_m'] = samples
                state.geometry[key] = geometry
            state.geometry.move_to_end(key)
            while len(state.geometry) > 16: state.geometry.popitem(last=False)
            value = _public_geometry(state.geometry[key])
            value.update(lane_id=identity, role=('predecessor' if identity in prior else 'current' if identity==lane.lane_id else 'forward'),
                source='sdk_matched_opendrive', map_digest=digest, geometry_valid=True, drivable_verified=True,
                native_sample_rechecked=True, travel_direction_verified=True, travel_matches_declared_direction=True,
                declared_road_s_direction=model['declared_road_s_direction'],
                geometry_evidence=evidence_factory(environment, 'sdk_matched_opendrive', 'map', False))
            # Full road observation is separately reported. Consumers still
            # check the entire chosen trajectory against the original view union.
            same_plane = [v for v in coverage if number(v.get('reference_z_m'))
                          and all(abs(p[2]-v['reference_z_m']) <= .1 for name in
                                  ('center_line','left_boundary','right_boundary') for p in value[name])]
            covered = (query is not None and query.contains(value['polygon'], same_plane,
                       [v[:2] for v in value['left_boundary']], [v[:2] for v in value['right_boundary']]))
            value['coverage_verified'] = bool(covered)
            value['evidence'] = evidence_factory(environment, perception.target_source, 'verified_fusion', bool(covered))
            roads.append(value)
            work.check()
        require(any(v['role']=='current' for v in roads), 'ROUTE_CURRENT_REGION_UNAVAILABLE')
        last = chosen[-1]
        source_end = min(high, distances[spans[last][1]])
        published_reference = _public_geometry(window3(reference, max(0.,low), source_end))
        require(outline_contains_path(published_reference, simple_outline(
                [v[:2] for v in roads[0]['left_boundary']]+[v[:2] for v in reversed(roads[0]['right_boundary'])]))
                if len(roads)==1 else True, 'ROUTE_REFERENCE_OUTSIDE_SOURCE')
        work.check()
        source = environment.status.get('objects', {})
        require(source.get('usable') is True and source.get('source') == perception.target_source
                and isinstance(perception.target_source, str) and perception.target_source.startswith('sensor:'),
                'ROUTE_CURRENT_SENSOR_SOURCE_UNAVAILABLE')
        current_map = work.query(mapper._read)
        require(current_map.get('verified') is True and current_map.get('semantic_verified') is True
                and current_map.get('digest') == digest, 'ROUTE_CURRENT_MAP_CHANGED')
        verified_coverage = all(v['coverage_verified'] for v in roads)
        result.update(geometry_valid=True, selected_route_verified=True, native_sample_rechecked=True,
            sensor_source_verified=True, coverage_verified=verified_coverage, roads=roads,
            reference=published_reference, selected_lane_ids=chosen, predecessor_lane_ids=prior,
            verified_joins=joins, source_lane_id=lane.lane_id, map_digest=digest,
            selection_context=[perception.case_id, perception.task_id, perception.scene_id],
            evidence=evidence_factory(environment, 'sdk_matched_opendrive', 'map', False),
            dynamic_evidence=evidence_factory(environment, perception.target_source, 'sensor', verified_coverage),
            source_observed_at_s=environment.dynamic_observed_at_s,
            source_valid_until_s=min(environment.valid_until, environment.dynamic_valid_until),
            reason='SDK_SELECTED_ROUTE_BOUNDARIES', source_end_status=lane.forward_reference_status,
            reference_window_start_m=max(0.,low), reference_window_end_m=source_end,
            query_count=work.queries, check_count=work.checks)
        # Cache only static selected native geometry after a complete successful
        # publication. Fresh evidence is attached above, never stored here.
        state.previous_ids = tuple(prior+chosen)
        state.previous_native = {v['lane_id']:{k:v[k] for k in ('center_line','left_boundary','right_boundary')}
                                 for v in models}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RuntimeError) as error:
        reason = str(error)
        invalid_current = (reason == 'ROUTE_CURRENT_MAP_CHANGED' or result.get('failure_lane_id') == perception.lane.lane_id
                           and reason.startswith(('ROUTE_NATIVE_', 'ROUTE_SELECTED_NATIVE_REFERENCE_MISMATCH')))
        result.update(reason=reason, query_count=work.queries, check_count=work.checks,
                      current_source_invalid=bool(invalid_current))
    result['elapsed_s'] = time.perf_counter()-work.started
    return result
