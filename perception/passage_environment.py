"""Source-bound junction corridors from the SDK-selected route and boundaries.

Static map geometry is never published as a clear exit or a right-of-way rule.
"""
import copy
import math
import time

from core.behavior_contract import require
from core.geometry import project_polyline, normalize_angle
from core.region_geometry import simple_outline, strip_cells


def read_passage_geometry(route_manager, ego, lane, junction, max_queries=12, deadline=None):
    """Fetch a bounded selected chain; keep actual cells instead of a hull."""
    from perception.route_manager import _vector_points, _clean_points
    result = copy.deepcopy(junction)
    result.update(corridor_geometry_verified=False, corridor_regions=[], conflict_polygons=[],
                  incoming_region=None,exit_region=[], exit_goal_pose=None, corridor_geometry_reason='PASSAGE_GEOMETRY_UNAVAILABLE')
    try:
        require(junction.get('entry_gate_verified') is True and junction.get('exit_gate_verified') is True
                and junction.get('selected_route_chain_verified') is True, 'PASSAGE_ROUTE_GATES_UNVERIFIED')
        identities = junction.get('selected_lane_ids')
        exit_id = junction.get('exit_lane_id')
        incoming_id=junction.get('source_lane_id') or lane.lane_id
        result['source_lane_id']=incoming_id
        require(isinstance(identities, list) and identities and isinstance(exit_id, str) and bool(exit_id)
                and incoming_id not in identities+[exit_id]
                and len(identities) + 2 <= max_queries, 'PASSAGE_CHAIN_BUDGET_OR_EXIT_UNKNOWN')
        map_reader = route_manager._maneuver_map
        require(map_reader.document is not None and map_reader.digest == junction.get('map_digest'),
                'PASSAGE_MAP_IDENTITY_MISMATCH')
        hdmap = route_manager.adapter.hdmap
        deadline = time.monotonic() + .005 if deadline is None else deadline
        checks = [0]
        def check():
            checks[0] += 1
            require(checks[0] <= 10000 and time.monotonic() < deadline,
                    'PASSAGE_MAP_QUERY_BUDGET_EXHAUSTED')
        native = []
        for identity in [incoming_id]+identities + [exit_id]:
            check()
            road, section, member = map_reader.document.lane(identity)
            require(member['type'] == 'driving' and member['permission_scope_verified'],
                    'PASSAGE_LANE_PERMISSION_UNKNOWN')
            sample = hdmap.getLaneSample(hdmap.pySimString(identity))
            require(getattr(sample, 'exists', False) is True, 'PASSAGE_LANE_SAMPLE_UNAVAILABLE')
            center, left, right = [tuple(_clean_points(_vector_points(v))) for v in
                (sample.laneInfo.centerLine, sample.laneInfo.leftBoundary, sample.laneInfo.rightBoundary)]
            require(all(2 <= len(v) <= 3000 for v in (center, left, right)), 'PASSAGE_LANE_SAMPLE_LIMIT')
            reference = (junction['entry_reference_point'] if identity in (incoming_id,identities[0]) else
                         junction['exit_reference_point'] if identity == exit_id else native[-1]['center_line'][-1])
            # SDK samples can be stored opposite to route travel. Compare the
            # selected verified join; never orient an outgoing lane from ego yaw.
            forward = math.hypot(center[0][0] - reference[0], center[0][1] - reference[1])
            backward = math.hypot(center[-1][0] - reference[0], center[-1][1] - reference[1])
            require(min(forward, backward) <= .3 and abs(forward - backward) > .1,
                    'PASSAGE_SAMPLE_JOIN_AMBIGUOUS')
            if (forward < backward if identity==incoming_id else backward < forward):
                center, left, right = tuple(reversed(center)), tuple(reversed(right)), tuple(reversed(left))
            require(all(len(v) >= 3 and abs(v[2] - ego.z) <= .1 for line in (center, left, right) for v in line),
                    'PASSAGE_SAMPLE_OTHER_HEIGHT')
            polygon = simple_outline([v[:2] for v in left] + [v[:2] for v in reversed(right)],
                                     max_checks=10000, check_step=check)
            cells = strip_cells([v[:2] for v in left], [v[:2] for v in right], check_step=check)
            if native:
                previous = native[-1]['center_line']
                require(math.hypot(previous[-1][0] - center[0][0], previous[-1][1] - center[0][1]) <= .3,
                        'PASSAGE_CHAIN_DISCONNECTED')
                a, b, c, d = previous[-2], previous[-1], center[0], center[1]
                require(abs(normalize_angle(math.atan2(b[1]-a[1], b[0]-a[0])
                    - math.atan2(d[1]-c[1], d[0]-c[0]))) < .35, 'PASSAGE_CHAIN_HEADING_MISMATCH')
            native.append(dict(lane_id=identity, center_line=center, polygon=polygon,
                               convex_cells=cells, left_boundary=left, right_boundary=right))
        require(time.monotonic() < deadline, 'PASSAGE_MAP_QUERY_BUDGET_EXHAUSTED')
        outgoing = native[-1]
        end = outgoing['center_line'][-1]
        before = outgoing['center_line'][-2]
        result.update(corridor_geometry_verified=True, corridor_regions=native[1:],incoming_region=native[0],
            conflict_polygons=[v['polygon'] for v in native[1:-1]], exit_region=outgoing['polygon'],
            exit_goal_pose=dict(x=end[0], y=end[1], body_heading_rad=math.atan2(end[1]-before[1], end[0]-before[0]),
                                reference_point='ego_rear_axle'), corridor_geometry_reason='SDK_SELECTED_CHAIN_BOUNDARIES')
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        result['corridor_geometry_reason'] = str(error)
    return result


def publish_passages(perception, environment, route_manager, coverage, query, evidence_factory):
    """Attach geometry and independently observed occupancy to existing records."""
    from perception.maneuver_environment import _covered, _touches, _plane_coverage
    updated = []
    deadline = time.monotonic() + .005
    mapper = getattr(route_manager,'_maneuver_map',None)
    sources = getattr(mapper,'passage_sources',{})
    digest = environment.map_semantics.get('digest')
    scope = (perception.case_id,perception.task_id,digest)
    originals = list(environment.junctions[:12])
    for original in originals:
        if (original.get('entry_gate_verified') is True and original.get('exit_gate_verified') is True
                and original.get('selected_route_chain_verified') is True):
            saved=copy.deepcopy(original)
            saved['source_lane_id']=perception.lane.lane_id
            sources[scope+(str(original.get('id')),)] = saved
    if len(sources)>16:
        for key in list(sources)[:-16]: sources.pop(key,None)
    if (environment.map_semantics.get('geometry_bound') is True and mapper is not None
            and mapper.digest==digest and mapper.document is not None):
        for key, saved in list(sources.items()):
            if key[:3]!=scope or perception.lane.lane_id not in saved.get('selected_lane_ids',[])+[saved.get('exit_lane_id')]:
                continue
            originals=[v for v in originals if str(v.get('id'))!=key[3]]
            originals.append(copy.deepcopy(saved))
    for original in originals[:12]:
        item = read_passage_geometry(route_manager, perception.ego, perception.lane, original, deadline=deadline)
        item['evidence'] = evidence_factory(environment, 'sdk_matched_opendrive', 'map', False)
        item['conflict_coverage_complete'] = False
        item['exit_coverage_verified'] = False
        item['exit_occupancy'] = 'unknown'
        if item['corridor_geometry_verified'] and environment.status['objects'].get('usable') is True:
            regions = [item['incoming_region']]+item['corridor_regions']
            visible = all(_covered(v['polygon'], _plane_coverage(v['center_line'], coverage), query)
                          for v in regions)
            item['conflict_coverage_complete'] = visible
            exit_lane = regions[-1]
            exit_visible = _covered(exit_lane['polygon'], _plane_coverage(exit_lane['center_line'], coverage), query)
            item['exit_coverage_verified'] = exit_visible
            if exit_visible:
                item['exit_occupancy'] = ('occupied' if any(_touches(o, cell) for o in environment.objects
                    for cell in exit_lane['convex_cells']) else 'clear')
            item['dynamic_evidence'] = evidence_factory(environment, perception.target_source, 'sensor', visible)
        updated.append(item)
    environment.junctions = updated
