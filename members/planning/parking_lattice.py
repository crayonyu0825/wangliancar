"""Bounded rear-axle pose search; every accepted path is re-timed and swept.

The lattice bins continuous bicycle poses only to bound duplicate work. It is
not complete and a failed/budgeted search is never evidence of no possible path.
Multiple directions produce stopped segments for a coordinator to authorize;
they never become a mixed-direction public Trajectory. No SDK is called here.
"""
import heapq
import math
import time

from core.geometry import normalize_angle
from core.interfaces import TrajectoryPoint
from core.validation import number
from members.planning.lane_change_generator import (
    _Failure, _Work, _require, _motion_parameters, _shape, _timed,
)
from members.planning.candidate_validation import ObstaclePrediction


class ParkingLatticeSearch(object):
    """Explicit numerical/search choices, separate from physical calibration."""
    def __init__(self, step_lengths_m, curvature_fractions, curvature_change_per_m,
                 xy_resolution_m, heading_resolution_rad, connector_distance_m,
                 max_nodes, max_expansions, max_path_length_m, max_direction_changes,
                 reverse_cost, direction_change_cost, permitted_directions,
                 direction_change_stop_duration_s,heuristic_weight=1.):
        self.step_lengths_m = step_lengths_m
        self.curvature_fractions = curvature_fractions
        self.curvature_change_per_m = curvature_change_per_m
        self.xy_resolution_m = xy_resolution_m
        self.heading_resolution_rad = heading_resolution_rad
        self.connector_distance_m = connector_distance_m
        self.max_nodes = max_nodes
        self.max_expansions = max_expansions
        self.max_path_length_m = max_path_length_m
        self.max_direction_changes = max_direction_changes
        self.reverse_cost = reverse_cost
        self.direction_change_cost = direction_change_cost
        self.permitted_directions = permitted_directions
        self.direction_change_stop_duration_s = direction_change_stop_duration_s
        self.heuristic_weight=heuristic_weight


def validate_lattice_search(search):
    _require(isinstance(search, ParkingLatticeSearch), 'PARKING_LATTICE_SEARCH_INVALID')
    for name in ('step_lengths_m', 'curvature_fractions'):
        values = getattr(search, name)
        _require(isinstance(values, (tuple, list)) and 1 <= len(values) <= 9
                 and all(number(v) for v in values) and len(set(values)) == len(values),
                 'PARKING_LATTICE_CHOICES_INVALID')
    _require(all(v > 0 for v in search.step_lengths_m)
             and all(abs(v) <= 1 for v in search.curvature_fractions),
             'PARKING_LATTICE_CHOICES_INVALID')
    for name in ('curvature_change_per_m', 'xy_resolution_m', 'heading_resolution_rad',
                 'connector_distance_m', 'max_path_length_m', 'reverse_cost',
                 'direction_change_cost', 'direction_change_stop_duration_s','heuristic_weight'):
        _require(number(getattr(search, name)) and getattr(search, name) > 0,
                 'PARKING_LATTICE_PARAMETER_INVALID')
    _require(search.heading_resolution_rad <= math.pi and search.reverse_cost >= 1
             and 1<=search.heuristic_weight<=10,
             'PARKING_LATTICE_PARAMETER_INVALID')
    _require(type(search.max_nodes) is int and 2 <= search.max_nodes <= 100000
             and type(search.max_expansions) is int and 1 <= search.max_expansions <= 100000
             and type(search.max_direction_changes) is int and 0 <= search.max_direction_changes <= 8,
             'PARKING_LATTICE_COUNT_INVALID')
    _require(isinstance(search.permitted_directions, (tuple, list))
             and 1 <= len(search.permitted_directions) <= 2
             and len(set(search.permitted_directions)) == len(search.permitted_directions)
             and all(type(v) is int and v in (-1, 1) for v in search.permitted_directions),
             'PARKING_LATTICE_DIRECTION_INVALID')
    return search


def _primitive(pose, curvature, target, direction, distance, search, spacing, work):
    """Integrate a curvature ramp as short exact constant-curvature arcs."""
    count = max(2, int(math.ceil(distance / spacing)))
    x, y, yaw = pose
    shape = [tuple(pose)]
    step = distance / count
    for unused in range(count):
        work.step()
        delta = search.curvature_change_per_m * step
        following = max(curvature-delta, min(curvature+delta, target))
        middle = (curvature+following)*.5
        end_yaw = yaw+direction*middle*step
        if abs(middle) < 1e-12:
            x += direction*step*math.cos(yaw)
            y += direction*step*math.sin(yaw)
        else:
            x += (math.sin(end_yaw)-math.sin(yaw))/middle
            y += (math.cos(yaw)-math.cos(end_yaw))/middle
        yaw, curvature = normalize_angle(end_yaw), following
        shape.append((x, y, yaw))
    return shape, curvature


def _lengths(shape):
    return [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(shape, shape[1:])]


def _key(pose, curvature, direction, changes, start, search, maximum):
    dx, dy = pose[0]-start[0], pose[1]-start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    # Local bins preserve equivalent translated/rotated numerical problems.
    return (int(round((c*dx+s*dy)/search.xy_resolution_m)),
            int(round((-s*dx+c*dy)/search.xy_resolution_m)),
            int(round(normalize_angle(pose[2]-start[2])/search.heading_resolution_rad)),
            min(range(len(search.curvature_fractions)),key=lambda i:abs(curvature/maximum-search.curvature_fractions[i])),
            direction, changes)


def _path_chunks(index, nodes, connector, direction):
    ending = nodes[index]
    edges = [(direction, ending['curvature'], connector)]
    node = ending
    while node['parent'] is not None:
        edges.append((node['direction'], node['start_curvature'], node['edge']))
        node = nodes[node['parent']]
    edges.reverse()
    chunks = []
    for travel, curvature, shape in edges:
        if chunks and chunks[-1]['motion_direction'] == travel:
            chunks[-1]['shape'].extend(shape[1:])
        else:
            chunks.append(dict(motion_direction=travel, initial_curvature=curvature,
                               shape=list(shape)))
    return chunks


def _shift_predictions(obstacles, offset):
    """Retain original finite horizon and growing uncertainty at segment t=0."""
    result = []
    for obj in obstacles:
        _require(number(obj.vx) and number(obj.vy) and number(obj.valid_until_s)
                 and obj.valid_until_s > offset, 'PREDICTION_HORIZON')
        result.append(ObstaclePrediction(obj.identity, obj.x+obj.vx*offset, obj.y+obj.vy*offset,
            obj.length, obj.width, obj.heading, obj.vx, obj.vy, obj.valid_until_s-offset,
            obj.position_uncertainty_m+obj.velocity_uncertainty_mps*offset, obj.velocity_uncertainty_mps))
    return result


def _certify(chunks, initial_speed, cap, vehicle, limits, access, coverage,
             goal_region, obstacles, parking_search, lattice, work):
    """Common absolute nominal clock includes explicit stopped gear dwell."""
    if sum(len(c['shape']) for c in chunks) > parking_search.max_points:
        return None
    from members.planning.parking_generator import _timed_parking
    for speed_scale in parking_search.speed_scales:
        segments, elapsed, failed = [], 0., False
        try:
            for i, chunk in enumerate(chunks):
                shape = chunk['shape']; lengths = _lengths(shape)
                points,timing = _timed_parking(shape, lengths, initial_speed if i == 0 else 0.,
                    cap*speed_scale, cap, chunk['initial_curvature'], vehicle, limits,
                    parking_search,work,chunk['motion_direction'],access,coverage,_shift_predictions(obstacles,elapsed))
                if i:
                    previous = segments[-1]['points'][-1]
                    hold_offset=segments[-1]['nominal_start_time_s']+previous.relative_time
                    hold = [TrajectoryPoint(previous.x, previous.y, 0., previous.heading, 0.),
                            TrajectoryPoint(previous.x, previous.y, 0., previous.heading,
                                            lattice.direction_change_stop_duration_s)]
                    for region, objects in ((access, _shift_predictions(obstacles,hold_offset)), (coverage, [])):
                        result = work.validate(hold, vehicle, limits, region, objects,
                                               chunk['motion_direction'],parking_search.execution_envelope)
                        if result['status'] != 'safe':
                            if result['reason_code'] == 'BUDGET_EXHAUSTED':
                                raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')
                            failed = True; break
                    if failed: break
                evidence = []
                for region, objects in ((access, _shift_predictions(obstacles,elapsed)), (coverage, [])):
                    result = work.validate(points, vehicle, limits, region, objects,
                                           chunk['motion_direction'],parking_search.execution_envelope)
                    evidence.append(result)
                    if result['status'] != 'safe':
                        if result['reason_code'] == 'BUDGET_EXHAUSTED':
                            raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')
                        failed = True; break
                if failed: break
                segments.append(dict(motion_direction=chunk['motion_direction'], points=points,
                    stop_distance=sum(lengths), initial_curvature_m_inv=chunk['initial_curvature'],
                    nominal_start_time_s=elapsed,
                    execution_robust=timing.get('execution_robust',False),
                    access_report=evidence[0], visibility_report=evidence[1],
                    direction_change_stop_duration_s=lattice.direction_change_stop_duration_s if i else 0.))
                elapsed += points[-1].relative_time+lattice.direction_change_stop_duration_s
            if failed: continue
            end = segments[-1]['points'][-1]
            probe = [TrajectoryPoint(end.x, end.y, 0., end.heading, t) for t in (0., 1.)]
            target = work.validate(probe, vehicle, limits, goal_region, [], chunks[-1]['motion_direction'],
                                   parking_search.execution_envelope)
            if target['status'] == 'safe':
                return dict(segments=segments, goal_pose_report=target,
                            execution_robust=all(s['execution_robust'] for s in segments),
                            nominal_duration_s=segments[-1]['nominal_start_time_s']+end.relative_time)
            if target['reason_code'] == 'BUDGET_EXHAUSTED':
                raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')
        except _Failure as error:
            if error.code == 'BUDGET_EXHAUSTED': raise
    return None


def _search_with_work(start, goal, initial_speed, initial_curvature, initial_direction,
                      cap, vehicle, limits, access, coverage, goal_region, obstacles,
                      parking_search, lattice, work, authorized_directions=None):
    validate_lattice_search(lattice)
    directions = tuple(lattice.permitted_directions if authorized_directions is None else authorized_directions)
    _require(initial_direction in directions and all(v in lattice.permitted_directions for v in directions),
             'PARKING_LATTICE_AUTHORIZATION_MISMATCH')
    _shift_predictions(obstacles,0.)  # Unknown velocity/horizon cannot enter search as static.
    probe=[TrajectoryPoint(start[0],start[1],0.,start[2],t) for t in (0.,1e-6)]
    start_reports=[]
    for region,objects in ((access,obstacles),(coverage,[])):
        checked=work.validate(probe,vehicle,limits,region,objects,initial_direction,parking_search.execution_envelope)
        start_reports.append(checked)
        if checked['status']!='safe':
            return dict(status=checked['status'],reason_code=checked['reason_code'],points=[],segments=[],
                needs_direction_authorization=False,authority='nominal_candidate_only',start_report=checked)
    maximum = math.tan(limits.max_front_steer_rad)/vehicle.wheelbase_m
    static = [o for o in obstacles if o.vx == 0. and o.vy == 0. and o.velocity_uncertainty_mps == 0.]
    root = dict(pose=tuple(start), curvature=initial_curvature, direction=initial_direction,
                changes=0, parent=None, distance=0., cost=0.)
    nodes, queue, best, serial = [root], [], {}, 0
    def heuristic(pose):
        return max(math.hypot(goal[0]-pose[0], goal[1]-pose[1]),
                   abs(normalize_angle(goal[2]-pose[2]))/maximum)
    heapq.heappush(queue, (lattice.heuristic_weight*heuristic(start), serial, 0))
    best[_key(start, initial_curvature, initial_direction, 0, start, lattice, maximum)] = 0.
    report = dict(status='inconclusive', reason_code='PARKING_LATTICE_EXHAUSTED', points=[],
                  segments=[], expanded=0, nodes=1, needs_direction_authorization=False,
                  authority='nominal_candidate_only', algorithm='bounded_hybrid_a_star')
    if tuple(start[:2])==tuple(goal[:2]) and abs(normalize_angle(start[2]-goal[2]))<1e-9:
        _require(initial_speed==0.,'MOVING_START_CANNOT_BECOME_STATIONARY_GOAL')
        checked=work.validate(probe,vehicle,limits,goal_region,[],initial_direction,parking_search.execution_envelope)
        if checked['status']=='safe':
            report.update(status='safe',reason_code='PARKING_LATTICE_STATIONARY_GOAL',points=probe,
                goal_pose_report=checked,nominal_duration_s=1e-6,
                execution_robust=parking_search.execution_envelope is not None,
                segments=[dict(motion_direction=initial_direction,points=probe,stop_distance=0.,
                    nominal_start_time_s=0.,initial_curvature_m_inv=initial_curvature,
                    execution_robust=parking_search.execution_envelope is not None,
                    direction_change_stop_duration_s=0.,access_report=start_reports[0],visibility_report=start_reports[1])])
            return report
        report.update(status=checked['status'],reason_code=checked['reason_code']); return report
    while queue:
        work.step()
        if report['expanded'] >= lattice.max_expansions:
            report['reason_code'] = 'PARKING_LATTICE_EXPANSION_LIMIT'; return report
        unused, unused, index = heapq.heappop(queue); node = nodes[index]
        key = _key(node['pose'], node['curvature'], node['direction'], node['changes'], start, lattice, maximum)
        if node['cost'] > best.get(key, float('inf'))+1e-9: continue
        report['expanded'] += 1
        distance_to_goal = math.hypot(goal[0]-node['pose'][0], goal[1]-node['pose'][1])
        if 1e-8 < distance_to_goal <= lattice.connector_distance_m:
            for tangent in parking_search.tangent_scales:
                for end_tangent in parking_search.end_tangent_scales:
                    work.step()
                    try:
                        connector, lengths = _shape(node['pose'], goal, node['curvature'], tangent,
                            parking_search, work, node['direction'], end_tangent)
                        if node['distance']+sum(lengths) > lattice.max_path_length_m: continue
                        chunks = _path_chunks(index, nodes, connector, node['direction'])
                        result = _certify(chunks, initial_speed, cap, vehicle, limits, access, coverage,
                            goal_region, obstacles, parking_search, lattice, work)
                        if result is not None:
                            report.update(result)
                            report.update(status='safe', reason_code='PARKING_LATTICE_READY',
                                needs_direction_authorization=len(result['segments']) > 1)
                            if len(result['segments']) == 1: report['points'] = result['segments'][0]['points']
                            return report
                    except _Failure as error:
                        if error.code == 'BUDGET_EXHAUSTED': raise
        for direction in directions:
            changes = node['changes']+int(direction != node['direction'])
            if changes > lattice.max_direction_changes: continue
            # The explicit first direction is part of the caller's request;
            # another direction requires a real stopped segment, even at rest.
            if node['parent'] is None and direction != initial_direction: continue
            for length in lattice.step_lengths_m:
                if node['distance']+length > lattice.max_path_length_m: continue
                for fraction in lattice.curvature_fractions:
                    work.step()
                    shape, curvature = _primitive(node['pose'], node['curvature'], maximum*fraction,
                        direction, length, lattice, parking_search.spacing_m, work)
                    cost = node['cost']+length*(lattice.reverse_cost if direction == -1 else 1.)
                    if direction != node['direction']: cost += lattice.direction_change_cost
                    key = _key(shape[-1], curvature, direction, changes, start, lattice, maximum)
                    if cost >= best.get(key, float('inf'))-1e-9: continue
                    try:
                        # Static geometric pruning only. This zero start is not
                        # the original vehicle speed; final certification uses
                        # the original speed and every dynamic prediction.
                        prune = _timed(shape, _lengths(shape), 0., cap*min(parking_search.speed_scales),
                            cap, node['curvature'], vehicle, limits, work, direction, True)
                        permitted = True
                        for region, objects in ((access, static), (coverage, [])):
                            checked = work.validate(prune, vehicle, limits, region, objects, direction)
                            if checked['status'] != 'safe':
                                if checked['reason_code'] == 'BUDGET_EXHAUSTED':
                                    raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')
                                permitted = False; break
                        if not permitted: continue
                    except _Failure as error:
                        if error.code == 'BUDGET_EXHAUSTED': raise
                        continue
                    if len(nodes) >= lattice.max_nodes:
                        report['reason_code'] = 'PARKING_LATTICE_NODE_LIMIT'; return report
                    following = dict(pose=shape[-1], curvature=curvature, direction=direction,
                        changes=changes, parent=index, edge=shape, start_curvature=node['curvature'],
                        distance=node['distance']+length, cost=cost)
                    nodes.append(following); best[key] = cost; serial += 1
                    heapq.heappush(queue, (cost+lattice.heuristic_weight*heuristic(shape[-1]), serial, len(nodes)-1))
                    report['nodes'] = len(nodes)
    return report


def search_parking_maneuver(start_pose, goal_pose, initial_speed_mps, initial_curvature_m_inv,
                            motion_direction, speed_cap_mps, vehicle, limits, access_corridor,
                            coverage, goal_corridor, obstacles, parking_search, lattice_search, budget):
    """Nominal whole maneuver, including explicit stopped direction changes.

    A multiple-segment result is not executable through the single-direction
    public contract. The coordinator must authorize and replan each segment
    from actual stopped pose/curvature with fresh sources and actual feedback.
    """
    report = dict(status='invalid', reason_code='INVALID_INPUT', points=[], segments=[], checks=0,
                  authority='nominal_candidate_only', needs_direction_authorization=False)
    work = None; started=time.monotonic()
    try:
        from members.planning.parking_generator import _search_parameters
        from members.planning.crossing_corridor import CrossingCorridor
        _motion_parameters(start_pose, initial_speed_mps, initial_curvature_m_inv, speed_cap_mps,
                           vehicle, limits, budget, allow_initial_overspeed=True)
        _search_parameters(parking_search)
        _require(type(motion_direction) is int and motion_direction in (-1, 1), 'PARKING_DIRECTION_INVALID')
        _require(isinstance(goal_pose, (tuple, list)) and len(goal_pose) == 3
                 and all(number(v) for v in goal_pose), 'PARKING_REAR_AXLE_GOAL_UNAVAILABLE')
        _require(isinstance(access_corridor, CrossingCorridor), 'PARKING_ACCESS_CORRIDOR_REQUIRED')
        _require(coverage is not None and goal_corridor is not None, 'PARKING_COVERAGE_OR_GOAL_REGION_MISSING')
        work = _Work(budget)
        report.update(_search_with_work(start_pose, goal_pose, initial_speed_mps, initial_curvature_m_inv,
            motion_direction, speed_cap_mps, vehicle, limits, access_corridor, coverage, goal_corridor,
            obstacles, parking_search, lattice_search, work))
    except _Failure as error:
        report.update(status=error.status, reason_code=error.code, points=[], segments=[])
    except (ValueError, TypeError, AttributeError, OverflowError, ZeroDivisionError) as error:
        report.update(status='invalid', reason_code='INVALID_INPUT', details=str(error), points=[], segments=[])
    finally:
        report['checks'] = work.count if work else 0
        report['elapsed_s']=time.monotonic()-started
    return report
