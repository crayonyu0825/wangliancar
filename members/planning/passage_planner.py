"""Whole-body passage admission on the selected path, with bounded work.

Geometry and time windows are explicit inputs. This module selects no right
of way, invents no visibility and never enlarges an authorized window.
"""
import math
import time

from core.behavior_contract import finite, require
from core.geometry import normalize_angle
from core.region_geometry import convex_cells, rectangle
from members.planning.candidate_validation import _polygon_gap, ValidationBudget


class PassageConstraints(object):
    def __init__(self, identifier, conflict_polygons, exit_region,
                 allowed_windows_s, entry_margin_m, exit_margin_m,
                 right_of_way=False, waiting=False, already_inside=False,
                 sample_spacing_m=.25, exiting=False):
        require(isinstance(identifier, str) and bool(identifier), 'PASSAGE_ID_REQUIRED')
        require(all(type(v) is bool for v in (right_of_way, waiting, already_inside,exiting)),
                'PASSAGE_AUTHORITY_FLAGS_INVALID')
        require(all(finite(v) and v >= 0 for v in (entry_margin_m, exit_margin_m))
                and finite(sample_spacing_m) and 0 < sample_spacing_m <= 1.,
                'PASSAGE_GEOMETRY_RESOLUTION_INVALID')
        require(isinstance(conflict_polygons, (list, tuple)) and 1 <= len(conflict_polygons) <= 64,
                'PASSAGE_CONFLICT_GEOMETRY_REQUIRED')
        require(isinstance(allowed_windows_s, (list, tuple)) and len(allowed_windows_s) <= 64,
                'PASSAGE_WINDOWS_INVALID')
        previous = -1.
        for window in allowed_windows_s:
            require(len(window) == 2 and all(finite(v) and v >= 0 for v in window)
                    and previous <= window[0] < window[1], 'PASSAGE_WINDOWS_INVALID')
            previous = window[1]
        self.identifier, self.conflict_polygons, self.exit_region = identifier, conflict_polygons, exit_region
        self.allowed_windows_s = tuple(tuple(v) for v in allowed_windows_s)
        self.entry_margin_m, self.exit_margin_m = entry_margin_m, exit_margin_m
        self.right_of_way, self.waiting, self.already_inside = right_of_way, waiting, already_inside
        self.sample_spacing_m = sample_spacing_m
        self.exiting=exiting


class _Work(object):
    def __init__(self, budget):
        require(isinstance(budget, ValidationBudget), 'PASSAGE_BUDGET_REQUIRED')
        self.budget, self.count = budget, 0

    def step(self):
        self.count += 1
        require(self.count <= self.budget.max_checks and time.monotonic() < self.budget.deadline_monotonic_s,
                'PASSAGE_BUDGET_EXHAUSTED')

    def remaining(self):
        self.step()
        return ValidationBudget(self.budget.max_checks - self.count, self.budget.max_depth,
                                self.budget.min_interval_s, self.budget.deadline_monotonic_s)


def path_stations(shape):
    require(isinstance(shape, (list, tuple)) and 2 <= len(shape) <= 2000
            and all(len(v) == 3 and all(finite(x) for x in v) for v in shape), 'PASSAGE_PATH_INVALID')
    stations = [0.]
    for a, b in zip(shape, shape[1:]):
        distance = math.hypot(b[0] - a[0], b[1] - a[1])
        require(distance > 1e-9, 'PASSAGE_PATH_DEGENERATE')
        stations.append(stations[-1] + distance)
    return stations


def pose_at(shape, stations, station):
    require(0 <= station <= stations[-1] + 1e-9, 'PASSAGE_PROGRESS_OUTSIDE_PATH')
    for index, (start, end) in enumerate(zip(stations, stations[1:])):
        if station <= end:
            u = max(0., min(1., (station - start) / (end - start)))
            a, b = shape[index:index + 2]
            return (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]),
                    normalize_angle(a[2] + u * normalize_angle(b[2] - a[2])))
    return tuple(shape[-1])


def _body(pose, vehicle, envelope, extra=0.):
    radius = math.hypot(max(vehicle.front_offset_m, vehicle.rear_offset_m), vehicle.half_width_m)
    margin = envelope.position_error_m + 2 * radius * math.sin(min(math.pi, envelope.heading_error_rad) / 2) + extra
    return rectangle(pose[0], pose[1], pose[2], vehicle.front_offset_m + margin,
                     vehicle.rear_offset_m + margin, vehicle.half_width_m + margin)


def body_occupancy_intervals(shape, vehicle, conflict_polygons, envelope, spacing, work):
    """Conservative s intervals, including front/rear overhang and turns.

    Each midpoint polygon is compared with the full possible corner movement
    over that subinterval. This cannot miss a thin crossing between samples.
    The returned intervals can be wider than the true intersection.
    """
    stations = path_stations(shape)
    cells = []
    for polygon in conflict_polygons:
        work.step()
        cells.extend(convex_cells(polygon, max_checks=work.budget.max_checks - work.count))
    occupied = []
    radius = math.hypot(max(vehicle.front_offset_m, vehicle.rear_offset_m), vehicle.half_width_m)
    for index, (a, b) in enumerate(zip(shape, shape[1:])):
        length = stations[index + 1] - stations[index]
        yaw = abs(normalize_angle(b[2] - a[2]))
        count = max(1, int(math.ceil((length + radius * yaw) / spacing)))
        require(count <= work.budget.max_checks - work.count, 'PASSAGE_BUDGET_EXHAUSTED')
        for step in range(count):
            work.step()
            low = stations[index] + length * step / count
            high = stations[index] + length * (step + 1) / count
            pose = pose_at(shape, stations, .5 * (low + high))
            body = _body(pose, vehicle, envelope)
            movement = .5 * (length + radius * yaw) / count
            touches = False
            for cell in cells:
                work.step()
                if _polygon_gap(body, cell) <= movement + 1e-8:
                    touches = True
                    break
            if touches:
                if occupied and abs(occupied[-1][1] - low) <= 1e-8:
                    occupied[-1] = (occupied[-1][0], high)
                else:
                    occupied.append((low, high))
    return stations, occupied


def _trim(shape, stations, bound):
    result = [tuple(shape[0])]
    for index in range(1, len(shape)):
        if stations[index] < bound - 1e-9:
            result.append(tuple(shape[index]))
        else:
            result.append(pose_at(shape, stations, bound))
            break
    return result


def _window_check(points, intervals, envelope, windows, work):
    stations = [0.]
    for a, b in zip(points, points[1:]):
        stations.append(stations[-1] + math.hypot(b.x - a.x, b.y - a.y))
    used = []
    for low, high in intervals:
        times = []
        for index, (a, b) in enumerate(zip(points, points[1:])):
            work.step()
            if stations[index] <= high and stations[index + 1] >= low:
                times.extend((max(0., a.relative_time - envelope.early_time_s),
                              b.relative_time + envelope.late_time_s))
        require(times, 'PASSAGE_CONFLICT_NOT_FULLY_TRAVERSED')
        occupied = (min(times), max(times))
        require(any(start <= occupied[0] and occupied[1] <= end for start, end in windows),
                'PASSAGE_EXECUTION_OUTSIDE_WINDOW')
        used.append(occupied)
    return used


def plan_passage(shape, initial_speed, speed_cap, vehicle, limits, corridor, coverage,
                 obstacles, envelope, constraints, budget, search=None, spatial_speed_limits=None):
    """Solve a wait/enter/authorized exit while retaining every obstacle."""
    from members.planning.dynamic_speed import solve_dynamic_speed
    started = time.monotonic()
    result = dict(status='invalid', reason_code='PASSAGE_INVALID', points=[], checks=0,
                  authority='explicit_selected_route_only', stop_required=False, stop_distance=-1.)
    work = None
    try:
        require(isinstance(constraints, PassageConstraints) and envelope is not None,
                'PASSAGE_CONTRACT_OR_EXECUTION_MODEL_REQUIRED')
        work = _Work(budget)
        stations, intervals = body_occupancy_intervals(shape, vehicle, constraints.conflict_polygons,
            envelope, constraints.sample_spacing_m, work)
        waiting = constraints.waiting or not constraints.right_of_way or not constraints.allowed_windows_s
        clearing_exit=False
        if not intervals and constraints.exiting and not waiting:
            actual=rectangle(*shape[0],vehicle.front_offset_m,vehicle.rear_offset_m,vehicle.half_width_m)
            clearing_exit=constraints.exit_region is not None and constraints.exit_region.body_clearance(
                actual,work.step,budget.max_checks-work.count)>1e-8
        require(intervals or constraints.already_inside or waiting or clearing_exit,
                'PASSAGE_CONFLICT_OUTSIDE_SELECTED_PATH')
        if waiting and not constraints.already_inside:
            bound = max(0., intervals[0][0] - constraints.entry_margin_m) if intervals else stations[-1]
            require(bound > 1e-6, 'PASSAGE_WAIT_BOUNDARY_ALREADY_REACHED')
            if not intervals:
                endpoint=_body(shape[-1],vehicle,envelope)
                require(all(_polygon_gap(endpoint,cell)>constraints.entry_margin_m for polygon in
                    constraints.conflict_polygons for cell in convex_cells(polygon)),
                    'PASSAGE_WAIT_FRONT_MARGIN_INSUFFICIENT')
            shape = _trim(shape, stations, bound)
            if spatial_speed_limits:
                from members.planning.dynamic_speed import _spatial_cap
                spatial_speed_limits=[(s,v) for s,v in spatial_speed_limits if s<bound]+[
                    (bound,_spatial_cap(bound,spatial_speed_limits,float('inf')))]
            result.update(stop_required=True, stop_distance=bound)
            solved = solve_dynamic_speed(shape, initial_speed, speed_cap, vehicle, limits, corridor,
                coverage, obstacles, work.remaining(), envelope, search=search, stop_required=True,
                terminal_speed_mps=0., spatial_speed_limits=spatial_speed_limits)
        else:
            require(constraints.right_of_way, 'PASSAGE_INSIDE_REQUIRES_EXPLICIT_EXIT_AUTHORITY')
            require(clearing_exit or intervals and intervals[-1][1] < stations[-1] - constraints.exit_margin_m,
                    'PASSAGE_ROUTE_DOES_NOT_CLEAR_TAIL')
            body = _body(shape[-1], vehicle, envelope, constraints.exit_margin_m)
            require(constraints.exit_region is not None and constraints.exit_region.body_clearance(
                body, work.step, budget.max_checks - work.count) > 1e-8, 'PASSAGE_EXIT_CANNOT_CONTAIN_BODY')
            solved = solve_dynamic_speed(shape, initial_speed, speed_cap, vehicle, limits, corridor,
                coverage, obstacles, work.remaining(), envelope, search=search,
                progress_time_windows=[(a, b, t0, t1) for a, b in intervals
                                       for t0, t1 in constraints.allowed_windows_s],
                window_progress_error_included=True,spatial_speed_limits=spatial_speed_limits)
        work.count += solved['checks']
        result['dynamic_report']={k:v for k,v in solved.items() if k!='points'}
        require(solved['status'] == 'safe', solved['reason_code'])
        if not waiting or constraints.already_inside:
            result['body_occupancy_windows_s'] = _window_check(solved['points'], intervals,
                envelope, constraints.allowed_windows_s, work)
        work.step()
        result.update(status='safe', reason_code='PASSAGE_WAIT_READY' if result['stop_required']
                      else 'PASSAGE_ADMISSION_READY', points=solved['points'], progress_intervals=intervals,
                      execution_robust=solved.get('execution_robust', False))
    except (ValueError, TypeError, AttributeError, OverflowError, ZeroDivisionError) as error:
        code = str(error)
        result.update(status='inconclusive' if 'BUDGET' in code else 'unsafe', reason_code=code, points=[])
    finally:
        result['checks'] = work.count if work else 0
        result['elapsed_s'] = time.monotonic() - started
    return result
