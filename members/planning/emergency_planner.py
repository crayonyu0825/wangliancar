"""Bounded brake/authorized-offset candidates, using the same P02/P05 model."""
import math
import time
import copy

from core.behavior_contract import finite, require
from members.planning.lane_change_generator import _shape, _Work, _Failure
from members.planning.dynamic_speed import solve_dynamic_speed, DynamicSearch
from members.planning.candidate_validation import ValidationBudget


def generate_emergency_candidates(start_pose, initial_speed, initial_curvature, speed_cap,
                                  vehicle, limits, regions, obstacles, search, budget,
                                  envelope, dynamic_search=None):
    """Only supplied rule-authorized regions can create lateral candidates.

    The caller sends the first-cycle brake before invoking this search on a
    later cycle. Deadline/node exhaustion returns no executable candidate.
    """
    started = time.monotonic()
    result = dict(status='unsafe', reason_code='NO_CERTIFIED_EMERGENCY_CANDIDATE', candidates=[],
                  checks=0, nodes=0, elapsed_s=0., budget_exhausted=False,
                  immediate_brake_required=True,attempts=[])
    work = None
    try:
        require(isinstance(regions, (list, tuple)) and 1 <= len(regions) <= 5
                and envelope is not None, 'EMERGENCY_AUTHORIZED_REGIONS_OR_EXECUTION_MODEL_REQUIRED')
        require(all(finite(v) for v in start_pose) and finite(initial_speed) and initial_speed >= 0,
                'EMERGENCY_START_MOTION_INVALID')
        work = _Work(budget)
        braking_search=copy.deepcopy(dynamic_search) if dynamic_search is not None else DynamicSearch()
        node_limit=braking_search.max_nodes
        samples=braking_search.acceleration_samples
        if samples is None:samples=(-limits.max_deceleration_mps2,-.5*limits.max_deceleration_mps2,0.)
        braking_search.acceleration_samples=tuple(v for v in samples if v<=0)
        for region in regions:
            work.step()
            require(isinstance(region, dict) and set(region) == {'action', 'goal', 'corridor', 'coverage', 'goal_region'}
                    and region['action'] in ('BRAKE', 'LEFT', 'RIGHT', 'BRAKE_LEFT', 'BRAKE_RIGHT'),
                    'EMERGENCY_REGION_INVALID')
            goal = region['goal']
            for scale in search.tangent_scales:
                work.step()
                try:
                    shape, distances = _shape(start_pose, goal, initial_curvature, scale, search, work)
                    sub = ValidationBudget(budget.max_checks - work.count, budget.max_depth,
                                           budget.min_interval_s, budget.deadline_monotonic_s)
                    remaining_nodes=node_limit-result['nodes']
                    if remaining_nodes<=0:
                        raise _Failure('EMERGENCY_NODE_BUDGET_EXHAUSTED','inconclusive')
                    attempt_search=copy.deepcopy(braking_search)
                    attempt_search.max_nodes=min(node_limit,remaining_nodes)
                    solved = solve_dynamic_speed(shape, initial_speed, speed_cap, vehicle, limits,
                        region['corridor'], region['coverage'], obstacles, sub, envelope,
                        search=attempt_search, stop_required=True, terminal_speed_mps=0.,allow_acceleration=False)
                    work.count += solved['checks']
                    result['nodes'] += solved['nodes']
                    result['attempts'].append(dict(action=region['action'],status=solved['status'],
                        reason_code=solved['reason_code'],checks=solved['checks'],nodes=solved['nodes'],
                        elapsed_s=solved['elapsed_s'],budget_exhausted='BUDGET' in solved['reason_code'],
                        rejected_edges=solved.get('rejected_edges',{}),
                        failed_constraints=solved.get('failed_constraints',[])))
                    if solved['status'] == 'safe':
                        from members.planning.passage_planner import _body
                        body = _body(goal, vehicle, envelope)
                        require(region['goal_region'].body_clearance(body, work.step,
                            budget.max_checks - work.count) > 1e-8, 'EMERGENCY_END_BODY_OUTSIDE_AUTHORITY')
                        require(all(b.speed<=a.speed+1e-9 for a,b in zip(solved['points'],solved['points'][1:])),
                                'EMERGENCY_PROFILE_REQUESTS_PROPULSION')
                        clearance = solved.get('legal_report',{}).get('clearance_lower_bound_m')
                        # All candidates satisfy hard constraints first. This
                        # finite risk ranking is not a probability of collision.
                        risk = 1. / max(1e-6, clearance) if finite(clearance) and clearance > 0 else 1e6
                        result['candidates'].append(dict(action=region['action'], goal=tuple(goal),
                            points=solved['points'], stop_distance=sum(distances), risk_cost=risk,
                            rule_cost=0., execution_robust=True, report=solved))
                        break
                    if 'BUDGET' in solved['reason_code']:
                        raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')
                except _Failure as error:
                    if 'BUDGET' in error.code:
                        raise
        work.step()
        if result['candidates']:
            result.update(status='safe', reason_code='EMERGENCY_CANDIDATES_READY')
            result['candidates'].sort(key=lambda v: (v['risk_cost'], v['rule_cost'], v['action']))
    except (ValueError, TypeError, AttributeError, OverflowError, _Failure) as error:
        code = error.code if isinstance(error, _Failure) else str(error)
        result.update(status='inconclusive' if 'BUDGET' in code else 'invalid', reason_code=code, candidates=[])
    finally:
        result['checks'] = work.count if work else 0
        result['elapsed_s'] = time.monotonic() - started
        result['budget_exhausted'] = 'BUDGET' in result['reason_code']
    return result
