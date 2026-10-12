"""Formal R08 bay/known-road scene; source geometry is not driving authority."""
import time
from fractions import Fraction

from core.boundary_lineage import encode_position
from core.behavior_contract import require
from core.maneuver_facts import ManeuverFacts
from core.geometry import opposes_direction
from core.validation import number
from members.planning.candidate_validation import CorridorRegion,ValidationBudget
from members.planning.crossing_corridor import CrossingCorridor
from members.planning.maneuver_scene import ManeuverGeometryCache
from members.planning.lane_change_generator import _Failure,_require,_source_predictions
from members.planning.parking_generator import generate_parking_segment
from members.planning.candidate_validation import validate_candidate


def validate_parking_path(perception,mission,stage,points,motion_direction,vehicle,limits,
                          budget,prediction_envelopes,clock=None,intermediate_segment=False,
                          execution_envelope=None):
    """Public current-source recheck for the captain's independent veto.

    Reuse P02's nominal swept-body validator, not a new trajectory or an
    execution acknowledgement. Budget is separate from the planner's search.
    """
    clock=clock or time.monotonic
    report=dict(status='invalid',reason_code='PARKING_MONITOR_INPUT_INVALID',checks=0)
    started=clock()
    try:
        require(stage in ('APPROACH','POSITION','REVERSE_ENTRY','ALIGN','PARKED_DWELL','EXIT_PREPARE','EXIT'),
                'PARKING_STAGE_UNSUPPORTED')
        require(type(motion_direction) is int and motion_direction in (-1,1) and (
            stage in ('ALIGN','PARKED_DWELL','EXIT_PREPARE') or motion_direction==(
            -1 if stage=='REVERSE_ENTRY' else 1)),
            'PARKING_STAGE_DIRECTION_MISMATCH')
        require(isinstance(budget,ValidationBudget) and type(budget.max_checks) is int and budget.max_checks>0
                and number(budget.deadline_monotonic_s) and clock()<budget.deadline_monotonic_s,
                'PARKING_MONITOR_BUDGET_UNAVAILABLE')
        scene=read_parking_scene(perception,mission.space_id,mission.road_lane_ids,
            None if intermediate_segment else stage in ('REVERSE_ENTRY','ALIGN','PARKED_DWELL'),mission.access_edge_index,clock)
        require(scene['map_digest']==mission.map_digest,'PARKING_MISSION_MAP_MISMATCH')
        predictions=_source_predictions(scene['objects'],prediction_envelopes,clock())
        deadline=min(budget.deadline_monotonic_s,
            time.monotonic()+scene['source_valid_until_s']-clock())
        for region,objects in ((scene['access_corridor'],predictions),(scene['coverage'],[])):
            require(report['checks']<budget.max_checks,'PARKING_MONITOR_BUDGET_EXHAUSTED')
            remaining=ValidationBudget(budget.max_checks-report['checks'],budget.max_depth,
                                       budget.min_interval_s,deadline)
            result=validate_candidate(points,motion_direction,vehicle,limits,region,objects,remaining,
                                      execution_envelope=execution_envelope)
            report['checks']+=result['checks']
            report.update(status=result['status'],reason_code=result['reason_code'])
            if result['status']!='safe': return report
        latest=read_parking_scene(perception,mission.space_id,mission.road_lane_ids,
            None if intermediate_segment else stage in ('REVERSE_ENTRY','ALIGN','PARKED_DWELL'),mission.access_edge_index,clock)
        require(all(latest[k]==scene[k] for k in ('bay','roads','objects','map_digest','source_valid_until_s'))
                and latest['coverage'].polygons==scene['coverage'].polygons,
                'PARKING_MONITOR_SOURCES_CHANGED_DURING_CHECK')
        require(started<=clock()<scene['source_valid_until_s']
                and time.monotonic()<deadline,'PARKING_MONITOR_SOURCE_EXPIRED_DURING_CHECK')
        report.update(source_valid_until_s=scene['source_valid_until_s'])
        return report
    except (_Failure,ValueError,TypeError,AttributeError,KeyError,OverflowError) as error:
        report.update(status='invalid',reason_code=error.code if isinstance(error,_Failure) else str(error))
        return report


def read_parking_scene(perception,space_id,road_lane_ids,goal_in_bay,access_edge_index,clock=None,geometry_cache=None):
    require((type(goal_in_bay) is bool or goal_in_bay is None) and isinstance(road_lane_ids,(list,tuple))
            and 1<=len(road_lane_ids)<=3 and len(set(road_lane_ids))==len(road_lane_ids)
            and all(isinstance(v,str) and v for v in road_lane_ids),'PARKING_ROAD_SET_INVALID')
    require(type(access_edge_index) is int and 0<=access_edge_index<=3,'PARKING_ACCESS_EDGE_UNAVAILABLE')
    facts=ManeuverFacts(perception,clock)
    allowed=[perception.lane.lane_id]+facts.neighbors()
    require(all(v in allowed for v in road_lane_ids),'PARKING_ROAD_UNAVAILABLE')
    bay=facts.parking(space_id)
    roads=[facts.road(v,False) for v in road_lane_ids]
    raw=bay['boundary_knots']+[p for road in roads for key in ('center_line','left_boundary','right_boundary') for p in road[key]]
    views=facts.coverage(raw)
    require(bool(views),'PARKING_COMPLETE_VISIBILITY_UNAVAILABLE')
    knots=bay['boundary_knots']
    # Upstream explicitly chooses a nominal access edge; SDK a-d front is not
    # automatically the access side of a parallel bay. Other edges stay closed
    # for this supplied constraint, not because map paint is a physical wall.
    # It does not establish lawful road access or create parking permission.
    i=access_edge_index
    descriptor=((knots[i],knots[(i+1)%4]),(knots[(i+3)%4],knots[(i+2)%4]),'left',
                ((encode_position(0,Fraction(0)),encode_position(0,Fraction(1))),))
    polygons=[v['polygon'] for v in roads]+[bay['boundary']]
    goal_polygons=(polygons if goal_in_bay is None else
                   [bay['boundary']] if goal_in_bay else [v['polygon'] for v in roads])
    if geometry_cache is None:
        access=CrossingCorridor(tuple(tuple(tuple(p) for p in v) for v in polygons),descriptor,None)
        coverage=CorridorRegion([v['polygon'] for v in views])
        goal=CorridorRegion(goal_polygons)
    else:
        require(isinstance(geometry_cache,ManeuverGeometryCache) and geometry_cache.max_entries>=3,
                'PARKING_GEOMETRY_CACHE_NEEDS_THREE_ENTRIES')
        scope=(perception.case_id,perception.task_id,perception.scene_id,bay['map_digest'])
        key=(space_id,tuple(road_lane_ids),access_edge_index)
        access=geometry_cache.region(scope,('parking_access',key),polygons,descriptor)
        coverage=geometry_cache.region(scope,('parking_view',key),[v['polygon'] for v in views])
        goal=geometry_cache.region(scope,('parking_goal',key,goal_in_bay),goal_polygons)
    deadlines=[facts.env.valid_until,facts.env.dynamic_valid_until,bay['evidence']['valid_until_s']]
    deadlines += [v['geometry_evidence']['valid_until_s'] for v in roads]
    deadlines += [v['valid_until_s'] for v in views]
    deadlines += [v['evidence']['valid_until_s'] for v in bay['objects']]
    facts.check_current(True)
    return dict(bay=bay,roads=roads,objects=bay['objects'],access_corridor=access,
                coverage=coverage,goal_corridor=goal,map_digest=bay['map_digest'],
                source_valid_until_s=min(deadlines),authority='bay_access_geometry_and_current_visibility_only')


def plan_parking_candidate(perception,space_id,stage,goal_pose,motion_direction,initial_curvature_m_inv,
                           speed_cap_mps,vehicle,limits,search,budget,prediction_envelopes,
                           road_lane_ids,access_edge_index,clock=None,geometry_cache=None,
                           intermediate_segment=False):
    """Generate one stage from formal facts, never advance or change gears.

    Requires a known current map/lane and explicit upstream goal/constraints.
    Actual parking-lot/aisle coverage and lawful access remain integration gates.
    """
    clock=clock or time.monotonic
    try:
        _require(stage in ('APPROACH','POSITION','REVERSE_ENTRY','ALIGN','EXIT'),'PARKING_STAGE_UNSUPPORTED')
        goal_in_bay=None if intermediate_segment else stage in ('REVERSE_ENTRY','ALIGN')
        _require(type(motion_direction) is int and motion_direction in (-1,1)
                 and (stage=='ALIGN' or motion_direction==(-1 if goal_in_bay else 1)),
                 'PARKING_STAGE_DIRECTION_MISMATCH')
        _require(all(number(v) for v in (perception.ego.vx,perception.ego.vy,perception.ego.speed))
                 and not opposes_direction(perception.ego.vx,perception.ego.vy,perception.ego.heading,
                                            perception.ego.speed,motion_direction),
                 'ACTUAL_MOTION_OPPOSES_PARKING_STAGE')
        scene=read_parking_scene(perception,space_id,road_lane_ids,goal_in_bay,access_edge_index,clock,geometry_cache)
        facts=ManeuverFacts(perception,clock)
        source_lane=perception.lane.lane_id
        now=clock()
        predictions=_source_predictions(scene['objects'],prediction_envelopes,now)
        _require(isinstance(budget,ValidationBudget) and number(budget.deadline_monotonic_s),
                 'VALIDATION_BUDGET_INVALID')
        effective=ValidationBudget(budget.max_checks,budget.max_depth,budget.min_interval_s,
            min(budget.deadline_monotonic_s,time.monotonic()+scene['source_valid_until_s']-clock()))
        result=generate_parking_segment((perception.ego.x,perception.ego.y,perception.ego.heading),goal_pose,
            perception.ego.speed,initial_curvature_m_inv,motion_direction,speed_cap_mps,vehicle,limits,
            scene['access_corridor'],scene['coverage'],scene['goal_corridor'],predictions,search,effective)
        facts.check_current(True)
        current_bay=facts.parking(space_id)
        _require(current_bay['boundary_knots']==scene['bay']['boundary_knots']
                 and current_bay['entrance_edge']==scene['bay']['entrance_edge'],
                 'PARKING_BAY_CHANGED_DURING_SEARCH')
        _require(perception.lane.lane_id==source_lane and facts.map_digest()==scene['map_digest'],
                 'PARKING_SOURCE_IDENTITY_CHANGED_DURING_SEARCH')
        _require(clock()<scene['source_valid_until_s'],'PARKING_SOURCE_EXPIRED_DURING_SEARCH')
        result.update(space_id=space_id,stage=stage,frame_id=perception.frame_id,
                      nominal_start_time_s=now,source_valid_until_s=scene['source_valid_until_s'],
                      map_digest=scene['map_digest'])
        return result
    except (_Failure,ValueError,TypeError,AttributeError,KeyError,OverflowError) as error:
        return dict(status=error.status if isinstance(error,_Failure) else 'invalid',
                    reason_code=error.code if isinstance(error,_Failure) else str(error),points=[],
                    authority='nominal_candidate_only')
