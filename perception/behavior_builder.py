"""Captain-owned formal observations. Unknown metadata never proves coverage."""
from core.behavior_protocol import TaskContext, GoalPose
from core.behavior_environment import (Evidence,LaneCorridor,ObservationCoverage,BehaviorEnvironment)


def build_behavior_environment(p, raw, route_manager):
    context=TaskContext(p.case_id,p.task_id,p.scene_id,p.session_id)
    now=p.observed_at_s
    observations=raw.get('behavior_observations',{})
    if not isinstance(observations,dict):
        raise ValueError('behavior observations must be a dictionary')
    header=(observations.get('frame_id'),observations.get('observed_at_s'),observations.get('valid_until_s'))
    fresh=(observations.get('clock_id')=='process_monotonic' and type(header[0]) is int and 0<=header[0]<=p.frame_id
           and type(header[1]) in (int,float) and header[1]<=now
           and type(header[2]) in (int,float) and now<header[2]<=p.valid_until
           and now-header[1]<=0.5)
    source_frame=header[0] if fresh else p.frame_id
    source_time=header[1] if fresh else now
    source_until=min(header[2],p.valid_until) if fresh else p.valid_until
    kind=observations.get('source_kind','unknown')
    allowed=(kind=='synthetic' if p.observation_mode=='model' else kind in ('sensor','verified_fusion'))
    source=observations.get('source','unknown')
    official=(p.targets_valid and p.source_status.get('targets',{}).get('usable') is True
              and p.target_source.startswith('sensor:'))
    if p.observation_mode=='production':
        if kind=='sensor':
            official=official and source==p.target_source
        elif kind=='verified_fusion':
            inputs=observations.get('input_sources',[])
            official=official and isinstance(inputs,list) and p.target_source in inputs
    dynamic=observations.get('dynamic_coverage',{})
    proof=Evidence(context,source_frame,source_time,source_until,usable=bool(fresh and allowed and official),
        coverage_verified=dynamic.get('coverage_complete') is True,
        source=source,source_kind=kind if allowed else 'unknown')
    timing=observations.get('execution_timing',{})
    timing_kind=timing.get('source_kind','unknown')
    timing_allowed=(timing_kind=='synthetic' if p.observation_mode=='model' else timing_kind in ('planning','verified_fusion'))
    timing_evidence=Evidence(context,source_frame,source_time,source_until,
        usable=bool(fresh and timing_allowed and timing.get('bounds_verified') is True),
        coverage_verified=timing.get('bounds_verified') is True,source=timing.get('source','unknown'),
        source_kind=timing_kind if timing_allowed else 'unknown')
    coverage=ObservationCoverage(proof,dynamic.get('regions',()),dynamic.get('scope',()),
        dynamic.get('proof_id',''),dynamic.get('proof_kind','unknown'),
        dynamic.get('position_uncertainty_m'),dynamic.get('velocity_uncertainty_mps'),dynamic.get('prediction_horizon_s'),
        timing.get('max_early_s'),timing.get('max_late_s'),timing_evidence)
    lanes=[]
    lane_data=observations.get('lanes',[])
    supplied_lanes=bool(lane_data)
    if not lane_data and hasattr(route_manager,'behavior_lane_observations'):
        lane_data=route_manager.behavior_lane_observations(p.ego)
    for lane in lane_data:
        map_kind=lane.get('source_kind','unknown')
        allowed_map=(map_kind=='synthetic' if p.observation_mode=='model' else map_kind in ('map','verified_fusion'))
        evidence=Evidence(context,source_frame if supplied_lanes else p.frame_id,
            source_time if supplied_lanes else now,source_until if supplied_lanes else p.valid_until,
            usable=allowed_map and lane.get('geometry_verified') is True and (fresh or not supplied_lanes),
            coverage_verified=lane.get('geometry_verified') is True,
            source=lane.get('source','unknown'),source_kind=map_kind if allowed_map else 'unknown')
        lanes.append(LaneCorridor(lane['lane_id'],lane['center_line'],lane['left_boundary'],lane['right_boundary'],
            evidence,lane.get('side','current'),lane.get('same_direction') is True,
            lane.get('marking','UNKNOWN'),lane.get('crossing_range_m')))
    task=observations.get('task',{})
    task_kind=task.get('source_kind','unknown')
    task_allowed=(task_kind=='synthetic' if p.observation_mode=='model' else task_kind in ('task','verified_fusion'))
    task_evidence=Evidence(context,source_frame,source_time,source_until,
        usable=bool(fresh and task_allowed and task.get('route_verified') is True),
        coverage_verified=task.get('route_verified') is True,source=task.get('source','unknown'),
        source_kind=task_kind if task_allowed else 'unknown')
    destination=task.get('destination')
    goal=GoalPose(destination['x'],destination['y'],destination['body_heading_rad']) if destination else None
    return BehaviorEnvironment(lanes,coverage,observations.get('road_regions',()),
        task.get('return_lane_id'),task.get('route_id'),goal,task.get('stop_duration_s',0.0),
        task.get('stop_strict',False),task.get('stop_rule_id'),task_evidence,task.get('desired_lane_id'))
