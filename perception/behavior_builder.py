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
    parking=None
    parking_raw=observations.get('parking_task')
    if parking_raw is not None:
        from core.parking_protocol import ParkingBay,ParkingTask
        bays=[]
        for bay in parking_raw.get('bays',[]):
            bays.append(ParkingBay(bay['id'],bay['boundary'],bay['body_heading_rad'],bay.get('occupancy','unknown'),proof))
        exit_pose=parking_raw['exit_goal']
        parking=ParkingTask(parking_raw['id'],task_evidence,bays,parking_raw.get('free_regions',()),
            GoalPose(exit_pose['x'],exit_pose['y'],exit_pose['body_heading_rad']),
            parking_raw.get('minimum_dwell_s',10.0),parking_raw.get('ego_target_id'))
    junction=None
    junction_raw=observations.get('intersection_task')
    if junction_raw is not None:
        from core.intersection_protocol import IntersectionTask
        junction=IntersectionTask(junction_raw['id'],task_evidence,junction_raw['route'],junction_raw['entry_s_m'],
            junction_raw['exit_s_m'],junction_raw['conflict_regions'],junction_raw['drivable_region'],
            junction_raw['exit_region'],junction_raw.get('permission_windows_s',()),
            junction_raw.get('right_of_way','UNKNOWN'),junction_raw.get('signal_permits') is True,
            junction_raw.get('speed_cap_mps',2.0),junction_raw.get('exit_gap_m',4.0))
    emergency=None
    emergency_raw=observations.get('emergency_task')
    if emergency_raw is not None:
        from core.emergency_protocol import EmergencyTask
        emergency=EmergencyTask(emergency_raw['danger_id'],proof,emergency_raw.get('dangerous') is True,
            emergency_raw.get('allowed_regions',()),emergency_raw.get('lateral_offsets_m',(0.0,)),
            emergency_raw.get('return_lane_id'),task_evidence if emergency_raw.get('allowed_regions_verified') is True
            and emergency_raw.get('rule_permissions_verified') is True else None)
    progress=None
    progress_raw=observations.get('task_progress')
    if progress_raw is not None:
        from core.task_protocol import TaskRouteFacts
        destination_raw=progress_raw.get('destination_goal')
        destination_pose=GoalPose(destination_raw['x'],destination_raw['y'],destination_raw['body_heading_rad']) if destination_raw else None
        progress=TaskRouteFacts(task_evidence,progress_raw['route_id'],progress_raw['segments'],progress_raw['segment_id'],
            progress_raw['progress_s_m'],progress_raw['reference_end_reason'],progress_raw.get('destination_region'),destination_pose,
            progress_raw.get('speed_cap_mps',3.0),progress_raw.get('selected_branch_verified') is True,
            progress_raw.get('required_obligations',()))
    return BehaviorEnvironment(lanes,coverage,observations.get('road_regions',()),
        task.get('return_lane_id'),task.get('route_id'),goal,task.get('stop_duration_s',0.0),
        task.get('stop_strict',False),task.get('stop_rule_id'),task_evidence,task.get('desired_lane_id'),parking,junction,emergency,progress)
