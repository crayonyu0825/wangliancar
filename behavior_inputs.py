"""Captain-owned installation of explicitly configured maneuver models.

JSON verification flags and references are the captain's assertions, not proof
created by this loader. No file/verified model means no new capabilities. The
current formal Sensor/map sources remain independently mandatory every frame.
"""
import copy
import hashlib
import json
import math
import os
import re
import time

from core.behavior_contract import finite,require,GoalPose
from core.behavior_evidence import Evidence
from core.driving_mission import DrivingMission,DrivingDecisionInputs
from core.parking_mission import ParkingMission
from core.maneuver_facts import ManeuverFacts
from lane_change_integration import LaneChangeIntegrationInputs,LANE_CHANGE_CAPABILITIES
from members.decision.behaviors.lane_change_runtime import LaneChangeDecisionInputs
from members.planning.candidate_validation import VehicleGeometry,MotionLimits,ExecutionEnvelope,ValidationBudget
from members.planning.dynamic_speed import DynamicSearch
from members.planning.lane_change_generator import LaneChangeSearch,PredictionEnvelope
from members.planning.lane_change_behavior import LaneChangePlanningInputs
from members.planning.maneuver_scene import ManeuverGeometryCache
from members.planning.driving_behavior import DrivingPlanningInputs
from driving_integration import DrivingIntegrationInputs,DRIVING_ACTIONS
from parking_integration import ParkingIntegrationInputs,PARKING_ACTIONS
from members.decision.behaviors.parking_runtime import ParkingDecisionInputs
from members.planning.parking_behavior import ParkingPlanningInputs
from members.planning.parking_generator import ParkingSearch
from members.planning.parking_lattice import ParkingLatticeSearch,validate_lattice_search


VERSION='behavior-inputs-v1'


def _text(value):
    return isinstance(value,str) and bool(value.strip())


def _numbers(value,positive=True):
    return isinstance(value,dict) and all(finite(v) and (v>0 if positive else v>=0) for v in value.values())


class ConfiguredBehaviorInputs(object):
    """A frozen local model plus per-packet source-bound behavior providers.

    GPS steering conversion is permitted only with the explicit verified
    raw-to-front-wheel-angle mapping. Control command calibration is separate.
    Source metadata must say steering was actually present, not defaulted.
    """
    def __init__(self,config,document=None,clock=None):
        self.config,self.clock=config,clock or time.monotonic
        self.document=copy.deepcopy(document)
        self.lane_change_provider=None; self.lane_change_actions=()
        self.driving_provider=None; self.driving_actions=()
        self.parking_provider=None; self.parking_actions=()
        self.vehicle_geometry=None
        self.status='NOT_CONFIGURED'; self.reference=''
        if document is None:
            return
        document=self.document
        require(isinstance(document,dict) and document.get('version')==VERSION,'BEHAVIOR_CONFIG_VERSION')
        require(type(document.get('verified')) is bool,'BEHAVIOR_CONFIG_VERIFICATION_REQUIRED')
        if not document['verified']:
            self.status='UNVERIFIED_MODEL_DISABLED'; return
        require(_text(document.get('verification_reference')),'BEHAVIOR_CONFIG_REFERENCE_REQUIRED')
        self.reference=document['verification_reference']
        require(getattr(config,'control_calibrated',False) is True,'BEHAVIOR_CONFIG_CONTROL_REQUIRED')
        scope=document.get('scope')
        require(isinstance(scope,dict) and set(scope)=={'vehicle_id','case_id','task_id','map_digest'}
                and all(_text(scope[k]) for k in scope)
                and re.fullmatch('[0-9a-f]{32}',scope['map_digest']) is not None
                and scope['vehicle_id']==config.vehicle_id,'BEHAVIOR_CONFIG_SCOPE_REQUIRED')
        model=document.get('model')
        require(isinstance(model,dict),'BEHAVIOR_CONFIG_MODEL_REQUIRED')
        for key in ('model_id','verification_reference'):
            require(_text(model.get(key)),'BEHAVIOR_CONFIG_MODEL_REFERENCE_REQUIRED')
        require(model.get('coordinate_frame')=='world_xy_metres'
                and model.get('body_reference')=='rear_axle','BEHAVIOR_CONFIG_REFERENCE_POINT')
        steering=model.get('gps_steering')
        require(isinstance(steering,dict) and steering.get('verified') is True
                and _text(steering.get('verification_reference'))
                and finite(steering.get('scale_to_front_rad')) and steering['scale_to_front_rad']>0
                and type(steering.get('sign')) is int and steering['sign'] in (-1,1)
                and finite(steering.get('raw_abs_max')) and steering['raw_abs_max']>0,
                'BEHAVIOR_CONFIG_STEERING_MAPPING_UNVERIFIED')
        values=model.get('vehicle')
        require(isinstance(values,dict) and set(values)=={'front_offset_m','rear_offset_m','half_width_m','wheelbase_m'}
                and _numbers(values),'BEHAVIOR_CONFIG_VEHICLE_REQUIRED')
        self.vehicle_geometry=VehicleGeometry(**values)
        limits=model.get('limits')
        require(isinstance(limits,dict) and set(limits)=={'max_speed_mps','max_acceleration_mps2',
                'max_deceleration_mps2','max_lateral_acceleration_mps2','max_front_steer_rad',
                'max_steer_rate_rad_s','heading_tolerance_rad','distance_tolerance_m'}
                and _numbers(limits) and limits['max_front_steer_rad']<math.pi/2,
                'BEHAVIOR_CONFIG_LIMITS_REQUIRED')
        self.limits=MotionLimits(**limits)
        for field,actual in (('control_wheelbase_m',values['wheelbase_m']),
                            ('control_front_steer_max_rad',limits['max_front_steer_rad'])):
            configured=getattr(config,field,None)
            require(finite(configured) and abs(configured-actual)<=1e-9,'BEHAVIOR_CONFIG_CONTROL_GEOMETRY_MISMATCH')
        envelope=model.get('execution_envelope')
        require(isinstance(envelope,dict) and set(envelope)=={'position_error_m','heading_error_rad','early_time_s','late_time_s'}
                and _numbers(envelope,False) and envelope['position_error_m']>0
                and envelope['heading_error_rad']<=math.pi,'BEHAVIOR_CONFIG_EXECUTION_BOUNDS_REQUIRED')
        self.envelope=ExecutionEnvelope(**envelope)
        require(finite(model.get('speed_error_mps')) and model['speed_error_mps']>=0,
                'BEHAVIOR_CONFIG_SPEED_BOUND_REQUIRED')
        prediction=model.get('prediction')
        require(isinstance(prediction,dict) and _text(prediction.get('source'))
                and prediction['source'].startswith('sensor:')
                and _text(prediction.get('verification_reference'))
                and finite(prediction.get('horizon_s')) and prediction['horizon_s']>0
                and all(finite(prediction.get(k)) and prediction[k]>=0 for k in
                    ('position_uncertainty_m','velocity_uncertainty_mps')),
                'BEHAVIOR_CONFIG_PREDICTION_MODEL_REQUIRED')
        for field in ('planning_budget','monitor_budget'):
            b=model.get(field)
            require(isinstance(b,dict) and set(b)=={'max_checks','max_depth','min_interval_s','max_elapsed_s'}
                    and type(b['max_checks']) is int and b['max_checks']>0
                    and type(b['max_depth']) is int and 0<=b['max_depth']<=40
                    and all(finite(b[k]) and b[k]>0 for k in ('min_interval_s','max_elapsed_s')),
                    'BEHAVIOR_CONFIG_BUDGET_REQUIRED')
        search=model.get('lane_search')
        require(isinstance(search,dict) and set(search)=={'distances_m','tangent_scales','speed_scales',
                'spacing_m','max_points','projection_ambiguity_m'},'BEHAVIOR_CONFIG_LANE_SEARCH_REQUIRED')
        for key in ('distances_m','tangent_scales','speed_scales'):
            require(isinstance(search[key],list) and 1<=len(search[key])<=16
                    and all(finite(v) and v>0 for v in search[key]),'BEHAVIOR_CONFIG_LANE_SEARCH_INVALID')
        require(all(finite(search[k]) and search[k]>0 for k in ('spacing_m','projection_ambiguity_m'))
                and type(search['max_points']) is int and 2<=search['max_points']<=2000,
                'BEHAVIOR_CONFIG_LANE_SEARCH_INVALID')
        self.search=LaneChangeSearch(**search)
        dynamic=model.get('dynamic_search')
        require(isinstance(dynamic,dict),'BEHAVIOR_CONFIG_DYNAMIC_SEARCH_REQUIRED')
        self.dynamic=DynamicSearch(**dynamic)
        cache=model.get('geometry_cache')
        require(isinstance(cache,dict) and set(cache)=={'max_entries','preparation_max_checks'},
                'BEHAVIOR_CONFIG_GEOMETRY_CACHE_REQUIRED')
        self.cache=ManeuverGeometryCache(**cache)
        self.model_source='configured-model:'+hashlib.sha256(json.dumps(model,sort_keys=True).encode('utf-8')).hexdigest()
        driving=document.get('driving',{})
        require(isinstance(driving,dict) and type(driving.get('enabled',False)) is bool,
                'BEHAVIOR_CONFIG_DRIVING_TASK_INVALID')
        if driving.get('enabled',False):
            require(driving.get('verified') is True and _text(driving.get('verification_reference'))
                    and _text(driving.get('mission_id')) and driving.get('clock_id')=='process_monotonic',
                    'BEHAVIOR_CONFIG_DRIVING_AUTHORITY_UNVERIFIED')
        self.driving_provider=self._driving
        self.driving_actions=DRIVING_ACTIONS
        self._configure_parking(document.get('parking',{'enabled':False}))
        task=document.get('overtake')
        require(isinstance(task,dict) and type(task.get('enabled')) is bool,'BEHAVIOR_CONFIG_OVERTAKE_TASK_REQUIRED')
        self.status='CONFIGURED_MOTION_MODEL_REQUIRE_CURRENT_SOURCES'
        if not task['enabled']: return
        require(all(_text(task.get(k)) for k in ('mission_id','return_lane_id','verification_reference'))
                and type(task.get('allow_command_lights')) is bool,'BEHAVIOR_CONFIG_OVERTAKE_TASK_UNVERIFIED')
        require(finite(task.get('speed_cap_mps')) and 0<task['speed_cap_mps']<=self.limits.max_speed_mps,
                'BEHAVIOR_CONFIG_OVERTAKE_SPEED_LIMIT')
        options=set(task)-{'enabled','mission_id','return_lane_id','verification_reference','speed_cap_mps'}
        allowed={'confirmation_s','min_advantage_mps','transition_distance_m','min_front_gap_m',
            'min_rear_gap_m','headway_s','maneuver_time_s','acceleration_uncertainty_mps2',
            'indicator_lead_s','settle_frames','allow_command_lights',
            'maneuver','target_lane_id','return_required'}
        require(options<=allowed,'BEHAVIOR_CONFIG_OVERTAKE_UNKNOWN_OPTION')
        action=task.get('maneuver','OVERTAKE'); target=task.get('target_lane_id')
        returning=task.get('return_required',True)
        require(action in ('OVERTAKE','AVOID','MERGE') and type(returning) is bool
                and (target is None or _text(target))
                and (action!='MERGE' or target is not None and not returning),
                'BEHAVIOR_CONFIG_LANE_CHANGE_TASK_ACTION_OR_TARGET_INVALID')
        self.task_options={k:task[k] for k in options}
        self.task_source='configured-task:'+hashlib.sha256(json.dumps(task,sort_keys=True).encode('utf-8')).hexdigest()
        self.lane_change_provider=self._lane_change
        self.lane_change_actions=LANE_CHANGE_CAPABILITIES
        self.status='CONFIGURED_LANE_CHANGE_ASSERTIONS_REQUIRE_CURRENT_SOURCES'

    def _configure_parking(self,task):
        require(isinstance(task,dict) and type(task.get('enabled')) is bool,
                'BEHAVIOR_CONFIG_PARKING_TASK_INVALID')
        if not task['enabled']: return
        require(task.get('verified') is True and _text(task.get('verification_reference')),
                'BEHAVIOR_CONFIG_PARKING_AUTHORITY_UNVERIFIED')
        mission=task.get('mission')
        require(isinstance(mission,dict) and mission.get('task_verified') is True
                and all(_text(mission.get(k)) for k in ('mission_id','source'))
                and type(mission.get('space_id')) is int and mission['space_id']>=0
                and isinstance(mission.get('road_lane_ids'),list) and 1<=len(mission['road_lane_ids'])<=3
                and all(_text(v) for v in mission['road_lane_ids'])
                and len(set(mission['road_lane_ids']))==len(mission['road_lane_ids'])
                and type(mission.get('access_edge_index')) is int and 0<=mission['access_edge_index']<=3
                and finite(mission.get('body_heading_rad')),
                'BEHAVIOR_CONFIG_PARKING_MISSION_UNVERIFIED')
        for key in ('approach_pose','position_pose','exit_goal'):
            pose=mission.get(key)
            require(isinstance(pose,dict) and set(pose)=={'x','y','body_heading_rad'}
                    and all(finite(v) for v in pose.values()),'BEHAVIOR_CONFIG_PARKING_POSE_REQUIRED')
        tolerances=task.get('tolerances')
        require(isinstance(tolerances,list) and len(tolerances)==3
                and all(finite(v) and v>0 for v in tolerances)
                and finite(task.get('speed_cap_mps')) and 0<task['speed_cap_mps']<=self.limits.max_speed_mps
                and all(finite(task.get(k)) and task[k]>=0 for k in
                    ('position_uncertainty_m','velocity_uncertainty_mps'))
                and type(task.get('geometry_max_checks')) is int and task['geometry_max_checks']>0,
                'BEHAVIOR_CONFIG_PARKING_LIMITS_REQUIRED')
        search=task.get('search')
        require(isinstance(search,dict) and set(search)=={'tangent_scales','end_tangent_scales',
                'spacing_m','max_points','speed_scales','terminal_straight_m','lattice'},
                'BEHAVIOR_CONFIG_PARKING_SEARCH_REQUIRED')
        for name in ('tangent_scales','end_tangent_scales','speed_scales','terminal_straight_m'):
            values=search[name]
            require(isinstance(values,list) and 1<=len(values)<=8
                    and all(finite(v) and (v>=0 if name=='terminal_straight_m' else v>0) for v in values),
                    'BEHAVIOR_CONFIG_PARKING_SEARCH_INVALID')
        require(all(v<=1 for v in search['speed_scales']) and finite(search['spacing_m'])
                and search['spacing_m']>0 and type(search['max_points']) is int
                and 3<=search['max_points']<=2000,
                'BEHAVIOR_CONFIG_PARKING_SEARCH_INVALID')
        require(len(search['tangent_scales'])*len(search['end_tangent_scales'])*
                len(search['speed_scales'])*len(search['terminal_straight_m'])<=64,
                'BEHAVIOR_CONFIG_PARKING_SEARCH_INVALID')
        lattice=search['lattice']
        require(isinstance(lattice,dict) and set(lattice)=={'step_lengths_m','curvature_fractions',
            'curvature_change_per_m','xy_resolution_m','heading_resolution_rad','connector_distance_m',
            'max_nodes','max_expansions','max_path_length_m','max_direction_changes','reverse_cost',
            'direction_change_cost','permitted_directions','direction_change_stop_duration_s','heuristic_weight'},
            'BEHAVIOR_CONFIG_PARKING_LATTICE_REQUIRED')
        lattice=ParkingLatticeSearch(**lattice)
        validate_lattice_search(lattice)
        self.parking_search=ParkingSearch(tangent_scales=search['tangent_scales'],speed_scales=search['speed_scales'],
            spacing_m=search['spacing_m'],max_points=search['max_points'],end_tangent_scales=search['end_tangent_scales'],
            terminal_straight_m=search['terminal_straight_m'],lattice_search=lattice,
            execution_envelope=self.envelope,dynamic_search=self.dynamic)
        self.parking_tolerances=tuple(tolerances)
        self.parking_task_source='configured-parking:'+hashlib.sha256(json.dumps(task,sort_keys=True).encode('utf-8')).hexdigest()
        self.parking_provider=self._parking; self.parking_actions=PARKING_ACTIONS

    def budget(self,name,deadline,now):
        b=self.document['model'][name]
        return ValidationBudget(b['max_checks'],b['max_depth'],b['min_interval_s'],
                                min(deadline,now+b['max_elapsed_s']))

    def model_snapshot(self,p,context,expected_type=LaneChangePlanningInputs,**input_kwargs):
        """Reusable captain model read; no task or Sensor facts are fabricated."""
        require(self.document is not None and self.document.get('verified') is True,'BEHAVIOR_MODEL_UNVERIFIED')
        scope=self.document['scope']; model=self.document['model']; now=self.clock()
        require(context.key()[:3]==(p.case_id,p.task_id,p.scene_id)
                and (p.case_id,p.task_id)==(scope['case_id'],scope['task_id']), 'BEHAVIOR_MODEL_TASK_SCOPE_MISMATCH')
        facts=ManeuverFacts(p,self.clock)
        require(facts.map_digest()==scope['map_digest'],'BEHAVIOR_MODEL_MAP_SCOPE_MISMATCH')
        objects=facts.objects()
        require(p.target_source==model['prediction']['source'],'BEHAVIOR_MODEL_SENSOR_SCOPE_MISMATCH')
        gps=p.source_status.get('gps',{})
        require(isinstance(gps,dict) and gps.get('usable') is True and gps.get('read_ok') is True
                and 'steering' in gps.get('motion_fields',()) and finite(p.ego.steering),
                'BEHAVIOR_MODEL_CURRENT_STEERING_UNAVAILABLE')
        steering=model['gps_steering']
        require(abs(p.ego.steering)<=steering['raw_abs_max'],'BEHAVIOR_MODEL_STEERING_OUTSIDE_CALIBRATED_RANGE')
        front=p.ego.steering*steering['scale_to_front_rad']*steering['sign']
        require(abs(front)<=self.limits.max_front_steer_rad,'BEHAVIOR_MODEL_FRONT_STEERING_LIMIT')
        curvature=math.tan(front)/self.vehicle_geometry.wheelbase_m
        deadline=min(p.valid_until,facts.env.valid_until,facts.env.dynamic_valid_until)
        produced=facts.env.observed_at_s
        prediction=model['prediction']
        envelopes={v['id']:PredictionEnvelope(prediction['horizon_s'],prediction['position_uncertainty_m'],
                      prediction['velocity_uncertainty_mps']) for v in objects}
        snapshot=expected_type(context,p.frame_id,produced,deadline,model['model_id'],curvature,
            self.vehicle_geometry,self.limits,self.search,self.budget('planning_budget',deadline,now),
            envelopes,self.cache,execution_envelope=self.envelope,dynamic_search=self.dynamic,
            speed_error_mps=model['speed_error_mps'],**input_kwargs)
        facts.check_current(True)
        require(self.clock()<deadline,'BEHAVIOR_MODEL_SOURCE_EXPIRED_DURING_READ')
        return snapshot

    def _lane_change(self,p,context):
        m=self.model_snapshot(p,context)
        task=self.document['overtake']
        model=Evidence(context,p.frame_id,m.produced_at_s,m.valid_until_s,True,True,
                       self.model_source,source_kind='task')
        route=Evidence(context,p.frame_id,m.produced_at_s,m.valid_until_s,True,True,
                       self.task_source,source_kind='task')
        d=LaneChangeDecisionInputs(model,route,m.model_id,task['mission_id'],task['return_lane_id'],
            tuple(getattr(m.vehicle,k) for k in ('front_offset_m','rear_offset_m','half_width_m')),
            task['speed_cap_mps'],**self.task_options)
        return LaneChangeIntegrationInputs(d,m,self.budget('monitor_budget',m.valid_until_s,self.clock()))

    def _driving(self,p,context):
        scope=self.document['scope']
        base=self.model_snapshot(p,context)
        e=Evidence(context,p.frame_id,base.produced_at_s,base.valid_until_s,True,True,
                   self.model_source,source_kind='task')
        task=self.document.get('driving',{})
        configured=task.get('enabled') is True
        require(not configured or finite(task.get('speed_cap_mps'))
                and 0<task['speed_cap_mps']<=self.limits.max_speed_mps,'BEHAVIOR_CONFIG_DRIVING_SPEED_LIMIT')
        goals=[]
        if configured:
            for option in task.get('emergency_goals',()):
                goals.append(dict(action=option['action'],goal=GoalPose(**option['goal']),
                                  lane_ids=option['lane_ids']))
        # Generic model installation authorizes speed retiming, not route choice
        # or priority. No passage/side goal is inferred from case or lane IDs.
        mission=DrivingMission(task['mission_id'] if configured else 'configured-motion-model',
            scope['map_digest'],e,e,
            passage_id=task.get('passage_id') if configured else None,
            allowed_windows_abs_s=task.get('allowed_windows_abs_s',()) if configured else (),
            right_of_way=task.get('right_of_way','UNKNOWN') if configured else 'UNKNOWN',
            speed_cap_mps=task.get('speed_cap_mps',self.limits.max_speed_mps) if configured else self.limits.max_speed_mps,
            entry_margin_m=task.get('entry_margin_m',.3) if configured else .3,
            exit_margin_m=task.get('exit_margin_m',.3) if configured else .3,
            emergency_goals=goals,immediate_brake_cycles=task.get('immediate_brake_cycles',1) if configured else 1)
        d=DrivingDecisionInputs(mission,base.model_id,tuple(getattr(base.vehicle,k) for k in
            ('front_offset_m','rear_offset_m','half_width_m')),base.limits.max_deceleration_mps2,
            execution_position_error_m=base.execution_envelope.position_error_m,
            execution_heading_error_rad=base.execution_envelope.heading_error_rad)
        # Create through the declared constructor instead of attaching a mission
        # to an arbitrary member model or Perception object.
        model=DrivingPlanningInputs(base.context,base.frame_id,base.produced_at_s,base.valid_until_s,
            base.model_id,base.initial_curvature_m_inv,base.vehicle,base.limits,base.search,base.budget,
            base.prediction_envelopes,base.geometry_cache,execution_envelope=base.execution_envelope,
            dynamic_search=base.dynamic_search,speed_error_mps=base.speed_error_mps,mission=mission)
        return DrivingIntegrationInputs(d,model,self.budget('monitor_budget',base.valid_until_s,self.clock()))

    def _parking(self,p,context):
        scope=self.document['scope']
        base=self.model_snapshot(p,context)
        task=self.document['parking']; spec=task['mission']
        evidence=Evidence(context,p.frame_id,base.produced_at_s,base.valid_until_s,True,True,
                          self.parking_task_source+':'+spec['source'],source_kind='task')
        model_evidence=Evidence(context,p.frame_id,base.produced_at_s,base.valid_until_s,True,True,
                                self.model_source,source_kind='task')
        mission=ParkingMission(evidence,spec['mission_id'],scope['map_digest'],spec['space_id'],
            spec['road_lane_ids'],spec['access_edge_index'],spec['body_heading_rad'],
            *(GoalPose(**spec[k]) for k in ('approach_pose','position_pose','exit_goal')))
        d=ParkingDecisionInputs(model_evidence,mission,base.model_id,tuple(getattr(base.vehicle,k) for k in
            ('front_offset_m','rear_offset_m','half_width_m')),task['position_uncertainty_m'],
            task['velocity_uncertainty_mps'],task['geometry_max_checks'],self.parking_tolerances,
            task['speed_cap_mps'])
        model=ParkingPlanningInputs(mission,self.parking_tolerances,base.context,base.frame_id,base.produced_at_s,
            base.valid_until_s,base.model_id,base.initial_curvature_m_inv,base.vehicle,base.limits,
            self.parking_search,base.budget,base.prediction_envelopes,base.geometry_cache,
            execution_envelope=base.execution_envelope,dynamic_search=base.dynamic_search,
            speed_error_mps=base.speed_error_mps)
        return ParkingIntegrationInputs(d,model,self.budget('monitor_budget',base.valid_until_s,self.clock()))


def load_behavior_inputs(config,clock=None):
    path=getattr(config,'behavior_model_file','')
    if not path:
        return ConfiguredBehaviorInputs(config,clock=clock)
    require(os.path.getsize(path)<=1024*1024,'BEHAVIOR_CONFIG_TOO_LARGE')
    with open(path,'r',encoding='utf-8') as stream:
        document=json.load(stream)
    return ConfiguredBehaviorInputs(config,document,clock)
