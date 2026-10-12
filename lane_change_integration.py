"""Captain-installed paired lane business inputs and independent path veto.

No provider, lamp evidence policy, visibility certificate or capability is
enabled by constructing the ordinary runtime. Models are explicit assertions,
not calibration obtained from this adapter. All consumers use the same packet.
"""
import copy
import math
import time

from core.behavior_channel import read_channel, capabilities_dict, feedback_dict
from core.behavior_contract import BehaviorFrame, ExecutionFeedback, finite, require
from core.geometry import normalize_angle, opposes_direction
from core.interfaces import DecisionTarget, Trajectory, TrajectoryPoint, ControlOut
from core.safety_supervisor import SafetyAssessment
from core.execution_monitor import ExecutionMonitor
from core.validation import validate_output
from core.maneuver_facts import ManeuverFacts
from members.decision.behaviors.lane_change_runtime import LaneChangeDecisionInputs
from members.planning.lane_change_behavior import LaneChangePlanningInputs
from members.planning.maneuver_inputs import read_inputs
from members.planning.maneuver_scene import read_maneuver_scene
from members.planning.behavior_contract import assess_behavior_request, LANE_CHANGE_ACTIONS
from members.planning.candidate_validation import validate_candidate, ValidationBudget, ExecutionEnvelope
from members.planning.lane_change_generator import _source_predictions


LANE_CHANGE_CAPABILITIES=('LANE_CHANGE','PATH','LIGHTS','FEEDBACK')


def _planning_copy(value):
    result=copy.copy(value)
    for name in vars(value):
        if name!='geometry_cache':
            setattr(result,name,copy.deepcopy(getattr(value,name)))
    return result


class LaneChangeIntegrationInputs(object):
    def __init__(self,decision,planning,monitor_budget):
        require(isinstance(decision,LaneChangeDecisionInputs) and isinstance(planning,LaneChangePlanningInputs)
                and isinstance(monitor_budget,ValidationBudget),'LANE_CHANGE_INPUT_PAIR_UNAVAILABLE')
        self.decision,self.planning=decision.snapshot(),_planning_copy(planning)
        self.monitor_budget=copy.deepcopy(monitor_budget)

    def snapshot(self):
        return LaneChangeIntegrationInputs(self.decision,self.planning,self.monitor_budget)


class LaneChangeInputBridge(object):
    def __init__(self,provider,clock=None):
        require(callable(provider),'LANE_CHANGE_PAIR_PROVIDER_INVALID')
        self.provider,self.clock=provider,clock or time.monotonic
        self.settings=None
        self.key,self.value,self.error=None,None,None
        self.reason='NOT_PREPARED'
        self.lamp_key,self.lamp_frame,self.lamp_time,self.lamp_elapsed=None,None,None,0.
        self.execution=ExecutionMonitor()
        self.execution_tracking=None
        self.execution_fault=None

    @staticmethod
    def _key(p,context):
        return (id(p),context.key(),p.frame_id,p.timestamp)

    def prepare(self,p):
        try:
            context,unused,unused_replies=read_channel(p,self.clock())
            key=self._key(p,context)
            if key==self.key:
                return
            self.key,self.value,self.error=key,None,None
            pair=self.provider(p,context)
            if pair is None:
                self.reason='NO_LANE_CHANGE_TASK'; return
            require(isinstance(pair,LaneChangeIntegrationInputs),'LANE_CHANGE_INPUT_PAIR_UNAVAILABLE')
            pair=pair.snapshot()
            self._validate(pair,p,context)
            self.value,self.reason=pair,'CURRENT_PAIRED_LANE_CHANGE_INPUTS'
        except Exception as error:
            self.value,self.error=None,str(error)
            self.reason='LANE_CHANGE_PAIR:'+str(error)

    def _validate(self,pair,p,context):
        d=pair.decision.snapshot()
        m=read_inputs(lambda *args:pair.planning,p,context,self.clock,'LANE_CHANGE_PAIR',LaneChangePlanningInputs)
        now=self.clock()
        frame=BehaviorFrame(context,p.frame_id,now,p.valid_until,p.ego.x,p.ego.y,
                            p.ego.heading,p.ego.speed,usable=p.valid)
        e=d.model_evidence
        require(e.frame_id==p.frame_id and e.observed_at_s==m.produced_at_s
                and e.valid_until_s==m.valid_until_s
                and e.verified(frame,allowed_source_kinds=('task','verified_fusion'))
                and d.route_evidence.frame_id==p.frame_id
                and d.route_evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                'LANE_CHANGE_PAIR_SOURCE_OR_TASK_MISMATCH')
        require(d.model_id==m.model_id and d.extents==tuple(getattr(m.vehicle,k) for k in
                ('front_offset_m','rear_offset_m','half_width_m')) and d.speed_cap<=m.limits.max_speed_mps,
                'LANE_CHANGE_PAIR_MODEL_OR_GEOMETRY_MISMATCH')
        require(isinstance(m.execution_envelope,ExecutionEnvelope),'LANE_CHANGE_EXECUTION_ERROR_UNAVAILABLE')
        require(self.settings is not None,'LANE_CHANGE_RUNTIME_SETTINGS_UNAVAILABLE')
        for name,actual in tuple(zip(('front_offset_m','rear_offset_m','half_width_m'),d.extents))+(
                ('wheelbase_m',m.vehicle.wheelbase_m),('front_steer_max_rad',m.limits.max_front_steer_rad)):
            configured=getattr(self.settings,name,None)
            require(finite(configured) and finite(actual) and abs(configured-actual)<=1e-9,
                    'LANE_CHANGE_RUNTIME_CONTROL_GEOMETRY_MISMATCH')
        b=pair.monitor_budget
        require(type(b.max_checks) is int and b.max_checks>0 and type(b.max_depth) is int
                and 0<=b.max_depth<=40 and finite(b.min_interval_s) and b.min_interval_s>0
                and finite(b.deadline_monotonic_s) and now<b.deadline_monotonic_s,
                'LANE_CHANGE_MONITOR_BUDGET_UNAVAILABLE')
        facts=ManeuverFacts(p,self.clock)
        facts.map_digest(); facts.objects(); facts.check_current(True)
        require(self.clock()<min(e.valid_until_s,d.route_evidence.valid_until_s),
                'LANE_CHANGE_PAIR_EXPIRED_DURING_READ')

    def _read(self,p,context):
        require(self.key==self._key(p,context),'LANE_CHANGE_PAIR_NOT_PREPARED_FOR_FRAME')
        require(self.error is None,self.reason)
        if self.value is None:
            return None
        pair=self.value.snapshot(); self._validate(pair,p,context)
        return pair

    def decision_inputs(self,p,context):
        pair=self._read(p,context)
        return pair.decision if pair is not None else None

    def planning_inputs(self,p,context):
        pair=self._read(p,context)
        require(pair is not None,'LANE_CHANGE_PAIR_WITHDRAWN')
        return pair.planning

    def assess_execution(self,p,trajectory):
        """Check the previous sent path on every configured runtime frame.

        This separate hook covers the ordinary confirmation and passing paths
        too. The captain calls it once before the maneuver's current-path veto.
        """
        try:
            now=self.clock(); context,unused,unused_replies=read_channel(p,now)
            pair=self._read(p,context)
            if pair is None:
                require(not self.execution.history,'LANE_CHANGE_TRACKING_MODEL_WITHDRAWN')
                return SafetyAssessment()
            m=pair.planning
            self.execution.max_speed_mps=m.limits.max_speed_mps
            self.execution.speed_error_mps=m.speed_error_mps
            require(self.execution_fault is None,self.execution_fault)
            report=self.execution.observe(p,trajectory,m.execution_envelope,now)
            self.execution_tracking=report
            require(not report['active'],'LANE_CHANGE_TRACKING:'+report['reason_code'])
            return SafetyAssessment()
        except Exception as error:
            return SafetyAssessment('controlled_stop','LANE_CHANGE_TRACKING:'+str(error))

    def record_sent(self,p,trajectory,receipt,safety_active=False):
        """Only the captain's actual normal-command receipt arms tracking."""
        if safety_active or not isinstance(receipt,dict) or receipt.get('attempted') is not True or receipt.get('ok') is not True:
            return
        try:
            now=self.clock(); context,unused,unused_replies=read_channel(p,now)
            pair=self._read(p,context)
            if pair is None: return
            m=pair.planning
            self.execution.max_speed_mps=m.limits.max_speed_mps
            self.execution.speed_error_mps=m.speed_error_mps
            report=self.execution.record_sent(p,trajectory,m.execution_envelope,now,receipt)
            require(not report['active'],report['reason_code'])
        except Exception as error:
            self.execution_fault='LANE_CHANGE_SENT_TIMELINE:'+str(error)

    def observe_lights(self,p,decision,trajectory,control,transport,receipt,safety_active=False):
        """Command receipt duration, never lamp observation or path acceptance."""
        r=decision.behavior_request
        if not isinstance(r,dict) or r.get('maneuver') not in LANE_CHANGE_ACTIONS or r.get('stage') not in ('PREPARE','WAIT_GAP'):
            self.lamp_key,self.lamp_elapsed=None,0.; return
        try:
            now=self.clock(); context,unused,unused_replies=read_channel(p,now)
            pair=self._read(p,context); require(pair is not None,'LANE_CHANGE_PAIR_WITHDRAWN')
            validate_output(trajectory,Trajectory,decision); validate_output(control,ControlOut,trajectory)
            requested=r['light_intent']
            lamps=(control.left_signal,control.right_signal,control.hazard_signal)
            wanted=tuple(requested[k] for k in ('left_signal','right_signal','hazard_signal'))
            key=(context.key(),r['intent_id'],lamps)
            dt=0. if self.lamp_time is None else now-self.lamp_time
            distinct=self.lamp_frame is None or p.frame_id>self.lamp_frame
            sent=(isinstance(receipt,dict) and receipt.get('attempted') is True
                  and receipt.get('ok') is True and not safety_active and lamps==wanted)
            if key!=self.lamp_key or dt<0 or dt>.5 or not sent:
                self.lamp_elapsed=0.
            elif distinct:
                self.lamp_elapsed+=dt
            self.lamp_key=key
            if distinct:
                self.lamp_frame,self.lamp_time=p.frame_id,now
            deadline=min(p.valid_until,decision.valid_until,trajectory.valid_until,control.valid_until,r['valid_until_s'])
            require(now<deadline,'LANE_CHANGE_LAMP_SOURCE_EXPIRED')
            reply=ExecutionFeedback(context,r['intent_id'],r['stage'],r['revision'],'runtime',
                p.frame_id,p.frame_id,now,deadline,'EXECUTING',usable=True,
                reason_code='LAMP_COMMAND_SENT' if sent else 'LAMP_COMMAND_UNCONFIRMED',
                lights_confirmed=False,lights_duration_s=0.,
                lights_command_duration_s=self.lamp_elapsed if sent else 0.,
                lights_evidence_level='command_sent' if sent else 'unknown',
                command_send_status='sent' if sent else 'unknown')
            transport.feedbacks.append(feedback_dict(reply))
        except Exception:
            self.lamp_key,self.lamp_elapsed=None,0.

    def assess(self,p,decision,trajectory,control):
        """Independent current-output P02 sweep; never regenerate a substitute."""
        mode='controlled_stop'
        original=(p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,p.ego.speed,p.ego.vx,p.ego.vy)
        try:
            validate_output(decision,DecisionTarget,p)
            validate_output(trajectory,Trajectory,decision)
            checked=validate_output(copy.deepcopy(control),ControlOut,trajectory)
            require(all(getattr(checked,k)==getattr(control,k) for k in ('throttle','brake','steering')),
                    'LANE_CHANGE_MONITOR_CONTROL_NOT_NORMALIZED')
            now=self.clock(); context,caps,unused=read_channel(p,now)
            pair=self._read(p,context); require(pair is not None,'LANE_CHANGE_PAIR_WITHDRAWN')
            assessment=assess_behavior_request(decision.behavior_request,context.to_dict(),p.frame_id,
                now,min(p.valid_until,decision.valid_until),'process_monotonic',decision.behavior_active_identity,
                capabilities=capabilities_dict(caps),supported_actions=LANE_CHANGE_ACTIONS,
                current_lane_id=p.lane.lane_id,frame_usable=p.valid,frame_paused=False,runtime_lane_change=True)
            require(assessment['eligible'],assessment['reason_code'])
            r,m=assessment['request'],pair.planning
            first=trajectory.points[0]
            require(first.relative_time==0 and abs(first.x-p.ego.x)<=1e-9 and abs(first.y-p.ego.y)<=1e-9
                    and abs(normalize_angle(first.heading-p.ego.heading))<=1e-9
                    and abs(first.speed-p.ego.speed)<=1e-9 and trajectory.motion_direction==1
                    and not opposes_direction(p.ego.vx,p.ego.vy,p.ego.heading,p.ego.speed),
                    'LANE_CHANGE_MONITOR_ACTUAL_START_MISMATCH')
            prepare=r['stage'] in ('PREPARE','WAIT_GAP')
            staging=prepare or r['stage']=='REQUEST_PATH'
            identity={k:r[k] for k in ('intent_id','stage','revision')}
            if staging:
                require(trajectory.behavior_identity is None and (not prepare or trajectory.candidate_id is None),
                        'LANE_CHANGE_PREPARATION_EXECUTION_IDENTITY')
            else:
                require(trajectory.behavior_identity==identity and isinstance(trajectory.candidate_id,str)
                        and trajectory.candidate_id and isinstance(trajectory.behavior_feedback,dict)
                        and trajectory.behavior_feedback.get('candidate_id')==trajectory.candidate_id
                        and (r.get('candidate_id') is None or r['candidate_id']==trajectory.candidate_id),
                        'LANE_CHANGE_MONITOR_CANDIDATE_IDENTITY')
                goal=r['goal_pose']; end=trajectory.points[-1]
                require(math.hypot(end.x-goal['x'],end.y-goal['y'])<=m.limits.distance_tolerance_m+1e-9
                        and abs(normalize_angle(end.heading-goal['body_heading_rad']))<=m.limits.heading_tolerance_rad+1e-9,
                        'LANE_CHANGE_MONITOR_ORIGINAL_GOAL_MISMATCH')
            lamps=r['light_intent']
            require((control.left_signal,control.right_signal,control.hazard_signal)==tuple(
                lamps[k] for k in ('left_signal','right_signal','hazard_signal')),
                'LANE_CHANGE_MONITOR_INDICATOR_MISMATCH')
            cap=min(r['speed_cap_mps'],decision.target_speed)
            require(all(v.speed<=max(p.ego.speed,cap)+1e-9 for v in trajectory.points),
                    'LANE_CHANGE_MONITOR_SPEED_LIMIT_MISMATCH')
            spent=0; target_deadline=p.valid_until; current_target=False
            if not staging and p.lane.lane_id==r['target_lane_id']:
                # The SDK lane label alone does not establish completion. An
                # independent whole-body proof permits target-only continuation.
                target_scene=read_maneuver_scene(p,(r['target_lane_id'],),self.clock,m.geometry_cache)
                target_deadline=target_scene['source_valid_until_s']
                probe=[TrajectoryPoint(p.ego.x,p.ego.y,0.,p.ego.heading,at) for at in (0.,1.)]
                bound=ValidationBudget(pair.monitor_budget.max_checks,pair.monitor_budget.max_depth,
                    pair.monitor_budget.min_interval_s,min(pair.monitor_budget.deadline_monotonic_s,
                    time.monotonic()+target_deadline-self.clock()))
                report=validate_candidate(probe,1,m.vehicle,m.limits,target_scene['corridor'],[],bound,
                                          execution_envelope=m.execution_envelope)
                spent=report['checks']
                current_target=report['status']=='safe'
                require(current_target or report['status']=='unsafe' and report.get('constraint')=='corridor',
                        'LANE_CHANGE_MONITOR_TARGET_BODY:'+report['reason_code'])
            ids=(p.lane.lane_id,) if staging or current_target else (r['source_lane_id'],r['target_lane_id'])
            scene=target_scene if current_target else read_maneuver_scene(p,ids,self.clock,m.geometry_cache,
                                      None if staging else r['source_lane_id'])
            scene['source_valid_until_s']=min(scene['source_valid_until_s'],target_deadline)
            predictions=_source_predictions(scene['objects'],m.prediction_envelopes,self.clock())
            region=scene['corridor'] if staging or current_target else scene['legal_crossing_corridor']
            for corridor,objects in ((region,predictions),(scene['coverage'],[])):
                budget=ValidationBudget(pair.monitor_budget.max_checks-spent,pair.monitor_budget.max_depth,
                    pair.monitor_budget.min_interval_s,min(pair.monitor_budget.deadline_monotonic_s,
                    time.monotonic()+scene['source_valid_until_s']-self.clock()))
                report=validate_candidate(trajectory.points,1,m.vehicle,m.limits,corridor,objects,budget,
                                          execution_envelope=m.execution_envelope)
                spent+=report['checks']
                if report['status']=='unsafe':
                    mode='emergency_stop'
                require(report['status']=='safe','LANE_CHANGE_MONITOR_PATH:'+report['reason_code'])
            self._read(p,context)
            require(original==(p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,p.ego.speed,p.ego.vx,p.ego.vy)
                    and self.clock()<scene['source_valid_until_s'],'LANE_CHANGE_MONITOR_SOURCE_CHANGED_OR_EXPIRED')
            self.reason='CURRENT_LANE_CHANGE_PATH_RECHECKED'
            return SafetyAssessment()
        except Exception as error:
            self.reason='LANE_CHANGE_MONITOR:'+str(error)
            return SafetyAssessment(mode,self.reason)
