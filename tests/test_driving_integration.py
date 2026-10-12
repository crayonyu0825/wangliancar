"""Formal builder -> fixed entries -> independent guard -> synthetic plant.

Only SDK I/O, authored map bytes and explicit task/model sources are replaced.
The error bounds below are fixed prerequisites of this test, not platform
calibration. Actual body clearance, feedback and receipts are checked apart
from the planner's valid flag.
"""
import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.behavior_channel import BehaviorTransport
from core.behavior_evidence import Evidence
from core.behavior_contract import GoalPose
from core.driving_mission import DrivingMission,DrivingDecisionInputs
from core.interfaces import DecisionTarget,Trajectory,ControlOut
from core.safety_supervisor import SafetySupervisor
from core.validation import validate_output
from driving_integration import DrivingInputBridge,DrivingIntegrationInputs,DRIVING_ACTIONS
from members import decision_stub,planning_stub,control_stub
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from members.control.controller import ControlEngine
from members.planning.candidate_validation import ExecutionEnvelope
from members.planning.driving_behavior import DrivingPlanningInputs
from members.planning.dynamic_speed import DynamicSearch
from members.planning.lane_planner import PlannerSettings
from members.planning.maneuver_scene import ManeuverGeometryCache
from members.planning.lane_change_generator import PredictionEnvelope
from perception.maneuver_map import ManeuverMap
from perception.perception_builder import PerceptionBuilder
from tests.control_benchmark import vehicle as control_vehicle
from tests.test_control_maneuvers import BidirectionalPlant
from tests.test_maneuver_map import map_adapter,junction_document,document
from tests.test_maneuver_perception import INSTALL,profile,perception
from tests.test_planning_candidate_validation import vehicle,limits,budget
from tests.test_planning_lane_change_generator import search
from tests.test_route_continuation import manager
from perception.map_semantics import SPEED_LIMIT_SIGN_TYPE
from simone_platform.map_observations import MapObservationReader
from tests.test_map_observations import map_api,pt,Vector
from tests.test_maneuver_map import road


class DrivingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now=100.;self.frame=0;self.targets=[];self.source='sensor:perfect';self.authority='EGO_PRIORITY'
        self.windows=((100.,250.),);self.enabled=True;self.change=None;self.p=None
        self.traffic_signs=[]
        self.passage_id='9';self.emergency_goals=();self.two_lane=False
        self.single_lane=False
        clock=lambda:self.now
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        visibility=os.path.join(temporary.name,'synthetic_visibility.json')
        template=perception();template.case_id='case';template.task_id='task'
        with open(visibility,'w',encoding='utf-8') as stream:json.dump(profile(template),stream)
        self.adapter,unused=map_adapter(junction_document())
        self.api=self.adapter.hdmap
        self.api.segments={'1_0_-1':[(0.,0.),(20.,0.)],
                           '2_0_-1':[(20.,0.),(40.,0.)],'3_0_-1':[(40.,0.),(60.,0.)]}
        self.api.successors={'1_0_-1':['2_0_-1'],'2_0_-1':['3_0_-1']}
        self.api.predecessors={'2_0_-1':['1_0_-1'],'3_0_-1':['2_0_-1']}
        self.api.neighbor=''
        self.api.getRoadST=lambda native,point:NS(exists=True,s=point[0]-{'1':0.,'2':20.,'3':40.}[native.GetString()])
        self.adapter.config=NS(vehicle_id='0',sensor_timeout_ms=500,max_sensor_frame_gap=10,
                               pipeline_timeout_ms=200,sensor_visibility_file=visibility)
        self.adapter.read_traffic=lambda lane:[]
        self.route=manager(self.api);self.route.adapter=self.adapter;self.route._maneuver_map=ManeuverMap(self.adapter)
        self.builder=PerceptionBuilder(self.adapter,self.route,scene_override=6)
        self.builder.maneuver_builder.clock=clock
        self.builder.case_id,self.builder.task_id,self.builder.case_name='case','task','authored junction'
        self.calibration=control_vehicle();self.body=vehicle(3.9,.9,.9,self.calibration.wheelbase_m)
        self.motion=limits(max_speed_mps=3.,max_acceleration_mps2=1.,max_deceleration_mps2=2.,
                           max_front_steer_rad=self.calibration.front_steer_max_rad)
        self.settings=PlannerSettings(front_offset_m=3.9,rear_offset_m=.9,half_width_m=.9,
            wheelbase_m=self.body.wheelbase_m,front_steer_max_rad=self.motion.max_front_steer_rad,
            acceleration=self.motion.max_acceleration_mps2,deceleration=self.motion.max_deceleration_mps2)
        self.cache=ManeuverGeometryCache(12,1000000)
        self.bridge=DrivingInputBridge(self.inputs,clock);self.bridge.settings=self.settings
        self.transport=BehaviorTransport(DRIVING_ACTIONS,clock)
        ds=DecisionSettings(cruise_speed=2.,front_offset_m=3.9,half_width_m=.9)
        self.engine=DecisionEngine(ds,clock,driving_inputs_provider=self.bridge.decision_inputs)
        self.control=ControlEngine(self.calibration,clock=clock)
        for target,value in (('time.monotonic',clock),('members.decision_stub._ENGINE',self.engine),
                             ('members.planning_stub._settings_with_control_geometry',lambda:self.settings),
                             ('members.control_stub._engine',self.control)):
            manager_patch=patch(target,value);manager_patch.start();self.addCleanup(manager_patch.stop)
        planning_stub.configure_planning(driving_inputs_provider=self.bridge.planning_inputs)
        self.addCleanup(planning_stub.configure_planning)
        self.safety=SafetySupervisor(NS(),clock,self.settings,ds,driving_guard=self.bridge.assess,
                                     execution_guard=self.bridge.assess_execution)
        self.plant=BidirectionalPlant(x=5.,speed=1.,lag=.2)

    def inputs(self,p,context):
        if not self.enabled:return None
        evidence=Evidence(context,p.frame_id,self.now,p.valid_until,True,True,
                          'authored task authority',source_kind='task')
        model=Evidence(context,p.frame_id,self.now,p.valid_until,True,True,
                       'authored lagged bicycle bounds',source_kind='task')
        mission=DrivingMission('fixed-junction-mission',p.maneuver_environment.map_semantics['digest'],
            evidence,model,passage_id=self.passage_id,allowed_windows_abs_s=self.windows,right_of_way=self.authority,
            speed_cap_mps=3. if self.two_lane else 2.,entry_margin_m=.3,exit_margin_m=.3,
            emergency_goals=self.emergency_goals)
        d=DrivingDecisionInputs(mission,'synthetic-driving-v1',(3.9,.9,.9),2.)
        curvature=math.tan(self.plant.steering*self.calibration.front_steer_max_rad)/self.calibration.wheelbase_m
        m=DrivingPlanningInputs(context,p.frame_id,self.now,p.valid_until,'synthetic-driving-v1',curvature,
            self.body,self.motion,search(),budget(max_checks=1000000),
            {v.id:PredictionEnvelope(90.,.02,.002) for v in p.targets},geometry_cache=self.cache,
            execution_envelope=ExecutionEnvelope(.25,.03,.2,.6),
            dynamic_search=DynamicSearch(max_nodes=5000),speed_error_mps=.5,mission=mission)
        pair=DrivingIntegrationInputs(d,m,budget(max_checks=1000000))
        if self.change:self.change(pair)
        return pair

    def build(self):
        self.frame+=1
        self.api.current=('1_0_-1' if self.single_lane else
                          ('1_0_-2' if self.plant.y>=1.75 else '1_0_-1') if self.two_lane else
                          ('1_0_-1' if self.plant.x<20. else '2_0_-1' if self.plant.x<40. else '3_0_-1'))
        raw=dict(gps=dict(frame=self.frame,timestamp=self.frame*50,x=self.plant.x,y=self.plant.y,z=0.,
                    heading=self.plant.heading,vx=self.plant.speed*math.cos(self.plant.heading),
                    vy=self.plant.speed*math.sin(self.plant.heading),steering=self.plant.steering,
                    yaw_rate=self.plant.yaw_rate,age_ms=0),targets=self.targets,targets_valid=True,
                 targets_frame=self.frame,targets_timestamp=self.frame*50,targets_age_ms=0,
                 target_source=self.source,source_status={'gps':dict(read_ok=True,frame_id=self.frame,age_ms=0),
                    'targets':dict(read_ok=True,frame_id=self.frame,age_ms=0)},
                 sensor_configurations=[dict(INSTALL,id='perfect')],sensor_configurations_valid=True)
        raw.update(traffic_signs=self.traffic_signs,traffic_signs_valid=True)
        self.p=self.builder.build_from_raw(raw)
        return self.p

    def cycle(self,check=True,sent=True):
        p=self.build();self.transport.prepare(p);self.bridge.prepare(p)
        d=decision_stub.decide(p);t=planning_stub.plan(p,d);c=control_stub.compute_control(p,t)
        safety=self.safety.evaluate(p,d,t,c)
        if check:
            self.assertTrue(p.valid,p.errors);self.assertTrue(t.valid,(d.reason,t.errors,self.bridge.reason))
            self.assertTrue(c.valid,c.errors)
            validate_output(d,DecisionTarget,p);validate_output(t,Trajectory,d);validate_output(c,ControlOut,t)
            self.assertFalse(safety.active,(safety.reason,d.reason,t.reason,self.bridge.reason,self.plant.x,self.plant.speed))
        receipt={'attempted':sent,'ok':sent,'reason':'synthetic acknowledgment'}
        self.transport.observe(p,d,t,c,safety.active,receipt)
        self.bridge.record_sent(p,t,receipt,safety.active)
        return d,t,c,safety

    def advance(self,c):
        self.plant.step(c,.05,self.calibration);self.now+=.05

    def two_lane_emergency_world(self):
        """Author the task before danger; warm native binding by actual coasting."""
        data=document()
        metadata=dict(verified=True,digest=hashlib.md5(data).hexdigest(),source='synthetic_verified_map')
        self.adapter.read_map_document=lambda:(data,dict(metadata))
        self.api.segments={'1_0_-1':[(0.,0.),(120.,0.)],'1_0_-2':[(0.,3.5),(120.,3.5)]}
        self.api.neighbor='1_0_-2';self.api.successors={};self.api.predecessors={}
        self.api.getRoadST=lambda native,point:NS(exists=True,s=point[0])
        self.two_lane=True;self.passage_id=None;self.plant.speed=3.
        # No receipt or behavior progress is fabricated while the map producer
        # binds its bounded local broken-line range. GPS follows this plant.
        for unused in range(40):
            self.build();self.advance(ControlOut())
        start_x=self.plant.x
        self.emergency_goals=({'action':'BRAKE_LEFT','goal':GoalPose(start_x+12.,2.5,0.),
                               'lane_ids':('1_0_-1','1_0_-2')},)
        self.targets=[dict(id=9,type=1,x=start_x+7.2,y=-1.1,z=0.,vx=0.,vy=0.,
            length=.4,width=.6,height=1.5,heading=0.,probability=1.)]
        return start_x

    def crosswalk_world(self):
        """A native crossing and stopline, with enough actual downstream road."""
        data=document(road(length=24))
        metadata=dict(verified=True,digest=hashlib.md5(data).hexdigest(),source='synthetic_verified_map')
        self.adapter.read_map_document=lambda:(data,dict(metadata))
        self.api.segments={'1_0_-1':[(0.,0.),(24.,0.)]}
        self.api.neighbor='';self.api.successors={};self.api.predecessors={}
        self.api.getRoadST=lambda native,point:NS(exists=True,s=point[0])
        self.single_lane=True;self.passage_id='crosswalk:5';self.plant.x=4.
        native=map_api()
        stop=NS(id=4,type='stopLine',pt=pt(10.,0.),boundaryKnots=Vector([pt(10.,-2.),pt(10.,2.)]))
        walk=NS(id=5,type='crosswalk',pt=pt(13.,0.),
                boundaryKnots=Vector([pt(12.,-2.),pt(14.,-2.),pt(14.,2.),pt(12.,2.)]))
        native.getStoplineList=lambda signal,lane:Vector([stop])
        native.getCrosswalkList=lambda signal,lane:Vector([walk])
        self.adapter.read_map_observations=MapObservationReader(native).lane_objects

    def test_real_builder_produces_selected_cells_and_generated_enter_path(self):
        d,t,c,safety=self.cycle()
        junction=self.p.maneuver_environment.junctions[0]
        self.assertTrue(junction['corridor_geometry_verified'],junction)
        self.assertEqual(['2_0_-1','3_0_-1'],[r['lane_id'] for r in junction['corridor_regions']])
        self.assertEqual('ENTER',d.behavior_request['stage'])
        self.assertTrue(t.planning_metrics['execution_robust'])
        self.assertEqual(self.p.ego.speed,t.points[0].speed)
        self.assertGreater(t.points[-1].x-.9,40.3)
        self.assertLess(t.points[-1].x+3.9,60.-.3)
        self.assertTrue(t.planning_metrics['body_occupancy_windows_s'])
        self.assertFalse(any(r['goal_pose_arrived'] or r['hold_completed'] for r in self.transport.feedbacks))

    def test_authority_unknown_produces_actual_whole_body_wait_profile(self):
        self.authority='UNKNOWN';self.windows=()
        d,t,c,safety=self.cycle()
        self.assertEqual('WAIT_ENTRY',d.behavior_request['stage'])
        self.assertTrue(t.stop_required);self.assertEqual(0.,t.points[-1].speed)
        self.assertLess(max(p.x+3.9 for p in t.points),20.-.3)

    def test_real_map_speed_sign_is_preserved_through_dynamic_retiming(self):
        self.traffic_signs=[dict(id=42,type=SPEED_LIMIT_SIGN_TYPE,value='.6',unit='m/s',valid=True,
            is_dynamic=False,heading_valid=True,heading=math.pi,x=12.,y=0.,
            validities=[dict(road_id=1,section_index=0,from_lane_id=-1,to_lane_id=-1)])]
        d,t,c,safety=self.cycle()
        self.assertTrue(self.p.speed_limit_observations[0]['applicable'])
        for x in (12.,14.,18.):
            checked=False
            for a,b in zip(t.points,t.points[1:]):
                if a.x<=x<=b.x:
                    ratio=(x-a.x)/(b.x-a.x)
                    speed=math.sqrt(a.speed*a.speed+ratio*(b.speed*b.speed-a.speed*a.speed))
                    self.assertLessEqual(speed,.6+1e-8);checked=True
            self.assertTrue(checked,x)

    def test_closed_loop_actual_body_exits_before_task_completes(self):
        phases=set();completed=False;maximum_y=0.;sent=0
        for unused in range(650):
            d,t,c,safety=self.cycle()
            request=d.behavior_request
            if request:phases.add(request['stage'])
            sent+=1
            maximum_y=max(maximum_y,abs(self.plant.y))
            # Independent straight-world body edges, not planner goal flags.
            tail=self.plant.x-.9;front=self.plant.x+3.9
            if self.engine._behavior.driving.snapshot().get('phase')=='COMPLETE':
                self.assertGreater(tail,40.+.3)
                self.assertLess(front,60.-.3)
                completed=True;break
            self.advance(c)
        self.assertTrue(completed,(self.plant.x,self.plant.speed,phases,self.engine._behavior.driving.snapshot()))
        self.assertTrue({'ENTER','EXIT'}<=phases,phases)
        self.assertGreater(sent,100);self.assertLess(maximum_y,.1)

    def test_new_candidate_cannot_hide_actual_tracking_departure_from_last_sent(self):
        d,t,c,safety=self.cycle();self.advance(c);self.plant.y=1.
        d,t,c,safety=self.cycle(check=False)
        self.assertTrue(safety.active)
        self.assertIn('EXECUTION',safety.reason)
        self.assertTrue(self.bridge.monitor.history)

    def test_ground_truth_cannot_replace_fresh_sensor_admission(self):
        self.source='ground_truth'
        d,t,c,safety=self.cycle(check=False)
        self.assertFalse(self.p.maneuver_environment.status['objects']['usable'])
        self.assertTrue(safety.active or d.target_speed==0.)
        self.assertFalse(d.behavior_request and d.behavior_request['stage']=='ENTER')

    def test_invalid_initial_explicit_mission_source_requires_protection(self):
        self.change=lambda pair:setattr(pair.planning,'model_id','foreign-motion-model')
        d,t,c,safety=self.cycle(check=False)
        self.assertEqual(0.,d.target_speed)
        self.assertTrue(safety.active)
        self.assertIsNone(d.behavior_request)

    def test_missing_prediction_protocol_reports_rejection_without_exception(self):
        self.targets=[dict(id=9,type=1,x=65.,y=8.,z=0.,vx=0.,vy=0.,length=4.5,width=1.8,
                           height=1.5,heading=0.,probability=1.)]
        self.change=lambda pair:setattr(pair.planning,'prediction_envelopes',{})
        d,t,c,safety=self.cycle(check=False)
        self.assertFalse(t.valid)
        self.assertTrue(safety.active)
        self.assertTrue(any('PREDICTION_OBJECT_SET_MISMATCH' in error for error in t.errors),t.errors)

    def test_first_emergency_cycle_brakes_before_search_or_candidate_acceptance(self):
        self.targets=[dict(id=9,type=1,x=13.,y=0.,z=0.,vx=0.,vy=0.,length=4.5,width=1.8,
                           height=1.5,heading=0.,probability=1.)]
        with patch('members.planning.driving_behavior.generate_emergency_candidates',
                   side_effect=AssertionError('search must be deferred until a later actual frame')) as generate:
            d,t,c,safety=self.cycle(check=False)
        generate.assert_not_called()
        self.assertEqual('EMERGENCY',d.behavior_request['maneuver'])
        self.assertEqual('SELECT',d.behavior_request['stage'])
        self.assertTrue(t.emergency_stop);self.assertTrue(t.hazard_signal)
        self.assertEqual(0.,c.throttle);self.assertGreater(c.brake,0.);self.assertTrue(c.hazard_signal)
        self.assertIsNone(t.behavior_feedback)
        self.assertFalse(self.bridge.monitor.history)

    def test_authorized_lateral_emergency_actually_steers_while_braking(self):
        start_x=self.two_lane_emergency_world();selected=None;goal=None;commands=[]
        for unused in range(8):
            d,t,c,safety=self.cycle()
            self.assertEqual(0.,c.throttle);self.assertGreater(c.brake,0.)
            self.assertTrue(c.hazard_signal)
            r=d.behavior_request
            if r and r['stage']=='EXECUTE_BRAKE_LEFT':
                if selected is None:selected=t.candidate_id;goal=copy.deepcopy(r['goal_pose'])
                self.assertEqual(selected,t.candidate_id);self.assertEqual(goal,r['goal_pose'])
                self.assertEqual(self.p.ego.speed,t.points[0].speed)
                self.assertGreater(max(v.y for v in t.points),2.4)
                for a,b in zip(t.points,t.points[1:]):
                    self.assertLessEqual(b.speed,a.speed+1e-9)
                    self.assertGreater(b.relative_time,a.relative_time)
                commands.append(c.steering)
            # These corners are computed directly from measured body pose.
            # The two authored lanes together occupy this explicit rectangle.
            co,si=math.cos(self.plant.heading),math.sin(self.plant.heading)
            for longitudinal in (-.9,3.9):
                for lateral in (-.9,.9):
                    x=self.plant.x+longitudinal*co-lateral*si
                    y=self.plant.y+longitudinal*si+lateral*co
                    self.assertTrue(0.<x<120.);self.assertTrue(-1.75<y<5.25)
            self.assertFalse(any(v['goal_pose_arrived'] or v['hold_completed']
                                 for v in self.transport.feedbacks))
            self.advance(c)
        self.assertIsNotNone(selected)
        self.assertGreater(max(commands),.001,commands)
        self.assertGreater(self.plant.steering,.001)
        self.assertGreater(self.plant.y,1e-4)
        self.assertGreater(self.plant.heading,.001)
        self.assertGreater(self.plant.x,start_x)
        self.assertTrue(self.bridge.monitor.history)
        self.assertFalse(self.bridge.execution_tracking['active'])
        self.assertEqual('gps_reachable_path_progress',self.transport.feedbacks[-1]['evidence_level'])

    def test_duplicate_emergency_action_uses_selected_goal_authority(self):
        start_x=self.two_lane_emergency_world()
        selected_option=self.emergency_goals[0]
        # Both options are explicit task authorities. The first uses only the
        # current road and cannot authorize the second option's lateral goal.
        self.emergency_goals=({'action':'BRAKE_LEFT','goal':GoalPose(start_x+12.,2.5,.2),
            'lane_ids':('1_0_-1',)},selected_option)
        executed=False
        for unused in range(6):
            d,t,c,safety=self.cycle(check=False)
            r=d.behavior_request
            if r and r['stage']=='EXECUTE_BRAKE_LEFT':
                self.assertEqual(2.5,r['goal_pose']['y'])
                self.assertEqual(0.,r['goal_pose']['body_heading_rad'])
                self.assertTrue(t.valid,(t.reason,t.errors))
                self.assertFalse(t.emergency_stop,(t.reason,t.behavior_feedback))
                self.assertFalse(safety.active,safety.reason)
                self.assertEqual(r['candidate_id'],t.candidate_id)
                self.assertAlmostEqual(start_x+12.,t.points[-1].x)
                self.assertAlmostEqual(2.5,t.points[-1].y)
                self.assertEqual(0.,c.throttle);self.assertGreater(c.brake,0.)
                self.assertGreater(c.steering,.001)
                executed=True
                self.advance(c)
                break
            self.advance(c)
        self.assertTrue(executed,'The second same-action goal must be selected by the real policy.')
        self.assertGreater(self.plant.steering,.001)
        self.assertGreater(self.plant.y,0.)

    def reach_early_brake_hold(self,sent_hold=True):
        self.targets=[dict(id=9,type=1,x=13.,y=0.,z=0.,vx=0.,vy=0.,length=4.5,width=1.8,
                           height=1.5,heading=0.,probability=1.)]
        frozen=None
        for unused in range(60):
            d,t,c,safety=self.cycle(sent=sent_hold if self.plant.speed==0 else True)
            if d.behavior_request and d.behavior_request['stage']=='EXECUTE_BRAKE':
                goal=d.behavior_request['goal_pose']
                if frozen is None:frozen=copy.deepcopy(goal)
                self.assertEqual(frozen,goal)
            self.assertEqual(0.,c.throttle);self.assertGreater(c.brake,0.)
            if t.planning_metrics.get('emergency_outcome')=='verified_safe_brake_stop':
                self.assertEqual(0.,self.plant.speed)
                self.assertGreater(frozen['x']-self.plant.x,self.motion.distance_tolerance_m)
                self.assertLess(self.plant.x+3.9,13.-2.25)
                self.assertFalse(safety.active);self.assertFalse(self.bridge.execution_tracking['active'])
                return frozen,d,t,c
            self.advance(c)
        self.fail(('actual safe early stop not reported',self.plant.x,self.plant.speed,
                   self.engine._behavior.driving.snapshot()))

    def test_actual_early_brake_stop_recovers_without_claiming_frozen_goal_arrival(self):
        frozen,d,t,c=self.reach_early_brake_hold()
        reply=self.transport.feedbacks[-1]
        self.assertTrue(reply['actual_brake_stop_confirmed']);self.assertTrue(reply['actual_standstill_confirmed'])
        self.assertFalse(reply['goal_pose_arrived']);self.assertFalse(reply['hold_completed'])
        self.assertEqual(0.,reply['progress']);self.assertEqual('verified_actual_brake_stop',reply['evidence_level'])
        self.assertEqual(frozen,t.planning_metrics['original_goal_pose'])
        self.assertTrue(self.bridge.monitor.history)
        self.targets=[] # A fresh complete Sensor packet now verifies the danger has cleared.
        for unused in range(5):
            self.advance(c);d,t,c,safety=self.cycle()
            session=self.engine._behavior.driving.emergency.session
            if session.status=='COMPLETED':
                self.assertEqual('DANGER_CLEARED_ACTUAL_BRAKE_STOP_RETURN_TO_TASK',session.reason_code)
                break
        self.assertEqual('COMPLETED',self.engine._behavior.driving.emergency.session.status)
        self.assertGreater(frozen['x']-self.plant.x,self.motion.distance_tolerance_m)

    def test_early_brake_hold_without_successful_send_cannot_complete_emergency(self):
        frozen,d,t,c=self.reach_early_brake_hold(sent_hold=False)
        self.assertFalse(self.transport.feedbacks[-1]['actual_brake_stop_confirmed'])
        self.targets=[]
        for unused in range(2):
            self.advance(c);d,t,c,safety=self.cycle(sent=False,check=False)
            self.assertNotEqual('COMPLETED',self.engine._behavior.driving.emergency.session.status)
            for reply in self.transport.feedbacks:
                self.assertFalse(reply['actual_brake_stop_confirmed'])

    def test_actual_emergency_hold_shares_road_and_coverage_check_budget(self):
        from members.planning.candidate_validation import validate_candidate
        from members.planning.maneuver_scene import read_maneuver_scene
        from members.planning.lane_change_generator import _source_predictions
        # No successful hold receipt: the real policy must remain in the
        # original emergency while the next frame tests its planning budget.
        unused,d,hold,c=self.reach_early_brake_hold(sent_hold=False)
        m=self.bridge.value.planning
        scene=read_maneuver_scene(self.p,(self.p.lane.lane_id,),lambda:self.now,self.cache)
        targets=_source_predictions(scene['objects'],m.prediction_envelopes,self.now)
        counts=[]
        for region,objects in ((scene['corridor'],targets),(scene['coverage'],[])):
            report=validate_candidate(hold.points,1,m.vehicle,m.limits,region,objects,
                budget(max_checks=1000000),execution_envelope=m.execution_envelope)
            self.assertEqual('safe',report['status'],report)
            counts.append(report['checks'])
        shared_limit=sum(counts)-1
        self.assertGreaterEqual(shared_limit,max(counts))
        self.change=lambda pair:setattr(pair.planning.budget,'max_checks',shared_limit)
        self.advance(c)
        d,t,c,safety=self.cycle(check=False)
        self.assertTrue(t.emergency_stop,(counts,shared_limit,t.planning_metrics,d.behavior_request,t.reason))
        self.assertEqual('REJECTED',t.behavior_feedback['status'])
        self.assertIn('BUDGET',t.behavior_feedback['reason_code'])
        self.assertTrue(t.planning_metrics['budget_exhausted'])
        self.assertGreater(t.planning_metrics['checks'],shared_limit)
        self.assertEqual(0.,c.throttle);self.assertGreater(c.brake,0.)
        self.assertFalse(self.transport.feedbacks[-1]['actual_brake_stop_confirmed'])

    def test_neighbor_vehicle_removes_lateral_option_without_erasing_front_danger(self):
        start_x=self.two_lane_emergency_world()
        self.targets.append(dict(id=10,type=1,x=start_x+7.,y=2.8,z=0.,vx=0.,vy=0.,
            length=4.5,width=1.8,height=1.5,heading=0.,probability=1.))
        d,t,c,safety=self.cycle();self.advance(c)
        d,t,c,safety=self.cycle()
        self.assertEqual({9,10},set(v['id'] for v in self.p.maneuver_environment.objects))
        options=t.behavior_feedback['candidate_options']
        self.assertEqual(['BRAKE'],[v['action'] for v in options])
        self.assertEqual(0.,c.throttle);self.assertGreater(c.brake,0.)
        self.advance(c);d,t,c,safety=self.cycle()
        self.assertEqual('EXECUTE_BRAKE',d.behavior_request['stage'])
        self.assertAlmostEqual(0.,c.steering);self.assertGreater(c.brake,0.)

    def test_emergency_search_exhaustion_keeps_real_brake_without_candidate_authorization(self):
        self.targets=[dict(id=9,type=1,x=13.,y=0.,z=0.,vx=0.,vy=0.,length=4.5,width=1.8,
                           height=1.5,heading=0.,probability=1.)]
        self.change=lambda pair:setattr(pair.planning.budget,'max_checks',1)
        for unused in range(3):
            d,t,c,safety=self.cycle(check=False)
            self.assertTrue(t.emergency_stop);self.assertEqual(0.,c.throttle)
            self.assertGreater(c.brake,0.);self.assertTrue(c.hazard_signal)
            self.assertFalse((t.behavior_feedback or {}).get('candidate_options'))
            self.assertFalse(self.bridge.monitor.history)
            if d.behavior_request:self.assertFalse(d.behavior_request['stage'].startswith('EXECUTE_'))
            self.advance(safety.control if safety.active and safety.control else c)

    def test_native_crosswalk_builder_enters_and_actual_whole_body_exits(self):
        self.crosswalk_world();stages=set();completed=False;sent=0
        for unused in range(250):
            d,t,c,safety=self.cycle();sent+=1
            request=d.behavior_request
            if request:
                self.assertEqual('CROSSWALK',request['maneuver']);stages.add(request['stage'])
            if sent==1:
                self.assertEqual('ENTER',request['stage'])
                self.assertEqual('crosswalk:5',self.p.maneuver_environment.crossing_regions[0]['id'])
                self.assertEqual('clear',self.p.maneuver_environment.crossing_regions[0]['occupancy'])
                self.assertTrue(self.p.map_crosswalks_valid);self.assertTrue(self.p.map_stop_lines_valid)
                self.assertEqual(4,self.p.maneuver_environment.crossing_regions[0]['entry_stop_line']['id'])
                self.assertTrue(t.planning_metrics['execution_robust'])
            if self.engine._behavior.driving.snapshot().get('phase')=='COMPLETE':
                self.assertGreater(self.plant.x-.9,14.+.3)
                self.assertLess(self.plant.x+3.9,24.-.3)
                completed=True;break
            self.assertFalse(any(v['goal_pose_arrived'] or v['hold_completed'] for v in self.transport.feedbacks))
            self.advance(c)
        self.assertTrue(completed,(self.plant.x,self.plant.speed,self.engine._behavior.driving.snapshot()))
        self.assertTrue({'ENTER','EXIT'}<=stages,stages);self.assertGreater(sent,50)
        self.assertAlmostEqual(0.,self.plant.y,places=5)

    def test_native_crosswalk_unknown_authority_waits_before_actual_stopline(self):
        self.crosswalk_world();self.authority='UNKNOWN';self.windows=()
        d,t,c,safety=self.cycle()
        self.assertEqual('CROSSWALK',d.behavior_request['maneuver'])
        self.assertEqual('WAIT_ENTRY',d.behavior_request['stage'])
        self.assertTrue(t.stop_required);self.assertEqual(0.,t.points[-1].speed)
        self.assertLess(max(v.x+3.9 for v in t.points),10.-.3)
        self.assertTrue(t.planning_metrics['execution_robust'])
        self.assertFalse(any(v['goal_pose_arrived'] or v['hold_completed'] for v in self.transport.feedbacks))


if __name__=='__main__':unittest.main()
