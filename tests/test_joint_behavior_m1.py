"""Explicit model world through real fixed member entries; no output mocks."""
import math
import os
import unittest
from unittest.mock import patch

from core.behavior_protocol import TaskContext, RuntimeRequest, PlanningFeedback
from core.behavior_environment import (Evidence, LaneCorridor, ObservationCoverage, BehaviorEnvironment)
from core.behavior_exchange import BehaviorExchange
from core.interfaces import DecisionTarget, Trajectory, ControlOut
from core.validation import validate_output
from members import decision_stub, planning_stub, control_stub
from members.control.controller import ControlEngine
from members.decision.settings import DecisionSettings
from tests.control_benchmark import Plant, vehicle
from tests.test_decision import perception, add_target


class JointM1Tests(unittest.TestCase):
    def setUp(self):
        self.now=1000.0
        self.clock=patch('time.monotonic',lambda:self.now); self.clock.start(); self.addCleanup(self.clock.stop)
        self.overrides=patch.dict(os.environ,{
            'NEVC_VEHICLE_FRONT_OFFSET_M':'3.5','NEVC_VEHICLE_REAR_OFFSET_M':'0.9',
            'NEVC_VEHICLE_HALF_WIDTH_M':'0.9','NEVC_VEHICLE_WHEELBASE_M':'2.7',
            'NEVC_VEHICLE_FRONT_STEER_MAX_RAD':'0.5'})
        self.overrides.start(); self.addCleanup(self.overrides.stop)
        decision_stub.reset_decision(DecisionSettings(front_offset_m=3.5,half_width_m=0.9,cruise_speed=5.0))
        planning_stub.configure_planning(None)
        self.previous_engine=control_stub._engine
        control_stub._engine=ControlEngine(vehicle())
        self.addCleanup(setattr,control_stub,'_engine',self.previous_engine)
        self.exchange=BehaviorExchange(); self.plant=Plant(speed=1.4)
        self.lead_x=19.0; self.records=[]

    def sample(self,index,mode='model',covered=True,return_required=True):
        p=perception(speed=self.plant.speed,frame_id=index,ttl=0.2)
        p.case_id=p.case_name='joint-model'; p.task_id='model-route'; p.scene_id=17
        p.session_id='explicit-model-session'; p.observation_mode=mode; p.observed_at_s=self.now
        p.ego.frame_id=index; p.ego.timestamp=p.timestamp; p.ego.age_ms=0
        p.ego.x,p.ego.y,p.ego.heading=self.plant.x,self.plant.y,self.plant.heading
        p.ego.yaw_rate=self.plant.yaw_rate
        p.ego.vx,p.ego.vy=self.plant.speed*math.cos(self.plant.heading),self.plant.speed*math.sin(self.plant.heading)
        ctx=TaskContext(p.case_id,p.task_id,p.scene_id,p.session_id)
        proof=Evidence(ctx,index,self.now,self.now+0.2,True,covered,'explicit_model_world',source_kind='synthetic')
        lanes=[]
        current_y=0.0 if self.plant.y<1.75 else 3.5
        for lane_id,y in (('lane-1',0.0),('lane-2',3.5)):
            side='current' if y==current_y else 'left' if y>current_y else 'right'
            lanes.append(LaneCorridor(lane_id,[(-100,y),(400,y)],
                [(-100,y+1.75),(400,y+1.75)],[(-100,y-1.75),(400,y-1.75)],
                proof,side,True,'BROKEN',(0,500)))
        current=next(v for v in lanes if v.side=='current')
        p.lane.lane_id=current.lane_id; p.lane.center_line=current.center_line
        p.lane.left_boundary=current.left_boundary; p.lane.right_boundary=current.right_boundary
        target=add_target(p,self.lead_x-self.plant.x,1.4,lane_id='lane-1')
        target.x,target.y=self.lead_x,0.0
        coverage=ObservationCoverage(proof,[[(-95,-8),(395,-8),(395,9),(-95,9)]],
            ('front','side','rear'),'model-all-actors','model_complete_world',0.05,0.01,60.0,0.1,1.0,proof)
        p.behavior_environment=BehaviorEnvironment(lanes,coverage,
            [[(-90,-1.75),(390,-1.75),(390,5.25),(-90,5.25)]],
            'lane-1' if return_required else None,'model-route',task_evidence=proof)
        return p

    def step(self,index,**kwargs):
        p=self.sample(index,**kwargs)
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        decision=decision_stub.decide(p)
        self.assertTrue(decision.valid,decision.reason)
        validate_output(decision,DecisionTarget,p)
        trajectory=planning_stub.plan(p,decision)
        self.assertTrue(trajectory.valid,(index,decision.reason,trajectory.reason))
        validate_output(trajectory,Trajectory,decision)
        control=control_stub.compute_control(p,trajectory)
        self.assertTrue(control.valid,(index,control.errors,trajectory.reason))
        validate_output(control,ControlOut,trajectory)
        self.exchange.finish(p,trajectory,control,{'attempted':True,'ok':True})
        request=decision.behavior_request
        record=(index,round(self.plant.x,2),round(self.plant.y,2),round(self.plant.speed,2),
                None if request is None else (request.stage,request.authorization),
                [v.reason_code for v in trajectory.planning_feedback])
        self.records.append(record)
        self.plant.step(control,0.05,vehicle()); self.lead_x+=1.4*0.05; self.now+=0.05
        return p,decision,trajectory,control

    def test_evaluate_candidate_is_feedback_only_and_not_sent_to_control(self):
        seen=False
        for i in range(1,140):
            p,d,t,c=self.step(i)
            if isinstance(d.behavior_request,RuntimeRequest) and d.behavior_request.authorization=='EVALUATE_ONLY':
                self.assertIsNone(t.execution_path_id)
                self.assertTrue(all(abs(q.y)<1e-6 for q in t.points))
                self.assertTrue(c.left_signal)
                if any(v.status=='PLANNED' for v in t.planning_feedback):
                    seen=True; break
        self.assertTrue(seen,self.records[-20:])

    def test_expired_and_unmatched_evaluation_cannot_be_adopted(self):
        import copy
        initial=None
        for index in range(1,140):
            p,d,t,c=self.step(index)
            if any(v.status=='PLANNED' for v in t.planning_feedback):
                initial=index
                break
        self.assertIsNotNone(initial)
        p=self.sample(initial+1)
        self.exchange.prepare(p,planning_stub.planning_capabilities(p))
        decision=decision_stub.decide(p)
        self.assertEqual('EXECUTE_SELECTED',decision.behavior_request.authorization)
        for field,value in (('candidate_id','absent'),('valid_until_s',self.now-1.0)):
            changed=copy.deepcopy(decision)
            setattr(changed.behavior_request.candidate_reference,field,value)
            out=planning_stub.plan(p,changed)
            self.assertIsNone(out.execution_path_id)
            self.assertTrue(all(abs(q.y)<1e-6 for q in out.points))
            self.assertEqual('REJECTED',out.planning_feedback[0].status)

    def test_red_light_does_not_start_an_overtake(self):
        from tests.test_decision import add_red_light
        for index in range(1,100):
            p=self.sample(index)
            add_red_light(p,25.0)
            self.exchange.prepare(p,planning_stub.planning_capabilities(p))
            decision=decision_stub.decide(p)
            self.assertIsNone(decision.behavior_request)
            self.assertGreaterEqual(decision.stop_distance,0.0)
            self.now+=0.05

    def test_unknown_coverage_and_synthetic_production_sources_never_authorize(self):
        for mode,covered in (('model',False),('production',True)):
            control_stub._engine.reset()
            self.exchange=BehaviorExchange()
            decision_stub.reset_decision(DecisionSettings(front_offset_m=3.5,half_width_m=0.9,cruise_speed=5.0))
            for i in range(1,100):
                _,decision,trajectory,_=self.step(i,mode=mode,covered=covered)
                self.assertIsNone(decision.behavior_request)
                self.assertIsNone(trajectory.execution_path_id)
            self.assertLess(abs(self.plant.y),0.01)

    def test_real_entries_physically_overtake_and_return_to_task_lane(self):
        changed=False
        minimum_follow_gap=float('inf')
        for i in range(1,2001):
            p,d,t,c=self.step(i)
            changed=changed or self.plant.y>2.8
            if abs(self.plant.y)<1.8 and self.plant.x<self.lead_x:
                minimum_follow_gap=min(minimum_follow_gap,self.lead_x-2-self.plant.x-3.5)
            if (changed and self.plant.x-0.9>self.lead_x+2+10.5 and abs(self.plant.y)<0.25
                    and abs(self.plant.heading)<0.1
                    and decision_stub.decision_behavior_info()['runtime_behavior']['phase']=='COMPLETE'):
                break
        self.assertTrue(changed,self.records[-30:])
        self.assertGreater(self.plant.x-0.9,self.lead_x+12.5,self.records[-30:])
        self.assertLess(abs(self.plant.y),0.25,self.records[-30:])
        self.assertGreater(self.plant.speed,1.4)
        self.assertGreaterEqual(minimum_follow_gap,10.5-0.15)
        self.assertEqual('COMPLETE',decision_stub.decision_behavior_info()['runtime_behavior']['phase'])
        self.results=dict(minimum_follow_gap_m=minimum_follow_gap,elapsed_s=(i*0.05),
                          passed_margin_m=self.plant.x-0.9-self.lead_x-2,final_lateral_m=self.plant.y)


if __name__=='__main__':
    unittest.main()

class RuntimeModelM1Tests(unittest.TestCase):
    def test_real_builder_runtime_supervisor_sender_physically_overtake(self):
        import tempfile
        from types import SimpleNamespace
        from core.config import load_config
        from runtime import CaptainRuntime
        from perception.perception_builder import PerceptionBuilder
        from perception.route_manager import RouteManager
        from tests.test_pipeline_integration import SimulatedSDK,Vector

        class WorldSDK(SimulatedSDK):
            def __init__(self,config):
                super(WorldSDK,self).__init__(config,scene=17,frames=1450,sensor=True,target_velocity=1.4,initial_offset=0)
                self.x,self.heading,self.speed=0.0,0.0,1.4
            def targets(self,vehicle,sensor,data):
                ok=super(WorldSDK,self).targets(vehicle,sensor,data)
                if data.objects:
                    data.objects[0].posX=19.0+1.4*self.reads*0.05
                    data.objects[0].oriZ=0.0
                return ok
            def world(self,snapshot):
                lanes=[]
                current_y=0.0 if self.y<1.75 else 3.5
                for identity,y in (('1_0_-1',0.0),('1_0_-2',3.5)):
                    lanes.append(dict(lane_id=identity,center_line=[(-100,y),(400,y)],
                        left_boundary=[(-100,y+1.75),(400,y+1.75)],right_boundary=[(-100,y-1.75),(400,y-1.75)],
                        side='current' if y==current_y else 'left' if y>current_y else 'right',
                        same_direction=True,marking='BROKEN',crossing_range_m=[0,500],geometry_verified=True,
                        source='explicit_model_map',source_kind='synthetic'))
                return dict(clock_id='process_monotonic',frame_id=self.reads,observed_at_s=self.now,valid_until_s=self.now+0.2,
                    source='explicit_model_world',source_kind='synthetic',lanes=lanes,
                    dynamic_coverage=dict(coverage_complete=True,regions=[[(-95,-8),(395,-8),(395,9),(-95,9)]],
                        scope=['front','side','rear'],proof_id='model-all-actors',proof_kind='model_complete_world',
                        position_uncertainty_m=0.05,velocity_uncertainty_mps=0.01,prediction_horizon_s=60.0),
                    execution_timing=dict(max_early_s=0.1,max_late_s=1.0,bounds_verified=True,
                        source='explicit_model_control_bound',source_kind='synthetic'),
                    road_regions=[[(-90,-1.75),(390,-1.75),(390,5.25),(-90,5.25)]],
                    task=dict(route_id='model-route',return_lane_id='1_0_-1',route_verified=True,
                        source='explicit_model_task',source_kind='synthetic'))
            def install(self,adapter):
                super(WorldSDK,self).install(adapter)
                adapter.is_model_adapter=True
                adapter.behavior_observation_provider=self.world
                def line(y):
                    return Vector([SimpleNamespace(x=x,y=y,z=0.0) for x in (-100.0,400.0)])
                def sample(identity):
                    y=0.0 if str(identity)=='1_0_-1' else 3.5
                    return SimpleNamespace(exists=True,laneInfo=SimpleNamespace(
                        centerLine=line(y),leftBoundary=line(y+1.75),rightBoundary=line(y-1.75)))
                adapter.hdmap.getLaneSample=sample
                adapter.hdmap.getNearMostLane=lambda point:SimpleNamespace(exists=True,laneId='1_0_-1' if point[1]<1.75 else '1_0_-2')

        root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config=load_config(root); config.scene_id_override=17
        config.send_control=True; config.safety_brake_enabled=True; config.behavior_model_mode=True
        config.control_wheelbase_m=2.7; config.control_front_steer_max_rad=0.5
        logger=SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)
        env={'NEVC_VEHICLE_FRONT_OFFSET_M':'3.5','NEVC_VEHICLE_REAR_OFFSET_M':'0.9',
             'NEVC_VEHICLE_HALF_WIDTH_M':'0.9','NEVC_VEHICLE_WHEELBASE_M':'2.7',
             'NEVC_VEHICLE_FRONT_STEER_MAX_RAD':'0.5','NEVC_DECISION_CRUISE_SPEED':'5.0'}
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,env):
            config.runtime_dir=directory; sdk=WorldSDK(config)
            with patch('time.monotonic',lambda:sdk.now):
                runtime=CaptainRuntime(config,logger); sdk.install(runtime.adapter)
                builder=PerceptionBuilder(runtime.adapter,RouteManager(runtime.adapter,logger),17,model_mode=True)
                builder.case_id=builder.case_name='explicit-runtime-model'; builder.task_id='model-route'
                with patch.object(runtime,'_sleep_remaining',return_value=None):
                    runtime._loop(builder,False)
        self.assertTrue(any(y>2.8 for x,y,v in sdk.positions),sdk.positions[-5:])
        self.assertGreater(sdk.x-0.9,19.0+1.4*sdk.reads*0.05+12.5)
        self.assertLess(abs(sdk.y),0.25)
        self.assertTrue(any(item['signalLights'] for item in sdk.lights))
        self.assertTrue(any(command['throttle']>0 for command in sdk.sent))
