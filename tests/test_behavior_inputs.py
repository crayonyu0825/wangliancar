"""Explicit configuration and raw GPS presence gates; all proof is authored."""
import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from behavior_inputs import ConfiguredBehaviorInputs,load_behavior_inputs
from core.maneuver_facts import ManeuverFacts
from core.behavior_contract import TaskContext
from core.config import AppConfig
from members import planning_stub,decision_stub
from runtime import CaptainRuntime
from tests.test_safety_runtime import FakeAdapter
from tests import test_lane_change_integration as fixture
from tests import test_planning_parking_behavior as parking_fixture
from tests.test_planning_parking_lattice import choices
from parking_integration import ParkingInputBridge,PARKING_ACTIONS


class BehaviorInputConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.LaneChangeIntegrationTests(); self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.f=self.h.f; self.h.transport.prepare(self.f.p)
        self.config=NS(vehicle_id='0',control_calibrated=True,
            control_wheelbase_m=self.h.h.calibration.wheelbase_m,
            control_front_steer_max_rad=self.h.h.calibration.front_steer_max_rad)
        model=self.h.inputs(self.f.p,self.h.transport.context).planning
        self.document=dict(version='behavior-inputs-v1',verified=True,
            verification_reference='AUTHORED TEST ONLY',
            scope=dict(vehicle_id='0',case_id=self.f.p.case_id,task_id=self.f.p.task_id,
                       map_digest=ManeuverFacts(self.f.p,self.f.clock).map_digest()),
            model=dict(model_id=model.model_id,verification_reference='authored bicycle/Sensor bounds',
                coordinate_frame='world_xy_metres',body_reference='rear_axle',
                gps_steering=dict(verified=True,verification_reference='authored front angle mapping',
                    scale_to_front_rad=.5,sign=1,raw_abs_max=1.),
                vehicle=dict(vars(model.vehicle)),limits=dict(vars(model.limits)),
                execution_envelope=dict(vars(model.execution_envelope)),speed_error_mps=.5,
                prediction=dict(source=self.f.p.target_source,verification_reference='authored Sensor errors',
                    horizon_s=90.,position_uncertainty_m=.02,velocity_uncertainty_mps=.002),
                planning_budget=dict(max_checks=1000000,max_depth=20,min_interval_s=.00001,max_elapsed_s=.2),
                monitor_budget=dict(max_checks=1000000,max_depth=20,min_interval_s=.00001,max_elapsed_s=.2),
                lane_search=dict(vars(model.search)),dynamic_search={},
                geometry_cache=dict(max_entries=12,preparation_max_checks=1000000)),
            overtake=dict(enabled=True,mission_id='authored-mission',return_lane_id=fixture.SOURCE,
                verification_reference='authored return route',speed_cap_mps=3.,allow_command_lights=True))
        for name in ('distances_m','tangent_scales','speed_scales'):
            self.document['model']['lane_search'][name]=list(self.document['model']['lane_search'][name])
        gps=self.f.p.source_status['gps'];gps['read_ok']=True;gps['motion_fields']=['steering','yaw_rate','vx','vy']

    def loader(self): return ConfiguredBehaviorInputs(self.config,self.document,self.f.clock)

    def parking_task(self,mission=None):
        # Explicit authored geometry; never infer a real task from these numbers.
        lattice=dict(vars(choices()))
        for k in ('step_lengths_m','curvature_fractions','permitted_directions'):
            lattice[k]=list(lattice[k])
        def pose(value):
            return {k:getattr(value,k) for k in ('x','y','body_heading_rad')}
        spec=dict(task_verified=True,source='AUTHORED task configuration',mission_id='authored-parking',
            space_id=7,road_lane_ids=[fixture.SOURCE],access_edge_index=3,body_heading_rad=0.,
            approach_pose=dict(x=15.,y=0.,body_heading_rad=0.),
            position_pose=dict(x=16.,y=0.,body_heading_rad=0.),
            exit_goal=dict(x=26.,y=0.,body_heading_rad=0.))
        if mission is not None:
            spec.update(mission_id=mission.mission_id,space_id=mission.space_id,
                road_lane_ids=list(mission.road_lane_ids),access_edge_index=mission.access_edge_index,
                body_heading_rad=mission.body_heading_rad,approach_pose=pose(mission.approach_pose),
                position_pose=pose(mission.position_pose),exit_goal=pose(mission.exit_goal))
        return dict(enabled=True,verified=True,verification_reference='AUTHORED parking only',
            mission=spec,tolerances=[.15,.15,.05],position_uncertainty_m=.1,
            velocity_uncertainty_mps=.1,geometry_max_checks=1000000,speed_cap_mps=1.,
            search=dict(tangent_scales=[1.],end_tangent_scales=[1.],speed_scales=[1.],
                spacing_m=.2,max_points=1000,terminal_straight_m=[0.],lattice=lattice))

    def test_absent_and_unverified_file_install_no_new_capabilities(self):
        empty=load_behavior_inputs(self.config,self.f.clock)
        self.assertIsNone(empty.lane_change_provider);self.assertIsNone(empty.driving_provider)
        self.assertIsNone(empty.parking_provider)
        path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'config','behavior_model.example.json')
        self.config.behavior_model_file=path
        disabled=load_behavior_inputs(self.config,self.f.clock)
        self.assertEqual('UNVERIFIED_MODEL_DISABLED',disabled.status)
        self.assertEqual((),disabled.lane_change_actions);self.assertEqual((),disabled.driving_actions)
        self.assertEqual((),disabled.parking_actions)

    def test_verified_model_needs_raw_steering_presence_scope_and_real_sensor(self):
        loader=self.loader();p=self.f.p;context=self.h.transport.context
        pair=loader.lane_change_provider(p,context)
        self.assertEqual(p.frame_id,pair.planning.frame_id)
        self.assertEqual(0.,pair.planning.initial_curvature_m_inv)
        self.assertLessEqual(pair.planning.valid_until_s,p.valid_until)
        self.assertEqual(9,list(pair.planning.prediction_envelopes)[0])
        p.source_status['gps']['motion_fields']=[]
        with self.assertRaisesRegex(ValueError,'CURRENT_STEERING_UNAVAILABLE'):
            loader.lane_change_provider(p,context)
        p.source_status['gps']['motion_fields']=['steering']
        p.ego.steering=2.
        with self.assertRaisesRegex(ValueError,'OUTSIDE_CALIBRATED_RANGE'):
            loader.lane_change_provider(p,context)
        p.ego.steering=0.;p.target_source='ground_truth'
        with self.assertRaises(ValueError):loader.lane_change_provider(p,context)

    def test_unverified_mapping_or_missing_model_bound_cannot_be_enabled(self):
        for change in (lambda d:d['model']['gps_steering'].update(verified=False),
            lambda d:d['model'].update(speed_error_mps=None),
            lambda d:d.update(verification_reference=''),
            lambda d:d['model'].update(body_reference='vehicle_center')):
            original=copy.deepcopy(self.document);change(self.document)
            with self.assertRaises(ValueError):self.loader()
            self.document=original

    def test_generic_motion_model_does_not_invent_passage_or_side_goal(self):
        self.document['overtake']['enabled']=False
        loader=self.loader();self.assertIsNone(loader.lane_change_provider)
        pair=loader.driving_provider(self.f.p,self.h.transport.context)
        self.assertIsNone(pair.decision.mission.passage_id)
        self.assertEqual('UNKNOWN',pair.decision.mission.right_of_way)
        self.assertEqual((),pair.decision.mission.allowed_windows_abs_s)
        self.assertEqual((),pair.decision.mission.emergency_goals)
        self.assertEqual(pair.decision.mission.signature(),pair.planning.mission.signature())
        self.assertEqual(pair.planning.execution_envelope.position_error_m,pair.decision.execution_position_error_m)
        self.assertEqual(pair.planning.execution_envelope.heading_error_rad,pair.decision.execution_heading_error_rad)

    def test_lane_task_action_target_and_return_are_explicit_and_frozen(self):
        task=self.document['overtake']
        default=self.loader().lane_change_provider(self.f.p,self.h.transport.context).decision
        self.assertEqual(('OVERTAKE',None,True),
                         (default.maneuver,default.target_lane_id,default.return_required))
        task.update(maneuver='MERGE',target_lane_id=fixture.TARGET,return_required=False)
        loader=self.loader()
        task['target_lane_id']='changed-after-load'
        merge=loader.lane_change_provider(self.f.p,self.h.transport.context).decision
        self.assertEqual(('MERGE',fixture.TARGET,False),
                         (merge.maneuver,merge.target_lane_id,merge.return_required))
        self.assertNotEqual(default.binding(),merge.binding())
        for target,returning in ((None,False),(fixture.TARGET,True)):
            task.update(target_lane_id=target,return_required=returning)
            with self.assertRaisesRegex(ValueError,'TASK_ACTION_OR_TARGET_INVALID'):
                self.loader()

    def test_parking_pair_uses_formal_mission_search_and_same_source_deadline(self):
        h=parking_fixture.ParkingFixedEntryTests();h.setUp();self.addCleanup(h.doCleanups)
        p=h.f.p;context=h.transport.context
        p.source_status['gps'].update(read_ok=True,motion_fields=['steering'])
        self.document['scope'].update(case_id=p.case_id,task_id=p.task_id,
            map_digest=ManeuverFacts(p,h.f.clock).map_digest())
        self.document['model']['prediction']['source']=p.target_source
        self.document['overtake']['enabled']=False
        self.document['parking']=self.parking_task(h.h.mission())
        loader=ConfiguredBehaviorInputs(self.config,self.document,h.f.clock)
        self.assertEqual(PARKING_ACTIONS,loader.parking_actions)
        bridge=ParkingInputBridge(loader.parking_provider,h.f.clock);bridge.settings=h.settings
        bridge.prepare(p)
        d=bridge.decision_inputs(p,context);m=bridge.planning_inputs(p,context)
        bay=ManeuverFacts(p,h.f.clock).parking(d.mission.space_id)
        self.assertEqual(d.mission.binding(bay),m.mission.binding(bay))
        self.assertEqual(d.tolerances,m.tolerances)
        self.assertEqual(p.frame_id,m.frame_id)
        self.assertEqual(m.produced_at_s,d.model_evidence.observed_at_s)
        self.assertLessEqual(m.valid_until_s,p.valid_until)
        self.assertEqual(m.valid_until_s,d.mission.evidence.valid_until_s)
        self.assertEqual(.2,m.search.spacing_m)
        self.assertEqual((-1,1),tuple(m.search.lattice_search.permitted_directions))
        self.assertIsNotNone(m.search.execution_envelope)
        # A task assertion cannot keep a source alive or create a live bay.
        h.f.now=p.valid_until
        with self.assertRaises(ValueError):bridge.planning_inputs(p,context)

    def test_parking_configuration_cannot_fill_unknown_task_or_bay_observations(self):
        self.document['parking']=self.parking_task()
        loader=self.loader();p=self.f.p;context=self.h.transport.context
        spaces=copy.deepcopy(p.parking_spaces)
        pair=loader.parking_provider(p,context)
        self.assertEqual(spaces,p.parking_spaces)
        with self.assertRaises(ValueError):ManeuverFacts(p,self.f.clock).parking(pair.planning.mission.space_id)
        for change in (lambda d:d['parking']['mission'].update(task_verified=False),
                lambda d:d['parking']['mission']['position_pose'].update(x=None),
                lambda d:d['parking'].update(speed_cap_mps=None),
                lambda d:d['parking']['search'].update(spacing_m=0.)):
            original=copy.deepcopy(self.document);change(self.document)
            with self.assertRaises(ValueError):self.loader()
            self.document=original

    def test_configured_model_scope_change_is_a_source_error_not_an_inactive_task(self):
        self.document['parking']=self.parking_task()
        loader=self.loader();p=self.f.p
        p.task_id='unverified-new-task'
        context=TaskContext(p.case_id,p.task_id,p.scene_id,self.h.transport.context.session_id)
        for provider in (loader.lane_change_provider,loader.driving_provider,loader.parking_provider):
            with self.assertRaisesRegex(ValueError,'TASK_SCOPE_MISMATCH'):provider(p,context)

    def test_standard_runtime_installs_explicit_file_and_shared_body_geometry(self):
        self.document['parking']=self.parking_task()
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        path=os.path.join(directory.name,'model.json')
        with open(path,'w',encoding='utf-8') as stream:json.dump(self.document,stream)
        cal=self.h.h.calibration
        values=dict(behavior_model_file=path,control_calibrated='true',
            control_wheelbase_m=str(cal.wheelbase_m),control_front_steer_max_rad=str(cal.front_steer_max_rad),
            control_steering_sign=str(cal.steering_sign),control_throttle_per_mps='0.1',
            control_brake_per_mps='0.2',control_hold_brake='0.1',control_emergency_brake='0.8',
            control_max_throttle='0.25',control_max_brake='0.35')
        config=AppConfig(directory.name,values)
        logger=NS(warning=lambda *args:None,info=lambda *args:None)
        with patch('runtime.SimOneAdapter',lambda cfg,log:FakeAdapter(1)):
            runtime=CaptainRuntime(config,logger)
        self.assertIsNotNone(runtime.lane_changes);self.assertIsNotNone(runtime.driving)
        self.assertIsNotNone(runtime.parking)
        self.assertTrue(set(PARKING_ACTIONS)<=set(runtime.behaviors.actions))
        self.assertTrue(set(fixture.LANE_CHANGE_CAPABILITIES)<=set(runtime.behaviors.actions))
        self.assertEqual(3.9,decision_stub.decision_settings().front_offset_m)
        self.assertFalse(runtime.config.send_control);self.assertFalse(runtime.config.safety_brake_enabled)
        self.assertEqual(.9,planning_stub._CONTROL_GEOMETRY['rear_offset_m'])
        self.addCleanup(decision_stub.reset_decision)


if __name__=='__main__':unittest.main()
