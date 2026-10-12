"""Production parser of the formal extension; fixtures are explicitly model."""
import copy
import json
import unittest
import tests.test_joint_behavior_m1 as m1
from core.behavior_protocol import behavior_frame
from core.serialization import to_dict
from perception.behavior_builder import build_behavior_environment


class ObservationExtensionTests(unittest.TestCase):
    setUp=m1.JointM1Tests.setUp

    def raw(self):
        rectangle=[(-10,-10),(80,-10),(80,30),(-10,30)]
        return {'behavior_observations':dict(clock_id='process_monotonic',frame_id=1,observed_at_s=self.now,
            valid_until_s=self.now+0.2,source='model-world',source_kind='synthetic',road_regions=[rectangle],
            dynamic_coverage=dict(coverage_complete=True,regions=[rectangle],scope=['front','side','rear'],proof_id='model-all',
                proof_kind='model_complete_world',position_uncertainty_m=.05,velocity_uncertainty_mps=.01,prediction_horizon_s=60),
            execution_timing=dict(max_early_s=.1,max_late_s=1,bounds_verified=True,source='model-bound',source_kind='synthetic'),
            task=dict(route_id='route-a',route_verified=True,source='model-task',source_kind='synthetic'),
            parking_task=dict(id='park-a',bays=[dict(id=1,boundary=[(7,-1.5),(13,-1.5),(13,1.5),(7,1.5)],
                body_heading_rad=0,occupancy='empty')],free_regions=[rectangle],exit_goal=dict(x=25,y=0,body_heading_rad=0),ego_target_id=0),
            intersection_task=dict(id='junction-a',route=[(-10,0),(70,0)],entry_s_m=30,exit_s_m=40,
                conflict_regions=[[(20,-5),(30,-5),(30,5),(20,5)]],drivable_region=rectangle,
                exit_region=[(30,-2),(80,-2),(80,2),(30,2)],permission_windows_s=[(self.now,self.now+30)],
                right_of_way='EGO_PRIORITY',signal_permits=True),
            emergency_task=dict(danger_id='danger-a',dangerous=True,allowed_regions=[rectangle],lateral_offsets_m=[0,1.5]),
            task_progress=dict(route_id='route-a',segments=[['s-a','lane-1',0,25]],segment_id='s-a',progress_s_m=0,
                reference_end_reason='task_destination',destination_region=[(23,-2),(30,-2),(30,2),(23,2)],
                destination_goal=dict(x=25,y=0,body_heading_rad=0),selected_branch_verified=True,
                required_obligations=['signal:42']))}

    def test_parser_populates_all_public_facts_without_handwritten_parking_legs(self):
        p=m1.JointM1Tests.sample(self,1);env=build_behavior_environment(p,self.raw(),None)
        frame=behavior_frame(p,self.now)
        self.assertTrue(env.parking_task.verified(frame,'model'))
        self.assertTrue(env.intersection_task.verified(frame,'model'))
        self.assertTrue(env.emergency_task.verified(frame,'model'))
        self.assertEqual(('signal:42',),env.task_progress.required_obligations)
        result=json.loads(json.dumps(to_dict(env)))
        self.assertIn('parking_task',result);self.assertIn('intersection_task',result)
        self.assertNotIn('approach_pose',result['parking_task'])

    def test_synthetic_or_expired_sources_cannot_create_production_authority(self):
        p=m1.JointM1Tests.sample(self,1);p.observation_mode='production'
        env=build_behavior_environment(p,self.raw(),None);frame=behavior_frame(p,self.now)
        self.assertFalse(env.parking_task.verified(frame,'production'))
        self.assertFalse(env.intersection_task.verified(frame,'production'))
        self.assertFalse(env.emergency_task.verified(frame,'production'))
        p.observation_mode='model';raw=self.raw();raw['behavior_observations']['valid_until_s']=self.now-.01
        env=build_behavior_environment(p,raw,None)
        self.assertFalse(env.current(frame,'model'))


if __name__=='__main__':unittest.main()
