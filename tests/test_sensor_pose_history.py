"""Exact SDK headers and historical GPS poses through the production builder."""
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from perception.maneuver_environment import ManeuverEnvironmentBuilder
from perception.sensor_pose_history import MAX_POSES
from tests.test_maneuver_perception import perception,profile
from core.region_geometry import inside
from core.serialization import perception_to_dict


class Clock(object):
    def __init__(self): self.now=100.
    def __call__(self): return self.now


def headers(p,frame,timestamp,x=10.,heading=0.):
    p.frame_id,p.timestamp=frame,timestamp
    p.ego.frame_id,p.ego.timestamp=frame,timestamp
    p.ego.x,p.ego.heading=x,heading
    p.targets_frame_id,p.targets_timestamp=frame,timestamp
    p.valid_until=10000.
    p.source_status['gps'].update(frame_id=frame,timestamp=timestamp,clock='sdk_frame',age_ms=0)
    p.source_status['targets'].update(frame_id=frame,timestamp=timestamp,clock='sdk_frame',age_ms=0)
    return p


def old_packet(p,frame,timestamp):
    p.targets_frame_id,p.targets_timestamp=frame,timestamp
    p.source_status['targets'].update(frame_id=frame,timestamp=timestamp,clock='sdk_frame',age_ms=0)


class SensorPoseHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.path=os.path.join(self.directory.name,'visibility.json')
        self.clock=Clock()

    def builder(self,p,route=None):
        model=profile(p); model['profiles'][0]['body_region_m']=[[-10.,-5.],[10.,-5.],[10.,5.],[-10.,5.]]
        with open(self.path,'w',encoding='utf-8') as stream: json.dump(model,stream)
        return ManeuverEnvironmentBuilder(route or NS(),NS(vehicle_id='0',sensor_timeout_ms=500,
            sensor_visibility_file=self.path),self.clock)

    def seed(self):
        p=headers(perception(),1,50); builder=self.builder(p); builder.build(p)
        self.clock.now=100.1
        headers(p,2,100,x=60.)
        old_packet(p,1,50)
        return builder,p

    def test_old_packet_uses_saved_pose_and_original_observation_deadline(self):
        builder,p=self.seed(); env=builder.build(p)
        self.assertTrue(env.coverage_regions)
        region=env.coverage_regions[0]
        self.assertTrue(inside((10.,0.),region['polygon']))
        self.assertFalse(inside((60.,0.),region['polygon']))
        self.assertEqual(1,region['pose_frame_id'])
        self.assertEqual('sensor_pose_history_exact_frame',region['pose_source'])
        self.assertEqual(100.,env.dynamic_observed_at_s)
        self.assertEqual(100.5,env.dynamic_valid_until)
        self.assertEqual(60.,p.ego.x)

    def test_saved_heading_rotates_old_coverage_and_not_current_heading(self):
        p=headers(perception(),1,50,heading=math.pi/2); builder=self.builder(p); builder.build(p)
        self.clock.now=100.1; headers(p,2,100,x=60.)
        old_packet(p,1,50); env=builder.build(p)
        self.assertTrue(inside((10.,9.),env.coverage_regions[0]['polygon']))
        self.assertFalse(inside((19.,0.),env.coverage_regions[0]['polygon']))

    def test_repeat_never_rejuvenates_pose_when_reported_age_resets(self):
        builder,p=self.seed()
        self.clock.now=100.2; headers(p,1,50); builder.build(p)
        self.clock.now=100.3; headers(p,2,100,x=60.); old_packet(p,1,50)
        env=builder.build(p)
        self.assertEqual(100.,env.dynamic_observed_at_s)
        self.assertEqual(100.5,env.dynamic_valid_until)

    def test_expired_pose_cannot_be_reseeded_by_same_frame(self):
        builder,p=self.seed()
        self.clock.now=100.6; headers(p,1,50); builder.build(p)
        self.clock.now=100.7; headers(p,2,100,x=60.); old_packet(p,1,50)
        env=builder.build(p)
        self.assertFalse(env.coverage_regions)

    def test_missing_frame_is_unknown_without_interpolation(self):
        builder,p=self.seed(); old_packet(p,0,25)
        env=builder.build(p)
        self.assertFalse(env.coverage_regions)
        self.assertEqual('sensor_pose_history_unavailable',env.status['coverage']['reason'])

    def test_header_time_unknown_mismatch_and_different_clock_are_rejected(self):
        for key,value in (('timestamp',0),('timestamp',49),('clock','host_query'),('frame_id',2)):
            builder,p=self.seed(); p.source_status['targets'][key]=value
            if key=='timestamp': p.targets_timestamp=value
            env=builder.build(p)
            self.assertFalse(env.coverage_regions,key)
            self.assertEqual('sensor_pose_history_time_mismatch',env.status['coverage']['reason'])
            self.clock.now=100.

    def test_gps_without_verified_sdk_header_does_not_seed_old_pose(self):
        p=headers(perception(),1,50); p.source_status['gps']['clock']='host_query'
        builder=self.builder(p); builder.build(p)
        self.clock.now=100.1; headers(p,2,100,x=60.); old_packet(p,1,50)
        self.assertFalse(builder.build(p).coverage_regions)

    def test_actual_perception_builder_keeps_history_across_raw_snapshots(self):
        from perception.perception_builder import PerceptionBuilder
        from tests.test_perception import FakeAdapter,FakeRouteManager
        p=headers(perception(),1,50); self.builder(p)  # Write the synthetic profile.
        adapter=FakeAdapter(); adapter.config=NS(vehicle_id='0',sensor_timeout_ms=500,
            sensor_visibility_file=self.path,pipeline_timeout_ms=200,max_sensor_frame_gap=10)
        builder=PerceptionBuilder(adapter,FakeRouteManager(),scene_override=7)
        builder.case_id,builder.task_id='case','task'
        builder.maneuver_builder.clock=self.clock
        def raw(frame,timestamp,x,target_frame,target_timestamp):
            return dict(gps=dict(frame=frame,timestamp=timestamp,x=x,y=0.,z=0.,heading=0.,vx=0.,vy=0.,age_ms=0),
                targets=[],targets_valid=True,target_source='sensor:perfect',
                targets_frame=target_frame,targets_timestamp=target_timestamp,targets_age_ms=0,
                sensor_configurations=p.sensor_configurations,sensor_configurations_valid=True,
                source_status=dict(gps=dict(read_ok=True,frame_id=frame,timestamp=timestamp,age_ms=0,clock='sdk_frame'),
                    targets=dict(read_ok=True,frame_id=target_frame,timestamp=target_timestamp,age_ms=0,clock='sdk_frame')))
        with patch('perception.perception_builder.time.monotonic',self.clock):
            first=builder.build_from_raw(raw(1,50,10.,1,50))
            self.assertTrue(first.maneuver_environment.coverage_regions,(first.errors,first.maneuver_environment.status))
            self.clock.now=100.1
            second=builder.build_from_raw(raw(2,100,60.,1,50))
        region=second.maneuver_environment.coverage_regions[0]
        self.assertTrue(inside((10.,0.),region['polygon']))
        self.assertFalse(inside((60.,0.),region['polygon']))
        saved=json.loads(json.dumps(perception_to_dict(second),allow_nan=False))
        self.assertEqual('sensor_pose_history_exact_frame',saved['maneuver_environment']['coverage_regions'][0]['pose_source'])

    def test_installation_change_cannot_borrow_history_for_old_packet(self):
        builder,p=self.seed(); p.sensor_configurations[0]['x']=1.
        env=builder.build(p)
        self.assertFalse(env.coverage_regions)
        self.assertEqual('sensor_pose_history_installation_mismatch',env.status['coverage']['reason'])

    def test_duplicate_frame_with_different_pose_is_permanently_ambiguous(self):
        builder,p=self.seed(); headers(p,1,50,x=15.); builder.build(p)
        headers(p,2,100,x=60.); old_packet(p,1,50)
        env=builder.build(p)
        self.assertFalse(env.coverage_regions)
        self.assertEqual('sensor_pose_history_conflicting',env.status['coverage']['reason'])

    def test_case_task_scene_changes_do_not_reuse_history(self):
        for field,value in (('case_id','other'),('task_id','other'),('scene_id',8)):
            builder,p=self.seed(); setattr(p,field,value)
            self.assertFalse(builder.build(p).coverage_regions,field)
            self.clock.now=100.

    def test_regressed_gps_clears_history_and_retains_high_water_mark(self):
        builder,p=self.seed(); headers(p,3,150); builder.build(p)
        headers(p,2,100); old_packet(p,1,50)
        self.assertFalse(builder.build(p).coverage_regions)
        self.assertEqual(3,builder.pose_history.high_frame)
        self.assertEqual({},builder.pose_history.poses)

    def test_clock_regression_and_other_height_plane_withdraw_history(self):
        builder,p=self.seed(); self.clock.now=99.
        self.assertFalse(builder.build(p).coverage_regions)
        self.clock.now=100.; builder,p=self.seed(); p.ego.z=1.
        env=builder.build(p)
        self.assertFalse(env.coverage_regions)
        self.assertEqual('sensor_pose_history_plane_mismatch',env.status['coverage']['reason'])

    def test_history_is_bounded_and_missing_evicted_frame_stays_unknown(self):
        p=headers(perception(),1,50); builder=self.builder(p)
        for frame in range(1,MAX_POSES+3):
            headers(p,frame,frame*50); builder.build(p)
        self.assertEqual(MAX_POSES,len(builder.pose_history.poses))
        old_packet(p,1,50)
        self.assertFalse(builder.build(p).coverage_regions)

    def test_original_gps_age_also_bounds_old_source(self):
        p=headers(perception(),1,50); p.source_status['gps']['age_ms']=200
        builder=self.builder(p); builder.build(p)
        self.clock.now=100.1; headers(p,2,100,x=60.); old_packet(p,1,50)
        env=builder.build(p)
        self.assertAlmostEqual(99.8,env.dynamic_observed_at_s)
        self.assertAlmostEqual(100.3,env.dynamic_valid_until)

    def test_map_free_regions_use_old_visibility_and_old_deadline(self):
        from tests.test_local_road_regions import LocalRoadRegionTests
        unused,unused,route,p=LocalRoadRegionTests().fixture()
        headers(p,1,50); builder=self.builder(p,route); builder.build(p)
        self.clock.now=100.1; headers(p,2,100,x=60.); old_packet(p,1,50)
        p.lane=route.update(p.ego,True)
        env=builder.build(p)
        cells=[v for v in env.free_regions if v['kind']=='lane']
        # The current local crop is ahead of the old visibility window. It
        # cannot be declared clear using the new vehicle pose.
        self.assertFalse(cells)
        self.assertFalse(any(v['coverage_verified'] for v in env.road_regions))

    def test_height_tolerances_do_not_stack_between_old_pose_current_pose_and_map(self):
        from perception.road_regions import local_region
        builder,p=self.seed(); p.ego.z=.09
        geometry=local_region([(0.,0.,.18),(75.,0.,.18)],
            [(0.,1.75,.18),(75.,1.75,.18)],[(0.,-1.75,.18),(75.,-1.75,.18)],
            [(0.,0.,.18),(75.,0.,.18)])
        geometry.update(lane_id=p.lane.lane_id,role='current',geometry_valid=True,drivable_verified=True)
        builder.route_manager=NS(read_maneuver_map=lambda *a:(dict(semantic_verified=True,road_regions=[geometry]),[]))
        p.parking_spaces=[dict(id=3,boundary_knots=[(5.,-1.,.18),(15.,-1.,.18),(15.,1.,.18),(5.,1.,.18)])]
        env=builder.build(p)
        self.assertTrue(env.coverage_regions)
        self.assertEqual(0.,env.coverage_regions[0]['reference_z_m'])
        self.assertFalse(env.road_regions[0]['coverage_verified'])
        self.assertEqual('unknown',env.parking_spaces[0]['occupancy'])
        self.assertFalse(env.free_regions)


if __name__=='__main__': unittest.main()
