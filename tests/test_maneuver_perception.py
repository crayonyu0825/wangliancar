"""Real perception/SDK-shaped map chain; visibility models are synthetic."""
import copy
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS

from core.interfaces import Perception, Target, ManeuverEnvironment
from core.region_geometry import convex_polygon,simple_outline
from core.serialization import perception_to_dict
from perception.maneuver_environment import ManeuverEnvironmentBuilder
from perception.perception_builder import PerceptionBuilder
from scripts.replay_decision import _assign
from simone_platform.map_observations import parking_record
from tests.test_map_observations import parking, map_api
from tests.test_route_continuation import Map, NativeId, Vector, manager


INSTALL = dict(type="7",x=0.,y=0.,z=1.,roll=0.,pitch=0.,yaw=0.,hz=20)


def perception():
    p = Perception()
    p.valid,p.frame_id,p.timestamp,p.valid_until = True,1,50,100.2
    p.case_id,p.task_id,p.scene_id = "case","task",7
    p.ego.valid,p.ego.x,p.ego.frame_id,p.ego.age_ms = True,10.,1,0
    p.target_source,p.targets_valid,p.targets_frame_id,p.targets_age_ms = "sensor:perfect",True,1,0
    p.source_status = dict(gps=dict(usable=True,age_ms=0),targets=dict(usable=True,age_ms=0))
    p.sensor_configurations_valid = True
    p.sensor_configurations = [dict(INSTALL,id="perfect")]
    p.parking_spaces,p.parking_spaces_valid = [parking_record(parking())],True
    p.lane.valid,p.lane.lane_id,p.lane.lane_width = True,"1_0_-1",3.5
    p.lane.center_line = [(0.,0.,0.),(120.,0.,0.)]
    return p


def target(x=22.5,y=4.5,heading=0.,identifier=9):
    t = Target()
    t.id,t.x,t.y,t.heading,t.valid = identifier,x,y,heading,True
    t.length,t.width,t.source = 4.5,1.8,"sensor:perfect"
    return t


def profile(p):
    return dict(version="sensor-visibility-v1",profiles=[dict(
        task_id=p.task_id,case_id=p.case_id,vehicle_id="0",sensor_id="perfect",
        installation=copy.deepcopy(INSTALL),complete_detections=True,
        verification_reference="synthetic unoccluded Sensor model, not platform calibration",
        body_region_m=[[-100.,-20.],[150.,-20.],[150.,20.],[-100.,20.]])])


class LaneMap(Map):
    def __init__(self,opposite=False,reversed_sample=False):
        self.neighbor = "1_0_1" if opposite else "1_0_-2"
        points = [(0.,3.5),(120.,3.5)]
        if reversed_sample:
            points.reverse()
        super(LaneMap,self).__init__({"1_0_-1":[(0.,0.),(120.,0.)],self.neighbor:points},current="1_0_-1")
        self.pySimString = NativeId
        self.mark,self.shift,self.height,self.kind = "broken",0.,0.,"driving"

    def getLaneSample(self,native):
        sample = super(LaneMap,self).getLaneSample(native)
        if not sample.exists:
            return sample
        identity = self.identity(native)
        points = self.segments[identity]
        reverse = points[0][0]>points[-1][0]
        sign = -1 if reverse else 1
        offset = self.shift if identity==self.neighbor else 0.
        z = self.height if identity==self.neighbor else 0.
        sample.laneInfo.centerLine = Vector([NS(x=x,y=y,z=z) for x,y in points])
        sample.laneInfo.leftBoundary = Vector([NS(x=x,y=y+sign*1.75+offset,z=z) for x,y in points])
        sample.laneInfo.rightBoundary = Vector([NS(x=x,y=y-sign*1.75+offset,z=z) for x,y in points])
        return sample

    def getLaneLink(self,native):
        link = super(LaneMap,self).getLaneLink(native)
        if self.identity(native)=="1_0_-1":
            link.laneLink.leftNeighborLaneId = NativeId(self.neighbor)
        return link

    def getLaneWidth(self,*args):
        return NS(exists=True,width=3.5)

    def getLaneType(self,*args):
        return NS(exists=True,laneType=self.kind)

    def getRoadMark(self,*args):
        return NS(exists=True,left=NS(type=self.mark),right=NS(type="solid"))


class ManeuverPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name,"visibility.json")

    def build(self,p=None,model=None,route=None):
        p = p or perception()
        if model is not None:
            with open(self.path,"w",encoding="utf-8") as stream:
                json.dump(model,stream)
        config = NS(vehicle_id="0",sensor_timeout_ms=500,sensor_visibility_file=self.path if model is not None else "")
        return ManeuverEnvironmentBuilder(route or NS(),config,clock=lambda:100.).build(p)

    def test_empty_packet_does_not_prove_empty_space(self):
        env = self.build()
        self.assertTrue(env.valid)
        self.assertEqual("unknown",env.parking_spaces[0]["occupancy"])
        self.assertFalse(env.coverage_regions)
        self.assertFalse(env.free_regions)

    def test_visible_body_proves_occupied_without_full_coverage(self):
        p = perception()
        p.targets = [target()]
        env = self.build(p)
        space = env.parking_spaces[0]
        self.assertEqual("occupied",space["occupancy"])
        self.assertEqual([9],space["blocking_object_ids"])
        self.assertFalse(space["coverage_verified"])

    def test_full_matched_model_and_empty_fresh_packet_prove_clear(self):
        p = perception()
        env = self.build(p,profile(p))
        self.assertEqual("empty",env.parking_spaces[0]["occupancy"])
        self.assertEqual(1,len(env.free_regions))
        self.assertTrue(env.parking_spaces[0]["evidence"]["coverage_verified"])

    def test_clear_region_is_not_extended_beyond_visibility_polygon(self):
        p = perception()
        model = profile(p)
        model["profiles"][0]["body_region_m"] = [[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]]
        self.assertEqual("unknown",self.build(p,model).parking_spaces[0]["occupancy"])

    def test_overlapping_profiles_of_one_original_sensor_packet_cover_the_whole_bay(self):
        p = perception()
        model = profile(p)
        first = model["profiles"][0]
        first["body_region_m"] = [[9.,2.],[13.,2.],[13.,7.],[9.,7.]]
        second = copy.deepcopy(first)
        second["body_region_m"] = [[12.,2.],[16.,2.],[16.,7.],[12.,7.]]
        model["profiles"].append(second)
        env=self.build(p,model)
        self.assertEqual("empty",env.parking_spaces[0]["occupancy"])
        self.assertEqual('COVERAGE_COMPLETE_PACKET_UNION',env.status['coverage']['query_reason'])

    def test_task_case_vehicle_sensor_and_installation_must_match(self):
        p = perception()
        for key,value in (("task_id","other"),("case_id","other"),("vehicle_id","1"),("sensor_id","other")):
            model = profile(p)
            model["profiles"][0][key] = value
            self.assertFalse(self.build(p,model).coverage_regions,key)
        for key in INSTALL:
            model = profile(p)
            model["profiles"][0]["installation"][key] = "9" if key=="type" else INSTALL[key]+1
            self.assertFalse(self.build(p,model).coverage_regions,key)

    def test_no_complete_detection_claim_or_reference_means_unknown(self):
        p = perception()
        for key,value in (("complete_detections",False),("verification_reference","")):
            model = profile(p)
            model["profiles"][0][key] = value
            self.assertFalse(self.build(p,model).coverage_regions)

    def test_bad_models_fail_closed_and_do_not_poison_legacy_perception(self):
        p = perception()
        for shape in ([(0.,0.)]*4,[(0.,0.),(2.,0.),(1.,.5),(2.,2.),(0.,2.)],
                      [(float("nan"),0.),(1.,0.),(1.,1.)],[(0.,0.),(300.,0.),(0.,300.)]):
            model = profile(p)
            model["profiles"][0]["body_region_m"] = shape
            env = self.build(p,model)
            self.assertTrue(env.valid)
            self.assertFalse(env.coverage_regions)

    def test_stale_failed_future_unknown_age_and_ground_truth_cannot_clear(self):
        original = perception()
        for change in ("stale","failed","future","unknown_age","ground_truth","missing_installation"):
            p = copy.deepcopy(original)
            if change=="stale": p.source_status["targets"]["usable"] = False
            if change=="failed": p.targets_valid = False
            if change=="future": p.targets_frame_id = 2
            if change=="unknown_age": p.source_status["targets"]["age_ms"] = -1
            if change=="ground_truth": p.target_source = "ground_truth"
            if change=="missing_installation": p.sensor_configurations_valid = False
            self.assertEqual("unknown",self.build(p,profile(original)).parking_spaces[0]["occupancy"],change)

    def test_incomplete_object_geometry_or_duplicate_id_does_not_prove_clear(self):
        p = perception()
        bad = target(x=100.,y=10.)
        bad.width = 0.
        for objects in ([bad],[target(100.,10.),target(101.,10.)]):
            p.targets = objects
            env = self.build(p,profile(p))
            self.assertFalse(env.status["objects"]["usable"])
            self.assertEqual("unknown",env.parking_spaces[0]["occupancy"])

    def test_bad_other_object_does_not_hide_observed_parking_obstacle(self):
        p = perception()
        bad = target(100.,10.,identifier=10)
        bad.width = 0.
        p.targets = [target(),bad]
        self.assertEqual("occupied",self.build(p,profile(p)).parking_spaces[0]["occupancy"])

    def test_visible_xy_does_not_certify_space_on_another_height_layer(self):
        p = perception()
        for point in p.parking_spaces[0]["boundary_knots"]:
            point[2] = 4.
        env = self.build(p,profile(p))
        self.assertTrue(env.parking_spaces[0]["geometry_valid"])
        self.assertFalse(env.parking_spaces[0]["coverage_verified"])
        self.assertFalse(env.free_regions)
        p.ego.z = 4.
        self.assertEqual("empty",self.build(p,profile(p)).parking_spaces[0]["occupancy"])

    def test_touching_park_edge_and_unknown_heading_are_conservative(self):
        p = perception()
        for obj in (target(x=27.25),target(x=27.25,heading=None)):
            p.targets = [obj]
            self.assertEqual("occupied",self.build(p).parking_spaces[0]["occupancy"])
        p.targets = [target(x=100.)]
        self.assertEqual("unknown",self.build(p).parking_spaces[0]["occupancy"])

    def test_invalid_map_geometry_is_not_promoted(self):
        p = perception()
        p.parking_spaces[0]["boundary_knots"] = [[0.,0.,0.]]*4
        item = self.build(p,profile(p)).parking_spaces[0]
        self.assertFalse(item["geometry_valid"])
        self.assertEqual("unknown",item["occupancy"])

    def test_dynamic_deadline_and_original_observation_age_are_preserved(self):
        p = perception()
        p.source_status["targets"]["age_ms"] = 499
        env = self.build(p,profile(p))
        certificate = env.coverage_regions[0]
        self.assertAlmostEqual(99.501,certificate["observed_at_s"])
        self.assertAlmostEqual(100.001,certificate["valid_until_s"])
        self.assertLess(certificate["valid_until_s"],env.valid_until)

    def test_older_sensor_pose_cannot_locate_current_visibility_model(self):
        p = perception()
        p.frame_id,p.ego.frame_id = 2,2
        env = self.build(p,profile(p))
        self.assertTrue(env.status["objects"]["usable"])
        self.assertFalse(env.coverage_regions)
        self.assertEqual("sensor_pose_history_unavailable",env.status["coverage"]["reason"])

    def test_foreign_source_and_invalid_probability_do_not_certify_empty(self):
        p = perception()
        for change in ("source","probability"):
            p.targets = [target(100.,10.)]
            if change=="source": p.targets[0].source = "ground_truth"
            else: p.targets[0].probability = float("nan")
            self.assertFalse(self.build(p,profile(p)).coverage_regions)

    def test_visibility_file_replacement_and_deletion_discard_old_certificate(self):
        from perception.sensor_visibility import SensorVisibility
        p = perception()
        self.build(p,profile(p))
        reader = SensorVisibility(self.path)
        self.assertTrue(reader.regions(p,"0")[0])
        with open(self.path,"w",encoding="utf-8") as stream:
            stream.write('{"version":"wrong","profiles":[]}')
        self.assertFalse(reader.regions(p,"0")[0])
        os.remove(self.path)
        self.assertFalse(reader.regions(p,"0")[0])

    def test_invalid_or_expired_gps_cannot_publish_environment(self):
        p = perception()
        p.source_status["gps"]["usable"] = False
        self.assertFalse(self.build(p,profile(p)).valid)
        p = perception()
        p.valid_until = 99.
        self.assertFalse(self.build(p,profile(p)).valid)

    def test_neighbor_geometry_uses_native_ids_and_does_not_claim_full_marking_scope(self):
        p = perception()
        route = manager(LaneMap())
        p.lane = route.update(p.ego,True)
        item = self.build(p,profile(p),route).neighbor_lanes[0]
        self.assertTrue(item["geometry_valid"])
        self.assertTrue(item["same_direction"])
        self.assertTrue(item["crossing_allowed_at_ego"])
        self.assertFalse(item["crossing_range_verified"])
        self.assertTrue(item["dynamic_coverage_verified"])
        self.assertEqual("1_0_-2",item["lane_id"])

    def test_reversed_neighbor_sample_preserves_physical_sides(self):
        p = perception()
        route = manager(LaneMap(reversed_sample=True))
        p.lane = route.update(p.ego,True)
        item = route.read_neighbor_lanes(p.ego,p.lane)[0]
        self.assertEqual(0.,item["center_line"][0][0])
        self.assertEqual(5.25,item["left_boundary"][0][1])
        self.assertEqual(1.75,item["right_boundary"][0][1])

    def test_opposite_solid_unknown_type_bad_join_and_height_never_allow_crossing(self):
        for change in ("opposite","solid","unknown","join","height"):
            p = perception()
            api = LaneMap(opposite=change=="opposite")
            if change=="solid": api.mark = "solid"
            if change=="unknown": api.kind = "unknown"
            if change=="join": api.shift = .5
            if change=="height": api.height = 1.
            route = manager(api)
            p.lane = route.update(p.ego,True)
            item = route.read_neighbor_lanes(p.ego,p.lane)[0]
            self.assertFalse(item["crossing_allowed_at_ego"],change)

    def test_neighbor_map_failure_and_context_reset_do_not_reuse_old_geometry(self):
        p = perception()
        api = LaneMap()
        route = manager(api)
        p.lane = route.update(p.ego,True)
        self.assertTrue(route.read_neighbor_lanes(p.ego,p.lane)[0]["geometry_valid"])
        del api.segments[api.neighbor]
        route.set_route_hint([],False,("other","task"))
        self.assertFalse(route.read_neighbor_lanes(p.ego,p.lane)[0]["geometry_valid"])

    def test_crossing_occupancy_and_predicted_sweep_use_all_objects(self):
        p = perception()
        api = map_api()
        # These map facts are the same records the existing reader publishes.
        from simone_platform.map_observations import MapObservationReader
        values = MapObservationReader(api).lane_objects([p.lane.lane_id])
        p.map_crosswalks,p.map_crosswalks_valid = values["map_crosswalks"],True
        p.map_stop_lines,p.map_stop_lines_valid = values["map_stop_lines"],True
        p.targets = [target(45.,0.),target(45.,-10.,identifier=10)]
        p.targets[1].vy = 10.
        item = self.build(p,profile(p)).crossing_regions[0]
        self.assertEqual([9],item["object_ids"])
        self.assertEqual([9,10],item["predicted_object_ids"])
        self.assertEqual("occupied",item["occupancy"])
        self.assertEqual(30.,item["entry_distance_m"])
        self.assertFalse(item["rule_verified"])
        self.assertFalse(item["exit_coverage_verified"])

    def test_unrelated_stop_line_is_not_assigned_by_nearness(self):
        p = perception()
        from simone_platform.map_observations import MapObservationReader
        values = MapObservationReader(map_api()).lane_objects([p.lane.lane_id])
        p.map_crosswalks,p.map_crosswalks_valid = values["map_crosswalks"],True
        p.map_stop_lines,p.map_stop_lines_valid = values["map_stop_lines"],True
        p.map_stop_lines[0]["association_signal_ids"] = [999]
        self.assertIsNone(self.build(p,profile(p)).crossing_regions[0]["entry_distance_m"])

    def test_public_serialization_and_replay_keep_typed_environment(self):
        p = perception()
        p.maneuver_environment = self.build(p,profile(p))
        saved = json.loads(json.dumps(perception_to_dict(p),allow_nan=False))
        rebuilt = _assign(Perception(),saved,"perception")
        self.assertIsInstance(rebuilt.maneuver_environment,ManeuverEnvironment)
        self.assertEqual("empty",rebuilt.maneuver_environment.parking_spaces[0]["occupancy"])
        self.assertEqual(100.2,rebuilt.maneuver_environment.valid_until)

    def test_real_perception_builder_publishes_new_channel_without_changing_old_map(self):
        p = perception()
        route = manager(LaneMap())
        adapter = NS(config=NS(vehicle_id="0",sensor_timeout_ms=500,max_sensor_frame_gap=10,pipeline_timeout_ms=200),
                     read_traffic=lambda identity:[])
        raw = dict(gps=dict(frame=1,timestamp=50,x=10.,y=0.,z=0.,heading=0.,vx=0.,vy=0.,age_ms=0),
                   targets=[],targets_valid=True,targets_frame=1,targets_timestamp=50,targets_age_ms=0,
                   target_source="sensor:perfect",source_status=dict(gps=dict(read_ok=True,frame_id=1,age_ms=0),
                   targets=dict(read_ok=True,frame_id=1,age_ms=0)),parking_spaces=p.parking_spaces,parking_spaces_valid=True)
        output = PerceptionBuilder(adapter,route).build_from_raw(raw)
        self.assertTrue(output.valid)
        self.assertTrue(output.maneuver_environment.valid)
        self.assertEqual("unknown",output.parking_spaces[0]["occupancy"])
        self.assertEqual("unknown",output.maneuver_environment.parking_spaces[0]["occupancy"])
        self.assertEqual(output.frame_id,output.maneuver_environment.frame_id)
        self.assertEqual(1,len(output.maneuver_environment.neighbor_lanes))

    def test_shared_geometry_rejects_star_not_only_concave_polygon(self):
        points = [(math.cos(i*4*math.pi/5),math.sin(i*4*math.pi/5)) for i in range(5)]
        with self.assertRaises(ValueError):
            convex_polygon(points)

    def test_simple_map_outline_handles_concavity_and_rejects_crossing_or_budget(self):
        outline = [(0.,0.),(3.,0.),(3.,1.),(1.,1.),(1.,3.),(0.,3.)]
        self.assertEqual(outline,simple_outline(outline))
        with self.assertRaises(ValueError):
            simple_outline([(0.,0.),(3.,2.),(0.,2.),(3.,0.)])
        with self.assertRaises(ValueError):
            simple_outline(outline,max_checks=1)


if __name__=="__main__":
    unittest.main()
