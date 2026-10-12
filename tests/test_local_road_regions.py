"""Local SDK corridors and clipped Sensor evidence without a free-space hull."""
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.region_geometry import (simple_outline,convex_cells,inside,intersect_convex,
    outline_contains_point,outline_contains_segment,outline_contains_path,strip_cells)
from core.serialization import perception_to_dict
from core.interfaces import Perception
from perception.road_regions import local_region
from perception.route_manager import RouteManager
from perception.maneuver_environment import ManeuverEnvironmentBuilder,_touches
from scripts.replay_decision import _assign
from tests.test_maneuver_map import map_adapter,document,road
from tests.test_maneuver_perception import perception,profile,target


def area(region):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(region,region[1:]+region[:1])))/2.


class LocalRoadRegionTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.filename=os.path.join(self.directory.name,'visibility.json')

    def fixture(self,length=1000):
        adapter,metadata=map_adapter(document(road(length=length)))
        adapter.hdmap.segments['1_0_-1']=[(0.,0.),(float(length),0.)]
        adapter.hdmap.segments['1_0_-2']=[(0.,3.5),(float(length),3.5)]
        route=RouteManager(adapter,NS(warning=lambda *a:None))
        p=perception(); p.lane=route.update(p.ego,True)
        return adapter,metadata,route,p

    def build(self,route,p,model=None,clock=lambda:100.):
        if model is not None:
            with open(self.filename,'w',encoding='utf-8') as stream: json.dump(model,stream)
        config=NS(vehicle_id='0',sensor_timeout_ms=500,sensor_visibility_file=self.filename if model is not None else '')
        return ManeuverEnvironmentBuilder(route,config,clock).build(p)

    def test_long_road_local_region_can_be_observed_without_claiming_entire_lane_visible(self):
        unused,unused,route,p=self.fixture()
        env=self.build(route,p,profile(p))
        self.assertEqual(2,len(env.road_regions))
        self.assertTrue(all(v['geometry_valid'] and v['drivable_verified'] and v['coverage_verified'] for v in env.road_regions))
        self.assertFalse(env.neighbor_lanes[0]['dynamic_coverage_verified'])
        self.assertTrue(env.neighbor_lanes[0]['local_coverage_verified'])
        self.assertEqual(75.,env.neighbor_lanes[0]['local_geometry']['center_arc_end_m'])
        self.assertTrue([v for v in env.free_regions if v['kind']=='lane'])
        self.assertEqual(1000.,p.lane.center_line[-1][0])

    def test_partial_visibility_clips_cells_and_never_fills_unknown_road(self):
        unused,unused,route,p=self.fixture(); visibility=profile(p)
        visibility['profiles'][0]['body_region_m']=[[-20.,-20.],[20.,-20.],[20.,20.],[-20.,20.]]
        env=self.build(route,p,visibility)
        self.assertFalse(any(v['coverage_verified'] for v in env.road_regions))
        cells=[v for v in env.free_regions if v['kind']=='lane']
        self.assertTrue(cells)
        self.assertTrue(all(-10.-1e-8<=point[0]<=30.+1e-8 for cell in cells for point in cell['polygon']))
        self.assertFalse(any(inside((50.,0.),cell['polygon']) for cell in cells))

    def test_empty_packet_without_visibility_and_unknown_native_type_do_not_prove_free(self):
        adapter,unused,route,p=self.fixture()
        env=self.build(route,p)
        self.assertFalse([v for v in env.free_regions if v['kind']=='lane'])
        self.assertTrue(all(v['occupancy']=='unknown' for v in env.road_regions))
        adapter.hdmap.kind='unknown'; p.lane=route.update(p.ego,True)
        route._neighbor_samples.clear()
        env=self.build(route,p,profile(p))
        self.assertFalse(any(v['drivable_verified'] for v in env.road_regions))
        self.assertFalse([v for v in env.free_regions if v['kind']=='lane'])

    def test_full_vehicle_overlap_is_occupied_and_intersecting_cell_not_free(self):
        unused,unused,route,p=self.fixture(); p.targets=[target(20.,0.)]
        env=self.build(route,p,profile(p))
        current=next(v for v in env.road_regions if v['role']=='current')
        self.assertEqual('occupied',current['occupancy'])
        self.assertEqual([9],current['object_ids'])
        self.assertFalse(any(inside((20.,0.),v['polygon']) for v in env.free_regions if v['kind']=='lane'))

    def test_unknown_heading_circle_catches_corner_overlap(self):
        unused,unused,route,p=self.fixture(); p.targets=[target(20.,3.8,heading=None)]
        env=self.build(route,p,profile(p))
        current=next(v for v in env.road_regions if v['role']=='current')
        self.assertEqual([9],current['object_ids'])
        self.assertEqual('occupied',current['occupancy'])

    def test_far_objects_skip_exact_cell_queries_without_changing_clear_result(self):
        unused,unused,route,p=self.fixture(); p.targets=[target(500.,50.)]
        calls=[]
        def query(obj,region,horizon=0.):
            calls.append(region); return _touches(obj,region,horizon)
        with patch('perception.maneuver_environment._touches',side_effect=query):
            env=self.build(route,p,profile(p))
        self.assertTrue(all(v['occupancy']=='clear' for v in env.road_regions))
        # The one parking query remains; no lane-cell exact query is needed.
        self.assertEqual(1,len(calls))

    def test_object_in_unobserved_part_of_cell_does_not_erase_observed_clip(self):
        unused,unused,route,p=self.fixture(); p.targets=[target(60.,0.)]
        visibility=profile(p)
        visibility['profiles'][0]['body_region_m']=[[-20.,-20.],[20.,-20.],[20.,20.],[-20.,20.]]
        env=self.build(route,p,visibility)
        current=next(v for v in env.road_regions if v['role']=='current')
        self.assertEqual('occupied',current['occupancy'])
        self.assertTrue(any(inside((20.,0.),v['polygon']) for v in env.free_regions
                            if v['kind']=='lane' and v['lane_id']==p.lane.lane_id))

    def test_source_expiring_during_map_work_does_not_publish_clear_regions(self):
        unused,unused,route,p=self.fixture()
        class ExpiringClock(object):
            def __init__(self): self.calls=0
            def __call__(self):
                self.calls+=1; return 100. if self.calls==1 else 100.21
        env=self.build(route,p,profile(p),clock=ExpiringClock())
        self.assertFalse(env.free_regions)
        self.assertFalse(any(v['coverage_verified'] for v in env.road_regions))
        self.assertFalse(any(v.get('occupancy')=='empty' for v in env.parking_spaces))

    def test_expiry_is_source_deadline_and_not_renewed_by_static_geometry(self):
        unused,unused,route,p=self.fixture(); p.source_status['targets']['age_ms']=450
        env=self.build(route,p,profile(p))
        cells=[v for v in env.free_regions if v['kind']=='lane']
        self.assertTrue(cells)
        self.assertAlmostEqual(100.05,cells[0]['evidence']['valid_until_s'])
        self.assertAlmostEqual(99.55,cells[0]['evidence']['observed_at_s'])

    def test_identity_failure_does_not_publish_cached_local_geometry_as_observed(self):
        unused,metadata,route,p=self.fixture(); self.build(route,p,profile(p))
        metadata['verified']=False
        env=self.build(route,p,profile(p))
        self.assertEqual([],env.road_regions)
        self.assertFalse([v for v in env.free_regions if v['kind']=='lane'])
        self.assertNotIn('local_geometry',env.neighbor_lanes[0])

    def test_wrong_height_level_and_bad_other_target_keep_free_space_unknown(self):
        unused,unused,route,p=self.fixture()
        p.ego.z=1.
        env=self.build(route,p,profile(p))
        self.assertFalse([v for v in env.free_regions if v['kind']=='lane'])
        p.ego.z=0.; bad=target(40.,0.); bad.width=0.; p.targets=[bad]
        env=self.build(route,p,profile(p))
        self.assertFalse([v for v in env.free_regions if v['kind']=='lane'])

    def test_public_replay_preserves_new_regions_and_lane_type(self):
        unused,unused,route,p=self.fixture(); p.maneuver_environment=self.build(route,p,profile(p))
        saved=json.loads(json.dumps(perception_to_dict(p),allow_nan=False))
        rebuilt=_assign(Perception(),saved,'perception')
        self.assertTrue(rebuilt.lane.lane_type_valid)
        self.assertEqual(saved['maneuver_environment']['road_regions'],rebuilt.maneuver_environment.road_regions)

    def test_curve_crop_and_cell_union_preserve_inner_cutout(self):
        center=[(0.,0.,0.),(20.,0.,0.),(20.,20.,0.)]
        left=[(0.,1.75,0.),(18.25,1.75,0.),(18.25,20.,0.)]
        right=[(0.,-1.75,0.),(21.75,-1.75,0.),(21.75,20.,0.)]
        value=local_region(center,left,right,[(5.,0.,0.),(20.,15.,0.)])
        self.assertEqual(5.,value['center_arc_start_m']); self.assertEqual(35.,value['center_arc_end_m'])
        self.assertFalse(any(inside((10.,10.),cell) for cell in value['convex_cells']))
        self.assertTrue(any(inside((20.,10.),cell) for cell in value['convex_cells']))
        self.assertAlmostEqual(area(value['polygon']),sum(area(cell) for cell in value['convex_cells']))

    def test_multiple_concavities_are_not_accepted_from_only_endpoints_and_midpoint(self):
        ring=simple_outline([(0.,0.),(10.,0.),(10.,6.),(8.,6.),(8.,2.),(6.,2.),
                             (6.,6.),(4.,6.),(4.,2.),(2.,2.),(2.,6.),(0.,6.)])
        self.assertTrue(all(outline_contains_point(p,ring) for p in ((.5,4.),(5.,4.),(9.5,4.))))
        self.assertFalse(outline_contains_segment((.5,4.),(9.5,4.),ring))
        cells=convex_cells(ring)
        self.assertAlmostEqual(area(ring),sum(area(cell) for cell in cells))
        self.assertFalse(any(inside((3.,4.),cell) or inside((7.,4.),cell) for cell in cells))

    def test_convex_clipping_empty_edge_contact_and_clockwise_cover(self):
        triangle=[(0.,0.),(10.,0.),(0.,10.)]
        coverage=[(0.,0.),(0.,4.),(4.,4.),(4.,0.)]
        value=intersect_convex(triangle,coverage)
        self.assertAlmostEqual(16.,area(value))
        self.assertEqual([],intersect_convex(triangle,[(10.,0.),(12.,0.),(12.,2.),(10.,2.)]))
        self.assertEqual([],intersect_convex(triangle,[(20.,20.),(22.,20.),(22.,22.),(20.,22.)]))

    def test_geometry_work_budget_exhaustion_is_unknown(self):
        ring=simple_outline([(0.,0.),(6.,0.),(6.,2.),(2.,2.),(2.,6.),(0.,6.)])
        with self.assertRaises(ValueError): convex_cells(ring,max_checks=1)
        with self.assertRaises(ValueError): outline_contains_segment((.5,1.),(5.,1.),ring,max_checks=1)

    def test_path_sweep_handles_duplicates_end_cuts_and_concavities(self):
        ring=simple_outline([(0.,0.),(10.,0.),(10.,6.),(6.,6.),(6.,2.),(4.,2.),(4.,6.),(0.,6.)])
        self.assertTrue(outline_contains_path([(0.,1.),(0.,1.),(5.,1.),(10.,1.)],ring))
        self.assertFalse(outline_contains_path([(0.,4.),(10.,4.)],ring))
        self.assertFalse(outline_contains_path([(0.,0.),(10.,0.)],ring))
        self.assertTrue(outline_contains_path([(2.,1.),(2.,1.)],ring))
        self.assertFalse(outline_contains_path([(5.,4.),(5.,4.)],ring))
        with self.assertRaises(ValueError): outline_contains_path([(0.,1.),(10.,1.)],ring,max_checks=1)

    def test_dense_curve_cells_preserve_all_border_vertices_and_exact_area(self):
        def arc(radius,count):
            return [(radius*math.cos(i*math.pi/2/(count-1)),radius*math.sin(i*math.pi/2/(count-1)),0.) for i in range(count)]
        center,left,right=arc(100.,3001),arc(98.25,3001),arc(101.75,3001)
        value=local_region(center,left,right,[center[300],center[1500]])
        cells=value['convex_cells']
        self.assertGreater(len(cells),2400)
        self.assertAlmostEqual(area(value['polygon']),sum(area(cell) for cell in cells),places=7)
        vertices={v for cell in cells for v in cell}
        self.assertTrue(all(v[:2] in vertices for v in value['left_boundary']+value['right_boundary']))
        self.assertFalse(any(inside((0.,0.),cell) for cell in cells))

    def test_strip_mesh_preserves_unequal_sampling_and_rejects_degenerate_strip(self):
        left=[(0.,1.),(1.,1.),(2.,1.),(5.,1.),(10.,1.)]
        right=[(0.,-1.),(3.,-1.),(10.,-1.)]
        cells=strip_cells(left,right)
        self.assertAlmostEqual(20.,sum(area(cell) for cell in cells))
        self.assertEqual(set(left+right),{v for cell in cells for v in cell})
        with self.assertRaises(ValueError): strip_cells([(0.,0.),(10.,0.)],[(0.,0.),(10.,0.)])

    def test_adjacent_boundary_backtrack_is_not_a_simple_road_outline(self):
        with self.assertRaises(ValueError):
            simple_outline([(0.,0.),(10.,0.),(5.,0.),(10.,5.),(0.,5.)])


if __name__=='__main__': unittest.main()
