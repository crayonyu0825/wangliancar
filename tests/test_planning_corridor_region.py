"""Analytic region unions through the existing continuous candidate checker."""
import copy
import math
import random
import unittest
from unittest.mock import patch

from members.planning.candidate_validation import CorridorRegion,validate_candidate
from tests.test_planning_candidate_validation import point,vehicle,limits,budget,rectangle,arc,obstacle


def check(polygons,points=None,geometry=None,direction=1,motion=None,objects=None,work=None):
    return validate_candidate(points or [point(0.),point(2.,at=2.)],direction,
        geometry or vehicle(),motion or limits(),CorridorRegion(polygons),objects or [],work or budget())


class CorridorRegionTests(unittest.TestCase):
    def assert_collision(self,result):
        self.assertEqual('unsafe',result['status'],result)
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)
        self.assertIsNone(result['clearance_lower_bound_m'])

    def test_single_region_preserves_existing_convex_result(self):
        polygon=rectangle(-2.,-2.,5.,2.)
        region=check([polygon])
        old=validate_candidate([point(0.),point(2.,at=2.)],1,vehicle(),limits(),polygon,[],budget())
        self.assertEqual('safe',region['status'],region)
        self.assertAlmostEqual(old['clearance_lower_bound_m'],region['clearance_lower_bound_m'],places=8)

    def test_body_spans_adjacent_regions_without_a_false_internal_wall(self):
        result=check([rectangle(-2.,-2.,5.,0.),rectangle(-2.,0.,5.,2.)])
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(6,result['details']['exterior_edge_count'])
        self.assertGreater(result['clearance_lower_bound_m'],.6)

    def test_overlapping_regions_remove_only_covered_boundary_fragments(self):
        result=check([rectangle(-2.,-2.,5.,.2),rectangle(-2.,-.2,5.,2.)])
        self.assertEqual('safe',result['status'],result)
        self.assertGreater(result['clearance_lower_bound_m'],.6)

    def test_exact_shared_diagonal_allows_body_across_triangle_cells(self):
        result=check([[(-2.,-2.),(5.,-2.),(5.,2.)],[(-2.,-2.),(5.,2.),(-2.,2.)]])
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(4,result['details']['exterior_edge_count'])

    def test_partial_collinear_join_does_not_extend_to_unjoined_end(self):
        regions=[rectangle(-2.,-2.,12.,0.),rectangle(0.,0.,10.,2.)]
        self.assertEqual('safe',check(regions,[point(1.),point(2.,at=1.)])['status'])
        self.assert_collision(check(regions,[point(0.,speed=0.),point(0.,speed=0.,at=1.)]))

    def test_positive_gap_is_retained_even_below_float_geometry_tolerance(self):
        for gap in (.02,1e-12):
            self.assert_collision(check([rectangle(-2.,-2.,5.,0.),rectangle(-2.,gap,5.,2.)]))

    def test_regions_touching_only_at_one_point_do_not_cover_body(self):
        self.assert_collision(check([rectangle(-2.,-2.,0.,0.),rectangle(0.,0.,5.,2.)],
            [point(0.,speed=0.),point(0.,speed=0.,at=1.)]))

    def test_disconnected_components_do_not_cover_motion_between_them(self):
        result=check([rectangle(-2.,-2.,2.,2.),rectangle(8.,-2.,12.,2.)],
            [point(0.),point(10.,at=10.)],geometry=vehicle(.05,.05,.05,.04))
        self.assert_collision(result)
        self.assertTrue(2.<result['details']['relative_time_s']<8.)

    def test_concave_notch_is_found_between_clear_endpoints_and_midpoint(self):
        ring=[(-2.,-2.),(12.,-2.),(12.,2.),(4.9,2.),(4.9,-.1),(4.7,-.1),(4.7,2.),(-2.,2.)]
        result=check([ring],[point(0.),point(10.,at=10.)],geometry=vehicle(.05,.05,.05,.04))
        self.assert_collision(result)
        self.assertTrue(4.6<result['details']['relative_time_s']<5.)

    def test_union_hole_is_found_between_samples(self):
        regions=[rectangle(-2.,-2.,12.,-.2),rectangle(-2.,.2,12.,2.),
                 rectangle(-2.,-.2,4.7,.2),rectangle(4.9,-.2,12.,.2)]
        result=check(regions,[point(0.),point(10.,at=10.)],geometry=vehicle(.05,.05,.05,.04))
        self.assert_collision(result)
        self.assertTrue(4.6<result['details']['relative_time_s']<5.)

    def test_hole_inside_vehicle_is_rejected_even_with_all_corners_covered(self):
        regions=[rectangle(-2.,-2.,5.,-.1),rectangle(-2.,.1,5.,2.),
                 rectangle(-2.,-.1,.3,.1),rectangle(.4,-.1,5.,.1)]
        result=check(regions,[point(0.,speed=0.),point(0.,speed=0.,at=1.)],geometry=vehicle(2.,.5,.5,.8))
        self.assert_collision(result)
        self.assertEqual(0.,result['details']['relative_time_s'])

    def test_non_dyadic_edge_intersections_and_clockwise_inputs(self):
        regions=[[(-4.,0.),(0.,-3.),(4.,0.),(0.,3.)],rectangle(-6.,-1.,6.,1.)]
        for supplied in (regions,[list(reversed(v)) for v in regions]):
            result=check(supplied,[point(0.),point(3.,at=3.)],geometry=vehicle(.3,.3,.3,.2))
            self.assertEqual('safe',result['status'],result)

    def test_contained_duplicate_and_ordered_regions_do_not_change_clearance(self):
        outer=rectangle(-2.,-2.,5.,2.); inner=rectangle(-1.,-.5,4.,.5)
        results=[check(v) for v in ([outer],[inner,outer],[outer,inner],[outer,outer,inner])]
        self.assertTrue(all(v['status']=='safe' for v in results),results)
        self.assertTrue(all(abs(v['clearance_lower_bound_m']-results[0]['clearance_lower_bound_m'])<1e-8 for v in results))

    def test_nonconvex_source_is_not_mutated_or_replaced_by_hull(self):
        ring=[(-3.,-2.),(10.,-2.),(10.,6.),(6.,6.),(6.,2.),(-3.,2.)]
        original=copy.deepcopy(ring)
        result=check([ring])
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(original,ring)
        self.assert_collision(check([ring],[point(0.,4.,speed=0.),point(0.,4.,speed=0.,at=1.)]))

    def test_rotation_translation_and_reverse_keep_region_semantics(self):
        polygons=[rectangle(-4.,-2.,5.,0.),rectangle(-4.,0.,5.,2.)]
        points=[point(0.),point(-2.,at=2.)]
        def transform(x,y):
            c,s=math.cos(.7),math.sin(.7)
            return (1e6+x*c-y*s,-1e6+x*s+y*c)
        moved=[[transform(*v) for v in poly] for poly in polygons]
        samples=[point(*transform(v.x,v.y),heading=.7,at=v.relative_time) for v in points]
        result=check(moved,samples,direction=-1)
        self.assertEqual('safe',result['status'],result)

    def test_dynamic_object_and_prediction_horizon_are_still_checked(self):
        regions=[rectangle(-4.,-2.,10.,0.),rectangle(-4.,0.,10.,2.)]
        result=check(regions,objects=[obstacle(1.,3.,vy=-3.)])
        self.assertEqual('OBSTACLE_COLLISION',result['reason_code'],result)
        result=check(regions,objects=[obstacle(100.,0.,horizon=1.)])
        self.assertEqual('inconclusive',result['status'])
        self.assertEqual('PREDICTION_HORIZON',result['reason_code'])

    def test_invalid_topology_and_oversized_inputs_do_not_become_safe(self):
        invalid=[[],None,[[(0.,0.),(2.,2.),(0.,2.),(2.,0.)]],
                 [[(0.,0.),(10.,0.),(5.,0.),(10.,5.),(0.,5.)]],
                 [[(0.,0.),(1.,0.),(0.,float('nan'))]],
                 [rectangle()]*33]
        for regions in invalid:
            result=check(regions)
            self.assertEqual('invalid',result['status'],result)
            self.assertIsNone(result['clearance_lower_bound_m'])

    def test_count_and_elapsed_budgets_interrupt_topology_work(self):
        regions=[rectangle(-2.,-2.,5.,0.),rectangle(-2.,0.,5.,2.)]
        result=check(regions,work=budget(max_checks=10))
        self.assertEqual('inconclusive',result['status'])
        self.assertEqual('BUDGET_EXHAUSTED',result['reason_code'])
        ticks=iter(1.+i*.01 for i in range(1000))
        with patch('members.planning.candidate_validation.time.monotonic',side_effect=lambda:next(ticks)):
            result=check(regions,work=budget(deadline_monotonic_s=1.1))
        self.assertEqual('inconclusive',result['status'])
        self.assertEqual('BUDGET_EXHAUSTED',result['reason_code'])

    def test_unresolved_roundoff_is_inconclusive_not_certified_or_collision(self):
        x=1e9
        result=check([rectangle(x-2.,-2.,x+1.+4.*2.**-23,2.)],
            [point(x,speed=0.),point(x,speed=0.,at=1.)])
        self.assertEqual('inconclusive',result['status'],result)
        self.assertEqual('REGION_NUMERIC_RESOLUTION',result['reason_code'])

    def test_cached_topology_reuses_geometry_but_not_new_motion_or_obstacles(self):
        region=CorridorRegion([rectangle(-2.,-2.,5.,0.),rectangle(-2.,0.,5.,2.)])
        samples=[point(0.),point(2.,at=2.)]
        first=validate_candidate(samples,1,vehicle(),limits(),region,[],budget())
        second=validate_candidate(samples,1,vehicle(),limits(),region,[],budget())
        self.assertEqual('safe',second['status'],second)
        self.assertLess(second['checks'],first['checks'])
        third=validate_candidate(samples,1,vehicle(),limits(),region,[obstacle(1.,0.)],budget())
        self.assertEqual('OBSTACLE_COLLISION',third['reason_code'],third)
        fourth=validate_candidate(samples,1,vehicle(),limits(max_speed_mps=.5),region,[],budget())
        self.assertEqual('MOTION_LIMIT',fourth['reason_code'],fourth)

    def test_changed_geometry_and_expired_budget_cannot_borrow_cached_proof(self):
        polygons=[rectangle(-2.,-2.,5.,2.)]; region=CorridorRegion(polygons)
        samples=[point(0.),point(2.,at=2.)]
        self.assertEqual('safe',validate_candidate(samples,1,vehicle(),limits(),region,[],budget())['status'])
        polygons[0]=rectangle(-2.,-2.,5.,.1)
        self.assert_collision(validate_candidate(samples,1,vehicle(),limits(),region,[],budget()))
        result=validate_candidate(samples,1,vehicle(),limits(),region,[],budget(deadline_monotonic_s=.1))
        self.assertEqual('BUDGET_EXHAUSTED',result['reason_code'],result)

    def test_real_perception_local_curves_can_feed_union_without_cell_walls(self):
        from perception.road_regions import local_region
        def boundary(radius):
            return [(radius*math.cos(i*.8/100),radius*math.sin(i*.8/100),0.) for i in range(101)]
        current=local_region(boundary(100.),boundary(98.25),boundary(101.75),[boundary(100.)[10],boundary(100.)[90]])
        neighbor=local_region(boundary(103.5),boundary(101.75),boundary(105.25),[boundary(103.5)[10],boundary(103.5)[90]])
        radius=101.75
        samples=[point(radius*math.cos(a),radius*math.sin(a),heading=a+math.pi/2,at=radius*(a-.3)) for a in [.3+i*.01 for i in range(11)]]
        result=check([current['polygon'],neighbor['polygon']],samples,work=budget(max_checks=100000))
        self.assertEqual('safe',result['status'],result)

    def test_axis_distance_never_exceeds_independent_rectangle_distance(self):
        from members.planning.corridor_region import _BodyBoxes
        from members.planning.candidate_validation import _polygon_gap
        generator=random.Random(7721)
        def distance(p,a,b):
            dx,dy=b[0]-a[0],b[1]-a[1]
            t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
            return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)
        body=rectangle(-1.,-.5,2.,.5)
        edges=list(zip(body,body[1:]+body[:1]))
        for unused in range(250):
            # Each endpoint has y>1, so the whole segment is disjoint from
            # the axis-aligned body; distance follows the vertex/edge formula.
            a=(generator.uniform(-4.,4.),generator.uniform(1.,5.))
            b=(generator.uniform(-4.,4.),generator.uniform(1.,5.))
            length=math.hypot(b[0]-a[0],b[1]-a[1])
            edge={'points':[a,b],'normal':(-(b[1]-a[1])/length,(b[0]-a[0])/length)}
            exact=min([distance(p,a,b) for p in body]+
                      [distance(p,u,v) for p in (a,b) for u,v in edges])
            lower=_BodyBoxes(body).segment_gap(edge)
            self.assertGreater(lower,0.)
            self.assertLessEqual(lower,exact+1e-12)
            self.assertAlmostEqual(exact,_polygon_gap(body,[a,b]),places=11)
            self.assertLessEqual(_BodyBoxes(body).lower_bound(
                (min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1]))),exact+1e-12)

    def test_rotated_box_pruning_preserves_exhaustive_exterior_bound(self):
        from members.planning.candidate_validation import _rectangle
        from members.planning.corridor_region import _BodyBoxes
        region=CorridorRegion([rectangle(-10.,-4.,10.,0.),rectangle(-10.,0.,10.,4.)])
        prepared=region.prepare(lambda:None,100000)
        def all_edges(node):
            if node[1] is not None: return node[1]
            return all_edges(node[2][0])+all_edges(node[2][1])
        for heading in (0.,.3,.9,1.6,2.7):
            body=_rectangle(1.,.2,heading,2.,1.,.5)
            exhaustive=min(_BodyBoxes(body).segment_gap(edge) for edge in all_edges(prepared.root))-prepared.roundoff
            self.assertAlmostEqual(exhaustive,prepared.gap(body,(1.,.2),lambda:None),places=12)

    def test_three_regions_cannot_use_pairwise_seam_to_hide_a_hole(self):
        regions=[rectangle(-2.,-2.,5.,0.),rectangle(-2.,0.,.3,2.),rectangle(.4,0.,5.,2.)]
        self.assert_collision(check(regions,[point(0.,speed=0.),point(0.,speed=0.,at=1.)],
                                    geometry=vehicle(2.,.5,.5,.8)))
        regions.append(rectangle(.3,0.,.4,2.))
        self.assertEqual('safe',check(regions)['status'])

    def test_exact_validation_rejects_crossing_missed_by_float_precheck(self):
        # Suppress the approximate preliminary helper to verify that exact
        # union compilation independently rejects non-adjacent intersections.
        ring=[(0.,0.),(2.,2.),(0.,2.),(3.,0.)]
        with patch('core.corridor_region.simple_outline',side_effect=lambda raw,**kw:list(raw)):
            result=check([ring])
        self.assertEqual('invalid',result['status'],result)
        self.assertIn('exact region self intersection',result['details']['message'])

    def test_exact_validation_rejects_adjacent_backtracking(self):
        ring=[(0.,0.),(10.,0.),(5.,0.),(10.,5.),(0.,5.)]
        with patch('core.corridor_region.simple_outline',side_effect=lambda raw,**kw:list(raw)):
            result=check([ring])
        self.assertEqual('invalid',result['status'],result)
        self.assertIn('exact adjacent region edges overlap',result['details']['message'])

    def test_incomplete_compilation_is_not_cached(self):
        region=CorridorRegion([rectangle(-2.,-2.,5.,0.),rectangle(-2.,0.,5.,2.)])
        samples=[point(0.),point(2.,at=2.)]
        result=validate_candidate(samples,1,vehicle(),limits(),region,[],budget(max_checks=80))
        self.assertEqual('BUDGET_EXHAUSTED',result['reason_code'],result)
        self.assertIsNone(region._prepared)
        self.assertEqual('safe',validate_candidate(samples,1,vehicle(),limits(),region,[],budget())['status'])


if __name__=='__main__': unittest.main()
