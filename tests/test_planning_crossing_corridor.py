"""Closed-source perimeter constraints in the existing continuous P02 sweep.

Examples are authored nominal geometry, not verified live marking or control.
Expected prohibited contacts follow the stated x-ranges and body dimensions.
"""
import math
import unittest
from fractions import Fraction
from types import SimpleNamespace as NS

from core.boundary_lineage import encode_position
from members.planning.crossing_corridor import CrossingCorridor
from members.planning.candidate_validation import validate_candidate,CorridorRegion
from members.planning.maneuver_scene import read_maneuver_scene,ManeuverGeometryCache
from tests.test_planning_candidate_validation import point,vehicle,limits,budget,rectangle,obstacle
from tests import test_maneuver_fact_consumers as producer_fixture


def region(openings=((5,15),),polygons=None,length=20):
    left=((0.,0.,0.),(float(length),0.,0.))
    right=((0.,-3.,0.),(float(length),-3.,0.))
    parts=tuple((encode_position(0,Fraction(a,length)),encode_position(0,Fraction(b,length))) for a,b in openings)
    polygons=polygons or [rectangle(0.,-3.,float(length),0.),rectangle(0.,0.,float(length),3.)]
    return CrossingCorridor(tuple(tuple(tuple(v) for v in p) for p in polygons),
                            (left,right,'left',parts),2000000)


def check(corridor,points=None,heading=0.,objects=None,direction=1,work=None):
    return validate_candidate(points or [point(10.,speed=0.,heading=heading),point(10.,speed=0.,heading=heading,at=1.)],
        direction,vehicle(),limits(),corridor,objects or [],work or budget(max_checks=100000))


class CrossingCorridorTests(unittest.TestCase):
    def test_body_may_span_exactly_permitted_shared_opening(self):
        result=check(region())
        self.assertEqual('safe',result['status'],result)
        self.assertGreater(result['details']['source_perimeter_barrier_count'],0)
        self.assertGreater(result['clearance_lower_bound_m'],0.)

    def test_geometry_union_does_not_remove_solid_or_unknown_shared_parts(self):
        polygons=[rectangle(0.,-3.,20.,0.),rectangle(0.,0.,20.,3.)]
        samples=[point(3.,speed=0.),point(3.,speed=0.,at=1.)]
        self.assertEqual('safe',check(CorridorRegion(polygons),samples)['status'])
        self.assertEqual('CORRIDOR_COLLISION',check(region(polygons=polygons),samples)['reason_code'])

    def test_front_and_rear_are_checked_even_with_reference_point_in_opening(self):
        for x in (5.2,14.5):
            result=check(region(),[point(x,speed=0.),point(x,speed=0.,at=1.)])
            self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)

    def test_rotated_corner_cannot_touch_forbidden_end(self):
        # At y=.15, the rotated rear edge meets y=0 at
        # x=5.55+.15-.5*sqrt(2) < 5, inside the forbidden segment.
        samples=[point(5.55,.15,speed=0.,heading=math.pi/4),point(5.55,.15,speed=0.,heading=math.pi/4,at=1.)]
        self.assertEqual('CORRIDOR_COLLISION',check(region(),samples)['reason_code'])

    def test_positive_gap_between_permissions_is_preserved(self):
        result=check(region(((5,10),(Fraction(501,50),15))))
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)

    def test_exactly_touching_and_overlapping_permissions_remove_no_extra_space(self):
        for openings in (((5,10),(10,15)),((5,12),(10,15)),((5,15),(5,15))):
            result=check(region(openings))
            self.assertEqual('safe',result['status'],result)

    def test_unverified_portion_outside_published_window_stays_blocked(self):
        samples=[point(30.,speed=0.),point(30.,speed=0.,at=1.)]
        result=check(region(((10,20),),length=100),samples)
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)

    def test_empty_permission_list_does_not_authorize_shared_border(self):
        self.assertEqual('CORRIDOR_COLLISION',check(region(()))['reason_code'])

    def test_other_border_remains_prohibited_even_inside_larger_drivable_union(self):
        samples=[point(10.,-3.,speed=0.),point(10.,-3.,speed=0.,at=1.)]
        result=check(region(polygons=[rectangle(-5.,-6.,30.,6.)]),samples)
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)

    def test_lane_end_cannot_be_used_to_bypass_a_finite_permission_window(self):
        corridor=region(polygons=[rectangle(-5.,-6.,30.,6.)])
        result=check(corridor,[point(18.,-1.5),point(22.,-1.5,at=4.)])
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)
        self.assertTrue(0.<result['details']['relative_time_s']<4.)

    def test_gap_is_found_between_endpoints_and_midpoint(self):
        corridor=region(((5,Fraction(46,5)),(Fraction(93,10),15)))
        samples=[point(7.),point(13.,at=6.)]
        result=check(corridor,samples)
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)
        self.assertTrue(1.<result['details']['relative_time_s']<2.5)

    def test_reverse_uses_same_whole_body_restriction(self):
        result=check(region(),[point(12.),point(8.,at=4.)],direction=-1)
        self.assertEqual('safe',result['status'],result)

    def test_dynamic_target_and_prediction_horizon_still_apply(self):
        samples=[point(8.),point(12.,at=4.)]
        result=check(region(),samples,objects=[obstacle(10.,0.)])
        self.assertEqual('OBSTACLE_COLLISION',result['reason_code'],result)
        result=check(region(),samples,objects=[obstacle(100.,0.,horizon=1.)])
        self.assertEqual('PREDICTION_HORIZON',result['reason_code'],result)

    def test_frame_budget_resumes_source_barriers_before_any_safe_result(self):
        corridor=region()
        pending=0
        for unused in range(100):
            result=check(corridor,work=budget(max_checks=80))
            if result['reason_code']=='BUDGET_EXHAUSTED':
                pending+=1
                self.assertIsNone(result['clearance_lower_bound_m'])
            else: break
        self.assertGreater(pending,1)
        self.assertEqual('safe',result['status'],result)

    def test_source_records_are_immutable_bounded_and_ordered(self):
        normal=region()._source_descriptor
        cases=[(normal[0],normal[1],'unknown',normal[3]),
               (normal[0],normal[1],'left',((encode_position(0,Fraction(3,4)),encode_position(0,Fraction(1,4))),)),
               (normal[0],normal[1],'left',(([0,'1','4'],[0,'3','4']),))]
        for descriptor in cases:
            corridor=CrossingCorridor(region().polygons,descriptor,2000000)
            result=check(corridor)
            self.assertEqual('invalid',result['status'],result)

    def test_curved_right_side_opening_retains_original_vertices(self):
        def border(radius):
            return tuple((radius*math.cos(i*.006),radius*math.sin(i*.006),0.) for i in range(101))
        left,shared,right=border(98.25),border(101.75),border(105.25)
        polygons=tuple(tuple(p[:2] for p in a)+tuple(p[:2] for p in reversed(b))
                       for a,b in ((left,shared),(shared,right)))
        openings=((encode_position(25,Fraction(0)),encode_position(75,Fraction(0))),)
        corridor=CrossingCorridor(polygons,(left,shared,'right',openings),2000000)
        samples=[point(101.75*math.cos(a),101.75*math.sin(a),heading=a+math.pi/2,
                       at=101.75*(a-.25)) for a in (.25,.3,.35)]
        result=check(corridor,samples)
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(sum(len(p) for p in polygons),sum(len(p.points) for p in corridor._prepared.polygons))
        samples=[point(101.75*math.cos(.1),101.75*math.sin(.1),speed=0.,heading=.1+math.pi/2,at=t)
                 for t in (0.,1.)]
        self.assertEqual('CORRIDOR_COLLISION',check(corridor,samples)['reason_code'])

    def test_right_side_straight_permission_does_not_authorize_left_border(self):
        left=((0.,3.,0.),(20.,3.,0.)); right=((0.,0.,0.),(20.,0.,0.))
        openings=((encode_position(0,Fraction(1,4)),encode_position(0,Fraction(3,4))),)
        corridor=CrossingCorridor((tuple(rectangle(-5.,-6.,30.,6.)),),(left,right,'right',openings),2000000)
        self.assertEqual('safe',check(corridor)['status'])
        samples=[point(10.,3.,speed=0.,at=t) for t in (0.,1.)]
        self.assertEqual('CORRIDOR_COLLISION',check(corridor,samples)['reason_code'])


class FormalCrossingSceneTests(unittest.TestCase):
    def setUp(self):
        self.fixture=producer_fixture.FactConsumerTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def scene(self,cache=None):
        f=self.fixture
        return read_maneuver_scene(f.p,['1_0_-1','1_0_-2'],f.clock,cache)

    def test_actual_bound_producer_can_supply_whole_source_constraint(self):
        self.fixture.bound()
        scene=self.scene()
        samples=[point(10.,1.75,speed=0.),point(10.,1.75,speed=0.,at=1.)]
        result=check(scene['legal_crossing_corridor'],samples)
        self.assertEqual('safe',result['status'],result)
        samples=[point(1.2,1.75,speed=0.),point(1.2,1.75,speed=0.,at=1.)]
        self.assertEqual('safe',check(scene['corridor'],samples)['status'])
        self.assertEqual('CORRIDOR_COLLISION',check(scene['legal_crossing_corridor'],samples)['reason_code'])

    def test_sdk_marking_mismatch_blocks_later_piece_in_same_local_road(self):
        f=self.fixture
        f.adapter.hdmap.getRoadMark=lambda p,n:NS(exists=True,
            left=NS(type='broken' if p[0]<40. else 'solid',sOffset=0.),
            right=NS(type='broken' if p[0]<40. else 'solid',sOffset=0.))
        f.bound()
        scene=self.scene()
        samples=[point(45.,1.75,speed=0.),point(45.,1.75,speed=0.,at=1.)]
        self.assertEqual('safe',check(scene['corridor'],samples)['status'])
        result=check(scene['legal_crossing_corridor'],samples)
        self.assertEqual('CORRIDOR_COLLISION',result['reason_code'],result)

    def test_both_original_borders_must_match_current_source(self):
        f=self.fixture
        f.bound()
        self.scene()
        f.p.lane.right_boundary=[(0.,-5.,0.),(1000.,-5.,0.)]
        with self.assertRaises(ValueError): self.scene()

    def test_cache_retains_progress_for_road_visibility_and_source_constraint(self):
        f=self.fixture; f.bound()
        with self.assertRaises(ValueError): self.scene(ManeuverGeometryCache(2,100000))
        cache=ManeuverGeometryCache(3,100000)
        first=self.scene(cache)
        samples=[point(10.,1.75,speed=0.),point(10.,1.75,speed=0.,at=1.)]
        for unused in range(100):
            scene=self.scene(cache)
            self.assertIs(first['legal_crossing_corridor'],scene['legal_crossing_corridor'])
            result=check(scene['legal_crossing_corridor'],samples,work=budget(max_checks=80))
            if result['reason_code']!='BUDGET_EXHAUSTED': break
        self.assertEqual('safe',result['status'],result)
        f.now+=.21
        with self.assertRaises(ValueError): self.scene(cache)


if __name__=='__main__': unittest.main()
