"""Authored P03 trajectories; independent geometry/time expectations.

No SDK initialization, control send, actual calibration or scoring evidence.
"""
import math
import time
import unittest
from fractions import Fraction
from unittest.mock import patch

from core.boundary_lineage import encode_position
from members.planning.lane_change_generator import (
    LaneChangeSearch, PredictionEnvelope, generate_lane_change, plan_lane_change_candidate,
)
from members.planning.crossing_corridor import CrossingCorridor
from members.planning.candidate_validation import CorridorRegion
from members.planning.maneuver_scene import ManeuverGeometryCache
from tests.test_planning_candidate_validation import vehicle, limits, budget, rectangle, obstacle
from tests import test_maneuver_fact_consumers as producer_fixture
from tests.test_maneuver_perception import target


def search(**changes):
    values = dict(distances_m=(25.,), tangent_scales=(1.,), speed_scales=(1.,),
                  spacing_m=.4, max_points=1000, projection_ambiguity_m=.05)
    values.update(changes)
    return LaneChangeSearch(**values)


def crossing(side='left', openings=((0.,100.),), start=0., end=100., width=3.5):
    left=((start,width/2,0.),(end,width/2,0.))
    right=((start,-width/2,0.),(end,-width/2,0.))
    records=tuple((encode_position(0,Fraction(a-start)/Fraction(end-start)),
                   encode_position(0,Fraction(b-start)/Fraction(end-start))) for a,b in openings)
    polygons=[rectangle(start,-width/2,end,width/2)]
    polygons.append(rectangle(start,width/2,end,1.5*width) if side=='left'
                    else rectangle(start,-1.5*width,end,-width/2))
    return CrossingCorridor(tuple(tuple(tuple(p) for p in v) for v in polygons),
                            (left,right,side,records),1000000)


class LaneChangeGeneratorTests(unittest.TestCase):
    def run_search(self, **changes):
        values=dict(start_pose=(10.,0.,0.), initial_speed_mps=2., initial_curvature_m_inv=0.,
                    target_reference=[(0.,3.5),(100.,3.5)], speed_cap_mps=3.,
                    vehicle=vehicle(3.9,.9,.9,2.9), limits=limits(),
                    crossing_corridor=crossing(), coverage=CorridorRegion([rectangle(-20.,-20.,120.,20.)]),
                    target_corridor=CorridorRegion([rectangle(0.,1.75,100.,5.25)]),
                    obstacles=[], search=search(), budget=budget(max_checks=1000000,deadline_monotonic_s=time.monotonic()+20.))
        values.update(changes)
        return generate_lane_change(**values)

    def assert_ready(self, result):
        self.assertEqual('safe',result['status'],result)
        self.assertEqual('safe',result['legal_report']['status'])
        self.assertEqual('safe',result['visibility_report']['status'])
        self.assertEqual('nominal_candidate_only',result['authority'])
        self.assertTrue(result['points'])

    def test_left_changes_lateral_position_with_actual_start_and_target_tangent(self):
        result=self.run_search(); self.assert_ready(result)
        points=result['points']
        self.assertEqual((10.,0.,0.,2.,0.),(points[0].x,points[0].y,points[0].heading,
                                        points[0].speed,points[0].relative_time))
        self.assertAlmostEqual(35.,points[-1].x)
        self.assertAlmostEqual(3.5,points[-1].y)
        self.assertAlmostEqual(0.,points[-1].heading)
        self.assertTrue(all(b.y>=a.y for a,b in zip(points,points[1:])))
        # With zero endpoint lateral derivatives, quintic Hermite y(u) is
        # 3.5*(10*u^3-15*u^4+6*u^5), independent of the code's coefficients.
        for i in (len(points)//4,len(points)//2,3*len(points)//4):
            u=float(i)/(len(points)-1)
            self.assertAlmostEqual(3.5*(10*u**3-15*u**4+6*u**5),points[i].y,places=10)

    def test_right_is_mirror_and_time_speed_are_preserved(self):
        left=self.run_search(); right=self.run_search(target_reference=[(0.,-3.5),(100.,-3.5)],
                                                      crossing_corridor=crossing('right'),
                                                      target_corridor=CorridorRegion([rectangle(0.,-5.25,100.,-1.75)]))
        self.assert_ready(left); self.assert_ready(right)
        for a,b in zip(left['points'],right['points']):
            self.assertAlmostEqual(a.x,b.x)
            self.assertAlmostEqual(a.y,-b.y)
            self.assertAlmostEqual(a.heading,-b.heading)
            self.assertAlmostEqual(a.speed,b.speed)
            self.assertAlmostEqual(a.relative_time,b.relative_time)

    def test_profile_accelerates_from_rest_without_a_speed_jump(self):
        result=self.run_search(initial_speed_mps=0.); self.assert_ready(result)
        points=result['points']; self.assertEqual(0.,points[0].speed)
        self.assertGreater(points[1].speed,0.)
        for a,b in zip(points,points[1:]):
            distance=math.hypot(b.x-a.x,b.y-a.y); dt=b.relative_time-a.relative_time
            self.assertGreater(dt,0.)
            self.assertAlmostEqual(distance,.5*(a.speed+b.speed)*dt,places=10)
            self.assertLessEqual((b.speed-a.speed)/dt,3.+1e-9)
            self.assertLessEqual((a.speed-b.speed)/dt,4.+1e-9)

    def test_search_skips_too_short_connector_and_finds_feasible_longer_one(self):
        result=self.run_search(search=search(distances_m=(3.,25.)))
        self.assert_ready(result)
        self.assertGreater(len(result['attempts']),1)
        self.assertEqual(25.,result['attempts'][-1]['distance_m'])

    def test_moving_rear_intrusion_rejects_nominal_crossing(self):
        # An object initially behind ego reaches its intermediate lateral path.
        result=self.run_search(obstacles=[obstacle(0.,3.5,length=4.,width=2.,vx=6.)])
        self.assertFalse(result['points'],result)
        self.assertTrue(any(a.get('legal_report',{}).get('reason_code')=='OBSTACLE_COLLISION'
                            for a in result['attempts']),result)

    def test_unknown_orientation_still_constrains_full_path(self):
        result=self.run_search(obstacles=[obstacle(22.,1.75,length=4.,width=2.,heading=None)])
        self.assertFalse(result['points'])
        self.assertEqual('OBSTACLE_COLLISION',result['attempts'][0]['legal_report']['reason_code'])

    def test_finite_prediction_horizon_cannot_be_extended_for_path(self):
        result=self.run_search(obstacles=[obstacle(80.,3.5,horizon=.2)])
        self.assertFalse(result['points'])
        self.assertEqual('PREDICTION_HORIZON',result['attempts'][0]['legal_report']['reason_code'])

    def test_observed_free_road_outside_complete_view_is_rejected(self):
        result=self.run_search(coverage=CorridorRegion([rectangle(-5.,-5.,25.,6.)]))
        self.assertFalse(result['points'])
        self.assertEqual('safe',result['attempts'][0]['legal_report']['status'])
        self.assertEqual('CORRIDOR_COLLISION',result['attempts'][0]['visibility_report']['reason_code'])

    def test_solid_or_unverified_shared_border_is_not_removed_by_road_union(self):
        for openings in ((),((0.,14.),),((60.,100.),)):
            result=self.run_search(crossing_corridor=crossing(openings=openings))
            self.assertFalse(result['points'],result)
            self.assertEqual('CORRIDOR_COLLISION',result['attempts'][0]['legal_report']['reason_code'])

    def test_source_caps_and_overhang_prevent_short_road_end_extrapolation(self):
        result=self.run_search(crossing_corridor=crossing(end=35.,openings=((0.,35.),)))
        self.assertFalse(result['points'])
        self.assertEqual('CORRIDOR_COLLISION',result['attempts'][0]['legal_report']['reason_code'])

    def test_plain_union_cannot_be_substituted_for_crossing_authority(self):
        result=self.run_search(crossing_corridor=CorridorRegion([rectangle()]))
        self.assertEqual('SOURCE_BOUND_CROSSING_CORRIDOR_REQUIRED',result['reason_code'])
        self.assertFalse(result['points'])

    def test_target_must_not_extrapolate_or_flip_direction(self):
        for reference,code in (([(20.,3.5),(100.,3.5)],'TARGET_REFERENCE_EXTRAPOLATION'),
                               ([(100.,3.5),(0.,3.5)],'TARGET_REFERENCE_OPPOSITE_DIRECTION')):
            result=self.run_search(target_reference=reference)
            self.assertEqual(code,result['reason_code']); self.assertFalse(result['points'])

    def test_loop_projection_has_no_arbitrary_branch_choice(self):
        reference=[(0.,3.5),(100.,3.5),(100.,-3.5),(0.,-3.5)]
        result=self.run_search(target_reference=reference)
        self.assertEqual('TARGET_REFERENCE_BRANCH_AMBIGUOUS',result['reason_code'])

    def test_repeated_vertices_are_cleaned_and_nonfinite_values_rejected(self):
        result=self.run_search(target_reference=[(0.,3.5),(0.,3.5),(100.,3.5)])
        self.assert_ready(result)
        for reference in ([(0.,3.5),(float('nan'),3.5)],[(0.,3.5),(0.,3.5)]):
            result=self.run_search(target_reference=reference)
            self.assertFalse(result['points']); self.assertEqual('invalid',result['status'])

    def test_missing_motion_geometry_and_initial_overspeed_are_rejected(self):
        for changes in (dict(initial_curvature_m_inv=None),dict(vehicle=None),dict(limits=None),
                        dict(initial_speed_mps=-1.),dict(initial_speed_mps=4.)):
            result=self.run_search(**changes)
            self.assertFalse(result['points']); self.assertEqual('invalid',result['status'])

    def test_caller_search_bounds_and_point_limit_are_enforced(self):
        for options in (search(max_points=True),search(speed_scales=(2.,)),search(spacing_m=0.),
                        search(distances_m=tuple(range(1,9)),tangent_scales=tuple(range(1,9)),speed_scales=(1.,.5))):
            result=self.run_search(search=options)
            self.assertEqual('invalid',result['status']); self.assertFalse(result['points'])
        result=self.run_search(search=search(max_points=3,spacing_m=.01))
        self.assertEqual('inconclusive',result['status']); self.assertFalse(result['points'])

    def test_one_budget_covers_generation_all_candidates_and_both_sweeps(self):
        for options in (budget(max_checks=10),budget(deadline_monotonic_s=time.monotonic()-1.)):
            result=self.run_search(budget=options)
            self.assertEqual('BUDGET_EXHAUSTED',result['reason_code']); self.assertFalse(result['points'])
            self.assertLessEqual(result['checks'],options.max_checks+1)
        first=self.run_search(); self.assert_ready(first)
        spent=first['checks']-first['visibility_report']['checks']
        result=self.run_search(budget=budget(max_checks=spent+5))
        self.assertEqual('BUDGET_EXHAUSTED',result['reason_code']); self.assertFalse(result['points'])

    def test_target_polyline_corner_is_not_given_an_arbitrary_tangent(self):
        result=self.run_search(target_reference=[(0.,3.5),(35.,3.5),(60.,10.)])
        self.assertFalse(result['points'])
        self.assertEqual('TARGET_TANGENT_AT_CORNER_UNKNOWN',result['attempts'][0]['reason_code'])

    def test_frozen_goal_cannot_select_one_tangent_at_a_changed_corner(self):
        result=self.run_search(target_reference=[(0.,3.5),(35.,3.5),(60.,10.)],
                               fixed_goal_pose=(35.,3.5,0.))
        self.assertFalse(result['points'])
        self.assertEqual('TARGET_TANGENT_AT_CORNER_UNKNOWN',result['reason_code'])

    def test_curved_lane_uses_actual_start_yaw_and_nominal_curvature(self):
        def arc(radius):
            return tuple((radius*math.cos(-.2+i*.06),radius*math.sin(-.2+i*.06),0.) for i in range(21))
        left,right,outer=arc(98.25),arc(101.75),arc(94.75)
        polygons=tuple(tuple(p[:2] for p in a)+tuple(p[:2] for p in reversed(b))
                       for a,b in ((left,right),(outer,left)))
        permission=((encode_position(0,Fraction(0)),encode_position(19,Fraction(1))),)
        region=CrossingCorridor(polygons,(left,right,'left',permission),1000000)
        reference=arc(96.5)
        options=dict(start_pose=(100.,0.,math.pi/2),initial_curvature_m_inv=.01,
            target_reference=reference,crossing_corridor=region,
            target_corridor=CorridorRegion([polygons[1]]),
            coverage=CorridorRegion([rectangle(40.,-30.,120.,110.)]),search=search(distances_m=(24.,)))
        result=self.run_search(**options)
        self.assert_ready(result)
        self.assertEqual(math.pi/2,result['points'][0].heading)
        self.assertGreater(result['points'][-1].heading,math.pi/2)
        self.assertLess(result['points'][-1].x,100.)
        # Replan on the curve from a later actual pose while keeping the
        # selected endpoint and its original tangent, rather than advancing it.
        index=len(result['points'])//3
        a,b=result['points'][index:index+2]
        curvature=(b.heading-a.heading)/math.hypot(b.x-a.x,b.y-a.y)
        options.update(start_pose=(a.x,a.y,a.heading),initial_speed_mps=a.speed,
                       initial_curvature_m_inv=curvature,fixed_goal_pose=result['target_pose'])
        replanned=self.run_search(**options)
        self.assert_ready(replanned)
        self.assertEqual(result['target_pose'],replanned['target_pose'])

    def test_terminal_body_must_fit_target_lane_separately_from_union(self):
        result=self.run_search(target_corridor=CorridorRegion([rectangle(0.,3.,100.,4.)]))
        self.assertFalse(result['points'])
        self.assertEqual('safe',result['attempts'][0]['legal_report']['status'])
        self.assertEqual('safe',result['attempts'][0]['visibility_report']['status'])
        self.assertEqual('CORRIDOR_COLLISION',result['attempts'][0]['target_pose_report']['reason_code'])

    def test_rotation_translation_and_start_in_lane_middle_preserve_maneuver(self):
        angle=.7; c,s=math.cos(angle),math.sin(angle)
        def transform(p):
            return (10000.+c*p[0]-s*p[1],-3000.+s*p[0]+c*p[1])
        normal=crossing()
        left,right,side,openings=normal._source_descriptor
        lines=tuple(tuple(transform(p)+(0.,) for p in line) for line in (left,right))
        region=CrossingCorridor(tuple(tuple(transform(p) for p in v) for v in normal.polygons),
                                (lines[0],lines[1],side,openings),1000000)
        start=transform((40.,0.))
        result=self.run_search(start_pose=start+(angle,),target_reference=[transform((0.,3.5)),transform((100.,3.5))],
            crossing_corridor=region,target_corridor=CorridorRegion([region.polygons[1]]),
            coverage=CorridorRegion([tuple(transform(p) for p in rectangle(-20.,-20.,120.,20.))]))
        self.assert_ready(result)
        goal=transform((65.,3.5))
        self.assertAlmostEqual(goal[0],result['points'][-1].x)
        self.assertAlmostEqual(goal[1],result['points'][-1].y)
        self.assertAlmostEqual(angle,result['points'][0].heading)

    def test_slow_candidate_search_rechecks_steering_rate_and_timing(self):
        result=self.run_search(initial_speed_mps=.3,limits=limits(max_steer_rate_rad_s=.02),
                               search=search(speed_scales=(1.,.1)))
        self.assert_ready(result)
        self.assertEqual(.1,result['attempts'][-1]['speed_scale'])
        self.assertEqual('INITIAL_STEERING_RATE_LIMIT',result['attempts'][0]['reason_code'])
        self.assertLessEqual(max(p.speed for p in result['points']),.3+1e-9)

    def test_initial_yaw_is_not_silently_snapped_to_source_reference(self):
        result=self.run_search(start_pose=(10.,.1,.04))
        self.assert_ready(result)
        self.assertEqual(.04,result['points'][0].heading)
        self.assertEqual(.1,result['points'][0].y)


class FormalLaneChangeCandidateTests(unittest.TestCase):
    def setUp(self):
        self.fixture=producer_fixture.FactConsumerTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.bound()

    def plan(self, **changes):
        f=self.fixture
        values=dict(perception=f.p,target_lane_id='1_0_-2',initial_curvature_m_inv=0.,speed_cap_mps=3.,
            vehicle=vehicle(3.9,.9,.9,2.9),limits=limits(),search=search(),
            budget=budget(max_checks=1000000),prediction_envelopes={},clock=f.clock)
        values.update(changes)
        return plan_lane_change_candidate(**values)

    def test_actual_formal_producer_to_certified_geometry_and_original_deadline(self):
        result=self.plan()
        self.assertEqual('safe',result['status'],result)
        self.assertEqual(self.fixture.p.frame_id,result['frame_id'])
        self.assertEqual('1_0_-1',result['source_lane_id'])
        self.assertEqual('1_0_-2',result['target_lane_id'])
        self.assertEqual(self.fixture.now,result['nominal_start_time_s'])
        self.assertLessEqual(result['source_valid_until_s'],self.fixture.p.valid_until)

    def test_empty_sensor_packet_without_actual_visibility_never_authorizes_candidate(self):
        self.fixture.p.maneuver_environment.coverage_regions=[]
        result=self.plan(); self.assertFalse(result['points'])
        self.assertNotEqual('safe',result['status'])

    def test_missing_prediction_bounds_not_inferred_from_probability(self):
        self.fixture.p.targets=[target(60.,3.5)]; self.fixture.build()
        result=self.plan(); self.assertEqual('PREDICTION_OBJECT_SET_MISMATCH',result['reason_code'])
        result=self.plan(prediction_envelopes={9:None})
        self.assertEqual('PREDICTION_ENVELOPE_UNAVAILABLE',result['reason_code'])

    def test_original_object_age_and_uncertainty_are_preserved(self):
        f=self.fixture; f.p.targets=[target(80.,3.5,heading=None)]
        f.p.source_status['targets']['age_ms']=40; f.build()
        f.p.targets[0].vx=2.; f.build()
        seen=[]
        def capture(*args):
            seen.extend(args[9])
            return dict(status='inconclusive',reason_code='SYNTHETIC_CAPTURE',points=[])
        with patch('members.planning.lane_change_generator.generate_lane_change',side_effect=capture):
            result=self.plan(prediction_envelopes={9:PredictionEnvelope(12.,.2,.3)})
        self.assertEqual('SYNTHETIC_CAPTURE',result['reason_code'])
        self.assertEqual(1,len(seen))
        self.assertAlmostEqual(80.+2.*.04,seen[0].x)
        self.assertAlmostEqual(.2+.3*.04,seen[0].position_uncertainty_m)
        self.assertAlmostEqual(12.-.04,seen[0].valid_until_s)
        self.assertIsNone(seen[0].heading)

    def test_source_expiry_during_search_withdraws_all_points(self):
        f=self.fixture
        from members.planning import lane_change_generator as generator
        original=generator.generate_lane_change
        def expired(*args):
            result=original(*args); f.now+=.21; return result
        with patch.object(generator,'generate_lane_change',side_effect=expired): result=self.plan()
        self.assertFalse(result['points']); self.assertNotEqual('safe',result['status'])

    def test_old_source_horizon_and_wrong_lane_identity_are_rejected(self):
        f=self.fixture; f.p.targets=[target(80.,3.5)]
        f.p.source_status['targets']['age_ms']=100; f.build()
        result=self.plan(prediction_envelopes={9:PredictionEnvelope(.05,.1,.1)})
        self.assertEqual('PREDICTION_SOURCE_HORIZON_EXPIRED',result['reason_code'])
        result=self.plan(target_lane_id=f.p.lane.lane_id)
        self.assertEqual('TARGET_ALREADY_CURRENT_LANE',result['reason_code'])

    def test_incremental_preparation_retains_progress_but_refreshes_source_each_call(self):
        cache=ManeuverGeometryCache(4,1000000); pending=0
        for unused in range(30):
            result=self.plan(geometry_cache=cache,budget=budget(max_checks=12600))
            if result['status']=='safe': break
            pending+=1
        self.assertGreater(pending,0)
        self.assertEqual('safe',result['status'],result)
        self.fixture.now+=.21
        expired=self.plan(geometry_cache=cache)
        self.assertFalse(expired['points']); self.assertNotEqual('safe',expired['status'])

    def test_three_entry_cache_is_rejected_without_restart_starvation(self):
        result=self.plan(geometry_cache=ManeuverGeometryCache(3,1000000))
        self.assertEqual('LANE_CHANGE_CACHE_NEEDS_FOUR_ENTRIES',result['reason_code'])

    def test_lane_or_map_change_during_search_withdraws_candidate(self):
        from members.planning import lane_change_generator as generator
        original=generator.generate_lane_change
        f=self.fixture
        def changed(*args):
            result=original(*args)
            f.p.maneuver_environment.map_semantics['digest']='a'*32
            return result
        with patch.object(generator,'generate_lane_change',side_effect=changed): result=self.plan()
        self.assertEqual('LANE_CHANGE_SOURCE_IDENTITY_CHANGED_DURING_SEARCH',result['reason_code'])
        self.assertFalse(result['points'])

    def test_actual_reverse_motion_is_not_reinterpreted_as_forward_speed(self):
        self.fixture.p.ego.vx=-2.; self.fixture.p.ego.speed=2.
        result=self.plan()
        self.assertEqual('ACTUAL_MOTION_OPPOSES_FORWARD_LANE_CHANGE',result['reason_code'])
        self.assertFalse(result['points'])


if __name__=='__main__':
    unittest.main()
