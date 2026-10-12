"""Resumable static geometry through actual P02 calls and formal scene reads.

All geometry/calibration and object examples are synthetic. The checks prove
budgeted preparation and current input gates, not real-time or driving ability.
"""
import copy
import math
import unittest
from unittest.mock import patch

from members.planning.candidate_validation import CorridorRegion,validate_candidate
from members.planning.maneuver_scene import ManeuverGeometryCache,read_maneuver_scene
from tests.test_planning_candidate_validation import point,vehicle,limits,budget,rectangle,obstacle
from tests import test_maneuver_fact_consumers as producer_fixture


def frozen(polygons):
    return tuple(tuple(tuple(p) for p in ring) for ring in polygons)


class IncrementalRegionTests(unittest.TestCase):
    def region(self,polygons=None,total=2000000):
        return CorridorRegion(frozen(polygons or [rectangle(-2.,-2.,5.,0.),rectangle(-2.,0.,5.,2.)]),
                              preparation_max_checks=total)

    def call(self,region,checks=80,points=None,objects=None,motion=None,**options):
        return validate_candidate(points or [point(0.),point(2.,at=2.)],1,vehicle(),motion or limits(),
            region,objects or [],budget(max_checks=checks,**options))

    def finish(self,region,checks=80,maximum=1000,**options):
        reports=[]
        for unused in range(maximum):
            report=self.call(region,checks,**options)
            reports.append(report)
            if report['reason_code']!='BUDGET_EXHAUSTED': return reports
        self.fail('Synthetic incremental preparation did not finish')

    def test_many_small_frame_budgets_complete_the_same_union(self):
        region=self.region()
        reports=self.finish(region)
        self.assertGreater(len(reports),1)
        self.assertTrue(all(v['status']=='inconclusive' and v['clearance_lower_bound_m'] is None
                            for v in reports[:-1]),reports)
        self.assertEqual('safe',reports[-1]['status'],reports[-1])
        original=CorridorRegion(copy.deepcopy(region.polygons))
        expected=self.call(original,checks=10000)
        self.assertAlmostEqual(expected['clearance_lower_bound_m'],reports[-1]['clearance_lower_bound_m'])

    def test_partial_geometry_is_never_published_as_prepared(self):
        region=self.region()
        report=self.call(region,checks=20)
        self.assertEqual('inconclusive',report['status'],report)
        self.assertFalse(report['details']['preparation_complete'])
        self.assertGreater(report['details']['preparation_checks'],0)
        self.assertIsNone(region._prepared)
        self.assertIsNone(region._fingerprint)

    def test_elapsed_deadline_interrupts_driver_and_keeps_progress(self):
        region=self.region()
        first=self.call(region,checks=20)
        previous=first['details']['preparation_checks']
        ticks=iter(1.+i*.001 for i in range(1000))
        with patch('members.planning.candidate_validation.time.monotonic',side_effect=lambda:next(ticks)):
            report=self.call(region,checks=10000,deadline_monotonic_s=1.009)
        self.assertEqual('BUDGET_EXHAUSTED',report['reason_code'],report)
        self.assertGreater(region._job_steps,previous)
        self.assertEqual('safe',self.finish(region)[-1]['status'])

    def test_expired_call_does_no_more_preparation(self):
        region=self.region()
        self.call(region,checks=20)
        previous=region._job_steps
        report=self.call(region,deadline_monotonic_s=.1)
        self.assertEqual('BUDGET_EXHAUSTED',report['reason_code'])
        self.assertEqual(previous,region._job_steps)

    def test_changed_input_discards_pending_geometry(self):
        region=self.region()
        self.call(region,checks=20)
        region.polygons=frozen([rectangle(-2.,-2.,5.,.1)])
        report=self.finish(region)[-1]
        self.assertEqual('CORRIDOR_COLLISION',report['reason_code'],report)

    def test_changed_input_cannot_reuse_completed_geometry(self):
        region=self.region()
        self.assertEqual('safe',self.finish(region)[-1]['status'])
        region.polygons=frozen([rectangle(-2.,-2.,5.,.1)])
        self.assertEqual('CORRIDOR_COLLISION',self.finish(region)[-1]['reason_code'])

    def test_cached_geometry_does_not_cache_obstacles(self):
        region=self.region()
        self.assertEqual('safe',self.finish(region)[-1]['status'])
        report=self.call(region,objects=[obstacle(1.,0.)])
        self.assertEqual('OBSTACLE_COLLISION',report['reason_code'],report)

    def test_cached_geometry_does_not_cache_limits_or_prediction_horizon(self):
        region=self.region()
        self.finish(region)
        report=self.call(region,motion=limits(max_speed_mps=.5))
        self.assertEqual('MOTION_LIMIT',report['reason_code'],report)
        report=self.call(region,objects=[obstacle(100.,0.,horizon=1.)])
        self.assertEqual('PREDICTION_HORIZON',report['reason_code'],report)

    def test_exact_self_intersection_stays_invalid_on_repeated_calls(self):
        region=self.region([[(0.,0.),(2.,2.),(0.,2.),(3.,0.)]])
        first=self.finish(region,checks=12)[-1]
        self.assertEqual('invalid',first['status'],first)
        self.assertIn('exact region self intersection',first['details']['message'])
        previous=region._job_steps
        self.assertEqual('invalid',self.call(region)['status'])
        self.assertEqual(previous,region._job_steps)

    def test_adjacent_backtracking_is_rejected_without_float_precheck(self):
        region=self.region([[(0.,0.),(10.,0.),(5.,0.),(10.,5.),(0.,5.)]])
        report=self.finish(region,checks=12)[-1]
        self.assertEqual('invalid',report['status'],report)
        self.assertIn('exact adjacent region edges overlap',report['details']['message'])

    def test_positive_gap_is_retained_after_many_resumes(self):
        region=self.region([rectangle(-2.,-2.,5.,0.),rectangle(-2.,.02,5.,2.)])
        report=self.finish(region,checks=40)[-1]
        self.assertEqual('CORRIDOR_COLLISION',report['reason_code'],report)

    def test_dense_curve_retains_all_native_vertices_and_proves_same_clearance(self):
        from perception.road_regions import local_region
        count=301
        def boundary(radius):
            return [(radius*math.cos(i*.8/(count-1)),radius*math.sin(i*.8/(count-1)),0.)
                    for i in range(count)]
        polygons=[]
        for radius in (100.,103.5):
            center=boundary(radius)
            polygons.append(local_region(center,boundary(radius-1.75),boundary(radius+1.75),
                                         [center[30],center[270]])['polygon'])
        region=self.region(polygons)
        samples=[point(101.75*math.cos(a),101.75*math.sin(a),heading=a+math.pi/2,
                       at=101.75*(a-.3)) for a in (.3,.31,.32)]
        reports=self.finish(region,checks=1000,points=samples)
        self.assertGreater(len(reports),10)
        self.assertEqual('safe',reports[-1]['status'],reports[-1])
        self.assertEqual(sum(len(v) for v in polygons),sum(len(v.points) for v in region._prepared.polygons))
        expected=self.call(CorridorRegion(polygons),checks=2000000,points=samples)
        self.assertEqual('safe',expected['status'],expected)
        self.assertAlmostEqual(expected['clearance_lower_bound_m'],reports[-1]['clearance_lower_bound_m'])

    def test_total_preparation_limit_cannot_be_reset_by_new_frame_budgets(self):
        region=self.region(total=30)
        report=self.finish(region,checks=20)[-1]
        self.assertEqual('REGION_PREPARATION_LIMIT',report['reason_code'],report)
        self.assertEqual('inconclusive',report['status'])
        self.assertIsNone(report['clearance_lower_bound_m'])
        previous=region._job_steps
        for unused in range(3):
            self.assertEqual('REGION_PREPARATION_LIMIT',self.call(region,checks=10000)['reason_code'])
        self.assertEqual(previous,region._job_steps)
        region.polygons=frozen([rectangle()])
        self.assertGreater(self.finish(region,checks=20)[-1]['details'].get('message','').find('REGION_TOTAL'),-1)

    def test_incremental_input_requires_deeply_immutable_valid_bounded_geometry(self):
        raw=[rectangle()]
        cases=[raw,tuple(raw),(tuple([list(p) for p in rectangle()]),),
               (((0.,0.),(1.,0.),(0.,float('nan'))),),frozen([rectangle()]*33),
               (((0.,0.),(1.,0.),(0.,0.)),)]
        for raw in cases:
            region=CorridorRegion(raw,preparation_max_checks=10000)
            report=self.finish(region,checks=20)[-1]
            self.assertEqual('invalid',report['status'],report)
        for total in (0,-1,True,1.,float('nan')):
            with self.assertRaises(ValueError): CorridorRegion(frozen([rectangle()]),preparation_max_checks=total)

    def test_cache_is_bounded_and_changed_scope_drops_old_jobs(self):
        cache=ManeuverGeometryCache(2,10000)
        first=cache.region(('case','task',1,'map'),('road',),[rectangle()])
        self.assertIs(first,cache.region(('case','task',1,'map'),('road',),[rectangle()]))
        for x in (10.,20.): cache.region(('case','task',1,'map'),('road',),[rectangle(xmin=x)])
        self.assertEqual(2,len(cache._entries))
        self.assertIsNot(first,cache.region(('case','task',1,'other-map'),('road',),[rectangle()]))
        self.assertEqual(1,len(cache._entries))
        cache.clear()
        self.assertEqual(0,len(cache._entries))

    def test_cache_budget_must_be_explicit_and_bounded(self):
        for entries,total in ((0,10),(1,10),(17,10),(True,10),(2,0),(2,True)):
            with self.assertRaises(ValueError): ManeuverGeometryCache(entries,total)


class FormalSceneCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture=producer_fixture.FactConsumerTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cache=ManeuverGeometryCache(4,100000)

    def scene(self):
        f=self.fixture
        return read_maneuver_scene(f.p,[f.p.lane.lane_id],f.clock,self.cache)

    def test_fresh_formal_reads_reuse_only_static_shape_and_keep_original_deadline(self):
        first=self.scene()
        samples=[point(10.),point(12.,at=2.)]
        f=self.fixture
        for unused in range(100):
            scene=self.scene()
            with patch('members.planning.candidate_validation.time.monotonic',f.clock):
                report=validate_candidate(samples,1,vehicle(),limits(),scene['corridor'],[],
                                          budget(max_checks=80,deadline_monotonic_s=f.p.valid_until))
            if report['reason_code']!='BUDGET_EXHAUSTED': break
        else: self.fail('Formal road preparation did not finish')
        self.assertEqual('safe',report['status'],report)
        self.assertIs(first['corridor'],scene['corridor'])
        self.assertEqual(100.2,scene['source_valid_until_s'])
        f.now+=.21
        with self.assertRaises(ValueError): self.scene()

    def test_missing_dynamic_coverage_cannot_borrow_a_cached_road(self):
        self.scene()
        self.fixture.p.maneuver_environment.coverage_regions=[]
        with self.assertRaises(ValueError): self.scene()

    def test_cached_object_with_replaced_geometry_is_rebuilt_from_current_facts(self):
        first=self.scene()
        first['corridor'].polygons=frozen([rectangle(-1000.,-1000.,1000.,1000.)])
        current=self.scene()
        self.assertIsNot(first['corridor'],current['corridor'])
        self.assertEqual(75.,max(p[0] for p in current['corridor'].polygons[0]))
        self.assertEqual(100.2,current['source_valid_until_s'])

    def test_independent_sensor_expiry_cannot_borrow_cached_static_geometry(self):
        original=self.scene()
        f=self.fixture
        f.now+=.1
        f.p.maneuver_environment.dynamic_valid_until=f.now-.01
        self.assertLess(f.now,f.p.valid_until)
        with self.assertRaises(ValueError): self.scene()
        self.assertIsNotNone(original['corridor'])

    def test_unverified_map_identity_cannot_borrow_cached_static_geometry(self):
        self.scene()
        self.fixture.p.maneuver_environment.map_semantics['geometry_bound']=False
        with self.assertRaises(ValueError): self.scene()

    def test_same_static_shape_with_new_object_keeps_new_source_state(self):
        from tests.test_maneuver_perception import target
        first=self.scene()
        f=self.fixture
        f.now+=.01; f.p.frame_id+=1; f.p.targets_frame_id=f.p.frame_id
        f.p.targets=[target(10.,0.)]
        f.build()
        current=self.scene()
        self.assertIs(first['corridor'],current['corridor'])
        self.assertEqual([],first['objects'])
        self.assertEqual(9,current['objects'][0]['id'])
        self.assertEqual(f.now,current['source_observed_at_s'])


if __name__=='__main__': unittest.main()
