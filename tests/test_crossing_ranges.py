"""SDK-shaped world-border binding, moving ego and bounded incremental work."""
import copy
import math
import unittest
from types import SimpleNamespace as NS

from perception.crossing_ranges import window3,points3,MAX_ATOMIC_STEPS
from perception.maneuver_map import ManeuverMap
from tests.test_maneuver_map import map_adapter,document,road,section,mark,lane,HEADER
from tests.test_maneuver_perception import perception
from tests.test_route_continuation import manager


class Clock(object):
    def __init__(self): self.now=100.
    def __call__(self): return self.now


def fixture(data=None):
    adapter,metadata=map_adapter(data)
    p=perception(); route=manager(adapter.hdmap)
    p.lane=route.update(p.ego,True)
    items=route.read_neighbor_lanes(p.ego,p.lane)
    model=ManeuverMap(adapter); clock=Clock(); model.crossing_ranges.clock=clock
    return adapter,metadata,p,items,model,clock


def run(p,items,model,limit=200):
    for index in range(limit):
        meta,unused=model.observe(p.ego,p.lane,items)
        if all(v['crossing_range_verified'] for v in items):
            return index+1,meta
    raise AssertionError('world range did not become verified: '+repr([v['crossing_scope_reason'] for v in items]))


class CrossingRangeTests(unittest.TestCase):
    def test_actual_route_reader_and_map_producer_bind_world_border_within_budget(self):
        adapter,unused,p,items,model,clock=fixture()
        queries=[]; road_st=adapter.hdmap.getRoadST; road_mark=adapter.hdmap.getRoadMark
        adapter.hdmap.getRoadST=lambda native,point:(queries.append(('st',point)),road_st(native,point))[1]
        adapter.hdmap.getRoadMark=lambda point,native:(queries.append(('mark',point)),road_mark(point,native))[1]
        for frame in range(100):
            before=len(queries)
            meta,unused=model.observe(p.ego,p.lane,items)
            self.assertLessEqual(meta['boundary_binding_budget']['atomic_steps'],MAX_ATOMIC_STEPS)
            self.assertLessEqual(len(queries)-before,MAX_ATOMIC_STEPS+1)  # One current-road-s query precedes binder.
            if items[0]['crossing_range_verified']: break
        self.assertTrue(items[0]['crossing_range_verified'])
        self.assertGreater(frame,0)
        self.assertTrue(items[0]['crossing_geometry_bound'])
        result=items[0]['crossing_boundary_window']
        self.assertEqual('sdk_piecewise_linear_boundary_v1',result['verification_model'])
        self.assertEqual(85.,result['boundary_arc_end_m'])
        for fragment in items[0]['crossing_ranges_world']:
            self.assertTrue(all(abs(v[1]-1.75)<1e-8 for v in fragment['shared_boundary_world']))
            self.assertEqual([v[0] for v in fragment['shared_boundary_world']],fragment['road_s_samples_m'])
        self.assertNotIn('free_regions',result)

    def test_constant_motion_does_not_restart_pending_window_each_frame(self):
        unused,unused,p,items,model,clock=fixture()
        for frame in range(100):
            p.ego.x=10.+frame*.2
            model.observe(p.ego,p.lane,items)
            if items[0]['crossing_range_verified']: break
        self.assertTrue(items[0]['crossing_range_verified'])
        self.assertLess(frame,30)

    def test_both_neighbors_progress_with_one_shared_frame_budget(self):
        lanes=lane(-1,mark())+lane(-2,mark())+lane(-3,mark(kind='solid'))
        data=document(road(sections='<laneSection s="0"><center>'+lane(0,mark(kind='solid'))+'</center><right>'+lanes+'</right></laneSection>'))
        adapter,unused,p,items,model,clock=fixture(data)
        p.lane.lane_id='1_0_-2'
        left=copy.deepcopy(items[0]); left['lane_id']='1_0_-1'
        right=copy.deepcopy(items[0]); right.update(lane_id='1_0_-3',side='right',
            center_line=[(0.,-3.5,0.),(120.,-3.5,0.)],
            left_boundary=[(0.,-1.75,0.),(120.,-1.75,0.)],
            right_boundary=[(0.,-5.25,0.),(120.,-5.25,0.)])
        right['marking_observation']['native_side']='right'
        adapter.hdmap.getRoadMark=lambda *a:NS(exists=True,left=NS(type='broken',sOffset=0.),right=NS(type='broken',sOffset=0.))
        items=[left,right]
        model.observe(p.ego,p.lane,items)
        jobs=list(model.crossing_ranges.jobs.values())
        self.assertEqual(2,len(jobs))
        self.assertEqual([False,False],[v.result is not None for v in jobs])
        count,meta=run(p,items,model)
        self.assertLess(count,60)
        self.assertTrue(all(v['crossing_ranges_world'] for v in items))
        self.assertLessEqual(meta['boundary_binding_budget']['atomic_steps'],MAX_ATOMIC_STEPS)

    def test_broken_to_solid_transition_is_not_extended_by_sdk_paint_length(self):
        data=document(road(sections=section(0,mark()+mark(30,'solid','none'))))
        adapter,unused,p,items,model,clock=fixture(data)
        def markings(point,native):
            value=NS(type='broken' if point[0]<30 else 'solid',sOffset=0. if point[0]<30 else 30.,length=3000.)
            return NS(exists=True,left=value,right=value)
        adapter.hdmap.getRoadMark=markings
        run(p,items,model)
        points=[v for f in items[0]['crossing_ranges_world'] for v in f['shared_boundary_world']]
        self.assertTrue(points)
        self.assertGreaterEqual(min(v[0] for v in points),.25)
        self.assertLessEqual(max(v[0] for v in points),29.75)

    def test_later_mismatch_is_unknown_and_does_not_erase_verified_earlier_fragment(self):
        adapter,unused,p,items,model,clock=fixture()
        adapter.hdmap.getRoadMark=lambda point,native:NS(exists=True,
            left=NS(type='broken' if point[0]<40 else 'solid',sOffset=0.,length=3.),
            right=NS(type='solid',sOffset=0.,length=3.))
        run(p,items,model)
        self.assertFalse(items[0]['crossing_boundary_window']['complete_marking_coverage'])
        self.assertLess(max(v[0] for f in items[0]['crossing_ranges_world'] for v in f['shared_boundary_world']),40.)

    def test_gap_far_from_ego_rejects_shape_even_when_pointwise_join_matches(self):
        unused,unused,p,items,model,clock=fixture()
        items[0]['right_boundary']=[(0.,1.75,0.),(10.,1.75,0.),(70.,2.25,0.),(120.,2.25,0.)]
        for unused in range(100): model.observe(p.ego,p.lane,items)
        self.assertFalse(items[0]['crossing_range_verified'])
        self.assertIn('SHARED_BOUNDARY',items[0]['crossing_scope_reason'])

    def test_far_height_mismatch_is_not_a_shared_crossable_border(self):
        unused,unused,p,items,model,clock=fixture()
        items[0]['right_boundary']=[(0.,1.75,0.),(10.,1.75,0.),(70.,1.75,.5),(120.,1.75,.5)]
        for unused in range(100): model.observe(p.ego,p.lane,items)
        self.assertFalse(items[0]['crossing_range_verified'])

    def test_nonmonotonic_road_s_does_not_bind_a_hairpin_to_wrong_station(self):
        adapter,unused,p,items,model,clock=fixture()
        adapter.hdmap.getRoadST=lambda native,point:NS(exists=True,s=point[0] if point[0]<40 else 80.-point[0])
        for unused in range(100): model.observe(p.ego,p.lane,items)
        self.assertFalse(items[0]['crossing_range_verified'])
        self.assertIn(items[0]['crossing_scope_reason'],('BOUNDARY_ROAD_S_NONMONOTONIC','BOUNDARY_OUTSIDE_LANE_SECTION'))

    def test_transient_native_query_failure_recovers_after_retry_interval(self):
        adapter,unused,p,items,model,clock=fixture(); original=adapter.hdmap.getRoadST
        adapter.hdmap.getRoadST=lambda native,point:NS(exists=False,s=0.) if abs(point[1])>1 else original(native,point)
        for unused in range(50): model.observe(p.ego,p.lane,items)
        self.assertFalse(items[0]['crossing_range_verified'])
        adapter.hdmap.getRoadST=original; clock.now+=.21
        run(p,items,model)
        self.assertTrue(items[0]['crossing_range_verified'])

    def test_identity_failure_withdraws_cached_border_and_all_world_fragments(self):
        adapter,metadata,p,items,model,clock=fixture(); run(p,items,model)
        metadata['verified']=False
        model.observe(p.ego,p.lane,items)
        self.assertFalse(items[0]['crossing_range_verified'])
        self.assertEqual([],items[0]['crossing_ranges_world'])
        self.assertEqual({},items[0]['crossing_boundary_window'])
        metadata['verified']=True
        run(p,items,model)

    def test_geometry_change_cannot_reuse_old_matching_id_cache(self):
        unused,unused,p,items,model,clock=fixture(); run(p,items,model)
        items[0]['right_boundary']=[(0.,1.75,0.),(10.,1.75,0.),(70.,2.25,0.),(120.,2.25,0.)]
        model.observe(p.ego,p.lane,items)
        self.assertFalse(items[0]['crossing_range_verified'])
        self.assertEqual([],items[0]['crossing_ranges_world'])

    def test_missing_native_side_and_restricted_rule_never_produce_crossable_fragment(self):
        unused,unused,p,items,model,clock=fixture()
        del items[0]['marking_observation']['native_side']
        model.observe(p.ego,p.lane,items)
        self.assertFalse(items[0]['crossing_range_verified'])
        unused,unused,p,items,model,clock=fixture(document(road(sections=section(0,mark(change='none')))))
        run(p,items,model)
        self.assertEqual([],items[0]['crossing_ranges_world'])
        self.assertFalse(items[0]['crossing_allowed_at_ego'])

    def test_road_s_decreasing_sample_keeps_physical_side_when_ego_reversed(self):
        adapter,unused,p,items,model,clock=fixture(document(road(rule='LHT')))
        p.ego.heading=math.pi; p.ego.x=110.
        route=manager(adapter.hdmap); p.lane=route.update(p.ego,True)
        items=route.read_neighbor_lanes(p.ego,p.lane)
        self.assertEqual('right',items[0]['side'])
        self.assertEqual('left',items[0]['marking_observation']['native_side'])
        run(p,items,model)
        values=items[0]['crossing_ranges_world'][0]['road_s_samples_m']
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))

    def test_wrong_way_world_alignment_does_not_authorize_a_lane_change(self):
        adapter,unused,p,items,model,clock=fixture()
        p.ego.heading=math.pi; p.ego.x=110.
        route=manager(adapter.hdmap); p.lane=route.update(p.ego,True)
        items=route.read_neighbor_lanes(p.ego,p.lane)
        run(p,items,model)
        self.assertTrue(items[0]['crossing_geometry_bound'])
        self.assertTrue(items[0]['travel_direction_verified'])
        self.assertFalse(items[0]['travel_matches_declared_direction'])
        self.assertFalse(items[0]['crossing_allowed_at_ego'])
        self.assertEqual([],items[0]['crossing_ranges_world'])

    def test_terminal_window_finishes_and_is_not_restarted_forever(self):
        unused,unused,p,items,model,clock=fixture(); p.ego.x=110.
        run(p,items,model)
        for unused in range(4):
            meta,unused=model.observe(p.ego,p.lane,items)
            self.assertTrue(items[0]['crossing_range_verified'])
            self.assertEqual(0,meta['boundary_binding_budget']['atomic_steps'])

    def test_prefetch_retains_verified_old_window_without_extending_it_early(self):
        adapter,unused,p,items,model,clock=fixture(document(road(length=1000)))
        p.lane.left_boundary=[(0.,1.75,0.),(1000.,1.75,0.)]
        items[0]['right_boundary']=list(p.lane.left_boundary)
        run(p,items,model)
        p.ego.x=50.
        model.observe(p.ego,p.lane,items)
        self.assertTrue(items[0]['crossing_range_verified'])
        self.assertEqual(85.,items[0]['crossing_boundary_window']['boundary_arc_end_m'])
        for unused in range(100):
            model.observe(p.ego,p.lane,items)
            if items[0]['crossing_boundary_window']['boundary_arc_end_m']>85.: break
        self.assertEqual(125.,items[0]['crossing_boundary_window']['boundary_arc_end_m'])

    def test_shape_clock_deadline_is_checked_between_atomic_work(self):
        unused,unused,p,items,model,clock=fixture()
        class AdvancingClock(object):
            def __init__(self): self.now=100.
            def __call__(self): self.now+=.002; return self.now
        model.crossing_ranges.clock=AdvancingClock()
        meta,unused=model.observe(p.ego,p.lane,items)
        self.assertLess(meta['boundary_binding_budget']['atomic_steps'],MAX_ATOMIC_STEPS)

    def test_clipping_preserves_native_vertex_and_height_without_extrapolation(self):
        source=points3([(0.,0.,0.),(10.,0.,1.),(10.,10.,2.)])
        self.assertEqual([(5.,0.,.5),(10.,0.,1.),(10.,5.,1.5)],window3(source,5.,15.))
        self.assertEqual(list(source),window3(source,-20.,500.))
        with self.assertRaises(ValueError): window3(source,25.,30.)
        with self.assertRaises(ValueError): points3([(0.,0.,0.),(0.,0.,1.)])


if __name__=='__main__': unittest.main()
