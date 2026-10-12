"""Authored rational edge examples and actual producer-to-reader replay.

This proves provenance within the nominal cleaned SDK polyline, not continuous
marking law, whole-vehicle clearance, coverage or driving permission.
"""
import copy
import json
import math
import unittest
from fractions import Fraction

from core.boundary_lineage import (native_points3,window_with_lineage,densify_with_lineage,
    decode_position,encode_position,exact_point,window_lineage_steps)
from core.maneuver_facts import ManeuverFacts
from core.interfaces import Perception
from core.serialization import perception_to_dict
from scripts.replay_decision import _assign
from perception.crossing_ranges import window3,densify3
from tests import test_maneuver_fact_consumers as producer_fixture


class BoundaryLineageTests(unittest.TestCase):
    def test_native_endpoint_survives_cancellation_in_window_and_sampling(self):
        native=((1.,0.,0.),(1e-20,0.,0.))
        points,positions=window_with_lineage(native,0.,1.)
        self.assertEqual(native[-1],points[-1])
        self.assertEqual(native[-1],window3(native,0.,1.)[-1])
        self.assertEqual(native[-1],densify3(list(native))[-1])
        dense,records=densify_with_lineage(native,points,positions,1.)
        self.assertEqual(native[-1],dense[-1])
        self.assertEqual((0,Fraction(1)),decode_position(records[-1],2))

    def test_clip_and_densify_keep_exact_edge_ratios_and_height(self):
        native=((0.,0.,2.),(8.,0.,6.),(16.,0.,10.))
        clipped,positions=window_with_lineage(native,3.,13.)
        self.assertEqual([(3.,0.,3.5),(8.,0.,6.),(13.,0.,8.5)],clipped)
        dense,records=densify_with_lineage(native,clipped,positions,2.)
        self.assertEqual(7,len(dense))
        self.assertEqual((0,Fraction(7,12)),decode_position(records[1],3))
        self.assertEqual((Fraction(14,3),Fraction(0),Fraction(13,3)),exact_point(native,records[1]))
        self.assertIn(native[1],dense)
        self.assertEqual((1,Fraction(5,8)),decode_position(records[-1],3))

    def test_json_keeps_rational_position_that_binary64_would_round(self):
        ratio=Fraction(2**60+1,2**61)
        record=encode_position(0,ratio)
        restored=json.loads(json.dumps(record))
        self.assertEqual((0,ratio),decode_position(restored,2))
        self.assertEqual(.5,float(ratio))
        self.assertNotEqual(Fraction(.5),ratio)

    def test_native_vertices_of_curved_border_are_kept_without_hull(self):
        native=tuple((math.cos(i*.01),math.sin(i*.01),i*.001) for i in range(50))
        points,positions=window_with_lineage(native,0.,100.)
        dense,records=densify_with_lineage(native,points,positions,.005)
        self.assertTrue(all(p in dense for p in native))
        self.assertEqual(len(dense),len(records))

    def test_removed_native_bend_cannot_be_replaced_by_one_sampled_chord(self):
        native=((0.,0.,0.),(1.,1.,0.),(2.,0.,0.))
        with self.assertRaises(ValueError):
            densify_with_lineage(native,[native[0],native[-1]],
                [encode_position(0,Fraction(0)),encode_position(1,Fraction(1))],.1)

    def test_window_never_extrapolates_beyond_native_endpoints(self):
        native=((0.,0.,0.),(8.,0.,0.))
        points,positions=window_with_lineage(native,-100.,100.)
        self.assertEqual(list(native),points)
        with self.assertRaises(ValueError): window_with_lineage(native,10.,20.)
        with self.assertRaises(ValueError): window_with_lineage(native,0.,float('inf'))

    def test_rational_decoder_rejects_noncanonical_or_unbounded_records(self):
        invalid=[(True,'0','1'),(-1,'0','1'),(1,'0','1'),(0,'2','1'),(0,'1','0'),
                 (0,'01','2'),(0,'2','4'),(0,'-1','2'),(0,'0.5','1'),(0,1,'2'),
                 (0,'1','9'*401),(0,'','1'),(0,'０','1'),None]
        for value in invalid:
            with self.assertRaises(ValueError,msg=repr(value)): decode_position(value,2)

    def test_native_cleanup_retains_prior_validity_and_vertical_unknowns(self):
        native=((0.,0.,0.),(0.,0.,0.),(1.,0.,0.))
        self.assertEqual((native[0],native[-1]),native_points3(native))
        for raw in ([(0.,0.,0.),(0.,0.,1.)],[(0.,0.,0.),(float('nan'),0.,0.)]):
            with self.assertRaises(ValueError): native_points3(raw)

    def test_sample_point_budget_is_unknown_not_truncated_geometry(self):
        native=((0.,0.,0.),(5000.,0.,0.))
        points,positions=window_with_lineage(native,0.,5000.)
        with self.assertRaises(ValueError): densify_with_lineage(native,points,positions,1.)

    def test_preparation_generator_yields_before_publishing_complete_window(self):
        native=((0.,0.,0.),(8.,0.,0.),(16.,0.,0.))
        job=window_lineage_steps(native,3.,13.)
        units=0
        while True:
            try: next(job); units+=1
            except StopIteration as completed:
                points,positions=completed.value
                break
        self.assertGreater(units,3)
        self.assertEqual(3.,points[0][0])
        self.assertEqual(13.,points[-1][0])


class ProducerLineageTests(unittest.TestCase):
    def setUp(self):
        self.fixture=producer_fixture.FactConsumerTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.bound()

    def read(self):
        return ManeuverFacts(self.fixture.p,self.fixture.clock).crossing('1_0_-2')

    def test_current_native_side_and_every_permitted_fragment_are_bound(self):
        value=self.read()
        self.assertEqual('native_edge_rational_v1',value['source_lineage_model'])
        self.assertEqual('left',value['side'])
        for fragment,positions in zip(value['permitted_fragments'],value['permitted_source_positions']):
            self.assertEqual(fragment,[tuple(float(v) for v in exact_point(value['source_boundary_world'],p))
                                       for p in positions])

    def test_public_json_replay_keeps_binding_and_original_expiry(self):
        original=self.read()
        restored=_assign(Perception(),json.loads(json.dumps(perception_to_dict(self.fixture.p))), 'perception')
        value=ManeuverFacts(restored,self.fixture.clock).crossing('1_0_-2')
        self.assertEqual(original['permitted_source_positions'],value['permitted_source_positions'])
        self.assertEqual(original['evidence']['valid_until_s'],value['evidence']['valid_until_s'])

    def test_missing_or_wrong_native_source_never_binds_a_permission(self):
        original=copy.deepcopy(self.fixture.p)
        for field,value in (('source_lineage_model','unknown'),('source_lane_id','other'),
                            ('source_boundary_side','right'),('source_boundary_world',[(0.,2.,0.),(1000.,2.,0.)])):
            self.fixture.p=copy.deepcopy(original)
            self.fixture.p.maneuver_environment.neighbor_lanes[0]['crossing_boundary_window'][field]=value
            with self.assertRaises(ValueError,msg=field): self.read()

    def test_world_point_and_native_position_must_agree(self):
        window=self.fixture.p.maneuver_environment.neighbor_lanes[0]['crossing_boundary_window']
        window['source_positions'][5]=(0,'1','2')
        with self.assertRaises(ValueError): self.read()

    def test_fragment_cannot_borrow_another_window_position_or_skip_a_vertex(self):
        neighbor=self.fixture.p.maneuver_environment.neighbor_lanes[0]
        neighbor['crossing_ranges_world'][0]['source_positions'][0]=(0,'1','2')
        with self.assertRaises(ValueError): self.read()

    def test_original_perception_native_geometry_is_rechecked(self):
        self.fixture.p.lane.left_boundary=[(0.,2.,0.),(1000.,2.,0.)]
        with self.assertRaises(ValueError): self.read()


if __name__=='__main__': unittest.main()
