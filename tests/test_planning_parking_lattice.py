"""Independent bounded parking-search examples, without SDK or vehicle sends."""
import math
import time
import unittest
from fractions import Fraction

from core.boundary_lineage import encode_position
from members.planning.candidate_validation import VehicleGeometry, MotionLimits, ValidationBudget, ObstaclePrediction, CorridorRegion
from members.planning.candidate_validation import ExecutionEnvelope
from members.planning.crossing_corridor import CrossingCorridor
from members.planning.parking_generator import ParkingSearch, generate_parking_segment
from members.planning.parking_lattice import ParkingLatticeSearch, search_parking_maneuver


def rectangle(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def choices(**changes):
    values = dict(step_lengths_m=(.8,), curvature_fractions=(-.8, 0., .8),
        curvature_change_per_m=2., xy_resolution_m=.3, heading_resolution_rad=.25,
        connector_distance_m=2., max_nodes=500, max_expansions=200,
        max_path_length_m=20., max_direction_changes=3, reverse_cost=1.,
        direction_change_cost=.5, permitted_directions=(-1, 1), direction_change_stop_duration_s=.5)
    values.update(changes)
    return ParkingLatticeSearch(**values)


def inputs(**changes):
    outer = rectangle(-12., -12., 12., 12.)
    knots = [tuple(p)+(0.,) for p in outer]
    descriptor = ((knots[0], knots[1]), (knots[3], knots[2]), 'left',
        ((encode_position(0, Fraction(0)), encode_position(0, Fraction(1))),))
    access = CrossingCorridor((outer,), descriptor, 1000000)
    values = dict(start_pose=(0., 0., 0.), goal_pose=(4., 0., 0.), initial_speed_mps=0.,
        initial_curvature_m_inv=0., motion_direction=1, speed_cap_mps=.4,
        vehicle=VehicleGeometry(.5, .2, .2, .4),
        limits=MotionLimits(1., 1., 1., 1., .65, 10., .2, 1e-7),
        access_corridor=access, coverage=CorridorRegion([outer]), goal_corridor=CorridorRegion([outer]),
        obstacles=[], parking_search=ParkingSearch((1.,), (1., .5), .15, 1000),
        lattice_search=choices(), budget=ValidationBudget(1000000, 18, .0001, time.monotonic()+20.))
    values.update(changes)
    return values


class ParkingLatticeTests(unittest.TestCase):
    def test_search_reaches_exact_goal_and_uses_original_speed(self):
        data = inputs(initial_speed_mps=.2)
        result = search_parking_maneuver(**data)
        self.assertEqual('safe', result['status'], result)
        self.assertEqual((4., 0., 0.), (result['points'][-1].x, result['points'][-1].y, result['points'][-1].heading))
        self.assertEqual(.2, result['points'][0].speed)
        self.assertEqual(0., result['points'][-1].speed)
        for a, b in zip(result['points'], result['points'][1:]):
            self.assertGreater(b.relative_time, a.relative_time)
            self.assertAlmostEqual(math.hypot(b.x-a.x, b.y-a.y),
                .5*(a.speed+b.speed)*(b.relative_time-a.relative_time), places=8)

    def test_opposite_longitudinal_goal_requires_authorized_direction_change(self):
        result = search_parking_maneuver(**inputs(goal_pose=(-2., 0., 0.), initial_speed_mps=.2,lattice_search=choices(
            connector_distance_m=1.5, curvature_fractions=(0.,), max_direction_changes=1)))
        self.assertEqual('safe', result['status'], result)
        self.assertTrue(result['needs_direction_authorization'])
        self.assertEqual([], result['points'])
        self.assertEqual([1, -1], [segment['motion_direction'] for segment in result['segments']])
        for segment in result['segments']:
            self.assertEqual(0., segment['points'][-1].speed)
        first, second = result['segments']
        self.assertEqual(0., second['points'][0].speed)
        self.assertEqual(0.,second['points'][0].relative_time)
        self.assertAlmostEqual(.5, second['nominal_start_time_s']-first['points'][-1].relative_time)

    def test_production_fallback_cannot_invent_an_unauthorized_reverse(self):
        data = inputs(goal_pose=(-2., 0., 0.), lattice_search=choices(curvature_fractions=(0.,),
            max_nodes=12, max_expansions=10))
        search = data.pop('parking_search'); search.lattice_search = data.pop('lattice_search')
        result = generate_parking_segment(search=search, **data)
        self.assertNotEqual('safe', result['status'], result)
        self.assertEqual([], result['points'])

    def test_budget_exhaustion_never_returns_a_partially_checked_path(self):
        for budget in (ValidationBudget(1, 10, .001, time.monotonic()+10.),
                       ValidationBudget(10000, 10, .001, time.monotonic()-1.)):
            result = search_parking_maneuver(**inputs(budget=budget))
            self.assertEqual('inconclusive', result['status'], result)
            self.assertEqual([], result['points']); self.assertEqual([], result['segments'])

    def test_unknown_velocity_is_not_a_static_pruning_fact(self):
        obstacle = ObstaclePrediction('unknown', 3., 0., .1, .1, 0., None, 0., 100., 0., 0.)
        result = search_parking_maneuver(**inputs(obstacles=[obstacle], lattice_search=choices(max_expansions=2)))
        self.assertNotEqual('safe', result['status'], result)
        self.assertEqual([], result['points'])

    def test_stationary_goal_applies_execution_body_margin_too(self):
        region=CorridorRegion([rectangle(-1.,-.25,1.,.25)])
        nominal=search_parking_maneuver(**inputs(goal_pose=(0.,0.,0.),goal_corridor=region))
        self.assertEqual('safe',nominal['status'],nominal)
        settings=ParkingSearch((1.,),(1.,),.15,1000,execution_envelope=ExecutionEnvelope(.1,0.,0.,0.))
        rejected=search_parking_maneuver(**inputs(goal_pose=(0.,0.,0.),goal_corridor=region,parking_search=settings))
        self.assertEqual('unsafe',rejected['status'],rejected)
        self.assertEqual([],rejected['points'])
        robust=search_parking_maneuver(**inputs(goal_pose=(0.,0.,0.),parking_search=settings))
        self.assertEqual('safe',robust['status'],robust)
        self.assertTrue(robust['execution_robust'])

    def test_explicit_choices_are_validated_without_defaults(self):
        for lattice in (choices(permitted_directions=(True,)), choices(max_nodes=True),
                        choices(curvature_change_per_m=None), choices(max_direction_changes=9)):
            result = search_parking_maneuver(**inputs(lattice_search=lattice))
            self.assertEqual('invalid', result['status'], result)

    def test_real_generator_fallback_routes_around_a_body_obstacle(self):
        obstacle=ObstaclePrediction('blocked_connector',2.,0.,.6,.8,0.,0.,0.,100.,0.,0.)
        data=inputs(obstacles=[obstacle],lattice_search=choices(permitted_directions=(1,),max_direction_changes=0))
        search=data.pop('parking_search'); lattice=data.pop('lattice_search')
        declined=generate_parking_segment(search=search,**data)
        self.assertNotEqual('safe',declined['status'])
        search.lattice_search=lattice
        accepted=generate_parking_segment(search=search,**data)
        self.assertEqual('safe',accepted['status'],accepted)
        self.assertEqual('NOMINAL_PARKING_LATTICE_SEGMENT_READY',accepted['reason_code'])
        self.assertGreater(max(abs(p.y) for p in accepted['points']),1.)
        # Independently compare every vehicle rectangle to the authored box.
        obstacle_corners=[(1.7,-.4),(2.3,-.4),(2.3,.4),(1.7,.4)]
        for point in accepted['points']:
            c,s=math.cos(point.heading),math.sin(point.heading)
            body=[(point.x+x*c-y*s,point.y+x*s+y*c) for x,y in
                  ((-.2,-.2),(.5,-.2),(.5,.2),(-.2,.2))]
            axes=[(1.,0.),(0.,1.),(c,s),(-s,c)]
            separated=False
            for ax,ay in axes:
                first=[x*ax+y*ay for x,y in body]
                second=[x*ax+y*ay for x,y in obstacle_corners]
                separated=separated or max(first)<min(second) or max(second)<min(first)
            self.assertTrue(separated,('body intersects obstacle',point.x,point.y,point.heading))

    def test_translation_rotation_and_mirror_preserve_the_same_physical_problem(self):
        for turn,mirror,dx,dy in ((.7,1,8.,-4.),(.0,-1,0.,0.)):
            data=inputs(goal_pose=(4.,0.,0.))
            def xy(point):
                x,y=point[0],mirror*point[1]; c,s=math.cos(turn),math.sin(turn)
                return (dx+c*x-s*y,dy+s*x+c*y)
            def pose(point): return xy(point)+(turn+mirror*point[2],)
            data['start_pose']=pose(data['start_pose']); data['goal_pose']=pose(data['goal_pose'])
            source=data['access_corridor']._source_descriptor
            descriptor=(tuple(xy(v)+(0.,) for v in source[0]),tuple(xy(v)+(0.,) for v in source[1]),source[2],source[3])
            rings=tuple(tuple(xy(v) for v in ring) for ring in data['access_corridor'].polygons)
            data['access_corridor']=CrossingCorridor(rings,descriptor,1000000)
            data['coverage']=CorridorRegion(rings); data['goal_corridor']=CorridorRegion(rings)
            result=search_parking_maneuver(**data)
            self.assertEqual('safe',result['status'],result)
            end=result['points'][-1]
            self.assertAlmostEqual(data['goal_pose'][0],end.x); self.assertAlmostEqual(data['goal_pose'][1],end.y)
            self.assertAlmostEqual(data['goal_pose'][2],end.heading)

    def test_finite_prediction_horizon_is_not_restarted_at_a_gear_change(self):
        target=ObstaclePrediction('far_but_short_horizon',8.,8.,.2,.2,0.,0.,0.,4.,0.,0.)
        result=search_parking_maneuver(**inputs(goal_pose=(-2.,0.,0.),initial_speed_mps=.2,
            obstacles=[target],lattice_search=choices(curvature_fractions=(0.,),max_expansions=10,
                max_direction_changes=1,max_path_length_m=3.7)))
        self.assertNotEqual('safe',result['status'],result); self.assertEqual([],result['segments'])

    def test_whole_terminal_body_must_fit_not_only_its_rear_axle(self):
        region=CorridorRegion([rectangle(3.9,-.1,4.1,.1)])
        result=search_parking_maneuver(**inputs(goal_corridor=region,lattice_search=choices(
            curvature_fractions=(0.,),max_expansions=4)))
        self.assertNotEqual('safe',result['status'],result); self.assertEqual([],result['points'])

    def test_search_keeps_original_measured_motion_and_model_objects(self):
        data=inputs(initial_speed_mps=.2)
        snapshot=(tuple(data['start_pose']),tuple(data['goal_pose']),data['initial_speed_mps'],
                  dict(vars(data['vehicle'])),dict(vars(data['limits'])),dict(vars(data['budget'])))
        result=search_parking_maneuver(**data); self.assertEqual('safe',result['status'])
        self.assertEqual(snapshot,(tuple(data['start_pose']),tuple(data['goal_pose']),data['initial_speed_mps'],
            dict(vars(data['vehicle'])),dict(vars(data['limits'])),dict(vars(data['budget']))))
        self.assertFalse(result.get('execution_robust',False))

    def test_explicit_execution_envelope_is_consumed_by_parking_timing_and_sweeps(self):
        data=inputs(); data['parking_search'].execution_envelope=ExecutionEnvelope(.05,.02,.1,.1)
        result=search_parking_maneuver(**data)
        self.assertEqual('safe',result['status'],result)
        self.assertTrue(result['execution_robust'])
        self.assertTrue(all(s['access_report']['execution_robust'] for s in result['segments']))


if __name__ == '__main__':
    unittest.main()
