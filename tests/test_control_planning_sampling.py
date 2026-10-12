"""Sampling invariants for the planner's spatial acceleration contract."""
import math
import unittest

from core.interfaces import TrajectoryPoint
from members.control.controller import ControlEngine
from members.control.parameters import ControllerSettings
from members.control.trajectory import PreparedPath
from members.planning.lane_planner import build_trajectory
from tests.test_control import calibration, inputs
from tests.test_planning import scene


def curved_points(sign=1., rotation=0.):
    points = []
    for i in range(65):
        x, y = 14.*math.sin(i/14.), sign*14.*(1.-math.cos(i/14.))
        points.append(TrajectoryPoint(x*math.cos(rotation)-y*math.sin(rotation),
                                     x*math.sin(rotation)+y*math.cos(rotation),
                                     5., 0., float(i)))
    return points


class ControlPlanningSamplingTests(unittest.TestCase):
    def test_redundant_near_start_points_do_not_manufacture_curve_braking(self):
        for sign in (-1., 1.):
            for rotation in (0., 1.2):
                raw = curved_points(sign, rotation)
                baseline = PreparedPath(raw, 12.).curve_speed_limit(0., ControllerSettings())
                a, b = raw[:2]
                for ratio in (.001, .005, .01):
                    extra = TrajectoryPoint(a.x+ratio*(b.x-a.x), a.y+ratio*(b.y-a.y),
                                            5., 0., ratio)
                    path = PreparedPath([a, extra]+raw[1:], 12.)
                    self.assertAlmostEqual(baseline,
                        path.curve_speed_limit(0., ControllerSettings()), delta=.03)

    def test_clipped_curve_window_remains_continuous_at_path_end(self):
        path = PreparedPath(curved_points(), 12.)
        endpoint = path.curvature_at(path.length)
        self.assertAlmostEqual(endpoint, path.curvature_at(path.length-.001), delta=.001)

    def test_point_speed_interpolation_preserves_acceleration_under_subdivision(self):
        for initial, acceleration, length in ((2., .5, 8.), (4., -1., 8.)):
            def point(s):
                return TrajectoryPoint(s, 0., math.sqrt(max(0., initial**2+2*acceleration*s)),
                                       0., float(s))
            sparse = PreparedPath([point(0.), point(length)], 12.)
            dense = PreparedPath([point(s) for s in (0., .3, 2., 6., length)], 12.)
            for s in (.1, .5, 1., 3., 7.9):
                expected = math.sqrt(max(0., initial**2+2*acceleration*s))
                self.assertAlmostEqual(expected, sparse.sample(s)[2])
                self.assertAlmostEqual(expected, dense.sample(s)[2])

    def test_acceleration_preview_is_independent_of_equivalent_point_density(self):
        commands = []
        settings = ControllerSettings()
        settings.launch_preview_m = settings.acceleration_preview_m = .5
        for positions in ((0., 8.), (0., .5, 2., 8.)):
            p, t = inputs(speed=2., x=0.,
                points=[(s, 0., math.sqrt(4.+2*s)) for s in positions])
            p.ego.vx = 2.
            c = ControlEngine(calibration(), settings).compute(p, t)
            self.assertTrue(c.valid, c.errors)
            self.assertAlmostEqual(math.sqrt(5.), c.diagnostics['reference_speed_mps'])
            self.assertAlmostEqual(1., c.diagnostics['reference_acceleration_mps2'])
            commands.append((c.throttle, c.brake))
        self.assertEqual(commands[0], commands[1])

    def test_deceleration_preview_preserves_the_planned_rate_and_stop_cap(self):
        settings = ControllerSettings()
        settings.acceleration_preview_m = .5
        for positions in ((0., 8.), (0., .5, 2., 8.)):
            p, t = inputs(speed=4., x=0.,
                points=[(s, 0., math.sqrt(max(0., 16.-2*s))) for s in positions])
            p.ego.vx = 4.
            t.target_speed, t.stop_required, t.stop_distance = 0., True, 8.
            c = ControlEngine(calibration(), settings).compute(p, t)
            self.assertTrue(c.valid, c.errors)
            self.assertAlmostEqual(-1., c.diagnostics['reference_acceleration_mps2'])
            self.assertEqual(0., c.throttle)
            self.assertGreater(c.brake, 0.)
            t.stop_distance = .2
            path = PreparedPath(t.points, 12.)
            reference, _ = path.speed_reference(0., t, settings)
            self.assertEqual(0., reference)

    def test_short_rising_first_segment_does_not_cancel_launch_preview(self):
        commands = []
        for phase in (.001, .01, .1, .5, .9):
            p, t = inputs(speed=2., x=0.,
                points=[(s, 0., math.sqrt(4.+2*s)) for s in (0., phase, 1., 10.)])
            p.ego.vx = 2.
            c = ControlEngine(calibration()).compute(p, t)
            self.assertTrue(c.valid, c.errors)
            self.assertAlmostEqual(math.sqrt(6.), c.diagnostics['reference_speed_mps'])
            self.assertAlmostEqual(1., c.diagnostics['reference_acceleration_mps2'])
            self.assertEqual(0., c.brake)
            self.assertGreater(c.throttle, 0.)
            commands.append(c.throttle)
        self.assertLess(max(commands)-min(commands), 1e-12)

    def test_rolling_launch_preview_cannot_skip_an_interior_stop(self):
        for speeds in ((0., 0., 2.), (2., 0., 2.)):
            p, t = inputs(speed=speeds[0], x=0.,
                points=[(s, 0., speed) for s, speed in zip((0., .5, 2.), speeds)])
            p.ego.vx = p.ego.speed
            if p.ego.speed == 0.:
                reference, _ = PreparedPath(t.points, 12.).speed_reference(0., t, ControllerSettings())
                self.assertEqual(0., reference)
                self.assertFalse(ControlEngine(calibration()).compute(p, t).valid)
                continue  # Two zero speeds cannot traverse a nonzero segment.
            c = ControlEngine(calibration()).compute(p, t)
            self.assertTrue(c.valid, c.errors)
            self.assertEqual(0., c.throttle)
            self.assertGreater(c.brake, 0.)

    def test_retained_zero_start_profile_keeps_launch_after_small_motion(self):
        for positions in ((0., 2., 8.), (0., .0001, .02, .2, 1., 2., 8.)):
            points = [(s, 0., min(.8, math.sqrt(2.*s))) for s in positions]
            for progress in (0., .0001, .001, .02, .2):
                p, t = inputs(speed=0., x=progress, points=points)
                t.target_speed = .8
                c = ControlEngine(calibration()).compute(p, t)
                self.assertTrue(c.valid, c.errors)
                self.assertGreater(c.diagnostics['reference_speed_mps'], .5)
                self.assertGreater(c.throttle, 0.)
                self.assertEqual(0., c.brake)

    def test_retained_launch_stops_at_first_falling_profile_and_internal_zero(self):
        points = [(0.,0.,0.), (.2,0.,.4), (.4,0.,.1),
                  (.6,0.,0.), (.8,0.,.8), (2.,0.,.8)]
        for progress in (.3, .4, .6):
            p, t = inputs(speed=.3, x=progress, points=points)
            t.target_speed = .8
            path = PreparedPath(t.points, 12.)
            reference, _ = path.speed_reference(progress, t, ControllerSettings())
            self.assertLessEqual(reference, path.sample(progress)[2])
        p, t = inputs(speed=0., x=.6, points=points)
        t.target_speed = .8
        c = ControlEngine(calibration()).compute(p, t)
        self.assertTrue(c.valid, c.errors)
        self.assertEqual(0., c.throttle)
        self.assertEqual('HOLD', c.diagnostics['state'])

    def test_neutral_stop_profile_can_release_stored_brake_integral(self):
        now = [0.]
        control = ControlEngine(calibration(), clock=lambda: now[0])
        p, t = inputs(speed=.18, x=0.)
        p.ego.vx = p.ego.speed
        self.assertTrue(control.compute(p, t).valid)
        control.integral = -.5
        now[0] += .05
        p, t = inputs(frame=2, speed=.18, x=0., points=[(0.,0.,.18),(1.,0.,0.)])
        p.ego.vx = p.ego.speed
        t.target_speed, t.stop_required, t.stop_distance, t.precision_stop = 0., True, 1., True
        c = control.compute(p, t)
        self.assertTrue(c.valid, c.errors)
        self.assertEqual(.18, c.diagnostics['reference_speed_mps'])
        self.assertEqual(0., c.diagnostics['reference_acceleration_mps2'])
        self.assertEqual(0., c.throttle)
        self.assertEqual(0., c.brake)
        self.assertEqual(0., control.integral)

    def test_short_stop_profile_can_preview_a_peak_before_its_midpoint(self):
        p, t = inputs(speed=.18, x=0.,
            points=[(0.,0.,.18),(.15,0.,.55),(.4,0.,0.)])
        p.ego.vx = p.ego.speed
        t.target_speed, t.stop_required, t.stop_distance, t.precision_stop = .8, True, .4, True
        c = ControlEngine(calibration()).compute(p, t)
        self.assertTrue(c.valid, c.errors)
        self.assertGreater(c.diagnostics['reference_speed_mps'], p.ego.speed)
        self.assertLessEqual(c.diagnostics['reference_speed_mps'], .4-.05)
        self.assertGreater(c.throttle, 0.)
        self.assertEqual(0., c.brake)

    def test_real_planner_curve_does_not_brake_for_a_short_first_segment(self):
        for sign in (-1., 1.):
            for phase in (.001, .01, .1, .4, .9, .99):
                p, d = scene(speed=4.1, target=8.)
                raw = curved_points(sign)
                p.lane.center_line = [(q.x, q.y) for q in raw]
                a, b = raw[:2]
                p.ego.x, p.ego.y = a.x+phase*(b.x-a.x), a.y+phase*(b.y-a.y)
                p.ego.heading = math.atan2(b.y-a.y, b.x-a.x)
                p.ego.vx, p.ego.vy = p.ego.speed*math.cos(p.ego.heading), p.ego.speed*math.sin(p.ego.heading)
                p.ego.frame_id, p.ego.age_ms = p.frame_id, 0
                p.source_status['gps'] = {'usable': True}
                t = build_trajectory(p, d)
                self.assertTrue(t.valid and not t.emergency_stop, t.reason)
                c = ControlEngine(calibration()).compute(p, t)
                self.assertTrue(c.valid, c.errors)
                self.assertGreater(c.diagnostics['curve_speed_limit_mps'], p.ego.speed)
                self.assertEqual(0., c.brake, (phase, c.diagnostics))
                self.assertGreaterEqual(c.diagnostics['reference_speed_mps'], p.ego.speed)


if __name__ == '__main__':
    unittest.main()
