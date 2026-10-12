"""Maneuver execution against a bidirectional lagged bicycle, not SimOne.

The synthetic upstream produces direction and speed/stop profiles explicitly.
It does not claim that today's decision/planning can produce parking paths.
"""

import math
import time
import unittest
from types import SimpleNamespace

from core.interfaces import ControlOut, Perception, Trajectory, TrajectoryPoint
from core.serialization import to_dict
from core.validation import validate_output
from members.control.controller import ControlEngine
from members.control.parameters import ControllerSettings
from simone_platform.simone_adapter import SimOneAdapter
from tests.control_benchmark import vehicle


class BidirectionalPlant(object):
    def __init__(self, x=0.0, y=0.0, heading=0.0, speed=0.0, direction=1, lag=0.3):
        self.x, self.y, self.heading = x, y, heading
        self.speed, self.direction, self.lag = speed, direction, lag
        self.throttle = self.brake = self.steering = self.yaw_rate = 0.0

    def step(self, output, dt, calibration):
        # Neutral coasts/brakes in the existing motion direction. A gear
        # command must not teleport positive velocity into negative velocity.
        if output.gear in (1, 2):
            desired = 1 if output.gear == 1 else -1
            if desired != self.direction:
                if self.speed > 0.05:
                    raise AssertionError("gear changed before observed standstill")
                self.direction = desired
        blend = 1.0 - math.exp(-dt / self.lag)
        for name in ("throttle", "brake", "steering"):
            previous = getattr(self, name)
            setattr(self, name, previous + blend * (getattr(output, name) - previous))
        acceleration = 3 * self.throttle - 6 * self.brake - 0.08 * self.speed
        next_speed = max(0.0, self.speed + dt * acceleration)
        if output.handbrake:
            next_speed = 0.0
        speed = self.direction * (self.speed + next_speed) / 2.0
        angle = self.steering * calibration.front_steer_max_rad / calibration.steering_sign
        self.yaw_rate = speed / calibration.wheelbase_m * math.tan(angle)
        mid_heading = self.heading + self.yaw_rate * dt / 2.0
        self.x += dt * speed * math.cos(mid_heading)
        self.y += dt * speed * math.sin(mid_heading)
        self.heading += dt * self.yaw_rate
        self.speed = next_speed


def maneuver_input(plant, points, frame, target=1.0, direction=1,
                   stop_distance=-1.0, precision=False, dwell=0.0, parking=False):
    p = Perception()
    p.valid, p.frame_id, p.timestamp = True, frame, frame * 50
    p.case_id = p.case_name = "synthetic-maneuver"
    p.valid_until = time.monotonic() + 30
    p.ego.valid, p.ego.frame_id, p.ego.age_ms = True, frame, 0
    p.ego.x, p.ego.y, p.ego.heading = plant.x, plant.y, plant.heading
    p.ego.speed, p.ego.yaw_rate = plant.speed, plant.yaw_rate
    p.ego.vx = plant.direction * plant.speed * math.cos(plant.heading)
    p.ego.vy = plant.direction * plant.speed * math.sin(plant.heading)
    p.ego.gear = 6  # Deliberately not an N/D/R/P command enum.
    p.source_status = {"gps": {"usable": True}}
    p.lane.valid = True
    t = Trajectory().bind(p)
    t.valid, t.target_speed, t.motion_direction = True, target, direction
    t.stop_required, t.stop_distance = stop_distance >= 0, stop_distance
    t.precision_stop, t.hold_duration_s = precision, dwell
    t.parking_brake_at_stop = parking
    t.points = [TrajectoryPoint(x, y, speed, 0.0, i * 0.2)
                for i, (x, y, speed) in enumerate(points)]
    return p, t


class ManeuverTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.0]
        self.calibration = vehicle()
        self.engine = ControlEngine(self.calibration, clock=lambda: self.now[0])

    def compute(self, p, t, step=0.05):
        result = self.engine.compute(p, t)
        self.assertTrue(result.valid, result.errors)
        validate_output(result, ControlOut, t)
        self.assertFalse(result.throttle > 0 and result.brake > 0)
        self.now[0] += step
        return result

    def test_reverse_straight_converges_from_both_sides_and_respects_speed_cap(self):
        points = [(-float(i), 0.0, 4.0) for i in range(81)]
        for y in (-0.6, 0.6):
            self.setUp()
            plant = BidirectionalPlant(y=y, heading=0.04, lag=0.4)
            peak = 0.0
            for frame in range(1, 401):
                p, t = maneuver_input(plant, points, frame, target=4, direction=-1)
                output = self.compute(p, t)
                plant.step(output, 0.05, self.calibration)
                peak = max(peak, plant.speed)
            self.assertLess(plant.x, -20.0)
            self.assertLess(abs(plant.y), 0.08)
            self.assertAlmostEqual(1.5, plant.speed, delta=0.12)
            self.assertLess(peak, 1.8)

    def test_reverse_curves_have_correct_physical_steering_across_lags(self):
        radius = 10.0
        for side in (-1, 1):
            points = [(-radius * math.sin(i * 0.01),
                       side * radius * (1 - math.cos(i * 0.01)), 1.0)
                      for i in range(231)]
            for lag in (0.1, 0.4):
                self.setUp()
                plant = BidirectionalPlant(lag=lag)
                errors = []
                for frame in range(1, 261):
                    p, t = maneuver_input(plant, points, frame, direction=-1)
                    output = self.compute(p, t)
                    errors.append(output.diagnostics.get("cross_track_abs_m", 0.0))
                    plant.step(output, 0.05, self.calibration)
                self.assertLess(max(errors), 0.25)
                self.assertGreater(side * plant.y, 3.0)
                self.assertLess(side * plant.heading, -0.6)

    def test_direction_change_brakes_stops_engages_then_moves(self):
        plant = BidirectionalPlant(speed=2.0)
        # First establish that this controller was driving forward.
        p, t = maneuver_input(plant, [(i * 5, 0, 2) for i in range(17)], 1, target=2)
        plant.step(self.compute(p, t), 0.05, self.calibration)
        phases = []
        reverse_command_at = None
        reverse_throttle_at = None
        for frame in range(2, 202):
            points = [(plant.x - i, 0.0, 1.0) for i in range(41)]
            p, t = maneuver_input(plant, points, frame, direction=-1)
            output = self.compute(p, t)
            phases.append(output.diagnostics["state"])
            if plant.speed > 0.05 and plant.direction == 1:
                self.assertEqual(0, output.gear)
                self.assertEqual(0, output.throttle)
            if output.gear == 2 and reverse_command_at is None:
                self.assertLessEqual(plant.speed, 0.05)
                reverse_command_at = self.now[0]
            if output.gear == 2 and output.throttle > 0 and reverse_throttle_at is None:
                reverse_throttle_at = self.now[0]
            plant.step(output, 0.05, self.calibration)
        self.assertIn("GEAR_STOP", phases)
        self.assertIn("GEAR_ENGAGE", phases)
        self.assertGreaterEqual(reverse_throttle_at - reverse_command_at, 0.3)
        self.assertEqual(-1, plant.direction)
        self.assertLess(plant.x, -4.0)

    def test_precision_reverse_stop_dwell_ten_seconds_then_forward_exit(self):
        plant = BidirectionalPlant(lag=0.3)
        endpoint = -6.0
        parked = None
        exit_started = None
        arrival_error = None
        for frame in range(1, 901):
            if parked is None:
                points = [(-i * 0.1, 0.0, min(1.0, math.sqrt(max(0.0, 2*(6-i*0.1)))))
                          for i in range(61)]
                p, t = maneuver_input(plant, points, frame, direction=-1,
                    stop_distance=max(0.0, plant.x-endpoint), precision=True,
                    dwell=10.0, parking=True)
            else:
                # Send the exit request immediately. The controller must retain
                # the already accepted minimum stop, not obey an early replan.
                points = [(plant.x + i, 0.0, 1.0) for i in range(41)]
                p, t = maneuver_input(plant, points, frame)
            output = self.compute(p, t)
            if output.diagnostics["state"] == "DWELL" and parked is None:
                parked = self.now[0]
                arrival_error = abs(plant.x-endpoint)
                self.assertLessEqual(plant.speed, 0.05)
            if parked is not None and self.now[0] - parked < 10.0:
                self.assertEqual(0, output.throttle)
                self.assertTrue(output.handbrake)
                self.assertLessEqual(plant.speed, 0.05)
            if parked is not None and output.throttle > 0 and exit_started is None:
                exit_started = self.now[0]
            plant.step(output, 0.05, self.calibration)
        self.assertIsNotNone(parked)
        self.assertLess(arrival_error, 0.16)
        self.assertGreaterEqual(exit_started - parked, 10.0)
        self.assertEqual(1, plant.direction)
        self.assertGreater(plant.x, 0.0)

    def test_follow_speed_and_stop_profiles_close_loop_on_stop_and_go(self):
        for curved in (False, True):
            with self.subTest(curved=curved):
                self.setUp()
                self._follow_stop_and_go(curved)

    def _follow_stop_and_go(self, curved):
        radius = 40.0
        plant = (BidirectionalPlant(x=radius, heading=math.pi/2, lag=0.3)
                 if curved else BidirectionalPlant(lag=0.3))
        lead_s = 30.0
        minimum_gap = float("inf")
        held = False
        lateral_errors = []
        for frame in range(1, 901):
            seconds = self.now[0]
            if seconds < 10:
                lead_speed = 3.0
            elif seconds < 14:
                lead_speed = max(0.0, 3.0 - 0.75*(seconds-10))
            elif seconds < 26:
                lead_speed = 0.0
            else:
                lead_speed = min(3.0, 0.75*(seconds-26))
            progress = radius*math.atan2(plant.y, plant.x) if curved else plant.x
            gap = lead_s - progress
            # Synthetic upstream spacing policy, independent from old decision
            # implementation. The controller only receives its numeric path.
            demand = max(0.0, min(4.0, lead_speed + 0.35*(gap-12-1.2*plant.speed)))
            remaining = max(0.0, gap-12)
            if demand == 0 and plant.speed == 0:
                points = [(plant.x, plant.y, 0), (plant.x, plant.y, 0)]
            else:
                points = []
                for i in range(41):
                    s = progress + remaining*i/40
                    x, y = ((radius*math.cos(s/radius), radius*math.sin(s/radius))
                            if curved else (s, 0.0))
                    points.append((x, y, min(demand if demand > 0 else plant.speed,
                        math.sqrt(max(0.0, 3*remaining*(1-i/40))))))
            p, t = maneuver_input(plant, points, frame, target=demand,
                                  stop_distance=remaining)
            output = self.compute(p, t)
            lateral_errors.append(output.diagnostics.get("cross_track_abs_m", 0.0))
            minimum_gap = min(minimum_gap, gap)
            if 23 <= seconds < 26:
                held = held or output.diagnostics["state"] == "HOLD"
            plant.step(output, 0.05, self.calibration)
            lead_s += 0.05 * lead_speed
        self.assertTrue(held)
        self.assertGreater(minimum_gap, 11.0)
        self.assertGreater(progress, 65.0)
        self.assertLess(max(lateral_errors), 0.3)
        self.assertAlmostEqual(3.0, plant.speed, delta=0.2)
        self.assertAlmostEqual(15.6, lead_s-progress, delta=1.0)

    def test_stale_input_restarts_continuous_parking_dwell(self):
        plant = BidirectionalPlant()
        points = [(0, 0, 0), (0, 0, 0)]
        for frame in range(1, 11):
            p, t = maneuver_input(plant, points, frame, target=0,
                                 stop_distance=0, precision=True, dwell=1, parking=True)
            self.assertEqual("DWELL", self.compute(p, t).diagnostics["state"])
        p, t = maneuver_input(plant, points, 11, target=0, stop_distance=0,
                             precision=True, dwell=1, parking=True)
        t.valid = False
        self.assertFalse(self.engine.compute(p, t).valid)
        self.assertEqual(0.0, self.engine.standstill.elapsed)
        self.now[0] += 0.05
        for frame in range(12, 29):
            p, t = maneuver_input(plant, [(0, 0, 1), (10, 0, 1)], frame)
            output = self.compute(p, t)
            self.assertEqual("DWELL", output.diagnostics["state"])
            self.assertEqual(0, output.throttle)

    def test_repeated_direction_changes_never_enable_early_propulsion(self):
        plant = BidirectionalPlant()
        for frame in range(1, 15):
            direction = -1 if frame % 3 else 1
            p, t = maneuver_input(plant, [(0, 0, 1), (direction*10, 0, 1)],
                                  frame, direction=direction)
            output = self.compute(p, t)
            self.assertEqual(0, output.throttle)
            self.assertIn(output.diagnostics["state"], ("GEAR_STOP", "GEAR_ENGAGE"))

    def test_parking_dwell_requires_arrival_position_and_heading(self):
        for heading, offset in ((0.3, 0.0), (0.0, 0.4)):
            self.setUp()
            plant = BidirectionalPlant(heading=heading, y=offset)
            p, t = maneuver_input(plant, [(0, 0, 0), (0, 0, 0)], 1, target=0,
                                 stop_distance=0, precision=True, dwell=10, parking=True)
            output = self.compute(p, t)
            self.assertEqual("HOLD", output.diagnostics["state"])
            self.assertFalse(self.engine.standstill.active)

    def test_reverse_emergency_preempts_gear_transition_and_dwell(self):
        plant = BidirectionalPlant(speed=1, direction=-1)
        p, t = maneuver_input(plant, [(0, 0, 0), (0, 0, 0)], 1,
                             target=0, direction=-1, stop_distance=0)
        t.emergency_stop, t.hazard_signal = True, True
        output = self.compute(p, t)
        self.assertEqual("EMERGENCY", output.diagnostics["state"])
        self.assertEqual(0, output.gear)
        self.assertEqual(0, output.throttle)
        self.assertEqual(self.calibration.emergency_brake, output.brake)
        self.assertTrue(output.hazard_signal)

    def test_moving_reverse_switches_back_to_forward_without_velocity_flip(self):
        plant = BidirectionalPlant()
        forward_seen = False
        for frame in range(1, 351):
            direction = -1 if frame < 120 else 1
            points = [(plant.x + direction*i, 0, 1) for i in range(41)]
            p, t = maneuver_input(plant, points, frame, direction=direction)
            output = self.compute(p, t)
            if frame == 120:
                self.assertGreater(plant.speed, 0.5)
                self.assertEqual(-1, plant.direction)
                self.assertEqual(0, output.gear)
            if direction == 1 and output.throttle > 0:
                forward_seen = True
                self.assertEqual(1, output.gear)
            plant.step(output, 0.05, self.calibration)
        self.assertTrue(forward_seen)
        self.assertEqual(1, plant.direction)
        self.assertAlmostEqual(1, plant.speed, delta=0.15)

    def test_precision_reverse_curve_reaches_stop_pose(self):
        radius, angle = 6.0, 0.7
        plant = BidirectionalPlant(lag=0.2)
        points = [(-radius*math.sin(angle*i/70),
                   radius*(1-math.cos(angle*i/70)),
                   min(0.8, math.sqrt(max(0.0, 2*radius*angle*(1-i/70)))))
                  for i in range(71)]
        dwell_seen = False
        for frame in range(1, 481):
            progress = radius*math.atan2(-plant.x, radius-plant.y)
            p, t = maneuver_input(plant, points, frame, target=0.8, direction=-1,
                stop_distance=max(0, radius*angle-progress), precision=True,
                dwell=1, parking=True)
            for i, point in enumerate(t.points):
                point.heading = -angle*i/70
            output = self.compute(p, t)
            dwell_seen = dwell_seen or output.diagnostics["state"] == "DWELL"
            plant.step(output, 0.05, self.calibration)
        self.assertTrue(dwell_seen)
        self.assertLess(math.hypot(plant.x-points[-1][0], plant.y-points[-1][1]), 0.16)
        self.assertAlmostEqual(-angle, plant.heading, delta=0.15)
        self.assertEqual(0, plant.speed)

    def test_low_reverse_request_can_start_without_precision_flag(self):
        plant = BidirectionalPlant()
        points = [(-i, 0, 0.15) for i in range(21)]
        for frame in range(1, 401):
            p, t = maneuver_input(plant, points, frame, target=0.15, direction=-1)
            output = self.compute(p, t)
            plant.step(output, 0.05, self.calibration)
        self.assertLess(plant.x, -0.5)
        self.assertGreater(plant.speed, 0.03)

    def test_inconsistent_velocity_cannot_arm_reverse(self):
        plant = BidirectionalPlant()
        p, t = maneuver_input(plant, [(0, 0, 1), (-10, 0, 1)], 1, direction=-1)
        p.ego.vx = 2.0
        result = self.engine.compute(p, t)
        self.assertFalse(result.valid)
        self.assertEqual(0, result.throttle)
        self.assertIn("magnitude disagrees", result.errors[0])

    def test_explicit_captain_reset_for_new_task_discards_previous_parking_dwell(self):
        plant = BidirectionalPlant()
        p, t = maneuver_input(plant, [(0, 0, 0), (0, 0, 0)], 1, target=0,
                             stop_distance=0, precision=True, dwell=10, parking=True)
        self.assertEqual("DWELL", self.compute(p, t).diagnostics["state"])
        p, t = maneuver_input(plant, [(0, 0, 1), (10, 0, 1)], 2)
        p.task_id = "new task"
        self.engine.reset()
        result = self.compute(p, t)
        self.assertGreater(result.throttle, 0)
        self.assertFalse(self.engine.standstill.active)

    def test_intent_fields_validate_serialize_and_reach_sdk_command(self):
        plant = BidirectionalPlant()
        points = [(-i*5, 0, 1) for i in range(7)]
        p, t = maneuver_input(plant, points, 1, direction=-1)
        t.left_signal = True
        output = self.compute(p, t)
        self.assertTrue(output.left_signal)
        self.assertEqual(-1, to_dict(t)["motion_direction"])
        calls = []
        adapter = SimOneAdapter(SimpleNamespace(vehicle_id="0"), SimpleNamespace())
        adapter.structs = SimpleNamespace(SimOne_Data_Control=lambda: SimpleNamespace(),
                                         SimOne_Data_Signal_Lights=lambda: SimpleNamespace())
        adapter.pnc_api = SimpleNamespace(SoSetDriveMode=lambda *args: True,
            SoSetDrive=lambda vehicle_id, native: calls.append(native) or True,
            SoSetSignalLights=lambda vehicle_id, native: calls.append(native) or True)
        self.assertTrue(adapter.send_control(output))
        self.assertEqual(0, calls[0].gear)
        self.assertEqual(2, calls[1].signalLights)
        self.assertFalse(calls[0].isManualGear)
        for frame in range(2, 18):
            p, t = maneuver_input(plant, points, frame, direction=-1)
            t.right_signal = True
            output = self.compute(p, t)
        calls[:] = []
        self.assertEqual(2, output.gear)
        self.assertGreater(output.throttle, 0)
        self.assertTrue(adapter.send_control(output))
        self.assertEqual(2, calls[0].gear)
        self.assertEqual(1, calls[1].signalLights)
        p, t = maneuver_input(plant, [(0, 0, 0), (0, 0, 0)], 18,
                             target=0, direction=-1, stop_distance=0,
                             precision=True, dwell=10, parking=True)
        t.hazard_signal = True
        output = self.compute(p, t)
        calls[:] = []
        self.assertTrue(adapter.send_control(output))
        self.assertTrue(calls[0].handbrake)
        self.assertEqual(4, calls[1].signalLights)
        t.motion_direction = True
        with self.assertRaises(ValueError):
            validate_output(t, Trajectory, p)
        t.motion_direction, t.hold_duration_s = -1, -1.0
        with self.assertRaises(ValueError):
            validate_output(t, Trajectory, p)


if __name__ == "__main__":
    unittest.main()
