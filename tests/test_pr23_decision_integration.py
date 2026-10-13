"""PR23 decision semantics on main's real fixed entries and formal transport.

Inputs, clocks, vehicle response and successful receipts are synthetic. No
strategy, planner, controller or monitor output is replaced by a test result.
"""
import unittest
from unittest.mock import patch

from core.behavior_channel import BehaviorTransport
from core.behavior_contract import GoalPose, BehaviorGoal
from core.behavior_evidence import Evidence
from core.maneuver_facts import ManeuverFacts
from core.task_route import TaskRouteFacts, TaskDecisionInputs
from members import decision_stub, planning_stub, control_stub
from members.decision.engine import DecisionEngine
from members.decision.settings import DecisionSettings
from members.decision.target_history import TargetHistory
from members.decision.behaviors.session import BehaviorSession
from tests import test_lane_change_integration as lane
from tests.test_decision_behavior_contract import frame, capabilities
from tests.test_control_maneuvers import BidirectionalPlant


class DecisionMigrationTests(unittest.TestCase):
    def setUp(self):
        self.h = lane.LaneChangeIntegrationTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.f = self.h.f

    def test_faster_lead_before_crossing_cancels_without_a_lateral_path(self):
        self.h.stage('PREPARE')
        self.h.lead.vx = 3.
        self.h.advance(self.f.p.ego.x + .05)
        d, t, c = self.h.cycle()
        state = self.h.engine._behavior.lane_changes
        self.assertEqual('IDLE', state.phase)
        self.assertEqual('OPPORTUNITY_WITHDRAWN_BEFORE_CROSSING', state.reason)
        self.assertIsNone(d.behavior_request)
        self.assertIsNone(t.behavior_identity)
        self.assertTrue(all(abs(v.y) < 1e-6 for v in t.points))

    def test_lead_speed_change_after_body_crosses_does_not_reset_the_intent(self):
        self.h.stage('EXECUTE')
        runtime = self.h.engine._behavior.lane_changes
        identifier = runtime.policy.session.intent_id
        self.h.lead.vx = 3.
        self.h.advance(self.f.p.ego.x + .05, y=1.5)
        self.h.cycle(check=False)
        self.assertEqual(identifier, runtime.policy.session.intent_id)
        self.assertNotEqual('IDLE', runtime.phase)

    def test_vanished_hazard_needs_distinct_original_sensor_clear_packets(self):
        self.f.p.targets = []
        self.f.build()
        history = TargetHistory(DecisionSettings(release_frames=2), self.f.clock)
        history._danger_memory[9] = (40., 0., self.f.p.ego.z, 4., 2., 0., ManeuverFacts(self.f.p, self.f.clock).map_digest())
        history.begin(self.f.p, self.f.now, True)
        self.assertTrue(history.missing_danger_constraints(self.f.p))
        history.begin(self.f.p, self.f.now, False)
        self.assertTrue(history.missing_danger_constraints(self.f.p))
        self.h.advance(self.f.p.ego.x + .05)
        self.h.transport.prepare(self.f.p)
        history.begin(self.f.p, self.f.now, True)
        self.assertEqual([], history.missing_danger_constraints(self.f.p))

    def test_unknown_coverage_and_history_expiry_cannot_clear_a_hazard(self):
        self.f.p.targets = []
        self.f.build()
        history = TargetHistory(DecisionSettings(release_frames=2), self.f.clock)
        history._danger_memory[9] = (40., 0., self.f.p.ego.z, 4., 2., 0., ManeuverFacts(self.f.p, self.f.clock).map_digest())
        for i in range(5):
            self.h.transport.prepare(self.f.p)
            self.f.p.maneuver_environment.coverage_regions = []
            history.begin(self.f.p, self.f.now, True)
            self.assertTrue(history.missing_danger_constraints(self.f.p))
            self.h.advance(self.f.p.ego.x + .05)
        self.assertIn(9, history._danger_memory)

    def test_verified_change_renewal_is_bounded_and_keeps_the_obligation(self):
        session = BehaviorSession(max_retries=0)
        f = frame()
        session.start(f, 'original', 'SIGNAL_STOP', 'STOP_AND_DWELL',
                      BehaviorGoal(
                          stop_obligation_id='signal:42', minimum_standstill_duration_s=6.))
        session.block(f, 'NO_PATH')
        identifier = session.intent_id
        self.assertTrue(session.renew_after_verified_change(frame(2, .1), 'occupied-to-clear:1'))
        session.block(frame(3, .2), 'NO_PATH')
        self.assertFalse(session.renew_after_verified_change(frame(4, .3), 'occupied-to-clear:1'))
        self.assertTrue(session.renew_after_verified_change(frame(5, .4), 'occupied-to-clear:2'))
        session.block(frame(6, .5), 'NO_PATH')
        self.assertFalse(session.renew_after_verified_change(frame(7, .6), 'occupied-to-clear:3'))
        self.assertEqual(identifier, session.intent_id)
        self.assertEqual('signal:42', session.goal.stop_obligation_id)

    def test_static_blockage_uses_real_stop_and_warning_chain_outside_scene_eleven(self):
        self.h.lead.vx = 0.
        self.f.build()
        transport = BehaviorTransport(('PATH', 'PATH_STOP', 'LIGHTS', 'FEEDBACK'), self.f.clock)
        engine = DecisionEngine(DecisionSettings(cruise_speed=1., front_offset_m=3.9, half_width_m=.9),
                                self.f.clock)
        planning_stub.configure_planning()
        with patch('members.decision_stub._ENGINE', engine):
            transport.prepare(self.f.p)
            d = decision_stub.decide(self.f.p)
            self.assertNotEqual(11, self.f.p.scene_id)
            self.assertEqual('BLOCKED_ROAD', d.behavior_request['maneuver'], d.reason)
            t = planning_stub.plan(self.f.p, d)
            self.assertTrue(t.valid, (t.reason, t.errors))
            c = control_stub.compute_control(self.f.p, t)
            self.assertTrue(c.valid, c.errors)
            self.assertTrue(t.stop_required)
            self.assertEqual(0., t.points[-1].speed)
            self.assertTrue(t.hazard_signal)
            self.assertTrue(c.hazard_signal)
            self.assertEqual(0., t.hold_duration_s)


class PassageAdmissionTests(unittest.TestCase):
    def setUp(self):
        from tests.test_driving_integration import DrivingIntegrationTests
        self.h = DrivingIntegrationTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)

    def test_current_occupied_conflict_region_denies_entry_even_with_authority(self):
        self.h.targets = [dict(id=9, type=1, x=30., y=0., z=0., vx=0., vy=0.,
            length=4., width=1.8, height=1.5, heading=0., probability=1.)]
        d, t, c, safety = self.h.cycle(check=False, sent=False)
        self.assertEqual('WAIT_ENTRY', d.behavior_request['stage'], d.reason)
        self.assertTrue(t.valid, t.errors)
        self.assertTrue(t.stop_required)
        self.assertLess(max(point.x+3.9 for point in t.points), 20.)

    def test_revoked_authority_inside_does_not_reuse_entry_stop_behind_car(self):
        self.h.cycle(sent=False)
        self.h.now += .05
        self.h.plant.x, self.h.plant.speed = 30., 1.
        self.h.authority, self.h.windows = 'UNKNOWN', ()
        d, t, c, safety = self.h.cycle(check=False, sent=False)
        self.assertEqual('INSIDE_PROTECTION', self.h.engine._behavior.driving.snapshot()['phase'],
                         self.h.engine._behavior.driving.snapshot())
        self.assertEqual('EMERGENCY_BRAKE', d.mode)
        self.assertIsNone(d.behavior_request)
        self.assertEqual(0., d.stop_distance)
        self.assertEqual(0., c.throttle)
        self.assertGreater(c.brake, 0.)


class TaskMigrationTests(unittest.TestCase):
    def setUp(self):
        self.h = lane.LaneChangeIntegrationTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.f = self.h.f
        self.f.p.targets = []
        self.f.build()
        self.end_reason = 'task_destination'
        self.required = ()
        self.enabled = True
        self.jump = None
        self.stop_rule = None
        self.destination_region = [(22., -1.5), (30., -1.5), (30., 1.5), (22., 1.5)]
        self.transport = BehaviorTransport(('PATH', 'PATH_STOP', 'TASK_ROUTE', 'LIGHTS', 'DWELL', 'FEEDBACK'), self.f.clock)
        self.engine = DecisionEngine(DecisionSettings(cruise_speed=1., front_offset_m=3.9, half_width_m=.9),
                                    self.f.clock, task_inputs_provider=self.inputs)
        manager = patch('members.decision_stub._ENGINE', self.engine)
        manager.start()
        self.addCleanup(manager.stop)
        planning_stub.configure_planning()

    def inputs(self, p, context):
        if not self.enabled:
            return None
        proof = Evidence(context, p.frame_id, self.f.now, p.valid_until, True, True,
                         'authored-task-route-model', source_kind='task')
        progress = min(25., max(0., p.ego.x)) if self.jump is None else self.jump
        facts = TaskRouteFacts(proof, 'route-a', [('segment-a', lane.SOURCE, 0., 25.)],
            'segment-a', progress, self.end_reason,
            self.destination_region,
            GoalPose(25., 0., 0.), 1., True, self.required)
        return TaskDecisionInputs(facts, (3.9, .9, .9), ManeuverFacts(p, self.f.clock).map_digest(), self.stop_rule)

    def step(self, plant):
        p = self.f.p
        self.transport.prepare(p)
        d = decision_stub.decide(p)
        self.assertTrue(d.valid, d.errors)
        t = planning_stub.plan(p, d)
        self.assertTrue(t.valid, (d.reason, t.reason, t.errors))
        c = control_stub.compute_control(p, t)
        self.assertTrue(c.valid, c.errors)
        self.transport.observe(p, d, t, c, False, {'attempted': True, 'ok': True})
        plant.step(c, .05, self.h.h.calibration)
        self.h.advance(x=plant.x, y=plant.y, heading=plant.heading, speed=abs(plant.speed))
        return d, t, c

    def test_real_fixed_entries_stop_at_destination_and_complete_from_actual_feedback(self):
        plant = BidirectionalPlant(x=self.f.p.ego.x, speed=self.f.p.ego.speed)
        for unused in range(1000):
            self.step(plant)
            if self.engine._behavior.task.phase == 'COMPLETE':
                break
        self.assertEqual('COMPLETE', self.engine._behavior.task.phase, self.engine._behavior.task.snapshot())
        self.assertAlmostEqual(25., plant.x, delta=.15)
        self.assertLess(abs(plant.speed), .05)

    def test_unfinished_obligation_keeps_actual_arrival_incomplete(self):
        self.required = ('signal:42',)
        plant = BidirectionalPlant(x=25.)
        self.h.advance(x=25., speed=0.)
        for unused in range(12):
            self.step(plant)
            self.assertNotEqual('COMPLETE', self.engine._behavior.task.phase)

    def test_local_map_end_stops_whole_body_without_completing_task(self):
        self.end_reason = 'map_end'
        plant = BidirectionalPlant(x=self.f.p.ego.x, speed=self.f.p.ego.speed)
        for unused in range(700):
            self.step(plant)
            self.assertNotEqual('COMPLETE', self.engine._behavior.task.phase)
        self.assertLessEqual(plant.x + 3.9, 25.)
        self.assertLess(abs(plant.speed), .05)

    def test_local_map_task_keeps_one_revision_while_only_stop_distance_changes(self):
        self.end_reason = 'map_end'
        plant = BidirectionalPlant(x=self.f.p.ego.x, speed=self.f.p.ego.speed)
        first = None
        for unused in range(5):
            d, t, c = self.step(plant)
            identity = (d.behavior_request['intent_id'], d.behavior_request['revision'])
            if first is None:
                first = identity
            self.assertEqual(first, identity)

    def test_unreachable_progress_and_withdrawn_source_protect(self):
        plant = BidirectionalPlant(x=self.f.p.ego.x, speed=self.f.p.ego.speed)
        self.step(plant)
        self.jump = 20.
        d, unused, unused = self.step(plant)
        self.assertEqual(0., d.target_speed)
        self.assertIn('UNREACHABLE', self.engine._behavior.task.reason)
        self.enabled = False
        d, unused, unused = self.step(plant)
        self.assertEqual(0., d.target_speed)
        self.assertIn('LOST', self.engine._behavior.task.reason)

    def test_configured_task_missing_on_first_frame_cannot_drive_without_its_route(self):
        self.enabled = False
        plant = BidirectionalPlant(x=self.f.p.ego.x, speed=self.f.p.ego.speed)
        d, t, c = self.step(plant)
        self.assertEqual(0., d.target_speed)
        self.assertEqual(0., c.throttle)
        self.assertIn('FORMAL_TASK_PROGRESS_LOST', d.reason)

    def test_verified_stop_rule_is_dispatched_on_green_without_a_scene_number(self):
        p = self.f.p
        traffic = p.traffic
        traffic.observed, traffic.valid, traffic.association_valid = True, True, True
        traffic.signal_presence, traffic.signal_id, traffic.signal_state = 'present', 42, 'GREEN'
        traffic.stop_line_distance = 20.
        traffic.signal_groups_valid = False
        self.stop_rule = dict(id='signal:42', minimum_duration_s=2., strict=True)
        self.transport.prepare(p)
        d = decision_stub.decide(p)
        self.assertEqual('RULE_STOP', d.behavior_request['maneuver'])
        self.assertAlmostEqual(2.001, d.behavior_request['minimum_standstill_duration_s'])
        self.assertNotEqual(20, p.scene_id)
        t = planning_stub.plan(p, d)
        self.assertTrue(t.valid, t.reason)
        self.assertEqual('signal:signal:42', t.stop_obligation_id)
        self.assertAlmostEqual(2.001, t.hold_duration_s)

    def test_rear_axle_in_destination_without_whole_body_cannot_complete(self):
        self.destination_region = [(24.9, -.4), (25.1, -.4), (25.1, .4), (24.9, .4)]
        plant = BidirectionalPlant(x=25.)
        self.h.advance(x=25., speed=0.)
        for unused in range(8):
            self.step(plant)
            self.assertNotEqual('COMPLETE', self.engine._behavior.task.phase)

    def test_rule_stop_uses_actual_continuous_time_and_new_id_has_its_own_duty(self):
        plant = BidirectionalPlant(x=0.)
        self.h.advance(x=0., speed=0.)
        p = self.f.p
        traffic = p.traffic
        traffic.observed, traffic.valid, traffic.association_valid = True, True, True
        traffic.signal_presence, traffic.signal_id, traffic.signal_state = 'present', 42, 'GREEN'
        traffic.stop_line_distance = 4.2
        traffic.signal_groups_valid = False
        self.stop_rule = dict(id='signal:42', minimum_duration_s=2., strict=True)
        self.required = ('signal:signal:42', 'signal:signal:43')
        for i in range(52):
            d, t, c = self.step(plant)
            if i < 38:
                self.assertEqual(0., c.throttle)
                self.assertNotIn('signal:42', self.engine._behavior.signal._completed)
        self.assertIn('signal:42', self.engine._behavior.signal._completed,
                      (self.engine._behavior.signal._phase, self.engine._behavior.signal.session.events[-5:],
                       self.h.h.engine.standstill.__dict__, d.reason, t.stop_distance, c.execution_observation))
        self.stop_rule = dict(id='signal:43', minimum_duration_s=2., strict=True)
        traffic.signal_id = 43
        for i in range(52):
            d, t, c = self.step(plant)
            if i < 38:
                self.assertEqual(0., c.throttle)
                self.assertNotIn('signal:43', self.engine._behavior.signal._completed)
        self.assertTrue({'signal:42', 'signal:43'} <= self.engine._behavior.signal._completed)
        self.assertTrue({'signal:signal:42', 'signal:signal:43'} <= self.engine._behavior.task.completed)


class ParkingContinuationTests(unittest.TestCase):
    def test_interrupted_parking_replans_from_actual_stop_and_keeps_original_goal(self):
        from tests.test_decision_parking_runtime import FixedParkingDecisionTests
        h = FixedParkingDecisionTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        p = h.f.p
        plant = BidirectionalPlant(p.ego.x, p.ego.y, p.ego.heading, lag=.3)
        def step():
            d = h.decide()
            t = planning_stub.plan(p, d)
            self.assertTrue(t.valid, (d.reason, t.errors))
            c = control_stub.compute_control(p, t)
            self.assertTrue(c.valid, c.errors)
            h.h.transport.observe(p, d, t, c, False, dict(attempted=True, ok=True))
            plant.step(c, .05, h.h.calibration)
            h.h.advance(GoalPose(plant.x, plant.y, plant.heading), plant.speed, plant.direction)
            return d, t, c
        for unused in range(20):
            d, t, c = step()
        session = h.engine._behavior.parking.policy.session
        identifier, revision, goal = session.intent_id, session.revision, session.goal.goal_pose.to_dict()
        h.enabled = False
        for unused in range(15):
            d, t, c = step()
            self.assertEqual(0., c.throttle)
        self.assertLess(plant.speed, .05)
        stopped = plant.x
        h.enabled = True
        for unused in range(3):
            d, t, c = step()
            if session.revision > revision:
                break
        self.assertEqual(identifier, session.intent_id)
        self.assertGreater(session.revision, revision,
                           (h.engine._behavior.parking.snapshot(), session.events[-6:], d.reason, plant.x, plant.speed))
        self.assertEqual(goal, session.goal.goal_pose.to_dict())
        self.assertAlmostEqual(t.points[0].x, stopped)
        for unused in range(50):
            step()
        self.assertGreater(plant.x, stopped+.1)


class ConfiguredDecisionMigrationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_behavior_inputs import BehaviorInputConfigurationTests
        self.h = BehaviorInputConfigurationTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)

    def test_verified_route_configuration_produces_current_main_task_inputs(self):
        from behavior_inputs import ConfiguredBehaviorInputs
        h = self.h
        p = h.f.p
        h.document['task_route'] = dict(enabled=True, verified=True,
            verification_reference='AUTHORED task route only', route_id='selected-route',
            reference_end_reason='task_destination', speed_cap_mps=1.,
            segments=[dict(id='leg-a', lane_id=lane.SOURCE, start_s_m=0., end_s_m=110.,
                           center_line=[[-10., 0.], [100., 0.]])],
            destination_goal=dict(x=80., y=0., body_heading_rad=0.),
            destination_region=[[78., -1.5], [85., -1.5], [85., 1.5], [78., 1.5]])
        loader = ConfiguredBehaviorInputs(h.config, h.document, h.f.clock)
        value = loader.task_provider(p, h.h.transport.context)
        self.assertIsInstance(value, TaskDecisionInputs)
        self.assertEqual(p.frame_id, value.facts.evidence.frame_id)
        self.assertLessEqual(value.facts.evidence.valid_until_s, p.valid_until)
        self.assertEqual('selected-route', value.facts.route_id)
        h.document['task_route']['verified'] = False
        with self.assertRaises(ValueError):
            ConfiguredBehaviorInputs(h.config, h.document, h.f.clock)

    def test_automatic_parking_selection_is_shared_with_main_planning(self):
        from behavior_inputs import ConfiguredBehaviorInputs
        from parking_integration import ParkingInputBridge
        from tests import test_planning_parking_behavior as fixture
        h = self.h
        park = fixture.ParkingFixedEntryTests()
        park.setUp()
        self.addCleanup(park.doCleanups)
        p, context = park.f.p, park.transport.context
        p.source_status['gps'].update(read_ok=True, motion_fields=['steering'])
        h.document['scope'].update(case_id=p.case_id, task_id=p.task_id,
                                   map_digest=ManeuverFacts(p, park.f.clock).map_digest())
        h.document['model']['prediction']['source'] = p.target_source
        h.document['overtake']['enabled'] = False
        task = h.parking_task(park.h.mission())
        spec = task['mission']
        original = {k: spec[k] for k in ('space_id', 'access_edge_index', 'body_heading_rad')}
        spec['candidate_spaces'] = [dict(original, space_id=999), original]
        for key in ('space_id', 'access_edge_index', 'body_heading_rad', 'approach_pose', 'position_pose'):
            spec.pop(key)
        h.document['parking'] = task
        loader = ConfiguredBehaviorInputs(h.config, h.document, park.f.clock)
        bridge = ParkingInputBridge(loader.parking_provider, park.f.clock)
        bridge.settings = park.settings
        bridge.prepare(p)
        d, m = bridge.decision_inputs(p, context), bridge.planning_inputs(p, context)
        self.assertEqual(original['space_id'], d.mission.space_id)
        self.assertEqual(d.mission.approach_pose.to_dict(), m.mission.approach_pose.to_dict())
        self.assertEqual(d.mission.position_pose.to_dict(), d.mission.approach_pose.to_dict())
        self.assertGreater(d.mission.approach_pose.x, p.ego.x)
        engine = DecisionEngine(DecisionSettings(front_offset_m=3.9, half_width_m=.9),
                                park.f.clock, parking_inputs_provider=bridge.decision_inputs)
        planning_stub.configure_planning(parking_inputs_provider=bridge.planning_inputs)
        out = engine.run(p)
        trajectory = planning_stub.plan(p, out)
        self.assertEqual('PARK', out.behavior_request['maneuver'])
        self.assertTrue(trajectory.valid, (out.reason, trajectory.reason, trajectory.errors))


class ParkingSelectionClosedLoopTests(unittest.TestCase):
    def test_selected_bay_actual_parking_dwell_exit_and_terminal_hold(self):
        import math
        from tests.test_planning_parking_segments import ParkingSegmentPipelineTests
        from members.planning.candidate_validation import ExecutionEnvelope
        from members.decision.behaviors.parking_selection import ParkingMissionSelector
        h = ParkingSegmentPipelineTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        h.execution_envelope = ExecutionEnvelope(.45, .02, .6, 1.5)
        h.speed_error_mps = 1.
        original = h.bridge.provider
        selector = [None]
        def inputs(p, context):
            pair = original(p, context)
            old = pair.decision.mission
            if selector[0] is None:
                selector[0] = ParkingMissionSelector(old.mission_id, old.road_lane_ids,
                    [dict(space_id=old.space_id, access_edge_index=old.access_edge_index,
                          body_heading_rad=old.body_heading_rad)], old.exit_goal, h.f.clock, 1000000)
            selected = selector[0].select(p, old.evidence, pair.planning, old.map_digest)
            pair.decision.mission = selected.snapshot()
            pair.planning.mission = selected.snapshot()
            return pair
        h.bridge.provider = inputs
        p = h.f.p
        plant = BidirectionalPlant(p.ego.x, p.ego.y, p.ego.heading, lag=.3)
        parked_at = exit_at = None
        seen_reverse = False
        completed = False
        for tick in range(1800):
            h.bridge.prepare(p)
            d = decision_stub.decide(p)
            policy = h.engine._behavior.parking.policy
            if policy is not None and policy.session.status == 'COMPLETED':
                self.assertEqual('STOP', d.mode)
                self.assertEqual(0., d.target_speed)
                self.assertIsNone(d.behavior_request)
                completed = True
                break
            t = planning_stub.plan(p, d)
            self.assertTrue(t.valid, (tick, d.reason, t.reason, t.errors, plant.x, plant.speed))
            c = control_stub.compute_control(p, t)
            self.assertTrue(c.valid, c.errors)
            assessed = h.safety.evaluate(p, d, t, c)
            self.assertFalse(assessed.active, (tick, assessed.reason, d.reason))
            receipt = dict(attempted=True, ok=True, reason='synthetic_vehicle_only')
            tracking = h.bridge.record_sent(p, t, receipt, assessed.active)
            self.assertFalse(tracking['active'], tracking)
            h.h.transport.observe(p, d, t, c, assessed.active, receipt)
            seen_reverse = seen_reverse or t.motion_direction == -1 and plant.speed > .05
            if h.h.engine.standstill.anchor is not None and parked_at is None:
                parked_at = h.f.now
            if d.behavior_request is not None and d.behavior_request['stage'] == 'EXIT' and c.throttle > 0 and exit_at is None:
                exit_at = h.f.now
            if parked_at is not None and h.f.now-parked_at < 10.:
                self.assertEqual(0., c.throttle)
                self.assertTrue(c.handbrake)
            plant.step(c, .05, h.h.calibration)
            h.h.curvature = math.tan(plant.steering*h.h.calibration.front_steer_max_rad /
                                     h.h.calibration.steering_sign) / h.h.calibration.wheelbase_m
            h.h.advance(GoalPose(plant.x, plant.y, plant.heading), plant.speed, plant.direction)
        self.assertTrue(completed, (plant.x, plant.speed, h.engine.behavior_diagnostics()))
        self.assertTrue(seen_reverse)
        self.assertIsNotNone(parked_at)
        self.assertIsNotNone(exit_at)
        self.assertGreaterEqual(exit_at-parked_at, 10.)
        self.assertAlmostEqual(34., plant.x, delta=.15)
