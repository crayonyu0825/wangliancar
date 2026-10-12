"""R06 consumer for existing D03 policies; never invent a maneuver intent."""
from core.maneuver_facts import ManeuverFacts
from core.behavior_contract import require
from members.decision.behaviors.observations import Evidence, MotionObject
from members.decision.behaviors.lane_change import LaneCandidate


def lane_candidates(perception, frame, clock=None, active_candidate=None):
    require(frame.context.key()[:3] == (perception.case_id, perception.task_id, perception.scene_id)
            and frame.frame_id == perception.frame_id, 'MANEUVER_DECISION_CONTEXT_MISMATCH')
    facts = ManeuverFacts(perception, clock)
    require(frame.current(facts.check_current()) and not frame.paused, 'MANEUVER_DECISION_FRAME_UNUSABLE')
    objects = [MotionObject(v['id'], v['x'], v['y'], v['vx'], v['vy'], v['length'], v['width'], v['heading'])
               for v in facts.objects()]
    candidates, failures = [], []
    digest=facts.map_digest()
    lane_ids=facts.neighbors()
    if active_candidate is not None:
        require(isinstance(active_candidate,LaneCandidate)
                and active_candidate.source_lane_id is not None
                and active_candidate.map_digest==digest
                and perception.lane.lane_id in (active_candidate.source_lane_id,active_candidate.lane_id),
                'MANEUVER_ACTIVE_LANE_PAIR_OR_MAP_MISMATCH')
        lane_ids=[active_candidate.lane_id]
    for lane_id in lane_ids:
        try:
            road=facts.road(lane_id)
            source_id=perception.lane.lane_id if active_candidate is None else active_candidate.source_lane_id
            crossing=None
            try:
                crossing=facts.crossing(lane_id,source_id)
                if active_candidate is not None:
                    require(crossing['side']==active_candidate.side,'MANEUVER_ORIGINAL_SIDE_CHANGED')
            except (ValueError,TypeError,KeyError,IndexError,OverflowError) as error:
                if active_candidate is None: raise
                # A fresh current target road can still prove settlement.
                # Missing original permission cannot authorize a crossing.
                crossing=None
                failures.append(dict(lane_id=lane_id,reason=str(error)))
            current=lane_id==perception.lane.lane_id
            if current:
                require(perception.lane.lane_width_valid is True,'MANEUVER_CURRENT_WIDTH_UNKNOWN')
                width=perception.lane.lane_width
            else:
                source=next(v for v in facts.env.neighbor_lanes if v['lane_id']==lane_id)
                width=source.get('width_m')
            evidence = road['evidence']
            candidate = LaneCandidate(lane_id, active_candidate.side if active_candidate is not None else crossing['side'],
                [p[:2] for p in road['center_line']],width,'BROKEN' if crossing else 'UNKNOWN',
                bool(crossing and crossing['permitted_fragments']), True,
                Evidence(frame.context, evidence['frame_id'], evidence['observed_at_s'], evidence['valid_until_s'],
                    usable=True, coverage_verified=True, source=evidence['source'], source_kind=evidence['source_kind']),
                True, objects, [p[:2] for p in road['left_boundary']], [p[:2] for p in road['right_boundary']],
                [p[:2] for p in crossing['shared_boundary_world']] if crossing else None,
                [[p[:2] for p in line] for line in crossing['permitted_fragments']] if crossing else None,
                source_lane_id=source_id,map_digest=digest,is_current_lane=current,
                object_age_s=frame.observed_at_s-facts.env.dynamic_observed_at_s)
            facts.check_current(True)
            candidates.append(candidate)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            failures.append(dict(lane_id=lane_id, reason=str(error)))
    facts.check_current(True)
    return candidates, failures
