"""Formal R06 snapshot for P02/P03 callers; no path or driving authority.

Road geometry and current complete visibility are separate constraints. Raw
object poses remain at source observation time; callers must supply a verified
prediction model and account for age, uncertainty and execution timing.
"""
from collections import OrderedDict

from core.maneuver_facts import ManeuverFacts
from core.behavior_contract import require
from members.planning.candidate_validation import CorridorRegion
from members.planning.crossing_corridor import CrossingCorridor


class ManeuverGeometryCache(object):
    """Bounded static preparation only; every scene read rechecks live facts."""
    def __init__(self,max_entries,preparation_max_checks):
        require(type(max_entries) is int and 2<=max_entries<=16
                and type(preparation_max_checks) is int and preparation_max_checks>0,
                'MANEUVER_GEOMETRY_CACHE_BUDGET_INVALID')
        self.max_entries,self.preparation_max_checks=max_entries,preparation_max_checks
        self.scope,self._entries=None,OrderedDict()

    def clear(self):
        self.scope=None; self._entries.clear()

    def region(self,scope,kind,polygons,source_descriptor=None):
        if scope!=self.scope:
            self.clear(); self.scope=scope
        frozen=tuple(tuple(tuple(p) for p in polygon) for polygon in polygons)
        key=(kind,frozen,source_descriptor)
        cached=self._entries.get(key)
        if (cached is None or cached.polygons!=frozen
                or cached.preparation_max_checks!=self.preparation_max_checks
                or getattr(cached,'_source_descriptor',None)!=source_descriptor):
            self._entries[key]=(CorridorRegion(frozen,preparation_max_checks=self.preparation_max_checks)
                if source_descriptor is None else CrossingCorridor(frozen,source_descriptor,self.preparation_max_checks))
        self._entries.move_to_end(key)
        while len(self._entries)>self.max_entries: self._entries.popitem(last=False)
        return self._entries[key]


def read_maneuver_scene(perception, lane_ids, clock=None, geometry_cache=None, source_lane_id=None):
    require(isinstance(lane_ids, (list, tuple)) and 1 <= len(lane_ids) <= 3
            and all(isinstance(v, str) and v for v in lane_ids)
            and len(lane_ids) == len(set(lane_ids)), 'MANEUVER_PLANNING_LANE_SET_INVALID')
    facts = ManeuverFacts(perception, clock)
    allowed = [perception.lane.lane_id] + facts.neighbors()
    require(all(v in allowed for v in lane_ids), 'MANEUVER_PLANNING_LANE_UNAVAILABLE')
    require(source_lane_id is None or (len(lane_ids)==2 and source_lane_id in lane_ids
            and perception.lane.lane_id in lane_ids), 'MANEUVER_ORIGINAL_LANE_PAIR_UNAVAILABLE')
    objects, roads, crossings = facts.objects(), [], []
    for lane_id in lane_ids:
        roads.append(facts.road(lane_id, require_coverage=False))
        if source_lane_id is None and lane_id != perception.lane.lane_id:
            crossings.append(facts.crossing(lane_id))
    if source_lane_id is not None:
        target=next(v for v in lane_ids if v!=source_lane_id)
        crossings.append(facts.crossing(target,source_lane_id))
    raw = [p for road in roads for name in ('center_line', 'left_boundary', 'right_boundary') for p in road[name]]
    views = facts.coverage(raw)
    require(bool(views), 'MANEUVER_PLANNING_COVERAGE_UNAVAILABLE')
    deadlines = [facts.env.valid_until, facts.env.dynamic_valid_until]
    deadlines += [road['geometry_evidence']['valid_until_s'] for road in roads]
    deadlines += [v['valid_until_s'] for v in views]
    deadlines += [v['evidence']['valid_until_s'] for v in objects + crossings]
    facts.check_current(True)
    if geometry_cache is None:
        corridor,coverage=(CorridorRegion([v['polygon'] for v in values]) for values in (roads,views))
    else:
        require(isinstance(geometry_cache,ManeuverGeometryCache),'MANEUVER_GEOMETRY_CACHE_INVALID')
        scope=(perception.case_id,perception.task_id,perception.scene_id,facts.map_digest())
        corridor=geometry_cache.region(scope,('road',tuple(lane_ids)),[v['polygon'] for v in roads])
        coverage=geometry_cache.region(scope,('visibility',tuple(lane_ids)),[v['polygon'] for v in views])
    facts.check_current(True)
    legal=None
    if len(lane_ids)==2 and perception.lane.lane_id in lane_ids and len(crossings)==1:
        crossing=crossings[0]
        borders=crossing['source_lane_boundaries']
        descriptor=(tuple(borders['left']),tuple(borders['right']),crossing['side'],
                    tuple((tuple(v[0]),tuple(v[-1])) for v in crossing['permitted_source_positions']))
        polygons=[v['polygon'] for v in roads]
        if geometry_cache is None:
            legal=CrossingCorridor(tuple(tuple(tuple(p) for p in v) for v in polygons),descriptor,None)
        else:
            require(isinstance(geometry_cache,ManeuverGeometryCache),'MANEUVER_GEOMETRY_CACHE_INVALID')
            require(geometry_cache.max_entries>=3,'MANEUVER_CROSSING_CACHE_NEEDS_THREE_ENTRIES')
            scope=(perception.case_id,perception.task_id,perception.scene_id,facts.map_digest())
            legal=geometry_cache.region(scope,('legal_crossing',tuple(lane_ids)),polygons,descriptor)
    facts.check_current(True)
    return dict(roads=roads, corridor=corridor,
                legal_crossing_corridor=legal,
                coverage=coverage, coverage_records=views,
                objects=objects, crossings=crossings, map_digest=facts.map_digest(),
                source_observed_at_s=facts.env.dynamic_observed_at_s,
                source_valid_until_s=min(deadlines), frame_id=perception.frame_id,
                authority='geometry_and_current_visibility_only')
