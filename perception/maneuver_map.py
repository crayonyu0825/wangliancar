"""Bind verified static semantics to the existing SDK lane/route observations."""
import copy
import math

from core.geometry import project_polyline
from core.interfaces import LaneContext
from core.route_segments import verified_spans
from perception.opendrive_semantics import OpenDriveSemantics,lane_identity
from perception.crossing_ranges import CrossingRanges
from perception.road_regions import RoadRegions


class ManeuverMap(object):
    def __init__(self,adapter):
        self.adapter,self.document,self.digest,self.parse_reason = adapter,None,None,"DOCUMENT_UNAVAILABLE"
        self.crossing_ranges = CrossingRanges()
        self.road_regions = RoadRegions()
        self.passage_sources = {}
        self._read()  # Parse at route-manager construction, before driving loop.

    def _read(self):
        reader = getattr(self.adapter,"read_map_document",None)
        if reader is None:
            return dict(verified=False,reason="MAP_DOCUMENT_API_UNAVAILABLE")
        try:
            data,metadata = reader()
            if not isinstance(metadata,dict) or metadata.get("verified") is not True or data is None:
                return metadata if isinstance(metadata,dict) else dict(verified=False,reason="INVALID_MAP_METADATA")
            digest = metadata.get("digest")
            if not isinstance(digest,str) or len(digest)!=32:
                return dict(verified=False,reason="MAP_DIGEST_UNAVAILABLE")
            if self.digest!=digest:
                self.passage_sources.clear()
                self.crossing_ranges.clear()
                self.road_regions.clear()
                self.document,self.digest,self.parse_reason = None,digest,"DOCUMENT_INVALID_OR_UNSUPPORTED"
                try:
                    self.document = OpenDriveSemantics(data)
                    self.parse_reason = "SEMANTIC_DOCUMENT_PARSED"
                except (ValueError,TypeError,KeyError,OverflowError,UnicodeError,IndexError):
                    pass
            result = dict(metadata)
            result.update(semantic_verified=self.document is not None,semantic_reason=self.parse_reason,
                unsupported_road_count=0 if self.document is None else len(self.document.road_errors))
            return result
        except Exception:
            return dict(verified=False,reason="MAP_DOCUMENT_QUERY_FAILED")

    def observe(self,ego,lane,neighbors):
        for item in neighbors:
            item.pop('crossing_to_current',None)
            item.update(marking_intervals_road_s=[],marking_semantics_verified=False,
                        crossing_range_verified=False,crossing_geometry_bound=False,
                        travel_direction_verified=False,travel_matches_declared_direction=False,
                        crossing_ranges_world=[],crossing_boundary_window={})
            item.pop("marking_map_digest",None)
            for key in ("local_geometry","local_coverage_verified","local_drivable_verified"):
                item.pop(key,None)
        metadata = self._read()
        if metadata.get("verified") is not True or metadata.get("semantic_verified") is not True:
            return metadata,[]
        if not ego.valid or not lane.valid or not getattr(self.adapter,"map_loaded",False):
            metadata.update(geometry_bound=False,geometry_reason="EGO_OR_CURRENT_LANE_UNAVAILABLE")
            return metadata,[]
        try:
            hdmap = self.adapter.hdmap
            road,section,member = self.document.lane(lane.lane_id)
            st = hdmap.getRoadST(hdmap.pySimString(road["id"]),hdmap.pySimPoint3D(ego.x,ego.y,ego.z))
            road_s = float(st.s)
            if (getattr(st,"exists",False) is not True or not math.isfinite(road_s)
                    or not section["start_m"]<=road_s<section["end_m"]):
                raise ValueError("current road-s or section unverified")
            metadata.update(geometry_bound=True,current_lane_id=lane.lane_id,current_road_s_m=road_s,
                            coordinate_system="opendrive_road_s")
            for item in neighbors:
                self._markings(item,lane,road_s)
            prior_sources=self._prior_sources(lane,neighbors,road_s)
            metadata["boundary_binding_budget"] = self.crossing_ranges.observe(
                ego,lane,neighbors,self.document,self.digest,hdmap,prior_sources)
            metadata["road_regions"] = self.road_regions.observe(ego,lane,neighbors,self.document,self.digest)
        except (AttributeError,ValueError,TypeError,KeyError,IndexError,OverflowError,RuntimeError):
            metadata.update(geometry_bound=False,geometry_reason="ROAD_S_OR_LANE_SECTION_UNVERIFIED")
        return metadata,self._junctions(ego,lane,metadata.get("geometry_bound") is True)

    def _prior_sources(self,lane,neighbors,road_s):
        """Refresh an already prepared original direction after SDK lane switch.

        Do not infer A->B permission from B->A. The original A binding must
        already exist and its native marking is freshly probed in the shared
        binder budget. No old dynamic observation is retained here.
        """
        result=[]
        for item in neighbors:
            if (item.get('geometry_valid') is not True or item.get('shared_boundary_verified') is not True
                    or item.get('same_direction') is not True
                    or str(item.get('lane_type','unknown')).lower().split('.')[-1]!='driving'
                    or type(item.get('native_sample_reversed')) is not bool
                    or item.get('side') not in ('left','right')
                    or not lane.lane_type_valid):
                continue
            source=LaneContext()
            source.lane_id=item['lane_id']
            for key in ('center_line','left_boundary','right_boundary'):
                setattr(source,key,item[key])
            side='left' if item['side']=='right' else 'right'
            native_side=('right' if side=='left' else 'left') if item['native_sample_reversed'] else side
            reverse=dict(source_lane_id=source.lane_id,lane_id=lane.lane_id,side=side,
                geometry_valid=True,shared_boundary_verified=True,same_direction=True,
                lane_type=lane.lane_type,left_boundary=copy.deepcopy(lane.left_boundary),right_boundary=copy.deepcopy(lane.right_boundary),
                marking_observation=dict(native_side=native_side))
            self._markings(reverse,source,road_s,defer_native=True)
            item['crossing_to_current']=reverse
            result.append((source,reverse))
        return result

    def _markings(self,item,lane,road_s,defer_native=False):
        item["marking_scope"] = "road_s_intervals"
        item["crossing_range_verified"] = False
        item["crossing_geometry_bound"] = False
        try:
            intervals = self.document.shared_markings(lane.lane_id,item["lane_id"])
            item["marking_intervals_road_s"] = intervals
            item["marking_map_digest"] = self.digest
            active = next(v for v in intervals if v["road_s_start_m"]<=road_s<v["road_s_end_m"])
            if defer_native:
                item.update(marking_semantics_verified=False,crossing_allowed_at_ego=False,
                            crossing_scope_reason='SOURCE_NATIVE_RECHECK_REQUIRED')
                return
            native = item.get("marking_observation",{})
            sdk_type = str(native.get("type","unknown")).lower().split(".")[-1]
            sdk_offset = native.get("section_s_offset_m")
            section = self.document.lane(lane.lane_id)[1]
            sdk_agrees = (type(sdk_offset) in (int,float) and math.isfinite(sdk_offset)
                and abs(sdk_offset-(active["road_s_start_m"]-section["start_m"]))<.001
                and sdk_type==active["type"])
            item.update(marking_semantics_verified=bool(active["semantic_verified"] and sdk_agrees),
                crossing_allowed_at_ego=bool(item.get("geometry_valid") and item.get("shared_boundary_verified")
                    and item.get("same_direction") and str(item.get("lane_type","unknown")).lower().split(".")[-1]=="driving"
                    and active["crossing_permitted"] and sdk_agrees),
                crossing_scope_reason="WORLD_BOUNDARY_INTERVAL_BINDING_REQUIRED" if sdk_agrees else "SDK_XML_MARKING_MISMATCH")
        except (KeyError,IndexError,ValueError,TypeError,StopIteration):
            item.update(marking_semantics_verified=False,crossing_allowed_at_ego=False,
                        crossing_scope_reason="ADJACENT_LANE_SEMANTICS_UNVERIFIED")

    def _endpoint(self,identity,point,contact):
        try:
            road,section,unused = self.document.lane(identity)
            index = lane_identity(identity)[1]
            if index!=(0 if contact=="start" else len(road["sections"])-1):
                return False
            hdmap = self.adapter.hdmap
            value = hdmap.getRoadST(hdmap.pySimString(road["id"]),hdmap.pySimPoint3D(*point))
            s = float(value.s)
            expected = 0. if contact=="start" else road["length_m"]
            return getattr(value,"exists",False) is True and math.isfinite(s) and abs(s-expected)<=.2
        except (AttributeError,TypeError,ValueError,KeyError,IndexError,OverflowError,RuntimeError):
            return False

    def _junctions(self,ego,lane,geometry_bound):
        reference = lane.forward_reference
        spans = verified_spans(len(lane.center_line),len(reference),lane.forward_lane_ids,lane.forward_lane_spans)
        if (not lane.forward_reference_valid or not spans or lane.forward_lane_ids[0]!=lane.lane_id
                or not reference or lane.center_line!=reference[:len(lane.center_line)]
                or not all(len(v)>=3 and all(type(x) in (int,float) and math.isfinite(x) for x in v[:3]) for v in reference)):
            return []
        origin = project_polyline(reference,ego.x,ego.y)
        if origin is None:
            return []
        distances = [0.]
        for a,b in zip(reference,reference[1:]):
            distances.append(distances[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
        groups = {}
        for identity in lane.forward_lane_ids:
            try:
                road,section,member = self.document.lane(identity)
                junction = road["junction_id"]
                if junction!="-1" and junction in self.document.junctions:
                    groups.setdefault(junction,[]).append(identity)
            except (KeyError,ValueError,IndexError,TypeError):
                continue
        result = []
        for junction,identities in groups.items():
            declared = copy.deepcopy(self.document.junctions[junction])
            start,end = spans[identities[0]][0],spans[identities[-1]][1]
            first,last = lane.forward_lane_ids.index(identities[0]),lane.forward_lane_ids.index(identities[-1])
            if lane.forward_lane_ids[first:last+1]!=identities:
                continue  # Re-entry into the same junction needs separate gates.
            previous = lane.forward_lane_ids[first-1] if first>0 else None
            following = lane.forward_lane_ids[last+1] if last+1<len(lane.forward_lane_ids) else None
            selected = []
            if previous:
                try:
                    a,b = lane_identity(previous),lane_identity(identities[0])
                    selected = [c for c in declared["connections"] if c["incoming_road_id"]==a[0]
                        and c["connecting_road_id"]==b[0]
                        and any(v["from_lane_id"]==a[2] and v["to_lane_id"]==b[2] for v in c["lane_links"])]
                except (TypeError,ValueError):
                    pass
            # A route's existing verified join supplies the world reference
            # point. This is an entry gate, not a stop line/parking goal.
            entry_bound = False
            chain_bound = bool(len(selected)==1 and any(chain["connecting_lane_ids"]==identities
                and chain["from_lane_id"]==lane_identity(previous)[2] for chain in selected[0]["lane_chains"]))
            if geometry_bound and len(selected)==1 and previous:
                contact = selected[0]["contact_point"]
                connector = self.document.lane(identities[0])[0]
                incoming_link = connector["predecessor" if contact=="start" else "successor"]
                entry_bound = (self._endpoint(identities[0],reference[start],contact)
                    and self._endpoint(previous,reference[start],incoming_link["contact_point"]))
            declared.update(kind="junction",source="sdk_matched_opendrive",map_digest=self.digest,
                selected_lane_ids=list(identities),selected_connection_ids=[v["id"] for v in selected],
                selected_connection_verified=len(selected)==1,selected_route_chain_verified=chain_bound,
                entry_gate_verified=entry_bound,
                entry_reference_point=tuple(reference[start]),entry_distance_m=distances[start]-origin["s"],
                exit_reference_point=tuple(reference[end]),exit_distance_m=distances[end]-origin["s"],
                exit_lane_id=following,exit_gate_verified=False,
                corridor_geometry_verified=False,conflict_coverage_complete=False,
                right_of_way="UNKNOWN",rule_verified=False,exit_coverage_verified=False)
            if geometry_bound and chain_bound and following:
                try:
                    outgoing = selected[0]["outgoing_road"]
                    contact = "end" if selected[0]["contact_point"]=="start" else "start"
                    last_road,last_section,last_lane = self.document.lane(identities[-1])
                    following_road,following_section,following_lane = self.document.lane(following)
                    link = last_lane["links"]["successor" if contact=="end" else "predecessor"]
                    declared["exit_gate_verified"] = bool(outgoing and outgoing["element_type"]=="road"
                        and outgoing["element_id"]==following_road["id"] and following_road["junction_id"]=="-1"
                        and link==lane_identity(following)[2]
                        and self._endpoint(identities[-1],reference[end],contact)
                        and self._endpoint(following,reference[end],outgoing["contact_point"]))
                except (ValueError,TypeError,KeyError,IndexError):
                    pass
            result.append(declared)
        return result
