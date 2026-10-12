"""Bounded OpenDRIVE semantics, using SDK geometry rather than rebuilding it.

The XML proves road/section intervals and declared connections only. It cannot
prove visibility, occupancy, right of way, or that a world trajectory is legal.
All public s values here are road-reference metres, never lane-polyline metres.
"""
import math
import re
import xml.etree.ElementTree as ET

MAX_BYTES = 16*1024*1024
MAX_NODES = 200000
MAX_DEPTH = 64


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite semantic coordinate")
    return result


def _identity(value):
    if not isinstance(value,str) or not re.fullmatch(r"0|[1-9][0-9]{0,17}",value):
        raise ValueError("unsupported SDK road/junction identity")
    return value


def lane_identity(value):
    if not isinstance(value,str) or not re.fullmatch(r"(0|[1-9][0-9]{0,17})_(0|[1-9][0-9]{0,5})_(-?[1-9][0-9]{0,5})",value):
        raise ValueError("unsupported SDK lane identity")
    road,section,lane = value.split("_")
    return road,int(section),int(lane)


def _tree(data):
    if (not isinstance(data,bytes) or not data or len(data)>MAX_BYTES
            or data.startswith((b"\xff\xfe",b"\xfe\xff"))
            or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper()):
        raise ValueError("unsupported or oversized semantic document")
    data.decode("utf-8-sig","strict")
    parser,depth,count,root = ET.XMLPullParser(events=("start","end")),0,0,None
    try:
        for offset in range(0,len(data),65536):
            parser.feed(data[offset:offset+65536])
            for event,element in parser.read_events():
                if event=="start":
                    count,depth = count+1,depth+1
                    if root is None:
                        root = element
                    if count>MAX_NODES or depth>MAX_DEPTH:
                        raise ValueError("semantic XML structure budget")
                else:
                    depth -= 1
        parser.close()
    except ET.ParseError:
        raise ValueError("malformed semantic XML")
    if root is None or root.tag!="OpenDRIVE" or depth!=0:
        raise ValueError("unsupported semantic XML root")
    headers = root.findall("header")
    if (len(headers)!=1 or headers[0].get("revMajor")!="1"
            or headers[0].get("revMinor") not in ("4","5","6","7","8")):
        raise ValueError("OpenDRIVE semantic version unverified")
    return root


def _road_link(element,kind):
    values = element.findall("link/"+kind)
    if not values:
        return None
    if len(values)!=1:
        raise ValueError("duplicate road link")
    value = values[0]
    result = dict(element_type=value.get("elementType"),element_id=_identity(value.get("elementId")),
                  contact_point=value.get("contactPoint"))
    if result["element_type"] not in ("road","junction"):
        raise ValueError("unsupported road link")
    if result["element_type"]=="road" and result["contact_point"] not in ("start","end"):
        raise ValueError("unresolved road contact point")
    return result


def _lane(element,start,end,side):
    lane_id = int(element.attrib["id"])
    if (abs(lane_id)>999999 or (side=="left" and lane_id<=0)
            or (side=="right" and lane_id>=0) or (side=="center" and lane_id!=0)):
        raise ValueError("invalid lane side/identity")
    records = element.findall("roadMark")
    if len(records)>2048:
        raise ValueError("marking record budget")
    offsets = [_number(v.attrib["sOffset"]) for v in records]
    if any(not 0<=v<end-start for v in offsets) or any(a>=b for a,b in zip(offsets,offsets[1:])):
        raise ValueError("ambiguous marking order or section scope")
    intervals = []
    if not offsets or offsets[0]>0:
        intervals.append(dict(road_s_start_m=start,road_s_end_m=start+(offsets[0] if offsets else end-start),
                              type="unknown",lane_change="unknown",semantic_verified=False,
                              reason="marking_missing"))
    for index,record in enumerate(records):
        kind,change = record.get("type","unknown"),record.get("laneChange","both")
        # A detailed/offset marking may override the simplified type and rule.
        supported = not list(record) and kind in ("broken","solid","none") and change in ("both","none","increase","decrease")
        intervals.append(dict(road_s_start_m=start+offsets[index],
            road_s_end_m=start+offsets[index+1] if index+1<len(offsets) else end,
            type=kind,lane_change=change,semantic_verified=supported,
            reason="plain_marking" if supported else "detailed_or_unknown_marking"))
    links = {}
    for name in ("predecessor","successor"):
        values = element.findall("link/"+name)
        if len(values)>1:
            raise ValueError("ambiguous lane link")
        links[name] = int(values[0].attrib["id"]) if values else None
    restricted = (element.get("direction","standard")!="standard"
        or element.get("dynamicLaneDirection","false")!="false"
        or element.get("dynamicLaneType","false")!="false"
        or bool(element.findall("access") or element.findall("rule")))
    return lane_id,dict(type=element.get("type","unknown"),permission_scope_verified=not restricted,
                       markings=intervals,links=links)


def _road(element):
    identity = _identity(element.attrib["id"])
    length = _number(element.attrib["length"])
    if not 0<length<=1.e7:
        raise ValueError("invalid road length")
    junction = element.get("junction")
    if junction!="-1":
        junction = _identity(junction)
    sections = element.findall("lanes/laneSection")
    if not 1<=len(sections)<=256:
        raise ValueError("lane section budget or missing sections")
    starts = [_number(v.attrib["s"]) for v in sections]
    if starts[0]!=0 or any(not 0<=v<length for v in starts) or any(a>=b for a,b in zip(starts,starts[1:])):
        raise ValueError("ambiguous lane section order")
    result = dict(id=identity,length_m=length,junction_id=junction,sections=[],
                  rule=element.get("rule","RHT"),predecessor=_road_link(element,"predecessor"),
                  successor=_road_link(element,"successor"))
    for index,section in enumerate(sections):
        if section.get("singleSide","false")!="false":
            raise ValueError("single-side lane section carry-over unsupported")
        start,end = starts[index],starts[index+1] if index+1<len(starts) else length
        lanes = {}
        for side in ("left","center","right"):
            for element_lane in section.findall(side+"/lane"):
                lane_id,record = _lane(element_lane,start,end,side)
                if lane_id in lanes or len(lanes)>=128:
                    raise ValueError("duplicate lane or lane budget")
                lanes[lane_id] = record
        if 0 not in lanes:
            raise ValueError("missing center lane")
        result["sections"].append(dict(start_m=start,end_m=end,lanes=lanes))
    return identity,result


def _junction(element,roads):
    identity = _identity(element.attrib["id"])
    result = dict(id=identity,type=element.get("type","default"),connections=[],priority_pairs=[],
                  topology_complete=False,reason="unsupported_junction_type")
    if result["type"] not in ("default","common"):
        return identity,result
    complete,covered,work = True,set(),0
    identities = set()
    for connection in element.findall("connection"):
        try:
            key = _identity(connection.attrib["id"])
            if key in identities or len(identities)>=2048:
                raise ValueError("duplicate connection or budget")
            identities.add(key)
            incoming,connector = _identity(connection.attrib["incomingRoad"]),_identity(connection.attrib["connectingRoad"])
            contact = connection.attrib["contactPoint"]
            if (incoming not in roads or connector not in roads or contact not in ("start","end")
                    or roads[connector]["junction_id"]!=identity or connection.get("type","default")!="default"):
                raise ValueError("connection endpoint unsupported")
            endpoint = roads[connector]["sections"][0 if contact=="start" else -1]["lanes"]
            pairs,seen = [],set()
            for link in connection.findall("laneLink"):
                a,b = int(link.attrib["from"]),int(link.attrib["to"])
                member = endpoint.get(b)
                if (not a or not b or b in seen or member is None or member["type"]!="driving"
                        or member["links"]["predecessor" if contact=="start" else "successor"]!=a):
                    raise ValueError("junction lane link unverified")
                pairs.append(dict(from_lane_id=a,to_lane_id=b))
                seen.add(b)
            if not pairs:
                raise ValueError("junction lane links missing")
            road = roads[connector]
            entry = road["predecessor" if contact=="start" else "successor"]
            if entry is None or entry["element_type"]!="road" or entry["element_id"]!=incoming:
                raise ValueError("junction incoming road link mismatch")
            incoming_lanes = roads[incoming]["sections"][0 if entry["contact_point"]=="start" else -1]["lanes"]
            if any(v["from_lane_id"] not in incoming_lanes or incoming_lanes[v["from_lane_id"]]["type"]!="driving" for v in pairs):
                raise ValueError("junction incoming lane missing")
            outgoing = road["successor" if contact=="start" else "predecessor"]
            if (outgoing is None or outgoing["element_type"]!="road" or outgoing["element_id"] not in roads):
                raise ValueError("junction outgoing road missing")
            exit_lanes = roads[outgoing["element_id"]]["sections"][0 if outgoing["contact_point"]=="start" else -1]["lanes"]
            chains = []
            for pair in pairs:
                current,chain = pair["to_lane_id"],[]
                order = range(len(road["sections"])) if contact=="start" else range(len(road["sections"])-1,-1,-1)
                local_covered = set()
                for index in order:
                    work += 1
                    if work>100000:
                        raise ValueError("junction lane traversal budget")
                    lane = road["sections"][index]["lanes"].get(current)
                    if not current or lane is None or lane["type"]!="driving" or not lane["permission_scope_verified"]:
                        raise ValueError("junction lane chain missing or direction restricted")
                    chain.append('%s_%d_%d'%(connector,index,current))
                    local_covered.add((connector,index,current))
                    current = lane["links"]["successor" if contact=="start" else "predecessor"]
                if not current or current not in exit_lanes or exit_lanes[current]["type"]!="driving":
                    raise ValueError("junction exit lane link missing")
                chains.append(dict(from_lane_id=pair["from_lane_id"],connecting_lane_ids=chain,
                                   outgoing_lane_id=current))
                covered.update(local_covered)
            result["connections"].append(dict(id=key,incoming_road_id=incoming,connecting_road_id=connector,
                contact_point=contact,lane_links=pairs,lane_chains=chains,outgoing_road=outgoing))
        except (KeyError,TypeError,ValueError,OverflowError):
            complete = False
    # Declared priority is a fact; signal/sign rules still need a producer.
    for priority in element.findall("priority"):
        try:
            high,low = _identity(priority.attrib["high"]),_identity(priority.attrib["low"])
            if high not in roads or low not in roads or high==low:
                raise ValueError("invalid priority pair")
            result["priority_pairs"].append(dict(high_road_id=high,low_road_id=low))
        except (KeyError,TypeError,ValueError):
            complete = False
    expected = {(key,index,lane_id) for key,road in roads.items() if road["junction_id"]==identity
                for index,section in enumerate(road["sections"]) for lane_id,lane in section["lanes"].items()
                if lane["type"]=="driving" and lane_id!=0}
    complete = complete and expected==covered
    result.update(topology_complete=complete and bool(result["connections"]),
                  reason="declared_lane_connections" if complete and result["connections"] else "incomplete_lane_connections")
    return identity,result


class OpenDriveSemantics(object):
    def __init__(self,data):
        root = _tree(data)
        self.roads,self.junctions,self.road_errors = {},{},{}
        elements = root.findall("road")
        if not 1<=len(elements)<=4096:
            raise ValueError("road count budget")
        seen = set()
        for element in elements:
            identity = _identity(element.get("id"))
            if identity in seen:
                raise ValueError("duplicate road identity")
            seen.add(identity)
            try:
                unused,record = _road(element)
                self.roads[identity] = record
            except (KeyError,TypeError,ValueError,OverflowError):
                self.road_errors[identity] = "ROAD_SEMANTICS_UNSUPPORTED_OR_INVALID"
        for element in root.findall("junction"):
            identity,record = _junction(element,self.roads)
            if identity in self.junctions or len(self.junctions)>=4096:
                raise ValueError("duplicate junction identity or budget")
            self.junctions[identity] = record
            declared = {v.get("id") for v in elements if v.get("junction")==identity}
            represented = {v["connecting_road_id"] for v in record["connections"]}
            if declared!=represented:
                record.update(topology_complete=False,reason="connector_roads_missing_from_connections")

    def lane(self,identity):
        road,section,lane = lane_identity(identity)
        record = self.roads[road]
        part = record["sections"][section]
        return record,part,part["lanes"][lane]

    def shared_markings(self,current,neighbor):
        a,b = lane_identity(current),lane_identity(neighbor)
        if a[:2]!=b[:2] or a[2]*b[2]<=0 or abs(a[2]-b[2])!=1:
            raise ValueError("not adjacent same-side lanes")
        road,section,source = self.lane(current)
        other = self.lane(neighbor)[2]
        owner = min((a[2],b[2]),key=abs)
        result = []
        for value in section["lanes"][owner]["markings"]:
            item = dict(value)
            change = item["lane_change"]
            direction = "increase" if b[2]>a[2] else "decrease"
            permission = (item["semantic_verified"] and item["type"]=="broken" and change in ("both",direction)
                and source["type"]==other["type"]=="driving"
                and source["permission_scope_verified"] and other["permission_scope_verified"]
                and road["rule"] in ("RHT","LHT"))
            item.update(crossing_permitted=permission,owner_lane_id=owner,
                        coordinate_system="opendrive_road_s",permission_direction=direction)
            result.append(item)
        return result
