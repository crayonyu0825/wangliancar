"""Exact map identity, XML semantics and actual captain entry (Python 3.6)."""
import copy
import ctypes
import hashlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.config import AppConfig
from core.serialization import perception_to_dict
from perception.opendrive_semantics import OpenDriveSemantics
from perception.maneuver_map import ManeuverMap
from perception.maneuver_environment import ManeuverEnvironmentBuilder
from simone_platform.map_document import MapDocument,MAX_DOCUMENT_BYTES
from simone_platform.simone_adapter import SimOneAdapter
from tests.test_maneuver_perception import perception,LaneMap
from tests.test_route_continuation import manager,NativeId


HEADER = '<OpenDRIVE><header revMajor="1" revMinor="4"/>'


def mark(offset=0,kind="broken",change="both",children=""):
    return '<roadMark sOffset="%s" type="%s" laneChange="%s">%s</roadMark>'%(offset,kind,change,children)


def lane(value,marks="",links="",kind="driving",extra=""):
    return '<lane id="%s" type="%s" %s><link>%s</link>%s</lane>'%(value,kind,extra,links,marks)


def section(start=0,marks=None):
    marks = mark() if marks is None else marks
    return '<laneSection s="%s"><center>%s</center><right>%s%s</right></laneSection>'%(
        start,lane(0,mark(kind="solid")),lane(-1,marks),lane(-2,mark(kind="solid")))


def road(identity="1",length=120,sections=None,junction="-1",links="",rule="RHT"):
    return '<road id="%s" length="%s" junction="%s" rule="%s"><link>%s</link><lanes>%s</lanes></road>'%(
        identity,length,junction,rule,links,section() if sections is None else sections)


def document(roads=None,junctions=""):
    return (HEADER+(road() if roads is None else roads)+junctions+'</OpenDRIVE>').encode('utf-8')


class NativeMap(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('openDrive',ctypes.c_char*128),('openDriveUrl',ctypes.c_char*256),('opendriveMd5',ctypes.c_char*128)]


class IdentityService(object):
    def __init__(self,data):
        self.identifier,self.digest,self.url,self.ok = b'map',hashlib.md5(data).hexdigest().encode('ascii'),b'',True
        self.reads = 0
    def SoGetHDMapData(self,value):
        self.reads += 1
        value.openDrive,value.opendriveMd5,value.openDriveUrl = self.identifier,self.digest,self.url
        return self.ok


STRUCTS = NS(SimOne_Data_Map=NativeMap)


def map_adapter(data=None):
    data = document() if data is None else data
    api = LaneMap()
    api.getRoadMark = lambda *args: NS(exists=True,left=NS(type='broken',sOffset=0.,length=3.),right=NS(type='solid',sOffset=0.,length=3.))
    api.getRoadST = lambda native,point: NS(exists=True,s=point[0]) if isinstance(native,NativeId) else None
    metadata = dict(verified=True,digest=hashlib.md5(data).hexdigest(),source='synthetic_verified_map')
    return NS(map_loaded=True,hdmap=api,read_map_document=lambda:(data,dict(metadata))),metadata


def junction_document(contact='start'):
    # Three collinear SDK lanes at x=0..20, 20..40, 40..60.
    incoming = road('1',20,section(),links='<successor elementType="junction" elementId="9"/>')
    outgoing = road('3',20,section(),links='<predecessor elementType="junction" elementId="9"/>')
    connector_lanes = '<laneSection s="0"><center>%s</center><right>%s</right></laneSection>'%(
        lane(0,mark(kind='none')),lane(-1,mark(kind='none'),'<predecessor id="-1"/><successor id="-1"/>'))
    connector_links = '<predecessor elementType="road" elementId="1" contactPoint="end"/><successor elementType="road" elementId="3" contactPoint="start"/>'
    connector = road('2',20,connector_lanes,'9',connector_links)
    j = '<junction id="9"><connection id="0" incomingRoad="1" connectingRoad="2" contactPoint="%s"><laneLink from="-1" to="-1"/></connection><priority high="1" low="3"/></junction>'%contact
    return document(incoming+connector+outgoing,j)


class MapDocumentTests(unittest.TestCase):
    def load(self,data=None):
        data = document() if data is None else data
        service = IdentityService(data)
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory,'selected.xodr')
            with open(filename,'wb') as stream: stream.write(data)
            value = MapDocument()
            result = value.load(service,STRUCTS,'3.0.0001',directory,filename,False)
        return value,service,result

    def test_exact_digest_and_native_abi_expose_only_small_metadata(self):
        value,service,result = self.load()
        self.assertTrue(result['verified'])
        self.assertEqual(document(),value.data)
        self.assertNotIn('url',result)
        self.assertNotIn('path',result)
        self.assertGreaterEqual(service.reads,2)

    def test_wrong_digest_explicit_file_does_not_use_other_document(self):
        data = document(); service = IdentityService(document(road(length=121)))
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory,'wrong.xodr')
            with open(filename,'wb') as stream: stream.write(data)
            value = MapDocument()
            result = value.load(service,STRUCTS,'3.0.0001',directory,filename,False)
        self.assertFalse(result['verified'])
        self.assertIsNone(value.data)

    def test_unverified_version_or_struct_never_calls_native(self):
        service = IdentityService(document())
        for structs,version in ((STRUCTS,'3.1'),(NS(SimOne_Data_Map=lambda:None),'3.0.0001')):
            result = MapDocument().load(service,structs,version,'.',allow_download=False)
            self.assertFalse(result['verified'])
        self.assertEqual(0,service.reads)

    def test_identity_changed_withdraws_bytes_until_explicit_reload(self):
        value,service,unused = self.load()
        service.identifier = b'other'
        self.assertFalse(value.verified(service,STRUCTS,'3.0.0001'))
        service.identifier = b'map'
        self.assertFalse(value.verified(service,STRUCTS,'3.0.0001'))
        self.assertIsNone(value.data)

    def test_transient_identity_failure_withdraws_authority_and_recovers_exact_map(self):
        value,service,unused = self.load()
        service.ok = False
        self.assertFalse(value.verified(service,STRUCTS,'3.0.0001'))
        self.assertFalse(value.metadata()['verified'])
        service.ok = True
        self.assertTrue(value.verified(service,STRUCTS,'3.0.0001'))

    def test_map_switch_during_hdmap_load_is_not_accepted(self):
        value = MapDocument(); service = IdentityService(document())
        result = value.load(service,STRUCTS,'3.0.0001','.',allow_download=False,loaded_identity=('other','0'*32))
        self.assertFalse(result['verified'])
        self.assertEqual('SDK_MAP_CHANGED_DURING_LOAD',result['reason'])

    def test_dtd_and_invalid_encoding_rejected_even_with_matching_digest(self):
        for data in (b'<!DOCTYPE OpenDRIVE [<!ENTITY x "bad">]><OpenDRIVE/>',b'\xff\xfe<xml>'):
            self.assertFalse(self.load(data)[2]['verified'])

    def test_file_budget_checked_before_read(self):
        service = IdentityService(document())
        with patch('simone_platform.map_document.os.path.getsize',return_value=MAX_DOCUMENT_BYTES+1), patch('builtins.open') as opened:
            result = MapDocument().load(service,STRUCTS,'3.0.0001','.',explicit_file='x',allow_download=False)
            opened.assert_not_called()
        self.assertFalse(result['verified'])

    def test_adapter_does_not_download_in_per_frame_read_or_pair_new_map_with_old_sdk(self):
        value,service,unused = self.load()
        adapter = SimOneAdapter(NS(),NS())
        adapter.map_loaded,adapter._map_document,adapter.service_api,adapter.structs,adapter.sdk_version = True,value,service,STRUCTS,'3.0.0001'
        with patch.object(value,'load',side_effect=AssertionError('per-frame download')):
            self.assertEqual(document(),adapter.read_map_document()[0])
            service.digest = b'0'*32
            self.assertIsNone(adapter.read_map_document()[0])
        self.assertFalse(adapter.read_map_document()[1]['verified'])

    def test_config_path_resolves_from_project(self):
        config = AppConfig(os.getcwd(),dict(map_document_file='maps/example.xodr'))
        self.assertEqual(os.path.join(os.getcwd(),'maps','example.xodr'),config.map_document_file)

    def test_sdk_url_download_is_bounded_digest_checked_and_private(self):
        data=document(); service=IdentityService(data)
        service.url=b'https://map.invalid/document?test_token=private_fixture'
        response=io.BytesIO(data); response.headers={'Content-Length':str(len(data))}
        with tempfile.TemporaryDirectory() as directory, patch('simone_platform.map_document.urllib.request.build_opener',return_value=NS(open=lambda *a,**kw:response)):
            value=MapDocument(); result=value.load(service,STRUCTS,'3.0.0001',directory)
        self.assertTrue(result['verified'])
        self.assertEqual('verified_sdk_url',result['source'])
        self.assertNotIn('private_fixture',repr(result))

    def test_sdk_url_wrong_digest_encoding_and_non_http_are_rejected(self):
        for headers,data in (({},b'wrong'),({'Content-Encoding':'gzip'},document()),({'Content-Length':str(MAX_DOCUMENT_BYTES+1)},document())):
            service=IdentityService(document()); service.url=b'https://map.invalid/document'
            response=io.BytesIO(data); response.headers=headers
            with tempfile.TemporaryDirectory() as directory, patch('simone_platform.map_document.urllib.request.build_opener',return_value=NS(open=lambda *a,**kw:response)):
                value=MapDocument(); result=value.load(service,STRUCTS,'3.0.0001',directory)
            self.assertFalse(result['verified'])
            self.assertIsNone(value.data)
        service=IdentityService(document()); service.url=b'file:///private/document'
        with tempfile.TemporaryDirectory() as directory, patch('simone_platform.map_document.urllib.request.build_opener') as opened:
            self.assertFalse(MapDocument().load(service,STRUCTS,'3.0.0001',directory)['verified'])
            opened.assert_not_called()

    def test_adapter_load_checks_identity_before_and_after_native_map_loading(self):
        data=document(); service=IdentityService(data)
        with tempfile.TemporaryDirectory() as directory:
            filename=os.path.join(directory,'map.xodr')
            with open(filename,'wb') as stream: stream.write(data)
            config=NS(sdk_dir=directory,map_document_file=filename,map_timeout_sec=1)
            adapter=SimOneAdapter(config,NS(info=lambda *a:None,warning=lambda *a:None))
            adapter.structs,adapter.service_api,adapter.sdk_version=STRUCTS,service,'3.0.0001'
            adapter.hdmap=NS(loadHDMap=lambda timeout:True)
            self.assertTrue(adapter.load_hdmap())
            self.assertTrue(adapter.read_map_document()[1]['verified'])
            def switched(timeout):
                service.identifier=b'new'
                return True
            adapter.hdmap.loadHDMap=switched
            self.assertTrue(adapter.load_hdmap())  # Optional semantics do not kill old pipeline.
            self.assertFalse(adapter.read_map_document()[1]['verified'])

    def test_optional_native_exception_withdraws_evidence_without_exposing_exception_text(self):
        def failed(value):
            raise RuntimeError('https://map.invalid/private_fixture')
        service=NS(SoGetHDMapData=failed)
        self.assertIsNone(MapDocument.selected_identity(service,STRUCTS,'3.0.0001'))
        result=MapDocument().load(service,STRUCTS,'3.0.0001','.',allow_download=False)
        self.assertFalse(result['verified'])
        self.assertNotIn('private_fixture',repr(result))


class OpenDriveSemanticTests(unittest.TestCase):
    def test_marking_offset_is_section_relative_and_ends_at_next_record(self):
        xml = document(road(length=120,sections=section(0)+section(50,mark(5)+mark(35,'solid','none'))))
        value = OpenDriveSemantics(xml).shared_markings('1_1_-1','1_1_-2')
        self.assertEqual([(50.,55.),(55.,85.),(85.,120.)],[(v['road_s_start_m'],v['road_s_end_m']) for v in value])
        self.assertEqual([False,True,False],[v['crossing_permitted'] for v in value])

    def test_inner_lane_owns_shared_border_in_both_crossing_directions(self):
        model = OpenDriveSemantics(document(road(sections=section(0,mark(change='increase')))))
        self.assertFalse(model.shared_markings('1_0_-1','1_0_-2')[0]['crossing_permitted'])
        value = model.shared_markings('1_0_-2','1_0_-1')[0]
        self.assertTrue(value['crossing_permitted'])
        self.assertEqual(-1,value['owner_lane_id'])

    def test_missing_lane_change_attribute_has_standard_both_default(self):
        model = OpenDriveSemantics(document(road(sections=section(0,'<roadMark sOffset="0" type="broken"/>'))))
        self.assertTrue(model.shared_markings('1_0_-1','1_0_-2')[0]['crossing_permitted'])

    def test_detailed_explicit_sway_and_unknown_marks_are_not_authority(self):
        for child in ('<type name="x"><line length="3"/></type>','<explicit/>','<sway a="1"/>'):
            model = OpenDriveSemantics(document(road(sections=section(0,mark(children=child)))))
            self.assertFalse(model.shared_markings('1_0_-1','1_0_-2')[0]['crossing_permitted'])
        model = OpenDriveSemantics(document(road(sections=section(0,mark(kind='broken solid')))))
        self.assertFalse(model.shared_markings('1_0_-1','1_0_-2')[0]['crossing_permitted'])

    def test_invalid_order_is_local_failure_and_duplicate_identity_is_fatal(self):
        bad = road('2',sections=section(0,mark(20)+mark(10)))
        model = OpenDriveSemantics(document(road()+bad))
        self.assertIn('1',model.roads)
        self.assertIn('2',model.road_errors)
        with self.assertRaises(ValueError): OpenDriveSemantics(document(road()+road()))

    def test_missing_marking_and_opposite_lane_do_not_allow_crossing(self):
        model = OpenDriveSemantics(document(road(sections=section(0,''))))
        self.assertFalse(model.shared_markings('1_0_-1','1_0_-2')[0]['crossing_permitted'])
        for target in ('1_0_1','1_0_-3','1_1_-2','legacy'):
            with self.assertRaises(ValueError): model.shared_markings('1_0_-1',target)

    def test_access_or_dynamic_direction_is_unknown_permission(self):
        for fragment in ('dynamicLaneDirection="true"','direction="both"'):
            data = document().replace(b'id="-2" type="driving" ',('id="-2" type="driving" '+fragment+' ').encode('ascii'))
            self.assertFalse(OpenDriveSemantics(data).shared_markings('1_0_-1','1_0_-2')[0]['crossing_permitted'])
        data = document().replace(b'<lane id="-2"',b'<lane id="-2"').replace(b'<link></link><roadMark sOffset="0" type="solid" laneChange="both"></roadMark></lane></right>',b'<link></link><access restriction="bus"/></lane></right>')
        self.assertFalse(OpenDriveSemantics(data).shared_markings('1_0_-1','1_0_-2')[0]['crossing_permitted'])

    def test_xml_root_version_entities_depth_and_malformed_are_rejected(self):
        for data in (b'<OpenDRIVE>',document().replace(b'revMinor="4"',b'revMinor="9"'),
                     b'<!DOCTYPE x><OpenDRIVE/>',b'<x/>',HEADER.encode('ascii')+b'<x>'*65+b'</x>'*65+b'</OpenDRIVE>'):
            with self.assertRaises(ValueError): OpenDriveSemantics(data)

    def test_junction_links_and_priority_are_declared_not_clearance(self):
        value = OpenDriveSemantics(junction_document()).junctions['9']
        self.assertTrue(value['topology_complete'])
        self.assertEqual([dict(high_road_id='1',low_road_id='3')],value['priority_pairs'])
        self.assertEqual('3',value['connections'][0]['outgoing_road']['element_id'])

    def test_missing_or_mismatched_junction_lane_links_are_incomplete(self):
        for data in (junction_document().replace(b'<laneLink from="-1" to="-1"/>',b''),
                     junction_document().replace(b'to="-1"',b'to="-2"'),junction_document('end')):
            self.assertFalse(OpenDriveSemantics(data).junctions['9']['topology_complete'])

    def test_unrepresented_connector_lane_and_missing_outgoing_link_are_incomplete(self):
        extra=lane(-2,mark(kind='none'),'<predecessor id="-2"/><successor id="-2"/>').encode('ascii')
        data=junction_document().replace(b'</right></laneSection></lanes></road><road id="3"',extra+b'</right></laneSection></lanes></road><road id="3"')
        self.assertFalse(OpenDriveSemantics(data).junctions['9']['topology_complete'])
        data=junction_document().replace(b'<successor elementType="road" elementId="3" contactPoint="start"/>',b'')
        self.assertFalse(OpenDriveSemantics(data).junctions['9']['topology_complete'])

    def test_center_lane_is_never_a_junction_driving_exit(self):
        data=junction_document().replace(b'<successor id="-1"/>',b'<successor id="0"/>')
        self.assertFalse(OpenDriveSemantics(data).junctions['9']['topology_complete'])


class ManeuverMapTests(unittest.TestCase):
    def neighbors(self,adapter,p):
        route = manager(adapter.hdmap)
        p.lane = route.update(p.ego,True)
        return route.read_neighbor_lanes(p.ego,p.lane)

    def test_native_road_s_and_marking_agreement_publish_semantics_without_fake_world_range(self):
        adapter,unused = map_adapter(); p = perception(); neighbors = self.neighbors(adapter,p)
        meta,junctions = ManeuverMap(adapter).observe(p.ego,p.lane,neighbors)
        self.assertTrue(meta['geometry_bound'])
        item = neighbors[0]
        self.assertTrue(item['marking_semantics_verified'])
        self.assertTrue(item['crossing_allowed_at_ego'])
        self.assertFalse(item['crossing_range_verified'])
        self.assertFalse(item['crossing_geometry_bound'])
        self.assertEqual(120.,item['marking_intervals_road_s'][0]['road_s_end_m'])

    def test_pointwise_broken_sdk_does_not_override_xml_solid_or_lane_change_none(self):
        for marks in (mark(kind='solid'),mark(change='none')):
            adapter,unused = map_adapter(document(road(sections=section(0,marks))))
            p=perception(); neighbors=self.neighbors(adapter,p)
            ManeuverMap(adapter).observe(p.ego,p.lane,neighbors)
            self.assertFalse(neighbors[0]['crossing_allowed_at_ego'])

    def test_native_offset_must_match_section_relative_record_not_lane_length(self):
        adapter,unused=map_adapter(); p=perception(); neighbors=self.neighbors(adapter,p)
        neighbors[0]['marking_observation']['section_s_offset_m']=3.
        ManeuverMap(adapter).observe(p.ego,p.lane,neighbors)
        self.assertFalse(neighbors[0]['marking_semantics_verified'])
        self.assertEqual('SDK_XML_MARKING_MISMATCH',neighbors[0]['crossing_scope_reason'])

    def test_unknown_native_neighbor_type_does_not_become_driving_from_xml(self):
        adapter,unused=map_adapter(); adapter.hdmap.kind='unknown'
        p=perception(); neighbors=self.neighbors(adapter,p)
        ManeuverMap(adapter).observe(p.ego,p.lane,neighbors)
        self.assertFalse(neighbors[0]['crossing_allowed_at_ego'])

    def test_source_failure_withdraws_old_semantic_intervals_and_recovers(self):
        adapter,metadata = map_adapter(); p=perception(); neighbors=self.neighbors(adapter,p)
        model=ManeuverMap(adapter)
        model.observe(p.ego,p.lane,neighbors)
        metadata['verified']=False
        meta,junctions=model.observe(p.ego,p.lane,neighbors)
        self.assertFalse(meta['verified'])
        self.assertEqual([],neighbors[0]['marking_intervals_road_s'])
        self.assertNotIn('marking_map_digest',neighbors[0])
        metadata['verified']=True
        self.assertTrue(model.observe(p.ego,p.lane,neighbors)[0]['geometry_bound'])

    def test_bad_road_s_does_not_publish_crossing_intervals(self):
        for s in (float('nan'),-1.,121.):
            adapter,unused=map_adapter(); adapter.hdmap.getRoadST=lambda *args:NS(exists=True,s=s)
            p=perception(); neighbors=self.neighbors(adapter,p)
            meta,unused=ManeuverMap(adapter).observe(p.ego,p.lane,neighbors)
            self.assertFalse(meta['geometry_bound'])
            self.assertEqual([],neighbors[0]['marking_intervals_road_s'])

    def test_real_builder_and_serialization_publish_static_channel(self):
        adapter,unused=map_adapter(); p=perception(); route=manager(adapter.hdmap)
        route.adapter=adapter; route._maneuver_map=ManeuverMap(adapter)
        p.lane=route.update(p.ego,True)
        env=ManeuverEnvironmentBuilder(route,clock=lambda:100.).build(p)
        self.assertTrue(env.map_semantics['semantic_verified'])
        p.maneuver_environment=env
        self.assertEqual(env.map_semantics,perception_to_dict(p)['maneuver_environment']['map_semantics'])

    def junction_case(self):
        adapter,unused=map_adapter(junction_document()); p=perception()
        p.lane.lane_id='1_0_-1'; p.lane.center_line=[(0.,0.,0.),(20.,0.,0.)]
        p.lane.forward_reference=[(0.,0.,0.),(20.,0.,0.),(40.,0.,0.),(60.,0.,0.)]
        p.lane.forward_lane_ids=['1_0_-1','2_0_-1','3_0_-1']
        p.lane.forward_lane_spans=[dict(lane_id=v,start_index=i,end_index=i+1) for i,v in enumerate(p.lane.forward_lane_ids)]
        p.lane.forward_reference_valid=True
        def st(native,point):
            return NS(exists=True,s=point[0]-{'1':0.,'2':20.,'3':40.}[native.GetString()])
        adapter.hdmap.getRoadST=st
        return adapter,p

    def test_route_selected_junction_has_verified_sdk_gates_but_unknown_occupancy_and_priority(self):
        adapter,p=self.junction_case()
        meta,values=ManeuverMap(adapter).observe(p.ego,p.lane,[])
        self.assertEqual(1,len(values)); j=values[0]
        self.assertTrue(j['entry_gate_verified'] and j['exit_gate_verified'])
        self.assertTrue(j['selected_route_chain_verified'])
        self.assertEqual(10.,j['entry_distance_m'])
        self.assertEqual(30.,j['exit_distance_m'])
        self.assertEqual('3_0_-1',j['exit_lane_id'])
        self.assertEqual('UNKNOWN',j['right_of_way'])
        self.assertFalse(j['rule_verified'] or j['exit_coverage_verified'] or j['conflict_coverage_complete'])

    def test_route_gate_requires_native_endpoint_and_correct_exit_lane_link(self):
        adapter,p=self.junction_case()
        adapter.hdmap.getRoadST=lambda *args:NS(exists=True,s=10.)
        j=ManeuverMap(adapter).observe(p.ego,p.lane,[])[1][0]
        self.assertFalse(j['entry_gate_verified'] or j['exit_gate_verified'])
        adapter,p=self.junction_case(); p.lane.forward_lane_ids[-1]='3_0_-2'
        p.lane.forward_lane_spans[-1]['lane_id']='3_0_-2'
        self.assertFalse(ManeuverMap(adapter).observe(p.ego,p.lane,[])[1][0]['exit_gate_verified'])

    def test_unverified_or_missing_route_spans_cannot_create_junction_gates(self):
        adapter,p=self.junction_case(); p.lane.forward_lane_spans=[]
        self.assertEqual([],ManeuverMap(adapter).observe(p.ego,p.lane,[])[1])

    def test_missing_exit_lane_semantics_preserves_junction_with_unknown_exit(self):
        adapter,p=self.junction_case(); p.lane.forward_lane_ids[-1]='3_0_-3'
        p.lane.forward_lane_spans[-1]['lane_id']='3_0_-3'
        j=ManeuverMap(adapter).observe(p.ego,p.lane,[])[1][0]
        self.assertTrue(j['entry_gate_verified'])
        self.assertFalse(j['exit_gate_verified'])


if __name__=='__main__':
    unittest.main()
