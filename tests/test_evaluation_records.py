"""SDK-shaped append output must survive the real task-JSON packaging contract."""
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simone_platform.evaluation import LocalEvaluation
from simone_platform.evaluation_records import EvaluationRecords,evaluation_directory
from tests.test_evaluation import NativeCall,adapter


TASK = '11111111-2222-3333-4444-555555555555'
OTHER_TASK = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'


def record(frame,category='mainvehicle',timestamp=None):
    return dict(category=category,timestamp=str(frame*10 if timestamp is None else timestamp),
                data=dict(frame=frame,pos=[frame,0.,0.],steering=0.,brake=0.))


class EvaluationRecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = os.path.join(self.directory.name,'evaluation')
        os.mkdir(self.root)
        self.backup = os.path.join(self.directory.name,'raw')
        self.delivery = EvaluationRecords(self.root,self.backup,logging.getLogger('evaluation-record-test'))
        self.addCleanup(self.delivery.close)
        self.delivery.bind_task(TASK)

    def path(self,suffix='.json',task=TASK):
        return os.path.join(self.root,task+suffix)

    def append(self,document,suffix='.json',task=TASK):
        with open(self.path(suffix,task),'a',encoding='utf-8') as stream:
            json.dump(document,stream,separators=(',',':'))
            stream.write('\n')

    def publish(self,generation=1):
        self.assertTrue(self.delivery.begin_save(True))
        try:return self.delivery.saved(generation,True)
        finally:self.delivery.end_save()

    def read(self):
        # Same strict single-JSON operation used by installed TaskServer.
        with open(self.path(),encoding='utf-8') as stream:return json.load(stream)

    def test_append_documents_and_separate_gps_become_one_unique_chronological_task(self):
        first,second = record(1),record(2)
        judge = record(9,'judge',15)
        self.append(dict(endreason=0))
        self.append(dict(records=[first]),'_mainvehicle.json')
        self.assertTrue(self.publish())
        self.append(dict(records=[judge],endreason=1))
        self.append(dict(records=[first,second]),'_mainvehicle.json')
        with open(self.path('_mainvehicle.json'),'rb') as stream:original_gps=stream.read()
        self.assertTrue(self.publish(2))
        data = self.read()
        self.assertEqual([first,judge,second],data['records'])
        self.assertEqual(3,data['total'])
        self.assertEqual(1,data['endreason'])
        self.assertNotIn('score',data)
        with open(self.path('_mainvehicle.json'),'rb') as stream:self.assertEqual(original_gps,stream.read())
        with open(os.path.join(self.backup,TASK,'sdk-judge-000002.jsonl'),encoding='utf-8') as stream:
            self.assertEqual(dict(records=[judge],endreason=1),json.load(stream))
        self.assertEqual(dict(mainvehicle=2,judge=1),self.delivery.status()['categories'])

    def test_cumulative_repeats_do_not_drop_different_events_at_the_same_timestamp(self):
        first,changed = record(1),record(1)
        changed['data']['brake'] = .5
        self.append(dict(records=[first,changed]))
        self.append(dict(records=[first,changed]))
        self.assertTrue(self.publish())
        self.assertEqual([first,changed],self.read()['records'])

    def test_rewritten_or_truncated_sdk_source_is_reconsumed_without_duplicate_records(self):
        self.append(dict(records=[record(1)]),'_mainvehicle.json')
        self.assertTrue(self.publish())
        with open(self.path('_mainvehicle.json'),'w',encoding='utf-8') as stream:
            json.dump(dict(records=[record(2)]),stream)
        self.assertTrue(self.publish(2))
        self.assertEqual([record(1),record(2)],self.read()['records'])

    def test_identical_judge_events_within_one_snapshot_keep_their_observed_multiplicity(self):
        event = record(9,'judge')
        self.append(dict(records=[event,event]))
        self.append(dict(records=[event,event]))
        self.assertTrue(self.publish())
        self.assertEqual([event,event],self.read()['records'])

    def test_partial_or_nonfinite_data_never_claims_success_or_overwrites_raw_source(self):
        for raw in ('{"records":[', '{"records":[{"category":"mainvehicle","timestamp":"10","data":NaN}]}'):
            with open(self.path(),'w',encoding='utf-8') as stream:stream.write(raw)
            self.assertFalse(self.publish())
            with open(self.path(),encoding='utf-8') as stream:self.assertEqual(raw,stream.read())
            self.assertFalse(self.delivery.status()['packaging_ready'])

    def test_empty_records_and_new_task_cannot_borrow_previous_task_evidence(self):
        self.append(dict(records=[record(1)]),'_mainvehicle.json')
        self.assertTrue(self.publish())
        self.delivery.bind_task(OTHER_TASK)
        self.assertFalse(self.publish(2))
        self.assertFalse(os.path.exists(self.path(task=OTHER_TASK)))
        self.assertEqual(0,self.delivery.status()['record_count'])
        self.assertEqual([record(1)],self.read()['records'])

    def test_background_delivery_never_blocks_regular_native_save_and_final_waits_for_delivery(self):
        calls = []
        def save():
            calls.append(True)
            self.append(dict(endreason=0))
            self.append(dict(records=[record(v) for v in range(1,len(calls)+1)]),'_mainvehicle.json')
            return True
        library = SimpleNamespace(InitEvaluationServiceWithLocalData=NativeCall(),SaveEvaluationRecord=NativeCall(callback=save))
        evaluation = LocalEvaluation(library,'3.0.0001',logging.getLogger(),records=self.delivery)
        entered,release = threading.Event(),threading.Event()
        original = self.delivery._consume
        def slow(*args):
            if not entered.is_set():
                entered.set();release.wait(2.)
            return original(*args)
        self.addCleanup(release.set)
        with patch.object(self.delivery,'_consume',side_effect=slow),patch('simone_platform.evaluation.time.monotonic',return_value=10.):
            evaluation.initialize('0')
            evaluation._next_save = 0.
            self.assertTrue(evaluation.flush())
            self.assertTrue(entered.wait(1.))
            evaluation._next_save = 0.
            start = time.perf_counter()
            self.assertTrue(evaluation.flush())
            self.assertLess(time.perf_counter()-start,.05)
            self.assertEqual(1,len(calls))
            release.set()
            self.assertTrue(evaluation.flush(force=True))
        self.assertEqual(2,evaluation.save_count)
        self.assertEqual([record(1),record(2)],self.read()['records'])
        self.assertTrue(self.delivery.status()['packaging_ready'])

    def test_adapter_delivers_records_before_native_termination(self):
        def save():
            self.append(dict(endreason=1))
            self.append(dict(records=[record(1)]),'_mainvehicle.json')
            return True
        library = SimpleNamespace(InitEvaluationServiceWithLocalData=NativeCall(),SaveEvaluationRecord=NativeCall(callback=save))
        platform = adapter(library)
        platform._evaluation = LocalEvaluation(library,'3.0.0001',logging.getLogger(),records=self.delivery)
        observed = []
        platform.service_api.SoTerminateSimOneAPI.callback = lambda:observed.append(self.read())
        platform.initialize()
        platform.shutdown()
        self.assertEqual([record(1)],observed[0]['records'])
        self.assertEqual(1,observed[0]['endreason'])

    def test_final_delivery_failure_cannot_report_sdk_save_as_complete_success(self):
        library = SimpleNamespace(InitEvaluationServiceWithLocalData=NativeCall(),SaveEvaluationRecord=NativeCall())
        evaluation = LocalEvaluation(library,'3.0.0001',logging.getLogger(),records=self.delivery)
        evaluation.initialize('0')
        self.assertFalse(evaluation.flush(force=True))
        self.assertFalse(evaluation.status()['last_save_ok'])
        self.assertEqual(1,evaluation.save_count)
        self.assertEqual('SDK_records_not_available',evaluation.status()['error'])

    def test_directory_resolves_installed_user_data_without_fixed_machine_paths(self):
        platform = os.path.join(self.directory.name,'platform')
        config_dir = os.path.join(platform,'Common','Conf')
        os.makedirs(config_dir)
        user = os.path.join(self.directory.name,'user')
        with open(os.path.join(config_dir,'simone.cfg'),'w') as stream:json.dump(dict(user_data=user),stream)
        config = SimpleNamespace(sdk_dir=os.path.join(platform,'SimOneAPI','lib','Win64'),evaluation_record_dir='')
        self.assertEqual(os.path.join(user,'evaluation'),evaluation_directory(config))
        config.evaluation_record_dir = self.root
        self.assertEqual(self.root,evaluation_directory(config))


if __name__=='__main__':unittest.main()
