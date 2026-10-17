"""Deliver authentic SDK records as one task JSON accepted by TaskServer.

The SDK 3.0.0001 save appends JSON documents and writes GPS separately.
Only SDK-produced records are merged; no score or Judge event is invented.
"""
import hashlib
import json
import math
import os
import re
import threading

from core.diagnostic_writer import DiagnosticWriter


TASK_ID = re.compile(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
MAX_DELTA_BYTES = 256*1024*1024
MAX_RECORDS = 200000


def evaluation_directory(config):
    explicit = getattr(config,'evaluation_record_dir','')
    if explicit:
        return os.path.abspath(explicit)
    platform = os.path.abspath(os.path.join(config.sdk_dir,'..','..','..'))
    root = os.path.join(os.path.expanduser('~'),'.simone')
    for relative in (os.path.join('Common','Conf','simone.cfg'),):
        filename = os.path.join(platform,relative)
        if os.path.isfile(filename):
            with open(filename,'r',encoding='utf-8-sig') as stream:
                data = json.loads(stream.read(1048577))
            if isinstance(data.get('user_data'),str) and data['user_data']:
                root = os.path.abspath(data['user_data'])
    directory = os.path.join(root,'evaluation')
    filename = os.path.join(platform,'Web','TaskServer','config','app_config.json')
    if os.path.isfile(filename):
        with open(filename,'r',encoding='utf-8-sig') as stream:
            data = json.loads(stream.read(1048577))
        override = data.get('dir',{}).get('evaluation')
        if isinstance(override,str) and override:
            directory = os.path.abspath(override)
    return directory


class EvaluationRecords(object):
    def __init__(self,directory,backup_dir,logger):
        self.directory,self.backup_dir,self.logger = directory,backup_dir,logger
        self.gate = threading.Lock()
        self.task_id = None
        self.records,self.cursors,self.metadata,self.multiplicity = {},{},{},{}
        self.saved_generation = self.published_generation = 0
        self.error,self.ready = 'task_scope_unavailable',False
        self.busy_skips = 0
        self.writer = DiagnosticWriter(self._work,capacity=2)

    def bind_task(self,task_id):
        task_id = task_id if isinstance(task_id,str) and TASK_ID.fullmatch(task_id) else None
        if task_id == self.task_id:
            return
        with self.gate:
            self.task_id = task_id
            self.records,self.cursors,self.metadata,self.multiplicity = {},{},{},{}
            self.saved_generation = self.published_generation = 0
            self.error,self.ready = 'task_scope_unavailable' if task_id is None else '',False

    def begin_save(self,force):
        acquired = self.gate.acquire(timeout=3.) if force else self.gate.acquire(False)
        if not acquired:
            self.busy_skips += 1
            if force:self.error = 'final_record_delivery_busy'
        return acquired

    def saved(self,generation,force):
        # Caller holds gate across the native save. Normalization never races
        # another save; regular saves skip a busy worker instead of waiting.
        self.saved_generation = generation
        self.ready = False
        return self._publish_locked(generation) if force else True

    def end_save(self):
        self.gate.release()

    def request(self):
        if self.task_id is not None:
            self.writer.submit(self.task_id,{'generation':self.saved_generation})

    def _work(self,task_id,unused):
        with self.gate:
            if task_id != self.task_id:
                return True
            return self._publish_locked(self.saved_generation)

    @staticmethod
    def _identity(stat):
        return stat.st_dev,stat.st_ino

    def _consume(self,filename,primary,generation,reset=False):
        if not os.path.isfile(filename):
            return
        stat = os.stat(filename)
        previous = self.cursors.get(filename)
        offset = (previous[1] if not reset and previous is not None and previous[0]==self._identity(stat)
                  and stat.st_size>=previous[1] else 0)
        with open(filename,'rb') as stream:
            stream.seek(offset)
            raw = stream.read(MAX_DELTA_BYTES+1)
        if len(raw)>MAX_DELTA_BYTES:
            raise ValueError('SDK evaluation delta exceeds bound')
        if not raw.strip():
            return
        # Keep the original primary documents before replacing the public task
        # JSON. The SDK's mainvehicle file itself remains untouched.
        if primary and self.backup_dir:
            folder = os.path.join(self.backup_dir,self.task_id)
            os.makedirs(folder,exist_ok=True)
            backup = os.path.join(folder,'sdk-judge-{0:06d}{1}.jsonl'.format(generation,'.full' if reset else ''))
            if not os.path.exists(backup):
                with open(backup,'xb') as stream:stream.write(raw)
        try:
            self._documents(raw,primary)
        except (ValueError,UnicodeError):
            # A vendor rewrite can retain the same inode and grow past the
            # cursor. Re-read once; never discard already observed records.
            if offset:
                return self._consume(filename,primary,generation,reset=True)
            raise
        self.cursors[filename] = self._identity(stat),offset+len(raw)

    def _documents(self,raw,primary):
        text,index = raw.decode('utf-8-sig'),0
        def reject_constant(value):raise ValueError('nonfinite SDK evaluation value')
        decoder = json.JSONDecoder(parse_constant=reject_constant)
        while index<len(text):
            while index<len(text) and text[index].isspace():index += 1
            if index==len(text):break
            document,index = decoder.raw_decode(text,index)
            if not isinstance(document,dict):
                raise ValueError('SDK evaluation document must be an object')
            records = document.get('records',[])
            if not isinstance(records,list):
                raise ValueError('SDK evaluation records must be an array')
            occurrences = {}
            for record in records:
                if (not isinstance(record,dict) or not isinstance(record.get('category'),str)
                        or not record['category'] or 'timestamp' not in record
                        or type(record['timestamp']) not in (str,int,float)
                        or not math.isfinite(float(record['timestamp']))):
                    raise ValueError('SDK evaluation record provenance missing')
                value = json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
                key = hashlib.sha256(value.encode('utf-8')).hexdigest()
                occurrences[key] = occurrences.get(key,0)+1
                if occurrences[key]>self.multiplicity.get(key,0):
                    if len(self.records)>=MAX_RECORDS:
                        raise ValueError('SDK evaluation unique record bound exceeded')
                    self.records[key,occurrences[key]] = record
                    self.multiplicity[key] = occurrences[key]
            if primary and 'endreason' in document:
                self.metadata['endreason'] = document['endreason']

    def _publish_locked(self,generation):
        if self.task_id is None:
            self.error = 'task_scope_unavailable'
            return False
        if generation<=self.published_generation and self.ready:
            return True
        filename = os.path.join(self.directory,self.task_id+'.json')
        try:
            self._consume(filename,True,generation)
            self._consume(os.path.join(self.directory,self.task_id+'_mainvehicle.json'),False,generation)
            if not self.records:
                self.error,self.ready = 'SDK_records_not_available',False
                return False
            ordered = sorted(self.records.values(),key=lambda record:float(record['timestamp']))
            document = dict(self.metadata,records=ordered,total=len(ordered))
            temporary = filename+'.captain-{0}.tmp'.format(os.getpid())
            with open(temporary,'w',encoding='utf-8') as stream:
                json.dump(document,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            os.replace(temporary,filename)
            stat = os.stat(filename)
            self.cursors[filename] = self._identity(stat),stat.st_size
            self.published_generation,self.ready,self.error = generation,True,''
            self.logger.info('评分记录已交付 task=%s records=%s generation=%s',
                             self.task_id,len(self.records),generation)
            return True
        except (OSError,ValueError,TypeError,UnicodeError,RecursionError) as error:
            self.error,self.ready = 'record_delivery_failed:'+type(error).__name__+':'+str(error),False
            self.logger.warning('评分记录交付失败 task=%s reason=%s',self.task_id,self.error)
            return False

    def status(self):
        categories = {}
        for record in list(self.records.values()):
            key = record['category']; categories[key] = categories.get(key,0)+1
        return dict(task_id=self.task_id,directory=self.directory,packaging_ready=self.ready,
                    sdk_save_generation=self.saved_generation,published_generation=self.published_generation,
                    record_count=len(self.records),categories=categories,error=self.error,busy_skips=self.busy_skips)

    def close(self):
        return self.writer.close()
