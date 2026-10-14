"""Small exact-frame GPS callback history; never interpolate Sensor poses."""
import copy
import ctypes
import math
import threading
import time
from collections import OrderedDict


class GpsPoseStream(object):
    def __init__(self, vehicle_id, timeout_s=.5, clock=None):
        self.vehicle_id, self.timeout_s = str(vehicle_id), timeout_s
        self.clock = clock or time.monotonic
        self.lock, self.samples = threading.Lock(), OrderedDict()
        self.scope, self.installations = None, {}
        self.high_frame, self.high_timestamp = -1, -1
        self.callback, self.kind = None, None
        self.accepting, self.status = True, 'not_registered'

    def set_scope(self, case_id, task_id):
        scope = (case_id,task_id,self.vehicle_id)
        with self.lock:
            if scope != self.scope:
                self.samples.clear()
                self.high_frame, self.high_timestamp = -1, -1
                self.scope = scope if case_id and task_id else None

    def set_installations(self, configurations, valid):
        fields = ('type','x','y','z','roll','pitch','yaw','hz')
        installs = {}
        if valid and isinstance(configurations,list) and len(configurations)<=128:
            for item in configurations:
                if (not isinstance(item,dict) or not all(k in item for k in ('id',)+fields)
                        or not isinstance(item['id'],str) or not item['id'] or item['id'] in installs
                        or not all(type(item[k]) in (int,float) and math.isfinite(item[k]) for k in fields[1:])):
                    installs = {}
                    break
                installs[item['id']] = {k:item[k] for k in fields}
        with self.lock:
            if installs != self.installations:
                self.samples.clear()
                self.installations = copy.deepcopy(installs)

    def register(self, native, kind, version):
        # A new SDK initialization starts a new acquisition epoch.
        with self.lock:
            self.accepting = True
            self.samples.clear()
            self.high_frame, self.high_timestamp = -1, -1
        if native is None or version != '3.0.0001':
            self.status = 'unsupported'
            self.accepting = False
            return
        try:
            if (ctypes.sizeof(kind) != 1193 or kind.posX.offset != 16
                    or kind.frame.offset != 8 or kind.oriZ.offset != 36
                    or kind.isGPSLost.offset != 1144):
                raise ValueError('unverified GPS callback ABI')
            self.kind = kind
            callback_type = ctypes.CFUNCTYPE(None,ctypes.c_char_p,ctypes.POINTER(kind))
            self.callback = callback_type(self.receive)
            register = native.SetGpsUpdateCB
            register.argtypes, register.restype = [callback_type], ctypes.c_bool
            self.status = 'registered' if register(self.callback) else 'registration_failed'
        except (ValueError,TypeError,AttributeError,OSError) as error:
            self.status = 'unavailable:'+str(error)
        if self.status != 'registered':
            self.accepting = False

    def receive(self, vehicle_id, pointer):
        try:
            identity = vehicle_id.decode('utf-8') if isinstance(vehicle_id,bytes) else str(vehicle_id)
            if identity != self.vehicle_id or not pointer:
                return
            gps = pointer.contents
            if gps.isGPSLost:
                return
            pose = dict(x=float(gps.posX),y=float(gps.posY),z=float(gps.posZ),
                        roll=float(gps.oriX),pitch=float(gps.oriY),heading=float(gps.oriZ))
            frame,timestamp = int(gps.frame),int(gps.timestamp)
            now = self.clock()
            if frame < 0 or timestamp <= 0 or not all(math.isfinite(v) for v in pose.values()):
                return
            with self.lock:
                if not self.accepting or self.scope is None or not self.installations:
                    return
                if (frame < self.high_frame or timestamp < self.high_timestamp
                        or frame > self.high_frame and timestamp <= self.high_timestamp):
                    self.samples.clear()
                    return  # A rollback cannot renew an old packet's lifetime.
                if frame == self.high_frame and frame not in self.samples:
                    return
                previous = self.samples.get(frame)
                record = dict(frame_id=frame,timestamp=timestamp,pose=pose,scope=self.scope,
                    installations=self.installations,observed_at_s=now,valid_until_s=now+self.timeout_s,
                    source='sdk_gps_callback',consistent=True)
                if previous is not None:
                    previous['consistent'] = (previous['consistent'] and previous['pose']==pose
                                               and previous['timestamp']==timestamp)
                else:
                    self.samples[frame] = record
                    self.high_frame, self.high_timestamp = frame,timestamp
                while len(self.samples)>128:
                    self.samples.popitem(last=False)
        except (ValueError,TypeError,AttributeError,OverflowError,UnicodeError):
            return

    def select(self, frame, timestamp):
        with self.lock:
            record = self.samples.get(frame)
            if (record is None or record['timestamp'] != timestamp or not record['consistent']
                    or self.clock() >= record['valid_until_s'] or not self.accepting):
                return None
            return copy.deepcopy(record)

    def close(self):
        with self.lock:
            self.accepting = False
            self.samples.clear()
