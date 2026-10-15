"""Exercise the verified native callback transport without loading a DLL."""
import ctypes
import unittest
from types import SimpleNamespace as NS

from simone_platform.gps_pose_stream import GpsPoseStream


class Gps(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('timestamp',ctypes.c_int64),('frame',ctypes.c_int),('version',ctypes.c_int),
        ('posX',ctypes.c_float),('posY',ctypes.c_float),('posZ',ctypes.c_float),
        ('oriX',ctypes.c_float),('oriY',ctypes.c_float),('oriZ',ctypes.c_float),
        ('padding',ctypes.c_char*1104),('isGPSLost',ctypes.c_bool),('tail',ctypes.c_char*48)]


class Register(object):
    def __call__(self,callback):
        self.callback = callback
        return True


class GpsPoseStreamTests(unittest.TestCase):
    def fixture(self):
        self.now = 100.
        stream = GpsPoseStream('0',clock=lambda:self.now)
        stream.set_scope('case','task')
        stream.set_installations([dict(id='sensor',type='sensorFusion',x=1.,y=0.,z=1.8,
                                      roll=0.,pitch=0.,yaw=0.,hz=10.)],True)
        native = NS(SetGpsUpdateCB=Register())
        stream.register(native,Gps,'3.0.0001')
        return stream,native,Gps(timestamp=500,frame=10,posX=10.)

    def test_native_typed_callback_copies_data_and_preserves_original_ttl(self):
        stream,native,gps = self.fixture()
        self.assertEqual('registered',stream.status)
        self.assertIsNone(native.SetGpsUpdateCB.callback._restype_)
        native.SetGpsUpdateCB.callback(b'0',ctypes.pointer(gps))
        self.now += .2
        native.SetGpsUpdateCB.callback(b'0',ctypes.pointer(gps))
        record = stream.select(10,500)
        self.assertEqual(100.,record['observed_at_s'])
        record['pose']['x'] = 999.
        self.assertEqual(10.,stream.select(10,500)['pose']['x'])
        gps.posX = 11.
        native.SetGpsUpdateCB.callback(b'0',ctypes.pointer(gps))
        self.assertIsNone(stream.select(10,500))

    def test_regression_cannot_reseed_old_frames_or_refresh_a_repeated_frame(self):
        stream,native,gps = self.fixture()
        callback = native.SetGpsUpdateCB.callback
        callback(b'0',ctypes.pointer(gps))
        gps.frame,gps.timestamp = 20,1000
        callback(b'0',ctypes.pointer(gps))
        self.now += .4
        gps.frame,gps.timestamp = 10,500
        callback(b'0',ctypes.pointer(gps))
        self.assertFalse(stream.samples)
        gps.frame,gps.timestamp = 20,1000
        callback(b'0',ctypes.pointer(gps))
        self.assertIsNone(stream.select(20,1000))
        gps.frame,gps.timestamp = 21,1050
        callback(b'0',ctypes.pointer(gps))
        self.assertIsNotNone(stream.select(21,1050))

    def test_close_new_initialization_and_wrong_vehicle_or_abi(self):
        stream,native,gps = self.fixture()
        native.SetGpsUpdateCB.callback(b'other',ctypes.pointer(gps))
        self.assertFalse(stream.samples)
        stream.close()
        native.SetGpsUpdateCB.callback(b'0',ctypes.pointer(gps))
        self.assertFalse(stream.samples)
        stream.register(native,Gps,'3.0.0001')
        native.SetGpsUpdateCB.callback(b'0',ctypes.pointer(gps))
        self.assertIsNotNone(stream.select(10,500))
        other = NS(SetGpsUpdateCB=Register())
        stream.register(other,ctypes.c_int,'3.0.0001')
        self.assertFalse(hasattr(other.SetGpsUpdateCB,'callback'))
        stream.register(other,Gps,'unknown')
        self.assertEqual('unsupported',stream.status)
