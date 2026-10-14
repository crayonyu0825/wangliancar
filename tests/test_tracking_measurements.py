import unittest

from core.interfaces import Perception,Trajectory,TrajectoryPoint
from core.tracking_measurements import TrackingMeasurements


class TrackingMeasurementsTests(unittest.TestCase):
    def fixture(self,frame=1,x=0.,speed=2.):
        p = Perception()
        p.case_id,p.task_id,p.scene_id = 'case','task',6
        p.frame_id,p.timestamp,p.valid_until,p.valid = frame,frame*100,200.,True
        p.ego.frame_id,p.ego.timestamp,p.ego.valid = frame,p.timestamp,True
        p.ego.x,p.ego.speed = x,speed
        p.source_status['gps'] = dict(usable=True,clock='sdk_frame',frame_id=frame,timestamp=p.timestamp)
        t = Trajectory().bind(p)
        t.valid = True
        t.points = [TrajectoryPoint(x,0.,speed,0.,0.),TrajectoryPoint(x+speed,0.,speed,0.,1.)]
        return p,t

    def test_measure_previous_actually_sent_plan_not_new_replanned_path(self):
        collector = TrackingMeasurements()
        p,t = self.fixture()
        sent = dict(attempted=True,ok=True,reason='sent')
        collector.record(p,t,sent,False,100.)
        p,t = self.fixture(2,x=.5,speed=2.5)
        collector.record(p,t,sent,False,100.1)
        result = collector.summary()
        self.assertEqual(1,result['sample_count'])
        self.assertAlmostEqual(.3,result['observed_maxima']['position_error_m'])
        self.assertAlmostEqual(.5,result['observed_maxima']['speed_error_mps'])
        self.assertEqual('OBSERVED_ONLY_NOT_CALIBRATED',result['status'])
        collector.record(p,t,sent,False,100.2)
        self.assertEqual(1,collector.count)

    def test_no_measurements_from_failed_safety_expired_or_other_task_command(self):
        for condition in ('failed','safety','expired','task'):
            collector = TrackingMeasurements()
            p,t = self.fixture()
            collector.record(p,t,dict(attempted=True,ok=condition!='failed',reason='sent'),condition=='safety',100.)
            p,t = self.fixture(2,x=.5)
            if condition=='expired':collector.previous['valid_until']=100.05
            if condition=='task':p.task_id='other'
            collector.record(p,t,dict(attempted=False,ok=False,reason='observe_mode'),False,100.1)
            self.assertEqual(0,collector.count,condition)
