"""An otherwise valid relabelled packet cannot erase an accepted dwell."""
import unittest
from unittest.mock import patch

from members.control.controller import ControlEngine
from tests.control_benchmark import vehicle
from tests.test_control_maneuvers import BidirectionalPlant,maneuver_input


class ControlIdentityTests(unittest.TestCase):
    def test_unfinished_dwell_survives_a_valid_new_task_label(self):
        now=[100.]
        with patch('time.monotonic',lambda:now[0]):
            engine=ControlEngine(vehicle(),clock=lambda:now[0])
            plant=BidirectionalPlant()
            p,t=maneuver_input(plant,[(0,0,0),(0,0,0)],1,target=0,
                stop_distance=0,precision=True,dwell=10,parking=True)
            t.stop_obligation_id='actual-old-bay'
            self.assertTrue(engine.compute(p,t).valid)
            self.assertTrue(engine.standstill.active)
            now[0]+=.05
            p,t=maneuver_input(plant,[(0,0,1),(10,0,1)],2,target=1.)
            p.task_id='untrusted-new-task'
            output=engine.compute(p,t)
            self.assertFalse(output.valid)
            self.assertEqual('actual-old-bay',engine.standstill.obligation_id)
            self.assertEqual(10,engine.standstill.duration)
            self.assertTrue(engine.standstill.active)
            # Explicit captain reset is the available new-session operation.
            engine.reset()
            self.assertFalse(engine.standstill.active)


if __name__=='__main__':unittest.main()
