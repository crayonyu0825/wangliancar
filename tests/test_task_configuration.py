import json
import os
import tempfile
import unittest

from simone_platform.task_configuration import installed_task_parameters


class TaskConfigurationTests(unittest.TestCase):
    def test_installed_task_dimensions_are_read_without_certifying_sensor_completeness(self):
        task = '11111111-2222-3333-4444-555555555555'
        with tempfile.TemporaryDirectory() as root:
            cache = os.path.join(root,'Module','VehicleDynamic','configJsonFromWeb')
            os.makedirs(cache)
            sdk = os.path.join(root,'SimOneAPI','lib','Win64')
            data = dict(data=dict(mainVehicleId=0,generalSensors=[],dynamics=dict(type='srvehicles',
                params=dict(Body=dict(AxisDistance=291.87,FrontToFrontAxis=100.,
                    RearAxisToRear=88.,ChassisWidth=180.)))))
            path = os.path.join(cache,task+'_CybertronBridgeSimOneDriver_WorkJsonFromWeb.txt')
            with open(path,'w') as stream:json.dump(data,stream)
            result = installed_task_parameters(sdk,task,'0')
            self.assertEqual('TASK_PARAMETERS_FOUND',result['status'])
            self.assertAlmostEqual(3.9187,result['vehicle']['front_offset_m'])
            self.assertEqual(.9,result['vehicle']['half_width_m'])
            self.assertTrue(result['missing_evidence'])
            self.assertEqual('TASK_PARAMETERS_UNAVAILABLE',installed_task_parameters(sdk,task,'1')['status'])
            self.assertEqual('TASK_ID_UNAVAILABLE',installed_task_parameters(sdk,'../other','0')['status'])
