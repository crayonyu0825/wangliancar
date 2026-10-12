"""Formal maneuver prediction cannot turn absent raw velocity into rest."""
import unittest

from tests import test_driving_integration as integration_fixture


class PlanningSourceFieldTests(unittest.TestCase):
    def test_absent_raw_velocity_rejects_formal_admission(self):
        case=integration_fixture.DrivingIntegrationTests('test_real_builder_produces_selected_cells_and_generated_enter_path')
        case.setUp()
        try:
            case.targets=[dict(id=90,type=1,x=65.,y=8.,z=0.,length=4.5,width=1.8,
                               heading=0.,probability=1.)]
            d,t,c,safety=case.cycle(check=False)
            self.assertFalse(case.p.source_status['targets']['motion_fields_complete'])
            self.assertEqual('formal_target_velocity_source_unknown',
                             case.p.maneuver_environment.status['objects']['reason'])
            self.assertFalse(d.behavior_request and d.behavior_request['stage']=='ENTER')
            self.assertTrue(safety.active or d.target_speed==0.)
        finally:
            case.doCleanups()


if __name__=='__main__':unittest.main()
