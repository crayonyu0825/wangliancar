"""Unsupported flat request goals cannot be silently planned as passage motion."""
import copy
import unittest

from core.behavior_contract import GoalPose
from members import planning_stub
from tests import test_driving_integration as fixture


class DrivingRequestConstraintTests(unittest.TestCase):
    def setUp(self):
        self.h=fixture.DrivingIntegrationTests();self.h.setUp();self.addCleanup(self.h.doCleanups)

    def test_unhandled_pose_or_adopted_candidate_is_rejected_by_both_consumers(self):
        d,t,c,unused=self.h.cycle(sent=False)
        self.assertEqual('ENTER',d.behavior_request['stage'])
        for field,value in (('goal_pose',GoalPose(999.,999.,0.).to_dict()),
                            ('candidate_id','unapproved-passage-candidate')):
            modified=copy.deepcopy(d);modified.behavior_request[field]=value
            result=planning_stub.plan(self.h.p,modified)
            self.assertFalse(result.valid)
            self.assertFalse(result.points)
            self.assertEqual('REJECTED',result.behavior_feedback['status'])
            self.assertIn('PASSAGE_UNSUPPORTED_FIXED_GOAL_OR_CANDIDATE',result.reason)
            # Check the original valid delivered path: the independent guard
            # must reject the request rather than relying on planner failure.
            assessment=self.h.bridge.assess(self.h.p,modified,t,c)
            self.assertTrue(assessment.active)
            self.assertIn('PASSAGE_UNSUPPORTED_FIXED_GOAL_OR_CANDIDATE',assessment.reason)


if __name__=='__main__':unittest.main()
