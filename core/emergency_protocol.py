"""Observed emergency identity and verified areas for brake-only or turn/brake."""
from core.behavior_protocol import finite,require
from core.behavior_environment import Evidence
from core.road_region import RoadRegion


class EmergencyTask(object):
    def __init__(self,danger_id,evidence,dangerous,allowed_regions,lateral_offsets_m=(0.0,),return_lane_id=None,permission_evidence=None):
        require(isinstance(danger_id,str) and bool(danger_id) and isinstance(evidence,Evidence)
                and type(dangerous) is bool,'invalid observed danger')
        require(isinstance(lateral_offsets_m,(list,tuple)) and bool(lateral_offsets_m)
                and len(lateral_offsets_m)<=5 and all(finite(v) and abs(v)<=4 for v in lateral_offsets_m),
                'emergency action permissions missing')
        require(return_lane_id is None or isinstance(return_lane_id,str) and bool(return_lane_id),'invalid recovery route')
        self.danger_id,self.evidence,self.dangerous=danger_id,evidence,dangerous
        self.allowed_regions=tuple(RoadRegion(p) for p in allowed_regions)
        self.lateral_offsets_m,self.return_lane_id=tuple(lateral_offsets_m),return_lane_id
        self.permission_evidence=permission_evidence

    def verified(self,frame,mode):
        kinds=('synthetic',) if mode=='model' else ('verified_fusion',)
        permissions=(mode=='model' or isinstance(self.permission_evidence,Evidence)
                     and self.permission_evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')))
        return permissions and bool(self.allowed_regions) and self.evidence.verified(frame,allowed_source_kinds=kinds)
