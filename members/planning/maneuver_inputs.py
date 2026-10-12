"""Common explicit model snapshot for source-gated maneuver consumers."""
import copy
from core.behavior_contract import TaskContext,finite,require


class ManeuverPlanningInputs(object):
    """Caller assertions, never proof of calibration or inferred curvature."""
    def __init__(self,context,frame_id,produced_at_s,valid_until_s,model_id,
                 initial_curvature_m_inv,vehicle,limits,search,budget,
                 prediction_envelopes,geometry_cache=None,clock_id='process_monotonic',
                 execution_envelope=None,dynamic_search=None,speed_error_mps=None):
        self.context,self.frame_id=context,frame_id
        self.produced_at_s,self.valid_until_s=produced_at_s,valid_until_s
        self.model_id,self.clock_id=model_id,clock_id
        self.initial_curvature_m_inv=initial_curvature_m_inv
        self.vehicle,self.limits,self.search=copy.deepcopy((vehicle,limits,search))
        self.budget,self.prediction_envelopes=copy.deepcopy((budget,prediction_envelopes))
        self.geometry_cache=geometry_cache
        self.execution_envelope,self.dynamic_search=copy.deepcopy((execution_envelope,dynamic_search))
        self.speed_error_mps=copy.deepcopy(speed_error_mps)


def read_inputs(provider,p,context,clock,prefix,expected_type):
    require(provider is not None,prefix+'_MODEL_INPUTS_UNAVAILABLE')
    value=provider(p,context)
    require(isinstance(value,expected_type),prefix+'_MODEL_INPUTS_INVALID')
    now=clock()
    require(isinstance(value.context,TaskContext) and value.context.key()==context.key()
            and type(value.frame_id) is int and value.frame_id==p.frame_id
            and value.clock_id=='process_monotonic'
            and finite(value.produced_at_s) and finite(value.valid_until_s)
            and value.produced_at_s<=now<value.valid_until_s<=p.valid_until,
            prefix+'_MODEL_SOURCE_MISMATCH_OR_EXPIRED')
    require(isinstance(value.model_id,str) and bool(value.model_id)
            and finite(value.initial_curvature_m_inv),prefix+'_MOTION_MODEL_UNAVAILABLE')
    result=copy.copy(value)
    for name in ('vehicle','limits','search','budget','prediction_envelopes',
                 'execution_envelope','dynamic_search','speed_error_mps'):
        setattr(result,name,copy.deepcopy(getattr(value,name)))
    return result
