"""Read task-bound installed simulator parameters without inventing calibration."""
import hashlib
import json
import math
import os
import re


def installed_task_parameters(sdk_dir, task_id, vehicle_id):
    if (not sdk_dir or not isinstance(task_id,str)
            or re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',task_id) is None):
        return dict(status='TASK_ID_UNAVAILABLE')
    root = os.path.abspath(os.path.join(sdk_dir,'..','..','..'))
    path = os.path.join(root,'Module','VehicleDynamic','configJsonFromWeb',
                        task_id+'_CybertronBridgeSimOneDriver_WorkJsonFromWeb.txt')
    try:
        with open(path,'rb') as stream:
            raw = stream.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:
            raise ValueError('task cache exceeds bound')
        data = json.loads(raw.decode('utf-8-sig'))['data']
        if str(data['mainVehicleId']) != str(vehicle_id):
            raise ValueError('task vehicle differs')
        dynamics = data['dynamics']
        if dynamics['type'] != 'srvehicles':
            raise ValueError('unsupported vehicle units/schema')
        body = dynamics['params']['Body']
        fields = ('AxisDistance','FrontToFrontAxis','RearAxisToRear','ChassisWidth')
        if not all(type(body[k]) in (int,float) and math.isfinite(body[k]) and body[k]>0 for k in fields):
            raise ValueError('invalid vehicle dimensions')
        geometry = dict(wheelbase_m=body['AxisDistance']/100.,
            front_offset_m=(body['AxisDistance']+body['FrontToFrontAxis'])/100.,
            rear_offset_m=body['RearAxisToRear']/100.,half_width_m=body['ChassisWidth']/200.)
        sensors = []
        for sensor in data['generalSensors']:
            spec = sensor['data']
            if sensor['type'] != 'sensorFusion':
                continue
            params = spec['params']
            sensors.append(dict(sensor_id=spec['id'],category=spec.get('category'),
                installation=dict(type=sensor['type'],x=spec['x'],y=spec['y'],z=spec['z'],
                    roll=spec['roll'],pitch=spec['pitch'],yaw=spec['yaw'],hz=params['frequency']),
                detection_range_m=params['detectionRange'],noise=params['error'],
                simulated_fusion=params.get('isFusionSim'),impact_factors=params.get('impactFactor',{}),
                complete_detections_verified=False))
        return dict(status='TASK_PARAMETERS_FOUND',task_id=task_id,vehicle_id=str(vehicle_id),
            source=path,sha256=hashlib.sha256(raw).hexdigest(),vehicle=geometry,sensors=sensors,
            gps_steering=dict(unit='degree',scale_to_front_rad=math.pi/180.,
                source='SimOneIOStruct.h: SimOne_Data_Gps.steering, SDK 3.0.0001'),
            visibility_explanation='Fusion confidence can filter occluded or weather-affected objects; range is not complete coverage',
            visibility_reference='Web/help/_sources/reference_manual/sensor_simuliation.rst.txt: Fusion target-level simulation',
            missing_evidence=['complete Sensor detection region under occlusion/weather',
                              'vehicle tracking position/heading/time/speed error bounds'])
    except (OSError,ValueError,TypeError,KeyError,UnicodeError,AttributeError) as error:
        return dict(status='TASK_PARAMETERS_UNAVAILABLE',reason=str(error),source=path)
