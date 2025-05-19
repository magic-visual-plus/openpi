# deploy model to real device
from loguru import logger
import os
import numpy as np
import random
import time
from openpi.serving.profiler import Dbg_Timer
from openpi.serving.network import ZMQResponseServer, ZMQResponseClient
from openpi.policies import policy as _policy


cur_dir = os.path.dirname(os.path.abspath(__file__))

def servo_infer(policy: _policy.Policy):
    zmq_server = ZMQResponseServer("0.0.0.0", 18000)
    logger.info("init server done")
    while True:
        # surround by try except to avoid server crash
        try:
            # receive obs_dict
            obs_dict = zmq_server.recv_request()
            logger.info(f'obs_dict{obs_dict}')
            # unsqueeze the obs_dict, current batch size is 1
            new_obs_dict = {}
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    print(f'key {key} value shape {value.shape}')
                    # delete the first dimension
                    new_obs_dict[key] = value[0]
                else:
                    new_obs_dict[key] = value
            
            obs_result = policy.infer(new_obs_dict)
            # append new axis to action_list
            action_list = obs_result['actions']
            # action_list = np.expand_dims(obs_result['actions'], axis=0)
            logger.info("infer action {}, shape {}", action_list, action_list.shape)
            obs_result = {"action": action_list}
            zmq_server.send_response(obs_result)
        except Exception as e:
            logger.exception("Error in server: {}", e)
            zmq_server.send_response({"error": str(e)})