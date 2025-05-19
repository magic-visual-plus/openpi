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
    step_count = 0
    use_wrist = policy.use_wrist
    logger.info("user wrist {}", use_wrist)
    zmq_server = ZMQResponseServer("0.0.0.0", 18000)
    logger.info("init server done")
    while True:
        obs_dict = zmq_server.recv_request()
        logger.info(f'obs_dict{obs_dict}')
        # obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        action_list = policy.infer(obs_dict)
        logger.info("step {} infer action {}", step_count, action_list)
        obs_result = {"action": action_list}
        zmq_server.send_response(obs_result)