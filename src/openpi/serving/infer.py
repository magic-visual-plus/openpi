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
            action_list = policy.infer(obs_dict)
            logger.info("infer action {}", action_list)
            obs_result = {"action": action_list}
            zmq_server.send_response(obs_result)
        except Exception as e:
            logger.exception("Error in server: {}", e)
            zmq_server.send_response({"error": str(e)})