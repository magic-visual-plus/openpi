from loguru import logger
from openpi.serving.network import ZMQResponseClient
from openpi.policies.aubo_policy import make_aubo_example
from openpi.serving.profiler import Dbg_Timer

ML_INFER_SERVER_HOST = "localhost"
ML_INFER_SERVER_PORT = 18000
# Create a ZMQResponseClient instance
zmq_client = ZMQResponseClient(ML_INFER_SERVER_HOST, ML_INFER_SERVER_PORT)
obs_dict = make_aubo_example()

# loop
for i in range(100):
    with Dbg_Timer("send_request"):
        # Send the request and receive the response
        #logger.info(f"obs_dict {obs_dict}")
        action_result = zmq_client.send_request(obs_dict)
        logger.info("action result {}", action_result)