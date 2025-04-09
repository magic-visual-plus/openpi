from openpi.policies import so100_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_so100_policy():
    config_name = "pi0_so100_low_mem_finetune"
    model_file_path = f'/opt/projects/openpi/checkpoints/{config_name}/pi0_so100_test/6000'
    config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(config, model_file_path)

    example = so100_policy.make_so100_example()
    for _ in range(config.model.action_horizon):
        outputs = policy.infer(example)
        print(f'outputs shape {outputs.keys()} result {outputs}')


if __name__ == '__main__':
    test_so100_policy()