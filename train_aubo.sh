# 计算数据的norm stat, 数据是按照你的config下索引的repo_id来的
# uv run scripts/compute_norm_stats.py --config_name pi0_aubo_low_mem_finetune
# 开启训练,model name会影响你保存后model的名称和你wandb任务的名称
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aubo_low_mem_finetune --exp-name=pi0_aubo_0519_test2 --checkpoint_base_dir=/root/autodl-tmp/checkpoints --fsdp_devices=4 --overwrite 