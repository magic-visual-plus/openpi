<!-- 安装 uv 环境 -->
pip install uv

<!-- 新建env -->
GIT_LFS_SKIP_SMUDGE=1 proxychains uv sync -v
GIT_LFS_SKIP_SMUDGE=1 uv pip install -v -e .

<!-- 设置文件路径和hugface路径 -->
export HF_LEROBOT_HOME=/root/autodl-fs/datasets
export HF_HOME=/root/autodl-fs/huggingface