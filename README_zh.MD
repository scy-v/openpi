# 安装uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'source $HOME/.local/bin/env' >> ~/.bashrc
source ~/.bashrc
```

# 下载安装openpi

```bash
git clone https://github.com/scy-v/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install -r requirements.txt
cd packages/openpi-client
uv pip install -e .
```


# 推理 
## 1. 远程服务器推理
### 参数说明:
- --policy.config: 指定config.py中用于推理所配置的TrainConfig的name
- --policy.dir: 指定模型checkpoint的路径

```bash
# 服务器运行：
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_path>
# 示例:
# uv run scripts/serve_policy.py policy:checkpoint --policy.config=server_infer --policy.dir=checkpoints/pi05_finetune_ur5e/task1/29999

# 本地运行：
python scripts/inference_ur_client.py 
```
## 2. 本地直接推理
```bash
#单臂
python scripts/inference_ur.py
#双臂
python scripts/inference_dual_ur.py
```


