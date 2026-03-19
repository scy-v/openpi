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
cd packages/openpi-client
uv pip install -e .
uv pip install -r requirements.txt
```

# 微调环境变量说明
## 1. `OBS_INDICES`
- 作用：指定如何提取并重新排列观测值（observation.state）的索引顺序。
- 格式：逗号分隔的索引列表（从1开始计数）。
- 示例：
    OBS_INDICES=1,3,2
  将原本第1、2、3位状态重新排列为第1、3、2位。
- 注意：索引必须在原始状态维度范围内。

## 2. `ACTION_INDICES`
- 作用：指定如何提取并重新排列动作（action）的索引顺序。
- 格式：逗号分隔的索引列表（从1开始计数）。
- 示例：
    ACTION_INDICES=2,1,3
  将原本第1、2、3位动作重新排列为第2、1、3位。
- 注意：索引必须在原始动作维度范围内。

## 3. `ACTION_SLICE`
- 作用：定义动作向末端位姿（FK，Forward Kinematics）转换的关节段。
- 格式：每两个数表示一个关节段的起止索引，-1表示保留原始动作段。
- 示例：
    ACTION_SLICE=6,-1
    ACTION_SLICE=6,-1,6,-1
  - 6,-1 → 对前6个关节角进行FK转换，剩余动作保持不变。
  - 6,-1,6,-1 → 对前6个关节角和第8-13个关节角分别进行FK转换，其余保持不变。
- 注意：必须成对提供数值（偶数个），否则报错。

## 4. `URDF_PATH`
- 作用：指定机械臂URDF文件路径，用于ACTION_SLICE执行FK时加载模型。
- 格式：相对路径或绝对路径字符串。
- 示例：
    URDF_PATH=urdf/ur5e/ur5e.urdf
- 注意：仅在ACTION_SLICE非空时需要配置，否则可忽略。

# 微调
### - 注意检查四个环境变量的使用以及config的配置内容, --config-name对应Config中所选配置类的name
## 1. 计算归一化结果


```bash
# 默认单臂 obs-关节角/action-关节角计算归一化
OBS_INDICES=1,2,3,4,5,6,7 uv run scripts/compute_norm_stats.py --config-name pi05_finetune_ur5e
# 默认单臂 obs-tcp_pose/action-关节角转tcp_pose计算归一化
OBS_INDICES=28,29,30,31,32,33,7 ACTION_SLICE=6,-1 URDF_PATH=urdf/ur5e/ur5e.urdf ACTION_INDICES=1,2,3,4,5,6,7 uv run scripts/compute_norm_stats.py --config-name pi05_finetune_ur5e
# 默认单臂 obs-tcp_pose tcp_speed tcp_force/action-关节角转tcp_pose计算归一化
 OBS_INDICES=28,29,30,31,32,33,7,34,35,36,37,38,39,43,44,45,46,47,48 ACTION_SLICE=6,-1 URDF_PATH=urdf/ur5e/ur5e.urdf ACTION_INDICES=1,2,3,4,5,6,7 uv run scripts/compute_norm_stats.py --config-name pi05_full_finetune_ur5e
```
## 2.微调
```bash
# --exp-name为实验名称，会显示在wandb上, 可以自定义，直接取数据集名称+自定义后缀即可
# CUDA_VISIBLE_DEVICES是选择空闲gpu设备

# 默认单臂 obs-关节角/action-关节角LORA微调
OBS_INDICES=1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_finetune_ur5e/pi05_full_finetune_ur5e  --exp-name=pick_greencube_into_trashbin_20251102_v02 --overwrite
# 默认单臂 obs-tcp_pose/action-关节角转tcp_pose微调
OBS_INDICES=28,29,30,31,32,33,7 CUDA_VISIBLE_DEVICES=0,1,2,3 ACTION_SLICE=6,-1 URDF_PATH=urdf/ur5e/ur5e.urdf ACTION_INDICES=1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_finetune_ur5e/pi05_full_finetune_ur5e --exp-name=move_reagent_bottle_20260317_v01_tcp --overwrite
# 默认单臂 obs-tcp_pose tcp_speed tcp_force/action-关节角转tcp_pose微调
OBS_INDICES=28,29,30,31,32,33,7,34,35,36,37,38,39,43,44,45,46,47,48 CUDA_VISIBLE_DEVICES=0,1,2,3 ACTION_SLICE=6,-1 URDF_PATH=urdf/ur5e/ur5e.urdf ACTION_INDICES=1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_finetune_ur5e/pi05_full_finetune_ur5e --exp-name=move_reagent_bottle_20260317_v01_tcp_force --overwrite
```

# 推理 
## 1. 远程服务器推理
### 参数说明:
- `--policy.config`: 指定`config.py`中用于推理所配置的`TrainConfig`的`name`
- `--policy.dir`: 指定模型`checkpoint`的路径
- 推理配置文件位于`scripts/config/`下
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


