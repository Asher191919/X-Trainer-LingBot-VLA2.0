# X-Trainer LingBot-VLA 2.0

本仓库为 LingBot-VLA 2.0 后训练添加了通用 X-Trainer 数据映射。

## 数据约定

通用机器人配置为 [`configs/robot_configs/xtrainer.yaml`](configs/robot_configs/xtrainer.yaml)，要求 LeRobot v2.1 数据集包含以下字段：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `observation.state` | `(14,)` | 左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪 |
| `action` | `(14,)` | 维度顺序与 `observation.state` 相同 |
| `observation.images.top` | image | 顶部相机 |
| `observation.images.left_wrist` | image | 左腕相机 |
| `observation.images.right_wrist` | image | 右腕相机 |

动作处理与 X-Trainer Pi0.5-JAX 保持一致：12 维机械臂关节动作由绝对目标转换为 delta action，两个夹爪动作保持 absolute。

## 训练流程

```text
X-Trainer 示教数据
  -> LeRobot 数据集
  -> 检查 14 维 state/action 和三路相机字段
  -> 计算 X-Trainer normalization statistics
  -> 全参微调或 LoRA
```

### 计算归一化统计

单个数据集：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name xtrainer \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/xtrainer.json \
  --data.data_ratio_for_norm_compute 1
```

多个数据集需要创建数据清单，每行填写一个数据集：

```text
xtrainer /path/to/lerobot_dataset_a
xtrainer /path/to/lerobot_dataset_b
```

然后将 `--data.data_name` 设为 `multi`，并通过 `--data.train_path` 传入该清单。

### 全参微调

训练配置和命令将在后续补充。

### LoRA

训练配置和命令将在后续补充。
