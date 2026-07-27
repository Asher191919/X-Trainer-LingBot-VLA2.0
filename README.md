# X-Trainer LingBot-VLA 2.0

本仓库为 LingBot-VLA 2.0 后训练添加了通用 X-Trainer 数据映射。

## 数据约定

通用机器人配置为 [`configs/robot_configs/xtrainer.yaml`](configs/robot_configs/xtrainer.yaml)，要求 LeRobot v2.1 数据集包含以下字段：


| 字段                             | 形状    | 含义                                        |
| ---------------------------------- | --------- | --------------------------------------------- |
| `observation.state`              | `(14,)` | 左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪 |
| `action`                         | `(14,)` | 维度顺序与`observation.state` 相同          |
| `observation.images.top`         | image   | 顶部相机                                    |
| `observation.images.left_wrist`  | image   | 左腕相机                                    |
| `observation.images.right_wrist` | image   | 右腕相机                                    |

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

全参训练配置位于 [`configs/vla/xtrainer/xtrainer.yaml`](configs/vla/xtrainer/xtrainer.yaml)。启动前需要填写模型、tokenizer、深度/视频教师、数据集和输出目录的实际路径。

```bash
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/xtrainer/xtrainer.yaml
```

脚本默认使用当前可见的全部 GPU。`global_batch_size` 会根据 GPU 进程数、`micro_batch_size` 和 `gradient_accumulation_steps` 自动计算。

### LoRA

训练配置和命令将在后续补充。

## 推理服务

仓库提供了与 X-Trainer Pi0.5-JAX `scripts/serve_policy.py` 类似的 WebSocket policy 服务入口：

```bash
cd /path/to/X-Trainer-LingBot-VLA2.0

python scripts/serve_policy.py \
  --model-path /path/to/checkpoint \
  --robot xtrainer \
  --port 8000
```

服务启动时会加载 checkpoint，并根据 `configs/robot_configs/<robot>.yaml` 初始化机器人字段映射和归一化器。默认使用 `xtrainer` 配置，客户端不需要在第一次推理前额外发送 reset 请求。

常用参数：


| 参数           | 默认值           | 说明                                                       |
| ---------------- | ------------------ | ------------------------------------------------------------ |
| `--model-path` | 必填             | 包含`.safetensors` 权重的 checkpoint 目录                  |
| `--robot`      | `xtrainer`       | `configs/robot_configs` 下的机器人配置名称，不包含 `.yaml` |
| `--norm-path`  | 配置文件中的路径 | 覆盖 normalization statistics 文件路径                     |
| `--host`       | `0.0.0.0`        | WebSocket 监听地址                                         |
| `--port`       | `8000`           | WebSocket 监听端口                                         |
| `--use-length` | `50`             | 每次模型前向后使用的动作数量                               |
| `--step-mode`  | 关闭             | 每次请求仅返回一个动作；默认返回完整 action chunk          |
| `--fp32`       | 关闭             | 使用 FP32；默认使用 BF16                                   |
| `--compile`    | 关闭             | 使用`torch.compile` 加速推理                               |

例如，启用逐步动作返回和 `torch.compile`：

```bash
python scripts/serve_policy.py \
  --model-path /path/to/checkpoint \
  --robot xtrainer \
  --step-mode \
  --use-length 50 \
  --compile \
  --port 8000
```

客户端可以使用仓库现有的 `WebsocketClientPolicy`：

```python
from deploy.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
result = policy.infer(observation)
```

`observation` 的字段需要与所选机器人 YAML 中的 `origin_keys` 对齐。对于默认的 `xtrainer` 配置，输入包括：

- `observation.state`：形状为 `(14,)` 的机器人状态。
- `observation.images.top`：顶部相机 RGB 图像。
- `observation.images.left_wrist`：左腕相机 RGB 图像。
- `observation.images.right_wrist`：右腕相机 RGB 图像。
- `task`：自然语言任务描述，具体字段处理以训练配置中的数据映射为准。

模型加载还需要满足以下目录约束：

- checkpoint 目录内包含一个或多个 `.safetensors` 文件。
- checkpoint 路径向上三级的位置存在训练时生成的 `lingbotvla_cli.yaml`。
- Qwen3-VL 基础模型路径可以从训练配置读取，也可以通过 `QWEN3VL_PATH` 环境变量覆盖。

服务启动后可通过以下接口检查进程是否存活：

```bash
curl http://127.0.0.1:8000/healthz
```
