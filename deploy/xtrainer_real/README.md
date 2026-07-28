# X-Trainer 真机客户端

本目录提供在 X-Trainer 真机上运行 LingBot-VLA 2.0 的最小 Linux 客户端。Dobot、RealSense 和 Feetech SDK 代码搬运自 `X-Trainer-Pi0.5-JAX/examples/xtrainer_real`，客户端不依赖 OpenPI runtime 或 OpenPI 客户端包。

## 安装依赖

在机器人控制机上安装轻量客户端依赖：

```bash
pip install -r deploy/xtrainer_real/requirements.txt
```

来源仓库中的 Feetech `scservo_sdk` 文件没有单独的上游许可证声明。独立分发本目录前，请先确认原始 SDK 的再分发条款。

## 启动推理服务

使用默认 action chunk 模式启动服务端。服务端的 `--use-length` 必须不小于客户端的 `--action-horizon`。

```bash
python scripts/serve_policy.py \
  --model-path /path/to/checkpoint \
  --robot xtrainer \
  --use-length 50 \
  --port 8000
```

使用本客户端时，不要向服务端传入 `--step-mode`。

## 启动真机客户端

```bash
python scripts/run_xtrainer_real.py \
  --host 192.168.1.10 \
  --task "fold the clothes" \
  --camera-top-serial TOP_SERIAL \
  --camera-left-wrist-serial LEFT_WRIST_SERIAL \
  --camera-right-wrist-serial RIGHT_WRIST_SERIAL \
  --action-horizon 25 \
  --control-hz 20
```

默认硬件地址与 Pi0.5 客户端保持一致：

- 左臂：`192.168.5.1`，夹爪串口 `/dev/ttyUSB1`，舵机 ID `21`。
- 右臂：`192.168.5.2`，夹爪串口 `/dev/ttyUSB0`，舵机 ID `22`。
- 状态与动作顺序：左臂 6 个关节、左夹爪、右臂 6 个关节、右夹爪。

运行 `python scripts/run_xtrainer_real.py --help` 可查看全部安全阈值和硬件参数。

## 首次真机测试

清空机器人工作空间，并从较小的 `--max-steps` 开始测试。确认第一次策略响应的形状为 `(H, 14)`，机械臂目标是以弧度表示的绝对关节角，并且两个夹爪值均位于 `[0, 1]`。

客户端会在动作发送到机器人前拒绝格式错误、NaN 或 Inf 动作，但无法判断一个数值有效的动作对于当前场景是否具备实际安全性。