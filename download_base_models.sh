#!/bin/bash
set -euo pipefail

echo "=== 下载 LingBot-VLA 2.0 基础模型 ==="

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

# 1. Qwen3-VL-4B-Instruct（视觉语言模型基础）
echo "正在下载 Qwen3-VL-4B-Instruct..."
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$MODELS_DIR/Qwen3-VL-4B-Instruct" \
  --local-dir-use-symlinks False

# 2. LingBot-VLA v2.6B（你的 VLA 模型，目前是 placeholder）
echo "正在下载 LingBot-VLA v2.6B..."
huggingface-cli download your-username/lingbot-vla-v2-6b \
  --local-dir "$MODELS_DIR/lingbot-vla-v2-6b" \
  --local-dir-use-symlinks False

# 3. MoRGBD（深度估计模型）
echo "正在下载 MoRGBD..."
huggingface-cli download MoGe/MoRGBD \
  --local-dir "$MODELS_DIR/MoRGBD" \
  --local-dir-use-symlinks False

# 4. Video-DINO（视频教师模型）
echo "正在下载 Video-DINO..."
huggingface-cli download facebook/video-dino \
  --local-dir "$MODELS_DIR/video-dino" \
  --local-dir-use-symlinks False

echo "✅ 所有基础模型已下载到 $MODELS_DIR"
echo "请根据实际情况修改下载脚本中的 Hugging Face 路径（尤其是 lingbot-vla-v2-6b）"
