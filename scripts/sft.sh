#!/bin/bash
# ============================================================================
# π₀.₅ PyTorch 单机多卡微调启动脚本
#
# 使用方式：
#   chmod +x scripts/sft.sh
#   bash scripts/sft.sh
#
# 可调参数说明见下方变量区。
# ============================================================================

set -euo pipefail

# ============================================================================
# 可调参数
# ============================================================================

# 训练配置名（config.py 中定义的 name 字段）
CONFIG_NAME="pi05_clothes"

# 实验名称，决定 checkpoint 保存路径 checkpoints/<config>/<exp_name>/
EXP_NAME="clothes_exp4"

# 使用的 GPU 数量（单机多卡）
NUM_GPUS=8

# 全局 batch size（代码会自动除以 GPU 卡数，每卡 = BATCH_SIZE / NUM_GPUS）
BATCH_SIZE=512

# 训练总步数
NUM_TRAIN_STEPS=30000

# 峰值学习率
PEAK_LR="5e-5"

# checkpoint 保存间隔（步）
SAVE_INTERVAL=600

# 是否覆盖已有同名 checkpoint：true=覆盖，false=报错
OVERWRITE=true

# 是否从上次 checkpoint 恢复训练：true=恢复，false=重新开始
RESUME=false

# 日志输出间隔（步）
LOG_INTERVAL=20

# PyTorch 训练精度：bfloat16 或 float32
TORCH_PRECISION="bfloat16"

# XLA 内存上限（仅 JAX 训练需要，PyTorch 可忽略）
XLA_MEM_FRACTION="0.9"

# wandb 日志：true=开启，false=关闭
WANDB_ENABLED="true"

# ============================================================================
# 执行训练
# ============================================================================

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Config:        ${CONFIG_NAME}"
echo "  Experiment:    ${EXP_NAME}"
echo "  GPUs:          ${NUM_GPUS}"
echo "  Batch/GPU:     ${BATCH_SIZE}"
echo "  Total Steps:   ${NUM_TRAIN_STEPS}"
echo "  Peak LR:       ${PEAK_LR}"
echo "  Save Interval: ${SAVE_INTERVAL}"
echo "  Resume:        ${RESUME}"
echo "  Overwrite:     ${OVERWRITE}"
echo "============================================"

RESUME_FLAG=""
if [ "${RESUME}" = "true" ]; then
    RESUME_FLAG="--resume"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE}" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

# Disable NVLS (NVLink SHARP) to prevent Cuda failure 401 on H200 with >2 GPUs
export NCCL_NVLS_ENABLE=0

# 单机多卡 torchrun 启动 PyTorch 训练
uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG_NAME}" \
    --exp_name "${EXP_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --num_train_steps "${NUM_TRAIN_STEPS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --log_interval "${LOG_INTERVAL}" \
    ${RESUME_FLAG} \
    ${OVERWRITE_FLAG}

echo "============================================"
echo "  Training finished."
echo "  Checkpoints: checkpoints/${CONFIG_NAME}/${EXP_NAME}/"
echo "============================================"
