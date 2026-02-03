#!/bin/bash
set -e  # 一旦出错立刻停

# export LIBERO_ROOT=/export/ra/sunxiaoquan/LIBERO-PRO
# export PYTHONPATH=${LIBERO_ROOT}:$(pwd):${PYTHONPATH}
# export LIBERO_CONFIG_PATH=/export/ra/sunxiaoquan/starVLA/examples/LIBERO
export LIBERO_ROOT=/root/starVLA/third_party/LIBERO-PRO
export PYTHONPATH=${LIBERO_ROOT}:$(pwd):${PYTHONPATH}
export LIBERO_CONFIG_PATH=$(pwd)/examples/LIBERO
# export MUJOCO_GL=osmesa

LIBERO_Python=python

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=/mnt/data/szeluresearch/models/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
model_name=$(basename $(dirname $(dirname "$your_ckpt")))
# =========================
# 🔥 日志控制开关
# =========================
export DEBUG=0              # 0 / 1
export LOG_LEVEL=INFO       # DEBUG / INFO / WARNING

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
folder_name="${folder_name}_${instruction_mode}"

# =========================
# 📁 Log & Video 目录
# =========================
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/libero_eval/${RUN_ID}_${model_name}"
mkdir -p ${LOG_DIR}

task_suite_name=libero_10
num_trials_per_task=50
video_out_path="eval_results/${task_suite_name}/${folder_name}/${RUN_ID}"
instruction_mode="task" # "null" or "task"
# =========================
# 🚀 启动评测（关键）
# =========================
${LIBERO_Python} examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.instruction-mode "$instruction_mode" \
    2>&1 | tee ${LOG_DIR}/eval.log