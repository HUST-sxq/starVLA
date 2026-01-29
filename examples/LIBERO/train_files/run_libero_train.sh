#!/usr/bin/env bash
set -euo pipefail

############################
# 0) GPU / Distributed basic
############################
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_LAUNCH_BLOCKING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600

############################
# 1) WANDB Settings
############################
export WANDB_ENTITY="sunxiaoquan_2002-huazhong-university-of-science-and-tech"
export WANDB_PROJECT="starVLA"
# export WANDB_MODE=disabled

############################
# 2) Paths / Configs
############################
Framework_name="QwenGR00T_WorldModel"
base_vlm="/mnt/data/szeluresearch/models/Qwen3-VL-4B-Instruct"
config_yaml="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
libero_data_root="/mnt/data/szeluresearch/datasets/libero"
# if data_mix is libero_all, it includes libero_data_root.
# if data_mix is data_mix="libero_goal/libero_10/libero_object/libero_spatial", it only train this.
data_mix="libero_10"

run_root_dir="/mnt/data/szeluresearch/models/starVLA"
run_id="$(date +%m%d_%H%M)_qwen3GR00T_WM_subtask"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

############################
# 3) Optional args
############################
# If you don't want to freeze any modules, keep it empty
# If you do want to freeze some modules, fill in the format supported by your project
# for example ：freeze_module_list="['vision_encoder','llm']"
freeze_module_list=""

extra_args=()
if [[ -n "${freeze_module_list}" ]]; then
  extra_args+=(--trainer.freeze_modules "${freeze_module_list}")
fi

############################
# 4) Logging Mode Settings
############################
log_dir="./logs/training/$(date +'%Y%m%d')"
mkdir -p "$log_dir"


# Define the log file with timestamp
log_file="${log_dir}/$(date +'%H%M').log"

# Choose logging mode
mode="logs"  # Options: "terminal", "logs", "both"

if [ "$mode" == "terminal" ]; then
  # Only output to terminal
  exec 2>&1
elif [ "$mode" == "logs" ]; then
  # Only output to log file
  exec > "$log_file" 2>&1
else
  # Output to both terminal and log file
  exec > >(tee -a "$log_file") 2>&1
fi

############################
# 5) Launch
############################
nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_port 29600 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 10000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "${extra_args[@]}" &

# Display process ID of the background job
echo "Training started in the background. Log file: $log_file"


##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
