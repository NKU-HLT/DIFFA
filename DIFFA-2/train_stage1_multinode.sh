#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL


export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

export PATH=/usr/local/cuda/bin:$PATH


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


export PATH="/path/to/miniconda3/bin:$PATH"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/diffa


GPUS_PER_NODE=8
NNODES=8
MASTER_ADDR=$1 
MASTER_PORT=$2 
NODE_RANK=$3 


# stage 1 : ASR task.
overwrite_output_dir="False" # Important!
output_dir="/path/to/projects/DIFFA-exp/stage1" #stage1_1e-4_20_4_64gpu_warmup1000

train_dataset_path="/path/to/projects/DIFFA-data/stage1/data.json"
llm_model="/path/to/models/LLaDA-8B-Instruct"
whisper_model="/path/to/models/whisper-large-v3"


mkdir -p $output_dir

# 创建日志目录和文件
log_dir="$output_dir/logs"
mkdir -p $log_dir

# 生成带时间戳的日志文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$log_dir/train_stage1_${timestamp}.log"

echo "训练开始时间: $(date)" | tee $log_file
echo "日志文件: $log_file" | tee -a $log_file
echo "输出目录: $output_dir" | tee -a $log_file
echo "训练数据: $train_dataset_path" | tee -a $log_file
echo "========================================" | tee -a $log_file

# 运行训练并保存日志（同时输出到屏幕和文件）
python3.10 -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE  \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    train.py \
    --deepspeed config/dp_config.json \
    --llm_model $llm_model\
    --whisper_model $whisper_model\
    \
    --output_dir $output_dir \
    --overwrite_output_dir $overwrite_output_dir \
    --remove_unused_columns "False" \
    --seed "42" \
    --do_train "True" \
    --bf16 "True" \
    \
    --learning_rate "1e-4" \
    --weight_decay "0.05" \
    --max_grad_norm "1.0" \
    --warmup_steps "1000" \
    \
    --per_device_train_batch_size "20" \
    --gradient_accumulation_steps "1" \
    --num_train_epochs "20" \
    --disable_tqdm "True" \
    --logging_steps "50" \
    --save_strategy "epoch" \
    --save_total_limit "20" \
    \
    --training_stage "1"\
    --train_dataset_path $train_dataset_path 2>&1 | tee -a $log_file

# 记录训练结束时间和退出状态
exit_code=${PIPESTATUS[0]}
echo "========================================" | tee -a $log_file
echo "训练结束时间: $(date)" | tee -a $log_file
echo "退出状态码: $exit_code" | tee -a $log_file

if [ $exit_code -eq 0 ]; then
    echo "训练成功完成！" | tee -a $log_file
else
    echo "训练失败，退出码: $exit_code" | tee -a $log_file
fi

echo "完整日志已保存到: $log_file"

# 退出时返回原始的退出码
exit $exit_code