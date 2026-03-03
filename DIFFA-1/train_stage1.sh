#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


export CUDA_VISIBLE_DEVICES="0,1,2,3"


# stage 1 : ASR task.
output_dir="exp/stage1"
train_dataset_path="data/librispeech_train_960/train.json"
llm_model=""
whisper_model=""

mkdir -p $output_dir

python -m torch.distributed.run --nproc_per_node=4  --master-port=12358 train.py \
    --deepspeed config/dp_config.json \
    --llm_model $llm_model\
    --whisper_model $whisper_model\
    \
    --output_dir $output_dir \
    --overwrite_output_dir "False" \
    --remove_unused_columns "False" \
    --seed "1" \
    --do_train "True" \
    --bf16 "True" \
    \
    --learning_rate "1e-4" \
    --weight_decay "0.05" \
    --max_grad_norm "1.0" \
    --warmup_steps "1000" \
    \
    --per_device_train_batch_size "16" \
    --gradient_accumulation_steps "2" \
    --num_train_epochs "10" \
    --disable_tqdm "True" \
    --logging_steps "20" \
    --save_strategy "epoch" \
    --save_total_limit "5" \
    \
    --training_stage "1"\
    --train_dataset_path $train_dataset_path