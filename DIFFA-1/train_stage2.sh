#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


export CUDA_VISIBLE_DEVICES="0,1,2,3"

# stage2 qa task
output_dir="exp/stage2" 
train_dataset_path="data/qwen3_ans/final_train.json"
llm_model=""
whisper_model=""
stage1_ckpt=""

mkdir -p $output_dir

python -m torch.distributed.run --nproc_per_node=4  --master-port=12357 train.py \
    --deepspeed config/dp_config.json \
    --llm_model $llm_model\
    --whisper_model $whisper_model\
    \
    --output_dir $output_dir \
    --overwrite_output_dir "True" \
    --remove_unused_columns "False" \
    --seed "1" \
    --do_train "True" \
    --bf16 "True" \
    \
    --lr_scheduler_type "cosine"\
    --learning_rate "5e-5" \
    --weight_decay "0.05" \
    --max_grad_norm "1.0" \
    --warmup_steps "2000" \
    \
    --per_device_train_batch_size "4" \
    --gradient_accumulation_steps "2" \
    --num_train_epochs "10" \
    --disable_tqdm "True" \
    --logging_steps "20" \
    --save_strategy "epoch" \
    --save_total_limit "1"\
    \
    --training_stage "2" \
    --train_dataset_path $train_dataset_path \
    --stage1_ckpt $stage1_ckpt