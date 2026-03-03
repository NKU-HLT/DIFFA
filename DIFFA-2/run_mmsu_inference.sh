#!/bin/bash
gpu=0

export PATH="/path/to/miniconda3/bin:$PATH"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/diffa



model_path=/path/to/projects/DIFFA-exp/stage4/debug_1211_wo_sft_toy_data/checkpoint-295
llm_path=/path/to/models/LLaDA-8B-Instruct
whisper_path=/path/to/models/whisper-large-v3
input_jsonl=/path/to/datasets/MMSU/question/mmsu.jsonl


steps=16
block_length=16 
max_new_tokens=16

output_dir=$model_path/origin_test_speed/mmsu
mkdir -p $output_dir
cp /path/to/projects/DIFFA-main/run_mmsu_inference.sh $$output_dir/



CUDA_VISIBLE_DEVICES=$gpu python3.10 ./inference_mmsu.py \
        --model_path $model_path \
        --llm_path $llm_path \
        --whisper_path $whisper_path \
        --input_jsonl $input_jsonl\
        --output_jsonl $output_dir/res.jsonl \
        --steps $steps \
        --block_length $block_length \
        --max_new_tokens $max_new_tokens \
        --use_lora \
        --accelerate "fast_dllm"


python3.10 ./mmsu/evaluate.py $output_dir/res.jsonl > $output_dir/final_score
tail -n 100 $output_dir/final_score
