#!/bin/bash
gpu=0

export PATH="/path/to/miniconda3/bin:$PATH"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/diffa2


model_path=/path/to/projects/DIFFA-exp/stage4
llm_path=/path/to/models/LLaDA-8B-Instruct
whisper_path=/path/to/models/whisper-large-v3
input_json=/path/to/datasets/MMAR/MMAR-meta.json

steps=16
block_length=16
max_new_tokens=16

output_dir=$model_path/fast_dllm/mmar
mkdir -p $output_dir

cp /path/to/projects/DIFFA-main/run_mmar_inference.sh $output_dir/

echo "start inference"

CUDA_VISIBLE_DEVICES=$gpu python3.10 ./inference_mmau.py \
        --model_path $model_path \
        --llm_path $llm_path \
        --whisper_path $whisper_path \
        --input_json_path $input_json\
        --output_json_path $output_dir/res.json \
        --steps $steps \
        --block_length $block_length \
        --max_new_tokens $max_new_tokens \
        --use_lora \
        --accelerate "fast_dllm"

python3.10 /path/to/datasets/MMAR/code/evaluation.py --input $output_dir/res.json > $output_dir/final_score
tail -n 100 $output_dir/final_score
