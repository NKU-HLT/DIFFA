gpu=0

model_path=""

steps=4
block_length=4
max_new_tokens=4

output_dir=$model_path/test_mmsu
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=$gpu python ./inference_stage2_mmsu.py \
        --model_path $model_path \
        --output_jsonl $output_dir/res.jsonl \
        --steps $steps \
        --block_length $block_length \
        --max_new_tokens $max_new_tokens

python ./mmsu/evaluate.py $output_dir/res.jsonl