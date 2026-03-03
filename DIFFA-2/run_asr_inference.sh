#!/bin/bash


gpu=0

export PATH="/path/to/miniconda3/bin:$PATH"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/diffa

MODEL_PATH="/path/to/projects/DIFFA-exp/stage1"

SUBSET="test_clean"

INSTRUCTION="Please transcribe the audio to text."


STEPS=128
BLOCK_LENGTH=128
MAX_NEW_TOKENS=128

WAV_SCP="/path/to/datasets/testset/${SUBSET}/wav.scp"

llm_path=/path/to/models/LLaDA-8B-Instruct
whisper_path=/path/to/models/whisper-large-v3

# use stage 1 ckpt.
OUTPUT_PATH=$MODEL_PATH/ASR/stage1/${SUBSET}_max${MAX_NEW_TOKENS}_steps${STEPS}_block${BLOCK_LENGTH}
OUTPUT_TEXT="$OUTPUT_PATH/text"
mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=$gpu python3.10 inference_stage1_asr.py \
  --model_path "$MODEL_PATH" \
  --llm_path $llm_path \
  --whisper_path $whisper_path \
  --wav_scp "$WAV_SCP" \
  --output_text "$OUTPUT_TEXT" \
  --instruction "$INSTRUCTION" \
  --steps $STEPS \
  --block_length $BLOCK_LENGTH \
  --max_new_tokens $MAX_NEW_TOKENS \
  --stage 1 \
  --accelerate "fast_dllm"


python tools/remove_marker.py -i "$OUTPUT_PATH/text" -o "$OUTPUT_PATH/text_clean"

python tools/compute-wer.py --char=1 --v=1 \
"/path/to/datasets/testset/${SUBSET}/text" \
"$OUTPUT_PATH/text_clean" > "$OUTPUT_PATH/wer"
tail "$OUTPUT_PATH/wer"