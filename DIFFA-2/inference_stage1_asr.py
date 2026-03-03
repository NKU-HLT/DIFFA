import os
import argparse
import json
from tqdm import tqdm
from loguru import logger

from src.utils import (
    read_audio,
    load_my_model,
    load_my_model_merged_lora,
    generate_response,
)

def build_asr_prompt(instruction: str) -> str:
    return instruction


def main():
    parser = argparse.ArgumentParser(description="ASR inference with wav.scp (utils version)")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--wav_scp", type=str, required=True, help="Path to wav.scp file")
    parser.add_argument("--output_text", type=str, required=True, help="Path to output text file")


    parser.add_argument(
        "--instruction",
        type=str,
        default="Please transcribe the audio to text.",
        help="Instruction for ASR",
    )

    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--llm_path", type=str, default="/path/to/models/LLaDA-8B-Instruct")
    parser.add_argument("--whisper_path", type=str, default="/path/to/models/whisper-large-v3")
    parser.add_argument("--use_lora", action="store_true", help="Load merged LoRA weights")
    parser.add_argument(
        "--accelerate",
        type=str,
        choices=["fast_dllm", None],
        default=None,
    )
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    # Step1: 加载模型
    if args.use_lora:
        logger.info("Loading with merged LoRA from DeepSpeed checkpoint")
        model, tokenizer, feature_extractor = load_my_model_merged_lora(
            args.model_path, args.whisper_path, args.llm_path, args.accelerate, args.stage
        )
    else:
        logger.info("Loading full model from DeepSpeed checkpoint")
        model, tokenizer, feature_extractor = load_my_model(
            args.model_path, args.whisper_path, args.llm_path, args.accelerate, args.stage
        )

    # Step2: 读取 wav.scp
    with open(args.wav_scp, "r", encoding="utf-8") as f:
        wav_entries = [line.strip().split(maxsplit=1) for line in f if line.strip()]

    prompt = build_asr_prompt(args.instruction)

    # Step3: 推理并写文件
    os.makedirs(os.path.dirname(args.output_text) or ".", exist_ok=True)

    with open(args.output_text, "w", encoding="utf-8") as fout_text, \
         open(args.output_text + ".jsonl", "w", encoding="utf-8") as fout_jsonl:

        for parts in tqdm(wav_entries, desc="Processing audios"):
            if len(parts) != 2:
                bad_line = " ".join(parts)
                logger.warning(f"Bad wav.scp line (skip): {bad_line}")
                continue

            audio_id, wav_path = parts
            result = {"audio_id": audio_id}

            # 检查文件是否存在
            if not os.path.exists(wav_path):
                logger.warning(f"音频文件不存在 - {wav_path}")
                result["error"] = "File not found"

                fout_text.write(f"{audio_id} [ERROR: File not found]\n")
                fout_text.flush()

                fout_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout_jsonl.flush()
                continue

            # 音频处理
            try:
                audio_array = read_audio(wav_path)
            except Exception as e:
                logger.error(f"音频处理失败 - {wav_path}: {e}")
                result["error"] = f"Audio processing failed: {str(e)}"

                fout_text.write(f"{audio_id} [ERROR: Audio processing failed]\n")
                fout_text.flush()

                fout_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout_jsonl.flush()
                continue

            # 模型推理
            try:
                pred = generate_response(
                    model,
                    tokenizer,
                    feature_extractor,
                    audio_array,
                    prompt,
                    args,
                )
                text = pred.strip() if isinstance(pred, str) else str(pred).strip()
                result["text"] = text

                print(f"{audio_id} {text}\n")

                fout_text.write(f"{audio_id} {text}\n")
                fout_text.flush()

            except Exception as e:
                logger.error(f"推理失败 - {audio_id}: {e}")
                result["error"] = f"Inference failed: {str(e)}"

                fout_text.write(f"{audio_id} [ERROR: Inference failed]\n")
                fout_text.flush()

            fout_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout_jsonl.flush()


if __name__ == "__main__":
    main()
