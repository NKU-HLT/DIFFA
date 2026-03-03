from argparse import ArgumentParser
import json
from tqdm import tqdm
from loguru import logger
from collections import OrderedDict
import string
import os
import torch
from loguru import logger

from src.utils import read_audio,load_my_model,extract_and_load_lora_weights,load_my_model_merged_lora,generate_response

def build_prompt(question, choices):
    """构建多选题prompt"""
    assert len(choices) <= 26, "Too many choices (max 26 supported: A-Z)"
    letters = string.ascii_uppercase
    
    choice_lines = "\n".join(
        f"{letters[i]}. {choice}" for i, choice in enumerate(choices)
    )
    
    prompt = f"""You are given the audio and a multiple-choice question. Choose the best answer based only on the speech content and style.

Question: {question}

Choices:
{choice_lines}
Answer:"""
    
    return prompt

import re


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./exp/checkpoint-30270",
                       help="Path to DIFFA checkpoint (with pytorch.bin)")
    parser.add_argument('--llm_path', type=str, default="/path/to/models/LLaDA-8B-Instruct",
                       help="Path to base LLM model")
    parser.add_argument('--whisper_path', type=str, default="/path/to/models/whisper-small",
                       help="Path to Whisper model")
    
    # LoRA相关参数
    parser.add_argument('--use_lora', action='store_true',
                       help="Extract and use LoRA weights from checkpoint")
    
    # 数据相关参数
    parser.add_argument('--input_json_path', type=str, 
                       default='/path/to/project/lla/data/mmau_test_mini.json')
    parser.add_argument('--output_json_path', type=str, required=True)
    
    # 生成参数
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--block_length', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=16)
    parser.add_argument(
                        '--accelerate',
                        type=str,
                        choices=['fast_dllm', None],
                        default=None,
                        )

    
    args = parser.parse_args()

    # 加载模型
    if args.use_lora:
        logger.info("Loading with merged LoRA from DeepSpeed checkpoint")
        model, tokenizer, feature_extractor = load_my_model_merged_lora(
            args.model_path, 
            args.whisper_path, 
            args.llm_path,
            args.accelerate
        )
    else:
        logger.info("Loading full model from DeepSpeed checkpoint")
        model, tokenizer, feature_extractor = load_my_model(
            args.model_path, 
            args.whisper_path, 
            args.llm_path,
            args.accelerate
        )

    # 加载测试数据
    logger.info(f"Loading test data from: {args.input_json_path}")
    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Total samples: {len(data)}")

    # 推理
    for item in tqdm(data, desc="Processing"):
        wav_path = item['audio_id']
        audio_array = read_audio(wav_path)
        question = item['question']
        choices = item['choices']
        prompt = build_prompt(question, choices)
        
        
        try:
            pred = generate_response(model, tokenizer, feature_extractor, audio_array, prompt, args)
            item['model_prediction'] = pred
        except Exception as e:
            logger.error(f"Error processing {wav_path}: {e}")
            item['model_prediction'] = "ERROR"

        print("=====")
        print(prompt)
        print(pred)    

    # 保存结果
    logger.info(f"Saving results to: {args.output_json_path}")
    os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
    with open(args.output_json_path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
    
    logger.info("Inference completed successfully!")


if __name__ == '__main__':
    main()