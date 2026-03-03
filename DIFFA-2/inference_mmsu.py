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



def main():

    parser = ArgumentParser(description="Easy Inference for your model, adapted for the new benchmark.")
    
    parser.add_argument('--input_jsonl', type=str, default="/path/to/datasets/MMSU/question/mmsu.jsonl", help="Path to the input JSONL file")
    parser.add_argument('--output_jsonl', type=str, required=True, help="Path to the output JSONL file")

    parser.add_argument('--model_path', type=str, default="./exp/checkpoint-30270", help="Path to your DIFFAModel checkpoint")
    parser.add_argument('--llm_path', type=str, default="/path/to/models/LLaDA-8B-Instruct", help="Path to the base LLM model")
    parser.add_argument('--whisper_path', type=str, default="/path/to/models/whisper-small", help="Path to the Whisper model")
    
    # LoRA相关参数
    parser.add_argument('--use_lora', action='store_true',
                       help="Extract and use LoRA weights from checkpoint")
    
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--block_length', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=4, help="Maximum new tokens to generate for the answer")

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
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(args.input_jsonl, "r", encoding='utf-8') as fin, open(args.output_jsonl, "w", encoding='utf-8') as fout:
        for line in tqdm(fin, total=total_lines, desc="Processing samples"):
            item = json.loads(line.strip())
            
            audio_path = item['audio_path']
            task_name = item['task_name']
            
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found, skipping: {audio_path}")
                continue
            
            question = item['question']
            question_prompts = 'Choose the most suitable answer from options A, B, C, and D to respond the question in next line, **you should only choose A or B or C or D.** Do not provide any additional explanations or content.'
            
            choices_list = []
            for key in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
                if key in item and item[key] is not None:
                    choices_list.append(item[key])
            
            letters = string.ascii_uppercase
            formatted_choices = "\n".join(
                f"{letters[i]}. {choice}" for i, choice in enumerate(choices_list)
            )

            instruction = f"{question_prompts}\n\nQuestion: {question}\n\n{formatted_choices}"

            audio_array = read_audio(audio_path)
            output = generate_response(model, tokenizer, feature_extractor, audio_array, instruction, args)
            


            result_item = {
                "id": item["id"],
                "audio_path": item["audio_path"],
                "question": question,
                "choice_a": item["choice_a"],
                "choice_b": item["choice_b"],
                "choice_c": item.get("choice_c"), 
                "choice_d": item.get("choice_d"),
                "answer_gt": item["answer_gt"],
                "response": output, 
                "task_name": task_name,
                "category": item["category"],
                "sub-category": item.get("sub-category"), 
                "sub-sub-category": item.get("sub-sub-category"),
                "linguistics_sub_discipline": item.get("linguistics_sub_discipline"),
            }
            
            fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
