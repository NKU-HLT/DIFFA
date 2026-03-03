import os
import argparse
import json
from tqdm import tqdm
import string

import torch
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel
from loguru import logger

from src.modeling_whisper_encoder import WhisperEncoder
from src.modeling_DIFFA import DIFFAModel


def load_my_model(model_path, whisper_path, llm_path):
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    model = DIFFAModel.from_pretrained(model_path).cuda()
    model.stage = 2
    model.whisper_model = WhisperEncoder.from_pretrained(whisper_path).cuda()
    model.llm_model = AutoModel.from_pretrained(llm_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer, feature_extractor


def generate_response(model, tokenizer, feature_extractor, audio_array, instruction, args):

    audio_tensor = torch.tensor(audio_array)
    inputs = feature_extractor(audio_tensor.numpy(), sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to("cuda")

    with torch.no_grad():
        response = model.generate(
            tokenizer,
            input_audio_features=inputs.input_features,
            input_audio_mask=inputs.attention_mask,
            system_prompt="You are a helpful voice assistant.",
            input_texts=[instruction],
            steps=args.steps,
            block_length=args.block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            mask_id=126336,
            max_new_tokens=args.max_new_tokens,
        )
    return response.strip()


def read_audio(wav_path):

    audio, sr = torchaudio.load(wav_path)
    if audio.shape[0] > 1:  
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:  
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    audio = audio.squeeze(0).numpy() 
    return audio



def main():

    parser = argparse.ArgumentParser(description="Easy Inference for your model, adapted for the new benchmark.")
    
    parser.add_argument('--input_jsonl', type=str, default="/path/to/datasets/MMSU/question/mmsu.jsonl", help="Path to the input JSONL file")
    parser.add_argument('--output_jsonl', type=str, required=True, help="Path to the output JSONL file")

    parser.add_argument('--model_path', type=str, default="./exp/checkpoint-30270", help="Path to your DIFFAModel checkpoint")
    parser.add_argument('--llm_path', type=str, default="/path/to/models/LLaDA-8B-Instruct", help="Path to the base LLM model")
    parser.add_argument('--whisper_path', type=str, default="/path/to/models/whisper-small", help="Path to the Whisper model")
    
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--block_length', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=4, help="Maximum new tokens to generate for the answer")

    args = parser.parse_args()

    model, tokenizer, feature_extractor = load_my_model(args.model_path, args.whisper_path, args.llm_path)

    with open(args.input_jsonl, "r", encoding='utf-8') as fin, open(args.output_jsonl, "w", encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Processing samples"):
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
