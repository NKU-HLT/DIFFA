from datasets import load_dataset, Audio
from argparse import ArgumentParser
import json
from tqdm import tqdm
from loguru import logger

import torch
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel

from src.modeling_whisper_encoder import WhisperEncoder
from src.modeling_DIFFA import DIFFAModel


def load_my_model(model_path, whisper_path, llm_path):
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    model = DIFFAModel.from_pretrained(model_path).cuda()
    model.stage = 2
    model.whisper_model = WhisperEncoder.from_pretrained(whisper_path).cuda()
    model.llm_model = AutoModel.from_pretrained(llm_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return model, tokenizer, feature_extractor


def generate_response(model, tokenizer, feature_extractor, audio_array, instruction, args):

    audio_array = torch.tensor(audio_array)
    inputs = feature_extractor(audio_array.numpy(), sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to("cuda")

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


import string

def build_prompt(question, choices):
    assert len(choices) <= 26, "Too many choices (max 26 supported: A-Z)"
    letters = string.ascii_uppercase  # 'A', 'B', 'C', ...
    
    choice_lines = "\n".join(
        f"{letters[i]}. {choice}" for i, choice in enumerate(choices)
    )
    
    prompt = f"""You are given the audio and a multiple-choice question. Choose the best answer based only on the speech content and style.

Question: {question}

Choices:
{choice_lines}
Answer:"""
    
    return prompt

# python inference_stage2_mmau.py --output_json_path 'xxx/mmau/res.json'
def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./exp/checkpoint-30270")
    parser.add_argument('--llm_path', type=str, default="/path/to/models/LLaDA-8B-Instruct")
    parser.add_argument('--whisper_path', type=str, default="/path/to/models/whisper-small")
    parser.add_argument('--input_json_path', type=str, default='/path/to/project/lla/data/mmau_test_mini.json')
    parser.add_argument('--output_json_path', type=str, required=True)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--block_length', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=16)
    args = parser.parse_args()



    model, tokenizer, feature_extractor = load_my_model(args.model_path, args.whisper_path, args.llm_path)

    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data):
        wav_path = item['audio_id']
        audio_array = read_audio(wav_path)
        question = item['question']
        choices = item['choices']
        prompt = build_prompt(question, choices)
        pred = generate_response(model, tokenizer, feature_extractor, audio_array, prompt, args)
        item['model_prediction'] = pred

    with open(args.output_json_path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
