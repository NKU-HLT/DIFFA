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
import os


def load_my_model(model_path, whisper_path, llm_path):
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    model = DIFFAModel.from_pretrained(model_path).cuda()
    model.stage = 2
    model.whisper_model = WhisperEncoder.from_pretrained(whisper_path).cuda()
    model.llm_model = AutoModel.from_pretrained(llm_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return model, tokenizer, feature_extractor


def generate_response(model, tokenizer, feature_extractor, audio_array, sampling_rate, instruction, args):
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio_array = resampler(torch.tensor(audio_array).unsqueeze(0)).squeeze(0)
    else:
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


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./exp/checkpoint-30270/checkpoint-30270")
    parser.add_argument('--llm_path', type=str, default="/path/to/models/LLaDA-8B-Instruct")
    parser.add_argument('--whisper_path', type=str, default="/path/to/models/whisper-small")
    parser.add_argument('--data', type=str, default='mmsu')
    parser.add_argument('--split', type=str, default='history')
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--block_length', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default="./exp/checkpoint-30270/voicebench_res")
    args = parser.parse_args()


    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load dataset
    dataset = load_dataset('/path/to/datasets/voicebench', args.data, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Load model components
    model, tokenizer, feature_extractor = load_my_model(args.model_path, args.whisper_path, args.llm_path)

    # Inference
    results = []
    for item in tqdm(dataset, total=len(dataset)):
        tmp = {k: v for k, v in item.items() if k != 'audio'}

        prompt = """
### Instruction:
Now, strictly follow my instructions: 
1. Do not repeat what you hear. 
2. Write an answer to response the question in the audio.

### Response:
"""
 
        audio = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']

        logger.info(f"Prompt: {prompt}")
        response = generate_response(model, tokenizer, feature_extractor, audio, sampling_rate, prompt, args)
        logger.info(f"Response: {response}")
        logger.info('====================================')

        tmp['response'] = response
        results.append(tmp)

    output_file = f'{args.output_dir}/myModel-{args.data}-{args.split}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
