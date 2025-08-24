from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from .base import VoiceAssistant
from loguru import logger
import torch

class MERaLiONAssistant(VoiceAssistant):
    def __init__(self):
        repo_id = "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION"

        self.processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=True,
            cache_dir='./cache',
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            use_safetensors=True,
            trust_remote_code=True,
            cache_dir='./cache',
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        conversation = [
            {"role": "user", "content": "<SpeechHere>"}
        ]
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(text=chat_prompt, audios=audio['array']).to('cuda')
        inputs['input_features'] = inputs['input_features'].bfloat16()
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = outputs[:, inputs['input_ids'].size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

