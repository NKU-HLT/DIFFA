from .base import VoiceAssistant
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers
import torch
from src.api.gpt import generate_text_chat
from openai import OpenAI

class Naive2Assistant(VoiceAssistant):
    def __init__(self):
        self.asr = self.load_asr()
        self.client = OpenAI()

    def load_asr(self):
        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, cache_dir='./cache'
        )
        model.to("cuda:0")

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda:0",
        )
        return pipe

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        transcript = self.asr(audio, generate_kwargs={"language": "english", 'return_timestamps': True})[
            'text'].strip()

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant who tries to help answer the user's question. Please note that the user's query is transcribed from speech, and the transcription may contain errors."},
            {"role": "user", "content": transcript},
        ]
        response = generate_text_chat(
            client=self.client,
            model='gpt-4o',
            messages=messages,
            max_tokens=max_new_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            n=1
        ).choices[0].message.content.strip()
        return response

    def generate_text(
        self,
        text,
    ):
        messages = [
            {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
            {"role": "user", "content": text},
        ]

        response = generate_text_chat(
            client=self.client,
            model='gpt-4o',
            messages=messages,
            max_tokens=2048,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            n=1
        ).choices[0].message.content.strip()
        return response

