from .base import VoiceAssistant
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers
import torch


class NaiveAssistant(VoiceAssistant):
    def __init__(self):
        self.asr = self.load_asr()
        self.llm = self.load_llm()

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

    def load_llm(self):
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
            # cache_dir='./cache',
        )
        return pipeline

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
        outputs = self.llm(
            messages,
            max_new_tokens=max_new_tokens,
        )
        response = outputs[0]["generated_text"][-1]['content']
        return response

    def generate_text(
        self,
        text,
    ):
        messages = [
            {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
            {"role": "user", "content": text},
        ]

        outputs = self.llm(
            messages,
            max_new_tokens=2048,
        )
        response = outputs[0]["generated_text"][-1]['content']
        return response




