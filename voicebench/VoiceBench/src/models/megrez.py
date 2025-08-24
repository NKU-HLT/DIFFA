from .base import VoiceAssistant
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers
import torch
from transformers import AutoModelForCausalLM
import io
import soundfile as sf
import base64
from loguru import logger
import numpy as np
import os
import tempfile


class MegrezAssistant(VoiceAssistant):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            'Infinigence/Megrez-3B-Omni',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            cache_dir='./cache',
        ).eval().cuda()
        self.model._get_or_init_processor()

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        messages = [
            {
                "role": "user",
                "content": {
                    "system": "You are Megrez-3B-Instruct, and you will provide detailed and constructive answers to the user's questions in English.",
                    "audio": temp_filename,
                },
            },
        ]

        try:
            response = self.model.chat(
                messages,
                sampling=False,
                max_new_tokens=max_new_tokens,
                temperature=0,
            )
        finally:
            os.remove(temp_filename)
        return response