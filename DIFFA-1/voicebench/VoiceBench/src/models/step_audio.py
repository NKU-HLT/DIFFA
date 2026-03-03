from .base import VoiceAssistant
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.models.src_step_audio.tokenizer import StepAudioTokenizer
from src.models.src_step_audio.utils import load_audio
from huggingface_hub import snapshot_download
import os
import tempfile
import soundfile as sf


class StepAssistant(VoiceAssistant):
    def __init__(self):
        if not os.path.exists("./cache/Step-Audio-Tokenizer"):
            snapshot_download(
                repo_id="stepfun-ai/Step-Audio-Tokenizer",
                local_dir="./cache/Step-Audio-Tokenizer",
            )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            'stepfun-ai/Step-Audio-Chat', trust_remote_code=True
        )
        self.encoder = StepAudioTokenizer("./cache/Step-Audio-Tokenizer")

        self.llm = AutoModelForCausalLM.from_pretrained(
            'stepfun-ai/Step-Audio-Chat',
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            # Write the audio data to the file
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        messages = [{"role": "user", "content": {"type": "audio", "audio": temp_filename}}]
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(text_with_audio, return_tensors="pt")
        outputs = self.llm.generate(
            token_ids, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1]: -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        return output_text

    def encode_audio(self, audio_path):
        audio_wav, sr = load_audio(audio_path)
        audio_tokens = self.encoder(audio_wav, sr)
        return audio_tokens

    def apply_chat_template(self, messages: list):
        text_with_audio = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio += f"<|BOT|>{role}\n{content}<|EOT|>"
            elif isinstance(content, dict):
                if content["type"] == "text":
                    text_with_audio += f"<|BOT|>{role}\n{content['text']}<|EOT|>"
                elif content["type"] == "audio":
                    audio_tokens = self.encode_audio(content["audio"])
                    text_with_audio += f"<|BOT|>{role}\n{audio_tokens}<|EOT|>"
            elif content is None:
                text_with_audio += f"<|BOT|>{role}\n"
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        if not text_with_audio.endswith("<|BOT|>assistant\n"):
            text_with_audio += "<|BOT|>assistant\n"
        return text_with_audio
