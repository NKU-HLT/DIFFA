from .base import VoiceAssistant
import transformers


class UltravoxAssistant(VoiceAssistant):
    def __init__(self):
        self.pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_4_1-llama-3_1-8b', trust_remote_code=True, cache_dir='./cache', device='cuda')

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people."
            },
        ]
        return self.pipe({'audio': audio['array'], 'turns': turns, 'sampling_rate': audio['sampling_rate']}, max_new_tokens=max_new_tokens)


class Ultravox0d5Assistant(UltravoxAssistant):
    def __init__(self):
        self.pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_5-llama-3_1-8b',
                                          trust_remote_code=True,
                                          cache_dir='./cache',
                                          device='cuda')
