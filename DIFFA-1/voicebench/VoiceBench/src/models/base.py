import torch
import time


class VoiceAssistant:
    @torch.no_grad()
    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_text(
        self,
        text,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_ttft(
        self,
        audio,
    ):
        tmp = time.perf_counter()
        self.generate_audio(audio, max_new_tokens=1)
        return time.perf_counter() - tmp
