from .base import VoiceAssistant

import torch, math
from transformers import MoshiForConditionalGeneration, AutoFeatureExtractor, AutoTokenizer
import librosa


class MoshiAssistant(VoiceAssistant):
    def __init__(self):
        self.device = 'cuda'
        self.dtype = torch.float16
        self.model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko", device_map=self.device, torch_dtype=self.dtype,
                                                              cache_dir='./cache')
        self.tokenizer = AutoTokenizer.from_pretrained('kmhf/hf-moshiko')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('kmhf/hf-moshiko')

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        assert audio['sampling_rate'] == 16000
        audio = audio['array']
        audio = librosa.resample(audio, orig_sr=16000, target_sr=self.feature_extractor.sampling_rate)

        user_input_values = self.feature_extractor(raw_audio=audio, sampling_rate=self.feature_extractor.sampling_rate,
                                              return_tensors="pt").to(device=self.device, dtype=self.dtype)

        # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
        moshi_input_values = torch.zeros_like(user_input_values.input_values)

        ratio = self.model.config.audio_encoder_config.frame_rate / self.model.config.sampling_rate

        # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
        num_tokens = math.ceil(moshi_input_values.shape[-1] * ratio)
        input_ids = torch.ones((1, num_tokens), device=self.device, dtype=torch.int64) * self.tokenizer.encode("<pad>")[0]

        output = self.model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values,
                                moshi_input_values=moshi_input_values, max_new_tokens=max_new_tokens, return_audio_waveforms=False)

        text_tokens = output.cpu().numpy()

        response = self.tokenizer.batch_decode(text_tokens, skip_special_tokens=True)[0]

        return response
